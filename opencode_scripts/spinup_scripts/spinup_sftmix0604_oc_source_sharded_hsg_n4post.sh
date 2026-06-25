#!/bin/bash
# =========================================================================
#  run_sftmix0604_opencode_source_sharded.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  Same as run_ultra_v3_GA_opencode_source_sharded.sh, but points at venkats'
#  sft_mix_0604_v2 step-0000485 HF checkpoint. Runs the opencode harness
#  FROM SOURCE (via Bun) instead of the OOB precompiled binary -- so harness
#  edits (e.g. the [COMPACTION_DEBUG] logs) and the MODEL_INPUT_LIMIT compaction
#  knob take effect.
#
#  Serves the model (TP=8, 2 nodes x 4 GPUs) per shard and drives opencode.
#  Defaults to NUM_SHARDS=15. Scale/override via env, e.g.:
#    MODEL_INPUT_LIMIT=65000 NUM_SHARDS=15 bash run_sftmix0604_opencode_source_sharded.sh
# =========================================================================
set -euo pipefail

# --- CLI args (optional): pin the harness to a git commit/ref --------------
#   --commit <ref> | --pin-commit <ref> | --commit=<ref>
# Equivalent to exporting OPENCODE_PIN_COMMIT=<ref>; the CLI flag takes
# precedence over the env var. Empty (default) => run from the LIVE working
# tree (current behavior). Any other args are ignored.
PIN_COMMIT_CLI=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit|--pin-commit)   PIN_COMMIT_CLI="${2:-}"; shift 2 ;;
    --commit=*|--pin-commit=*) PIN_COMMIT_CLI="${1#*=}"; shift ;;
    *) echo "WARN: ignoring unrecognized arg: $1" >&2; shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../opencode_scripts/spinup_scripts
OC_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"             # .../opencode_scripts
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"               # .../opencode-hsg  (live harness repo)

# ============================ CONFIG (env-overridable) ===================
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/runs/checkpoints/sft_mix_0604_v2_only_192k_128n_mem900g_20260605/eval/0000485/hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sft_mix_0604_v2_step0000485}"
DATASET="${DATASET:-/lustre/fsw/portfolios/llmservice/users/abhibhag/Gym/benchmarks/browsecomp/data/browsecomp_benchmark_400.jsonl}"

NUM_SHARDS="${NUM_SHARDS:-15}"
PARALLEL="${PARALLEL:-16}"
KEY_OFFSET="${KEY_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
QOS="${QOS:-interactive}"          # valid for llmservice_modelalignment_ppo: interactive/normal (NOT nemotron-priority)
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-llmservice_modelalignment_ppo}"
JOB_PREFIX="${JOB_PREFIX:-sftmix0604_ocsrc}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# Compaction-trigger override (ABLATION KNOB). Default EMPTY = stock opencode:
# no limit.input emitted, so compaction fires at the natural context-max_output (~99072).
# Set MODEL_INPUT_LIMIT=<N> to set model.limit.input=N. compaction.reserved is NOT set,
# so opencode uses its own default buffer (min(20000, max_output)=20000); trigger = N-20000.
MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-}"
# Max-steps override (ABLATION KNOB). Default EMPTY = stock opencode: agent.steps
# unset => unbounded (agent.steps ?? Infinity in session/prompt.ts). Set MAX_STEPS=<N>
# to cap each rollout at N agent steps via agent.build.steps=N in the per-shard opencode.json.
MAX_STEPS="${MAX_STEPS:-}"
# System-prompt knob. EMPTY = stock opencode source selection (system.ts model-id routing).
# Set SYSTEM_PROMPT_FILE=<abs path to a .txt> and the opencode source override
# (OPENCODE_SYSTEM_PROMPT_FILE, read in session/system.ts) uses that file's contents
# verbatim as the system prompt for EVERY rollout, overriding all model-id routing.
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
# Compaction kill-switch (ABLATION KNOB). Default EMPTY = compaction on. Set
# DISABLE_COMPACTION=1 to emit "compaction":{"auto":false} => NO compaction ever
# fires; overflowing rollouts ERROR OUT instead. MODEL_INPUT_LIMIT is a no-op when set.
DISABLE_COMPACTION="${DISABLE_COMPACTION:-}"
# Subagent kill-switch (ABLATION KNOB). Default 0 = subagents ENABLED (stock: the
# `task` tool is present so the model can spawn explore/general subagents). Set
# DISABLE_SUBAGENTS=1 to deny the `task` tool in agent.build.permission; opencode's
# Permission.disabled() then strips `task` from the request entirely (it won't
# appear in the model's tool schema / vLLM logs and cannot be invoked).
DISABLE_SUBAGENTS="${DISABLE_SUBAGENTS:-0}"
# Overflow STRATEGY (ABLATION KNOB). Default EMPTY = stock "summarize" (LLM summary).
# Set COMPACTION_STRATEGY=discard to emit "compaction":{"strategy":"discard"} => at the
# overflow trigger the harness drops all prior context and keeps only the last
# COMPACTION_KEEP_TOOL_CALLS tool calls verbatim, with NO summary LLM call.
COMPACTION_STRATEGY="${COMPACTION_STRATEGY:-}"
# k for strategy=discard. Default EMPTY = harness default (3). Set to override.
COMPACTION_KEEP_TOOL_CALLS="${COMPACTION_KEEP_TOOL_CALLS:-}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}"
TAVILY_MAX_RESULTS="${TAVILY_MAX_RESULTS:-5}"
TAVILY_SEARCH_DEPTH="${TAVILY_SEARCH_DEPTH:-advanced}"
TAVILY_INCLUDE_RAW_CONTENT="${TAVILY_INCLUDE_RAW_CONTENT:-true}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"

# --- RUN-FROM-SOURCE config (self-contained inside opencode-hsg) ---
# Harness source, bun, node, tavily-mcp, node_modules all live in THIS repo
# (opencode-hsg = REPO_ROOT). By DEFAULT the harness runs from the LIVE working
# tree so local edits take effect. Set OPENCODE_PIN_COMMIT (or pass --commit
# <ref>) to instead freeze a specific commit into a git-worktree snapshot and
# run every shard from that frozen source -- reproducible and immune to edits
# made after submit. TOOLCHAIN (bun/node/tavily) always comes from LIVE_REPO;
# only the harness SOURCE is pinned.
# First-run prereq: cd <opencode-hsg> && bun/bin/bun install   (its deps differ).
LIVE_REPO="${LIVE_REPO:-${REPO_ROOT}}"
OPENCODE_SRC_REPO="${OPENCODE_SRC_REPO:-${LIVE_REPO}}"
OPENCODE_BIN="${OPENCODE_BIN:-${OC_SCRIPTS_DIR}/opencode-src}"
BUN_BIN="${BUN_BIN:-${LIVE_REPO}/bun/bin/bun}"
NODE_BIN="${NODE_BIN:-${LIVE_REPO}/node/node-v22.11.0-linux-arm64/bin}"
TAVILY_MCP_ENTRY="${TAVILY_MCP_ENTRY:-${LIVE_REPO}/mcp/node_modules/tavily-mcp/build/index.js}"

# --- PINNED-COMMIT config (opt-in; empty => run from the live working tree) ---
# Resolved from the --commit/--pin-commit CLI flag if given, else the env var.
OPENCODE_PIN_COMMIT="${OPENCODE_PIN_COMMIT:-${PIN_COMMIT_CLI}}"
# Where frozen snapshots live (one git worktree per SHA, shared across runs).
OPENCODE_SNAPSHOT_BASE="${OPENCODE_SNAPSHOT_BASE:-${OC_SCRIPTS_DIR}/runs/_src_snapshots}"

DRIVER_PY="${DRIVER_PY:-${OC_SCRIPTS_DIR}/analysis_scripts/launch_opencode_evals.py}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"

GRADE_SCRIPT="${GRADE_SCRIPT:-${OC_SCRIPTS_DIR}/analysis_scripts/grade_with_glm5.sh}"
RUN_GRADING="${RUN_GRADING:-true}"
EXCLUDE_DOMAINS_FILE="${EXCLUDE_DOMAINS_FILE:-/lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json}"
TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"
WORK_BASE="${WORK_BASE:-${OC_SCRIPTS_DIR}/runs}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/dmosallanezh/containers/vllm-hsg-0.17.0.sqsh}"

RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# =========================================================================

# --- resolve pinned snapshot (SUBMIT TIME), only when a commit is requested ---
if [[ -n "${OPENCODE_PIN_COMMIT}" ]]; then
    command -v git >/dev/null 2>&1 || { echo "ERROR: git not on PATH (needed to pin a commit)" >&2; exit 1; }
    git -C "${LIVE_REPO}" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: LIVE_REPO is not a git repo: ${LIVE_REPO}" >&2; exit 1; }
    PIN_SHA="$(git -C "${LIVE_REPO}" rev-parse "${OPENCODE_PIN_COMMIT}")" || { echo "ERROR: cannot resolve commit '${OPENCODE_PIN_COMMIT}' in ${LIVE_REPO}" >&2; exit 1; }
    SNAPSHOT_DIR="${OPENCODE_SNAPSHOT_BASE}/${PIN_SHA}"
    if [[ -d "${SNAPSHOT_DIR}/packages/opencode/src" && -e "${SNAPSHOT_DIR}/node_modules" ]]; then
        echo "PIN: reusing existing snapshot ${SNAPSHOT_DIR} (commit ${PIN_SHA:0:9})"
    else
        # A half-built snapshot (tracked files present but node_modules symlinks
        # missing, e.g. an interrupted prior build) must NOT be reused -- rebuild it.
        [[ -d "${SNAPSHOT_DIR}/packages/opencode/src" && ! -e "${SNAPSHOT_DIR}/node_modules" ]] && {
            echo "PIN: snapshot ${PIN_SHA:0:9} is incomplete (no node_modules) -> rebuilding"
            git -C "${LIVE_REPO}" worktree remove --force "${SNAPSHOT_DIR}" 2>/dev/null || rm -rf "${SNAPSHOT_DIR}"
        }
        echo "PIN: creating snapshot for commit ${PIN_SHA:0:9} -> ${SNAPSHOT_DIR}"
        mkdir -p "${OPENCODE_SNAPSHOT_BASE}"
        # Detached worktree = clean checkout of the commit's tracked files.
        git -C "${LIVE_REPO}" worktree add --detach "${SNAPSHOT_DIR}" "${PIN_SHA}"
        # Symlink gitignored runtime deps so Bun resolves imports without re-install.
        ln -sfn "${LIVE_REPO}/node_modules" "${SNAPSHOT_DIR}/node_modules"
        while IFS= read -r pkg_nm; do
            [[ -d "${LIVE_REPO}/${pkg_nm}" ]] || continue
            mkdir -p "${SNAPSHOT_DIR}/$(dirname "${pkg_nm}")"
            ln -sfn "${LIVE_REPO}/${pkg_nm}" "${SNAPSHOT_DIR}/${pkg_nm}"
        done < <(cd "${LIVE_REPO}" && ls -d packages/*/node_modules 2>/dev/null)
        echo "PIN: snapshot ready (node_modules symlinked from live repo)"
    fi
    # Point the harness SOURCE at the frozen snapshot; toolchain stays in LIVE_REPO.
    OPENCODE_SRC_REPO="${SNAPSHOT_DIR}"
    PIN_LABEL="${PIN_SHA}"
    # If SYSTEM_PROMPT_FILE lives inside the live repo, remap it into the snapshot
    # so the frozen commit's prompt is used too (launchers point it at LIVE_REPO).
    if [[ -n "${SYSTEM_PROMPT_FILE:-}" && "${SYSTEM_PROMPT_FILE}" == "${LIVE_REPO}/"* ]]; then
        REL_PROMPT="${SYSTEM_PROMPT_FILE#${LIVE_REPO}/}"
        if [[ -f "${SNAPSHOT_DIR}/${REL_PROMPT}" ]]; then
            echo "PIN: remapping SYSTEM_PROMPT_FILE into snapshot (${REL_PROMPT})"
            SYSTEM_PROMPT_FILE="${SNAPSHOT_DIR}/${REL_PROMPT}"
        else
            echo "WARN: SYSTEM_PROMPT_FILE '${REL_PROMPT}' not present at commit ${PIN_SHA:0:9}; using live path (NOT pinned)" >&2
        fi
    fi
else
    PIN_LABEL="<live working tree>"
fi
# Entry follows OPENCODE_SRC_REPO (live or pinned); honors an explicit override.
OPENCODE_SRC_ENTRY="${OPENCODE_SRC_ENTRY:-${OPENCODE_SRC_REPO}/packages/opencode/src/index.ts}"
# =========================================================================

# --- sanity checks ---
[[ -f "${DATASET}" ]]      || { echo "ERROR: DATASET not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DRIVER_PY}" ]]    || { echo "ERROR: DRIVER_PY not found: ${DRIVER_PY}" >&2; exit 1; }
[[ -x "${VENV_PY}" ]]      || { echo "ERROR: VENV_PY not found: ${VENV_PY}" >&2; exit 1; }
[[ -f "${TAVILY_KEYS_FILE}" ]] || { echo "ERROR: TAVILY_KEYS_FILE not found: ${TAVILY_KEYS_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "ERROR: SYSTEM_PROMPT_FILE set but not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }
[[ -x "${NODE_BIN}/node" ]] || { echo "ERROR: node not found under NODE_BIN: ${NODE_BIN} (needed for tavily-mcp)" >&2; exit 1; }
[[ -f "${TAVILY_MCP_ENTRY}" ]] || { echo "ERROR: TAVILY_MCP_ENTRY not found: ${TAVILY_MCP_ENTRY} (pre-install: npm install tavily-mcp@latest)" >&2; exit 1; }
# --- run-from-source sanity checks ---
[[ -x "${BUN_BIN}" ]]      || { echo "ERROR: BUN_BIN not found/executable: ${BUN_BIN}" >&2; exit 1; }
[[ -f "${OPENCODE_SRC_ENTRY}" ]] || { echo "ERROR: opencode source entry not found: ${OPENCODE_SRC_ENTRY}" >&2; exit 1; }
[[ -x "${OPENCODE_BIN}" ]] || { echo "ERROR: opencode-src wrapper not found/executable: ${OPENCODE_BIN}" >&2; exit 1; }
[[ -d "${OPENCODE_SRC_REPO}/node_modules" ]] || { echo "ERROR: deps not installed: ${OPENCODE_SRC_REPO}/node_modules missing (run: cd ${OPENCODE_SRC_REPO} && ${BUN_BIN} install)" >&2; exit 1; }

# --- load tavily key pool ---
mapfile -t TAVILY_KEYS < <(grep -oE 'tvly-[A-Za-z0-9_-]+' "${TAVILY_KEYS_FILE}")
NKEYS=${#TAVILY_KEYS[@]}
(( NKEYS > 0 )) || { echo "ERROR: no tavily keys parsed from ${TAVILY_KEYS_FILE}" >&2; exit 1; }
(( NKEYS >= NUM_SHARDS )) || echo "WARN: only ${NKEYS} keys for ${NUM_SHARDS} shards -> keys will be reused" >&2

# --- run dir ---
RESUMING=0
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    [[ -d "${RESUME_RUN_DIR}" ]] || { echo "ERROR: RESUME_RUN_DIR not found: ${RESUME_RUN_DIR}" >&2; exit 1; }
    RUN_DIR="${RESUME_RUN_DIR%/}"
    RESUMING=1
else
    TS="$(date +%Y%m%d_%H%M%S)"
    SLUG="${SERVED_MODEL_NAME//\//__}"
    MODEL_GROUP="${MODEL_GROUP:-$(echo "${SERVED_MODEL_NAME}" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')}"
    RUN_DIR="${WORK_BASE}/sharded_source/${MODEL_GROUP}/${TS}_${SLUG}"
fi
mkdir -p "${RUN_DIR}"/{shards,shard_scripts,slurm_logs,vllm_logs}

# --- shard the dataset ---
EXISTING_SHARDS=$(find "${RUN_DIR}/shards" -maxdepth 1 -name 'shard*.jsonl' 2>/dev/null | wc -l)
if (( RESUMING == 1 )) && (( EXISTING_SHARDS > 0 )); then
    NUM_SHARDS=${EXISTING_SHARDS}
    echo "RESUME: reusing ${RUN_DIR} with ${NUM_SHARDS} existing shards (no re-shard);"
    echo "        driver will skip already-finished samples and run the rest."
else
    (( RESUMING == 1 )) && echo "RESUME: ${RUN_DIR} has no shard files yet -> sharding now."
    python3 - "${DATASET}" "${RUN_DIR}/shards" "${NUM_SHARDS}" <<'PY'
import json,sys,os
ds,outdir,ns=sys.argv[1],sys.argv[2],int(sys.argv[3])
os.makedirs(outdir,exist_ok=True)
recs=[l for l in open(ds) if l.strip()]
outs=[open(os.path.join(outdir,f"shard{i}.jsonl"),"w") for i in range(ns)]
counts=[0]*ns
for idx,line in enumerate(recs):
    rec=json.loads(line)
    rec["_sample_id"]=idx
    s=idx % ns
    outs[s].write(json.dumps(rec)+"\n"); counts[s]+=1
for o in outs: o.close()
print(f"sharded {len(recs)} records into {ns} shards (strided): {counts}")
PY
fi

echo "============================================"
echo "  Nemotron-Ultra opencode rollout [FROM SOURCE]$( ((RESUMING==1)) && echo '  [RESUME]' )"
echo "  Dataset:   ${DATASET}"
echo "  Shards:    ${NUM_SHARDS}  (x2 nodes = $((NUM_SHARDS*2)) nodes, TP=8)"
echo "  Parallel:  ${PARALLEL} rollouts/shard"
echo "  Keys:      ${NKEYS} (offset ${KEY_OFFSET})"
echo "  Retries:   ${MAX_RETRIES}"
echo "  InputLimit:${MODEL_INPUT_LIMIT:-<stock>} (compaction trigger = limit-20000 when set)"
echo "  Harness:   ${PIN_LABEL}"
echo "  Source:    ${OPENCODE_SRC_ENTRY}"
echo "  Bun:       ${BUN_BIN}  (from live repo)"
echo "  Wrapper:   ${OPENCODE_BIN}"
echo "  Run dir:   ${RUN_DIR}"
echo "============================================"

ALL_SHARD_JIDS=""
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    KEY_IDX=$(( (SHARD_ID + KEY_OFFSET) % NKEYS ))
    TAVILY_KEY="${TAVILY_KEYS[$KEY_IDX]}"
    SHARD_DIR="${RUN_DIR}/shard${SHARD_ID}"
    SHARD_JSONL="${RUN_DIR}/shards/shard${SHARD_ID}.jsonl"
    SCRIPT_FILE="${RUN_DIR}/shard_scripts/shard${SHARD_ID}.sh"
    mkdir -p "${SHARD_DIR}"

    cat > "${SCRIPT_FILE}" <<'SHARD_EOF'
#!/bin/bash
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

echo "JOB ${SLURM_JOB_ID}  shard ${SHARD_ID}/${NUM_SHARDS}  key ...${TAVILY_KEY: -6}"
MOUNTS="/lustre:/lustre"
CONTAINER_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS}"
VLLM_LOG="${RUN_DIR}/vllm_logs/vllm_shard${SHARD_ID}_${SLURM_JOB_ID}.log"

NUM_NODES=$SLURM_JOB_NUM_NODES
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

# --- Ray worker(s) ---
for ((i=1; i<NUM_NODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" ${CONTAINER_ARGS} \
        -o "${RUN_DIR}/vllm_logs/worker_shard${SHARD_ID}_${SLURM_JOB_ID}_${i}.log" \
        bash -c "
            export FLASHINFER_WORKSPACE_BASE=/tmp
            export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --block
        " &
done

# --- Ray head + vLLM serve ---
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" ${CONTAINER_ARGS} -o "${VLLM_LOG}" bash -c "
    set -euo pipefail
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL}
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
    EXPECTED=${NUM_NODES}
    for attempt in \$(seq 1 120); do
        COUNT=\$(ray status 2>/dev/null | grep -c 'node_' || true); COUNT=\${COUNT:-0}
        if [ \"\${COUNT}\" -ge \"\${EXPECTED}\" ] 2>/dev/null; then echo 'All workers connected'; break; fi
        sleep 10
    done
    ray status
    echo 'Starting vLLM serve...'
    vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL_NAME} \
        --tensor-parallel-size 8 \
        --trust-remote-code \
        --dtype bfloat16 \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.95 \
        --enable-expert-parallel \
        --mamba-ssm-cache-dtype float32 \
        --distributed-executor-backend ray \
        --no-enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser deepseek_r1 \
        --enable-log-requests \
        --compilation-config '{\"pass_config\": {\"fuse_allreduce_rms\": false}}' \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs 32 \
        --chat-template ${MODEL_PATH}/chat_template.jinja \
        --host 0.0.0.0 \
        --port ${SERVER_PORT}
" &

# --- wait for vLLM readiness ---
MODEL_BASE_URL="http://${HEAD_NODE_IP}:${SERVER_PORT}/v1"
echo "Waiting for vLLM at ${MODEL_BASE_URL} ..."
for i in $(seq 1 60); do
    if curl -s "${MODEL_BASE_URL}/models" >/dev/null 2>&1; then echo "vLLM ready"; break; fi
    if [[ $i -eq 60 ]]; then echo "ERROR: vLLM not ready within 30 min" >&2; exit 1; fi
    sleep 30
done

# --- per-shard opencode.json ---
# Ablation knob: empty MODEL_INPUT_LIMIT => stock opencode (no limit.input). Set it to
# inject model.limit.input=N only; opencode triggers compaction at (N - its own default
# reserved buffer ~20000). compaction.reserved is intentionally NOT set here.
if [[ -n "${MODEL_INPUT_LIMIT}" && "${MODEL_INPUT_LIMIT}" != "0" ]]; then
    INPUT_LIMIT_FRAG=", \"input\": ${MODEL_INPUT_LIMIT}"
else
    INPUT_LIMIT_FRAG=""
fi
# Empty MAX_STEPS => stock opencode (no agent.steps => unbounded). Set it to cap
# each rollout at N agent steps via agent.build.steps=N.
if [[ -n "${MAX_STEPS}" && "${MAX_STEPS}" != "0" ]]; then
    STEPS_FRAG="\"steps\": ${MAX_STEPS}, "
else
    STEPS_FRAG=""
fi
# Build the top-level "compaction" object from the ablation knobs:
#   DISABLE_COMPACTION=1        => "auto": false  (overflow.ts:isOverflow returns
#                                  false; NO compaction ever fires; overflowing
#                                  rollouts ERROR OUT). Mutually exclusive with the
#                                  discard knobs below (auto:false wins).
#   COMPACTION_STRATEGY=discard => "strategy": "discard"  (at the overflow trigger,
#                                  drop all prior context and keep only the last
#                                  keep_tool_calls tool calls verbatim; NO summary
#                                  LLM call). Default/empty => stock "summarize".
#   COMPACTION_KEEP_TOOL_CALLS=N => "keep_tool_calls": N  (k for the discard tail).
COMPACTION_FIELDS=""
if [[ "${DISABLE_COMPACTION:-}" == "1" ]]; then
    COMPACTION_FIELDS="\"auto\": false"
fi
if [[ -n "${COMPACTION_STRATEGY:-}" ]]; then
    COMPACTION_FIELDS="${COMPACTION_FIELDS:+${COMPACTION_FIELDS}, }\"strategy\": \"${COMPACTION_STRATEGY}\""
fi
if [[ -n "${COMPACTION_KEEP_TOOL_CALLS:-}" ]]; then
    COMPACTION_FIELDS="${COMPACTION_FIELDS:+${COMPACTION_FIELDS}, }\"keep_tool_calls\": ${COMPACTION_KEEP_TOOL_CALLS}"
fi
if [[ -n "${COMPACTION_FIELDS}" ]]; then
    COMPACTION_FRAG="\"compaction\": { ${COMPACTION_FIELDS} },"
else
    COMPACTION_FRAG=""
fi
# DISABLE_SUBAGENTS=1 => deny the `task` tool so the model cannot spawn subagents.
# Emitted as a permission rule in agent.build.permission; opencode strips denied
# tools (pattern "*", action "deny") from the request schema entirely.
if [[ "${DISABLE_SUBAGENTS:-0}" == "1" ]]; then
    TASK_PERM_FRAG=$'\n        "task": "deny",'
else
    TASK_PERM_FRAG=""
fi
cat > "${SHARD_DIR}/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  ${COMPACTION_FRAG}
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Ultra (vLLM)",
      "options": { "baseURL": "${MODEL_BASE_URL}", "apiKey": "${MODEL_API_KEY}" },
      "models": {
        "${SERVED_MODEL_NAME}": { "name": "Nemotron Ultra (local)", "limit": { "context": ${MAX_MODEL_LEN}${INPUT_LIMIT_FRAG}, "output": 32768 } }
      }
    }
  },
  "mcp": {
    "tavily": {
      "type": "local",
      "command": ["${NODE_BIN}/node", "${TAVILY_MCP_ENTRY}"],
      "enabled": true,
      "environment": {
        "TAVILY_API_KEY": "${TAVILY_KEY}",
        "DEFAULT_PARAMETERS": "{\"search_depth\":\"${TAVILY_SEARCH_DEPTH}\",\"include_raw_content\":${TAVILY_INCLUDE_RAW_CONTENT},\"max_results\":${TAVILY_MAX_RESULTS}}"
      }
    }
  },
  "agent": {
    "build": {
      ${STEPS_FRAG}"permission": {
        "*": "allow",${TASK_PERM_FRAG}
        "websearch": "deny",
        "webfetch": "deny"
      }
    }
  }
}
JSON

echo "Wrote ${SHARD_DIR}/opencode.json"
# Put bun, node/npx (for tavily-mcp), and the opencode-src wrapper on PATH.
export PATH="$(dirname "${BUN_BIN}"):${NODE_BIN}:$(dirname "${OPENCODE_BIN}"):${PATH}"
echo "bun: $(command -v bun || echo MISSING)  node: $(command -v node || echo MISSING)"
echo "opencode-src -> $(${OPENCODE_BIN} --version 2>/dev/null || echo MISSING)"

# --- System prompt: resolve, export (only when overridden), and DUMP the full
# prompt to this shard's log so every run records exactly which prompt it used. ---
if [[ -n "${SYSTEM_PROMPT_FILE}" ]]; then
    export OPENCODE_SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    EFFECTIVE_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    PROMPT_SOURCE="launcher override (OPENCODE_SYSTEM_PROMPT_FILE)"
else
    EFFECTIVE_PROMPT_FILE=""
    PROMPT_SOURCE="stock source default (system.ts model-id routing)"
fi
echo "================= SYSTEM PROMPT ================="
echo "source: ${PROMPT_SOURCE}"
echo "file:   ${EFFECTIVE_PROMPT_FILE:-<resolved inside opencode source>}"
if [[ -n "${EFFECTIVE_PROMPT_FILE}" && -f "${EFFECTIVE_PROMPT_FILE}" ]]; then
    echo "size:   $(wc -c < "${EFFECTIVE_PROMPT_FILE}") chars, $(wc -l < "${EFFECTIVE_PROMPT_FILE}") lines"
    echo "---------------- begin prompt ------------------"
    cat "${EFFECTIVE_PROMPT_FILE}"
    echo "----------------- end prompt -------------------"
fi
echo "================================================"

# --- drive opencode (FROM SOURCE) over this shard's questions ---
"${VENV_PY}" "${DRIVER_PY}" \
    --shard-jsonl  "${SHARD_JSONL}" \
    --shard-dir    "${SHARD_DIR}" \
    --shard-id     "${SHARD_ID}" \
    --run-dir      "${RUN_DIR}" \
    --opencode-bin "${OPENCODE_BIN}" \
    --model        "local/${SERVED_MODEL_NAME}" \
    --agent        "${AGENT:-build}" \
    --parallel     "${PARALLEL}" \
    --tavily-search-depth    "${TAVILY_SEARCH_DEPTH:-advanced}" \
    --tavily-max-results     "${TAVILY_MAX_RESULTS:-5}" \
    --tavily-include-raw-content "${TAVILY_INCLUDE_RAW_CONTENT:-true}" \
    --tavily-exclude-domains-file "${EXCLUDE_DOMAINS_FILE}" \
    |& tee "${RUN_DIR}/slurm_logs/driver_shard${SHARD_ID}_${SLURM_JOB_ID}.log"

echo "shard ${SHARD_ID} done -> ${SHARD_DIR}/trajectories_shard${SHARD_ID}.jsonl"
SHARD_EOF

    export SHARD_ID NUM_SHARDS SHARD_DIR SHARD_JSONL TAVILY_KEY RUN_DIR
    export MODEL_PATH SERVED_MODEL_NAME MODEL_API_KEY MAX_MODEL_LEN MODEL_INPUT_LIMIT MAX_STEPS VLLM_LOG_LEVEL
    export CONTAINER SERVER_PORT RAY_PORT EXCLUDE_DOMAINS_FILE SYSTEM_PROMPT_FILE DISABLE_COMPACTION
    export COMPACTION_STRATEGY COMPACTION_KEEP_TOOL_CALLS DISABLE_SUBAGENTS
    export TAVILY_MAX_RESULTS TAVILY_SEARCH_DEPTH TAVILY_INCLUDE_RAW_CONTENT
    export OPENCODE_BIN NODE_BIN TAVILY_MCP_ENTRY DRIVER_PY VENV_PY PARALLEL AGENT
    export BUN_BIN OPENCODE_SRC_REPO OPENCODE_SRC_ENTRY

    # Optional: gate the whole run behind another run's jobs. DEPENDENCY=<id[:id...]>
    # makes each shard's INITIAL job wait until those jobs terminate (any state).
    DEP_FLAG=""
    [[ -n "${DEPENDENCY:-}" ]] && DEP_FLAG="--dependency=afterany:${DEPENDENCY}"
    JID=$(sbatch --parsable ${DEP_FLAG} \
        -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" \
        --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
        --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
        --export=ALL "${SCRIPT_FILE}")
    echo "shard ${SHARD_ID}: submitted job ${JID} (key idx ${KEY_IDX}, ...${TAVILY_KEY: -6}${DEPENDENCY:+, afterany:${DEPENDENCY}})"

    for RETRY in $(seq 1 "${MAX_RETRIES}"); do
        JID=$(sbatch --parsable --dependency=afternotok:${JID} \
            -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" \
            --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
            --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
            --export=ALL "${SCRIPT_FILE}")
        echo "  retry ${RETRY}: job ${JID} (afternotok)"
    done
    ALL_SHARD_JIDS="${ALL_SHARD_JIDS:+${ALL_SHARD_JIDS} }${JID}"
done

echo ""
echo "All ${NUM_SHARDS} shard(s) submitted ($((NUM_SHARDS*2)) nodes)."
echo "Run dir: ${RUN_DIR}"

# --- Submit grading job after all shards finish ---
if [[ "${RUN_GRADING}" == "true" ]] && [[ -f "${GRADE_SCRIPT}" ]]; then
    GRADE_DEPS=$(echo "${ALL_SHARD_JIDS}" | tr ' ' ':')
    GRADE_JID=$(TRAJECTORIES="${RUN_DIR}/trajectories.jsonl" \
                OUTPUT="${RUN_DIR}/trajectories_graded_GLM-5-FP8.jsonl" \
                TOTAL_SAMPLES="${NUM_SHARDS_TOTAL_SAMPLES:-400}" \
                sbatch --parsable \
                    --dependency="afterany:${GRADE_DEPS}" \
                    --output "${RUN_DIR}/slurm_logs/%j_grade.out" \
                    "${GRADE_SCRIPT}" 2>/dev/null || true)
    if [[ -n "${GRADE_JID:-}" ]]; then
        echo "Grading job: ${GRADE_JID} (afterany all shards; output: trajectories_graded_GLM-5-FP8.jsonl)"
    else
        echo "WARN: failed to submit grading job (check GRADE_SCRIPT=${GRADE_SCRIPT})"
    fi
else
    echo "Grading: skipped (RUN_GRADING=${RUN_GRADING})"
fi
