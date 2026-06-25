#!/bin/bash
# =========================================================================
#  run_glm51_opencode_pinned_sharded.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  PROTOTYPE: same as run_glm51_opencode_source_sharded.sh, but instead of running
#  opencode straight from the LIVE working tree (where any edit after submit can
#  change an in-flight job), this PINS the harness to a specific git commit and
#  runs every shard from a frozen snapshot of that commit.
#
#  WHY: the source-sharded twin sets OPENCODE_SRC_REPO to the live repo, so jobs
#  read packages/opencode/src/** as it exists on disk WHEN THE JOB STARTS -- no
#  commit, no snapshot. This removes that race for reproducible ablations.
#
#  HOW (all at submit time, in this launcher -- the shard scripts are unchanged):
#    1. Resolve OPENCODE_PIN_COMMIT (default: current HEAD of the live repo).
#    2. `git worktree add --detach <snapshot>/<sha>` -> clean checkout of that
#       commit's TRACKED files (idempotent + shared across runs by SHA).
#    3. Symlink the gitignored runtime deps (root node_modules + each
#       packages/*/node_modules) from the live repo into the snapshot so Bun can
#       resolve imports without a re-install.
#    4. Point OPENCODE_SRC_REPO / OPENCODE_SRC_ENTRY at the snapshot; keep
#       BUN_BIN / NODE_BIN / TAVILY_MCP_ENTRY pointing at the live repo (those are
#       toolchain, not harness source, and are gitignored anyway).
#
#  SYSTEM PROMPT: SYSTEM_PROMPT_FILE works exactly as in the source twin. NOTE it
#  is NOT pinned by the snapshot unless its path points inside the snapshot. If
#  you want the prompt frozen too, point it at
#  ${SNAPSHOT}/packages/opencode/src/session/prompt/<file>.txt (the snapshot path
#  is printed at submit time), or leave it empty to use the pinned source default.
#
#  CAVEAT (acceptable for ablation testing; fix on migrate): the symlinked
#  node_modules are the LIVE repo's. Third-party npm deps are lockfile-pinned so
#  that's fine, but any bun-WORKSPACE package symlinked inside node_modules
#  resolves to the live repo's copy, NOT the snapshot's. Our ablation edits live
#  in packages/opencode/src (run directly from the snapshot via the wrapper's
#  `${REPO}/packages/opencode/src/index.ts`), so they ARE pinned; only cross-
#  package bare imports are not. To make it fully hermetic later, run
#  `bun install` inside the snapshot instead of symlinking node_modules.
#
#  Usage (defaults to pinning current HEAD):
#    bash run_glm51_opencode_pinned_sharded.sh
#  Pin an explicit commit / tag / branch:
#    OPENCODE_PIN_COMMIT=7e08d3025 bash run_glm51_opencode_pinned_sharded.sh
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (env-overridable) ===================
# NOTE: only GLM-5-FP8 is staged on this cluster; the old GLM-5.1-FP8 default path
# does not exist. The serve recipe (glm47/glm45 parsers) is identical for the family.
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ameyasunilm/models/GLM-5-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
DATASET="${DATASET:-/lustre/fsw/portfolios/llmservice/users/abhibhag/Gym/benchmarks/browsecomp/data/browsecomp_benchmark_400.jsonl}"

# Each shard now serves on 4 nodes (TP=4 DP=4), so total nodes = NUM_SHARDS*4.
NUM_SHARDS="${NUM_SHARDS:-10}"
PARALLEL="${PARALLEL:-16}"
KEY_OFFSET="${KEY_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
# Extra flags appended verbatim to `vllm serve` (e.g. "--enforce-eager"). Default empty.
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-nemotron_agents_dev}"
JOB_PREFIX="${JOB_PREFIX:-glm51_ocpin}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# Compaction-trigger override (ABLATION KNOB). Default EMPTY = stock opencode:
# no limit.input emitted, so compaction fires at the natural context-max_output (~99072).
# Set MODEL_INPUT_LIMIT=<N> to set model.limit.input=N. We do NOT set compaction.reserved,
# so opencode uses its own default buffer (min(20000, max_output) = 20000), and compaction
# triggers at (N - 20000). e.g. MODEL_INPUT_LIMIT=90000 => compaction triggers at ~70000.
MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-}"
# Max-steps override (ABLATION KNOB). Default EMPTY = stock opencode: agent.steps
# unset, so the harness allows unbounded steps (agent.steps ?? Infinity in
# session/prompt.ts). Set MAX_STEPS=<N> to cap each rollout at N agent steps by
# emitting agent.build.steps=N into the per-shard opencode.json.
MAX_STEPS="${MAX_STEPS:-}"
# Compaction kill-switch (ABLATION KNOB). Default EMPTY = compaction on. Set
# DISABLE_COMPACTION=1 to emit "compaction":{"auto":false} => NO compaction ever
# fires (overflow.ts:isOverflow returns false). Overrides the discard knobs below.
DISABLE_COMPACTION="${DISABLE_COMPACTION:-}"
# Overflow STRATEGY (ABLATION KNOB). Default EMPTY = stock "summarize" (LLM summary).
# Set COMPACTION_STRATEGY=discard to emit "compaction":{"strategy":"discard"} => at the
# overflow trigger the harness DISCARDS all prior context and keeps only the last
# COMPACTION_KEEP_TOOL_CALLS tool calls verbatim, with NO summary LLM call.
# REQUIRES an opencode source tree with the discard branch in session/compaction.ts
# (e.g. OPENCODE_SRC_REPO=<opencode-hsg>); otherwise the key is inert and the run
# falls back to summarize.
COMPACTION_STRATEGY="${COMPACTION_STRATEGY:-}"
# k for strategy=discard. Default EMPTY = harness default (3). Set to override.
COMPACTION_KEEP_TOOL_CALLS="${COMPACTION_KEEP_TOOL_CALLS:-}"
# System-prompt knob. EMPTY = stock opencode source selection (system.ts routes
# GLM-5.1 to its frankenstein fallback). Set SYSTEM_PROMPT_FILE=<abs path to a
# .txt> and the opencode source override (OPENCODE_SYSTEM_PROMPT_FILE) uses that
# file's contents verbatim as the system prompt for EVERY rollout in this run,
# overriding all model-id routing. Path must be readable from inside the job.
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}"
TAVILY_MAX_RESULTS="${TAVILY_MAX_RESULTS:-5}"
TAVILY_SEARCH_DEPTH="${TAVILY_SEARCH_DEPTH:-advanced}"
TAVILY_INCLUDE_RAW_CONTENT="${TAVILY_INCLUDE_RAW_CONTENT:-true}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"

# --- RUN-FROM-SOURCE config ---
# OPENCODE_SRC_REPO is the LIVE repo (toolchain + git history live here). The
# pinning block below overrides OPENCODE_SRC_REPO/ENTRY to point at a frozen
# snapshot; LIVE_REPO keeps a handle on the original for symlinking deps.
OPENCODE_SRC_REPO="${OPENCODE_SRC_REPO:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode}"
LIVE_REPO="${OPENCODE_SRC_REPO}"
BUN_BIN="${BUN_BIN:-${LIVE_REPO}/bun/bin/bun}"
# The wrapper that execs `bun run <entry>`; used in place of the OOB binary.
OPENCODE_BIN="${OPENCODE_BIN:-${SCRIPT_DIR}/opencode-src}"

NODE_BIN="${NODE_BIN:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/node/node-v22.11.0-linux-x64/bin}"
# Pre-installed tavily-mcp entrypoint (launched directly via node; avoids per-rollout `npx` download/race)
TAVILY_MCP_ENTRY="${TAVILY_MCP_ENTRY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/mcp/node_modules/tavily-mcp/build/index.js}"

# --- PINNED-COMMIT config (the difference vs the live source-sharded script) ---
# Commit/ref to pin. Empty => resolve the live repo's current HEAD at submit time.
OPENCODE_PIN_COMMIT="${OPENCODE_PIN_COMMIT:-}"
# Where frozen snapshots live (one git worktree per SHA, shared across runs).
OPENCODE_SNAPSHOT_BASE="${OPENCODE_SNAPSHOT_BASE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/runs/_src_snapshots}"

DRIVER_PY="${DRIVER_PY:-${SCRIPT_DIR}/launch_opencode_evals.py}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/.env/bin/python3}"

GRADE_SCRIPT="${GRADE_SCRIPT:-${SCRIPT_DIR}/grade_with_glm5.sh}"
RUN_GRADING="${RUN_GRADING:-true}"
EXCLUDE_DOMAINS_FILE="${EXCLUDE_DOMAINS_FILE:-/lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json}"
TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"
WORK_BASE="${WORK_BASE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/runs}"
# HSG-validated GLM-5 container (the old v0.20.1 default path does not exist).
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/lvega/sqsh/vllm-openai-v0.17.1.sqsh}"

RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# =========================================================================

# --- build / resolve the pinned snapshot (SUBMIT TIME) -------------------
command -v git >/dev/null 2>&1 || { echo "ERROR: git not on PATH (needed to pin a commit)" >&2; exit 1; }
git -C "${LIVE_REPO}" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: LIVE_REPO is not a git repo: ${LIVE_REPO}" >&2; exit 1; }
PIN_SHA="$(git -C "${LIVE_REPO}" rev-parse "${OPENCODE_PIN_COMMIT:-HEAD}")" || { echo "ERROR: cannot resolve commit '${OPENCODE_PIN_COMMIT:-HEAD}' in ${LIVE_REPO}" >&2; exit 1; }
SNAPSHOT_DIR="${OPENCODE_SNAPSHOT_BASE}/${PIN_SHA}"

if [[ -d "${SNAPSHOT_DIR}/packages/opencode/src" ]]; then
    echo "PIN: reusing existing snapshot ${SNAPSHOT_DIR} (commit ${PIN_SHA:0:9})"
else
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
# Point the harness source at the frozen snapshot; toolchain stays in LIVE_REPO.
OPENCODE_SRC_REPO="${SNAPSHOT_DIR}"
OPENCODE_SRC_ENTRY="${SNAPSHOT_DIR}/packages/opencode/src/index.ts"

# --- sanity checks ---
[[ -f "${DATASET}" ]]      || { echo "ERROR: DATASET not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DRIVER_PY}" ]]    || { echo "ERROR: DRIVER_PY not found: ${DRIVER_PY}" >&2; exit 1; }
[[ -x "${VENV_PY}" ]]      || { echo "ERROR: VENV_PY not found: ${VENV_PY}" >&2; exit 1; }
[[ -f "${TAVILY_KEYS_FILE}" ]] || { echo "ERROR: TAVILY_KEYS_FILE not found: ${TAVILY_KEYS_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "ERROR: SYSTEM_PROMPT_FILE set but not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }
[[ -x "${NODE_BIN}/npx" && -x "${NODE_BIN}/node" ]] || { echo "ERROR: node/npx not found under NODE_BIN: ${NODE_BIN} (needed for tavily-mcp)" >&2; exit 1; }
[[ -f "${TAVILY_MCP_ENTRY}" ]] || { echo "ERROR: TAVILY_MCP_ENTRY not found: ${TAVILY_MCP_ENTRY} (pre-install: npm install tavily-mcp@latest)" >&2; exit 1; }
# --- run-from-source sanity checks (now validate the SNAPSHOT) ---
[[ -x "${BUN_BIN}" ]]      || { echo "ERROR: BUN_BIN not found/executable: ${BUN_BIN} (run: curl -fsSL https://bun.sh/install | BUN_INSTALL=${LIVE_REPO}/bun bash -s bun-v1.3.14)" >&2; exit 1; }
[[ -f "${OPENCODE_SRC_ENTRY}" ]] || { echo "ERROR: snapshot entry not found: ${OPENCODE_SRC_ENTRY}" >&2; exit 1; }
[[ -x "${OPENCODE_BIN}" ]] || { echo "ERROR: opencode-src wrapper not found/executable: ${OPENCODE_BIN} (chmod +x it)" >&2; exit 1; }
[[ -e "${OPENCODE_SRC_REPO}/node_modules" ]] || { echo "ERROR: snapshot node_modules symlink missing: ${OPENCODE_SRC_REPO}/node_modules" >&2; exit 1; }

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
    # Model-group subdir (e.g. glm_5_1): served name with non-alnum -> underscore.
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
echo "  GLM-5.1 opencode rollout [PINNED SOURCE]$( ((RESUMING==1)) && echo '  [RESUME]' )"
echo "  Dataset:   ${DATASET}"
echo "  Shards:    ${NUM_SHARDS}  (x2 nodes = $((NUM_SHARDS*2)) nodes, TP=16)"
echo "  Parallel:  ${PARALLEL} rollouts/shard"
echo "  Keys:      ${NKEYS} (offset ${KEY_OFFSET})"
echo "  Sys prompt:${SYSTEM_PROMPT_FILE:-<stock pinned source default (frankenstein)>}"
echo "  InputLimit:${MODEL_INPUT_LIMIT:-<stock>}   MaxSteps:${MAX_STEPS:-<unbounded>}"
echo "  Retries:   ${MAX_RETRIES}"
echo "  Pinned to: ${PIN_SHA}"
echo "  Snapshot:  ${SNAPSHOT_DIR}"
echo "  Bun:       ${BUN_BIN}  (from live repo)"
echo "  Source:    ${OPENCODE_SRC_ENTRY}"
echo "  Wrapper:   ${OPENCODE_BIN}"
echo "  Run dir:   ${RUN_DIR}"
echo "============================================"

# --- submit-time system-prompt preview (full prompt is dumped per-shard at job start) ---
BANNER_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${OPENCODE_SRC_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt.txt}"
if [[ -f "${BANNER_PROMPT_FILE}" ]]; then
    echo "System prompt: ${BANNER_PROMPT_FILE}  ($(wc -c < "${BANNER_PROMPT_FILE}") chars)"
    echo "  ---------------- preview (first 40 lines) ----------------"
    sed -n '1,40p' "${BANNER_PROMPT_FILE}" | sed 's/^/  | /'
    echo "  ----------------------------------------------------------"
    echo "  (full prompt is logged per shard in slurm_logs at job start)"
else
    echo "System prompt: ${BANNER_PROMPT_FILE} (NOT readable at submit time -- check the path)"
fi
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
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --switches=1
set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

echo "JOB ${SLURM_JOB_ID}  shard ${SHARD_ID}/${NUM_SHARDS}  key ...${TAVILY_KEY: -6}"
MOUNTS="/lustre:/lustre"
CONTAINER_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS}"
VLLM_LOG="${RUN_DIR}/vllm_logs/vllm_shard${SHARD_ID}_${SLURM_JOB_ID}.log"
# Ray-head IP + readiness handshake files (on /lustre, visible to every node).
HEAD_IP_FILE="${RUN_DIR}/.ray_head_ip_${SLURM_JOB_ID}"
READY_FILE="${RUN_DIR}/.vllm_ready_${SLURM_JOB_ID}"

export VLLM_NO_USAGE_STATS=1
export RAY_CGRAPH_get_timeout=1800

# --- Serve GLM via Ray across 4 nodes (TP=4, DP=4) -- HSG-validated GLM-5 recipe
#     transplanted from browsecomp 0430_glm5_fp8_browsecomp_spinup_interleaved_reasoning_on.sh.
#     One --mpi=pmix srun spans the whole allocation: rank 0 starts the Ray head +
#     vLLM and waits; other ranks join as Ray workers. The srun is backgrounded so
#     this script can drive opencode once vLLM is ready, then tears the srun down. ---
srun ${CONTAINER_ARGS} \
    --export=ALL,MODEL_PATH,VLLM_LOG,SERVER_PORT,RAY_PORT,SERVED_MODEL_NAME,VLLM_NO_USAGE_STATS,RAY_CGRAPH_get_timeout,EXTRA_VLLM_FLAGS,HEAD_IP_FILE,READY_FILE \
    --mpi=pmix \
    bash -lc '
    pip install "transformers>=5.0.0" -q
    set -euo pipefail

    ray stop --force 2>/dev/null || true

    if [ "$SLURM_PROCID" -eq 0 ]; then
        rm -f "$HEAD_IP_FILE" "$READY_FILE"
        HEAD_IP=$(hostname -I | awk "{print \$1}")
        [ -z "$HEAD_IP" ] && HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
        if [ -z "$HEAD_IP" ]; then echo "ERROR: could not determine head node IP"; exit 1; fi
        echo "$HEAD_IP" > "$HEAD_IP_FILE"

        echo "=== [rank0] Starting Ray head on ${HEAD_IP}:${RAY_PORT} ==="
        ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --disable-usage-stats

        echo "=== [rank0] Starting vLLM (TP=4 DP=4 across ${SLURM_NNODES} nodes) ==="
        SERVE_PORT="${SERVER_PORT}"
        unset VLLM_PORT   # else vLLM may silently bump the port and the harness never connects
        export FLASHINFER_WORKSPACE_BASE=/tmp

        vllm serve "${MODEL_PATH}" \
            --trust-remote-code \
            --tensor-parallel-size 4 \
            --data-parallel-size 4 \
            --data-parallel-size-local 1 \
            --data-parallel-backend ray \
            --api-server-count 1 \
            --distributed-executor-backend ray \
            --gpu-memory-utilization 0.85 \
            --enable-prefix-caching \
            --port "${SERVE_PORT}" \
            --enable-auto-tool-choice \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
            --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" ${EXTRA_VLLM_FLAGS} >> "${VLLM_LOG}" 2>&1 &
        VLLM_PID=$!

        echo "=== [rank0] Waiting for server readiness ==="
        while ! grep -qE "Uvicorn running on|Application startup complete" "${VLLM_LOG}" 2>/dev/null; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "ERROR: vLLM server process died"; ray stop || true; rm -f "$HEAD_IP_FILE"; exit 1; fi
            sleep 2
        done
        echo "${HEAD_IP}" > "${READY_FILE}"
        echo "=== Server ready on http://${HEAD_IP}:${SERVE_PORT} ==="

        wait "$VLLM_PID"
        ray stop || true
        rm -f "$HEAD_IP_FILE"
    else
        for _ in $(seq 1 120); do [ -s "$HEAD_IP_FILE" ] && break; sleep 1; done
        if [ ! -s "$HEAD_IP_FILE" ]; then echo "ERROR: timed out waiting for Ray head IP file: $HEAD_IP_FILE"; exit 1; fi
        HEAD_IP=$(cat "$HEAD_IP_FILE")
        echo "=== [rank${SLURM_PROCID}] Waiting for Ray head ${HEAD_IP}:${RAY_PORT} ==="
        for _ in $(seq 1 120); do ray status --address "${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1 && break; sleep 2; done
        echo "=== [rank${SLURM_PROCID}] Starting Ray worker ==="
        ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats
        tail -f /dev/null
    fi
    ' &
SRUN_PID=$!

# --- wait for vLLM readiness (rank0 writes READY_FILE once Uvicorn is up) ---
echo "Waiting for vLLM readiness signal at ${READY_FILE} ..."
for i in $(seq 1 600); do
    if [ -s "${READY_FILE}" ] 2>/dev/null; then HEAD_IP=$(cat "${READY_FILE}"); echo "vLLM ready on ${HEAD_IP}:${SERVER_PORT}"; break; fi
    if ! kill -0 "$SRUN_PID" 2>/dev/null; then echo "ERROR: srun died before vLLM became ready" >&2; exit 1; fi
    if [[ $i -eq 600 ]]; then echo "ERROR: vLLM not ready within 20 min" >&2; exit 1; fi
    sleep 2
done
MODEL_BASE_URL="http://${HEAD_IP}:${SERVER_PORT}/v1"

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
# Build the top-level "compaction" object from the ablation knobs (all default
# EMPTY => no compaction key emitted => stock opencode behavior):
#   DISABLE_COMPACTION=1        => "auto": false  (NO compaction ever fires).
#   COMPACTION_STRATEGY=discard => "strategy": "discard"  (discard-all at overflow,
#                                  keeping the last keep_tool_calls calls verbatim,
#                                  NO summary LLM call).
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
cat > "${SHARD_DIR}/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  ${COMPACTION_FRAG}
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local GLM-5.1 (vLLM)",
      "options": { "baseURL": "${MODEL_BASE_URL}", "apiKey": "${MODEL_API_KEY}" },
      "models": {
        "${SERVED_MODEL_NAME}": { "name": "GLM-5.1 (local)", "limit": { "context": ${MAX_MODEL_LEN}${INPUT_LIMIT_FRAG}, "output": 32768 } }
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
        "*": "allow",
        "websearch": "deny",
        "webfetch": "deny"
      }
    }
  }
}
JSON

echo "Wrote ${SHARD_DIR}/opencode.json"
# Put bun, node/npx (for tavily-mcp), and the opencode-src wrapper on PATH for
# the driver + opencode subprocesses.
export PATH="$(dirname "${BUN_BIN}"):${NODE_BIN}:$(dirname "${OPENCODE_BIN}"):${PATH}"
echo "bun: $(command -v bun || echo MISSING)  node: $(command -v node || echo MISSING)  npx: $(command -v npx || echo MISSING)"
echo "PINNED snapshot: ${OPENCODE_SRC_REPO}"
echo "opencode-src -> $(${OPENCODE_BIN} --version 2>/dev/null || echo MISSING)"

# --- System prompt: resolve, export (only when overridden), and DUMP the full
# prompt to this shard's log so every run records exactly which prompt it used. ---
if [[ -n "${SYSTEM_PROMPT_FILE}" ]]; then
    export OPENCODE_SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    EFFECTIVE_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    PROMPT_SOURCE="launcher override (OPENCODE_SYSTEM_PROMPT_FILE)"
else
    # No override -> opencode's system.ts routing picks it. For glm-5.1 that is
    # the frankenstein fallback; surface that file (from the PINNED snapshot).
    EFFECTIVE_PROMPT_FILE="${OPENCODE_SRC_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt.txt"
    PROMPT_SOURCE="stock pinned-source default (system.ts -> frankenstein fallback for glm-5.1)"
fi
echo "================= SYSTEM PROMPT ================="
echo "source: ${PROMPT_SOURCE}"
echo "file:   ${EFFECTIVE_PROMPT_FILE}"
if [[ -f "${EFFECTIVE_PROMPT_FILE}" ]]; then
    echo "size:   $(wc -c < "${EFFECTIVE_PROMPT_FILE}") chars, $(wc -l < "${EFFECTIVE_PROMPT_FILE}") lines"
    echo "---------------- begin prompt ------------------"
    cat "${EFFECTIVE_PROMPT_FILE}"
    echo "----------------- end prompt -------------------"
else
    echo "WARN: prompt file not readable from job: ${EFFECTIVE_PROMPT_FILE}"
fi
echo "================================================"

# --- drive opencode (FROM PINNED SOURCE) over this shard's questions ---
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

# --- tear down the backgrounded vLLM/Ray srun so the job can exit ---
rm -f "${READY_FILE}" "${HEAD_IP_FILE}"
kill "${SRUN_PID}" 2>/dev/null || true
wait "${SRUN_PID}" 2>/dev/null || true
SHARD_EOF

    export SHARD_ID NUM_SHARDS SHARD_DIR SHARD_JSONL TAVILY_KEY RUN_DIR
    export MODEL_PATH SERVED_MODEL_NAME MODEL_API_KEY MAX_MODEL_LEN MODEL_INPUT_LIMIT MAX_STEPS VLLM_LOG_LEVEL
    export DISABLE_COMPACTION COMPACTION_STRATEGY COMPACTION_KEEP_TOOL_CALLS EXTRA_VLLM_FLAGS
    export CONTAINER SERVER_PORT RAY_PORT EXCLUDE_DOMAINS_FILE SYSTEM_PROMPT_FILE
    export TAVILY_MAX_RESULTS TAVILY_SEARCH_DEPTH TAVILY_INCLUDE_RAW_CONTENT
    export OPENCODE_BIN NODE_BIN TAVILY_MCP_ENTRY DRIVER_PY VENV_PY PARALLEL AGENT
    export BUN_BIN OPENCODE_SRC_REPO OPENCODE_SRC_ENTRY

    JID=$(sbatch --parsable \
        -A "${ACCOUNT}" -p "${PARTITION}" \
        --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
        --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
        --export=ALL "${SCRIPT_FILE}")
    echo "shard ${SHARD_ID}: submitted job ${JID} (key idx ${KEY_IDX}, ...${TAVILY_KEY: -6})"

    for RETRY in $(seq 1 "${MAX_RETRIES}"); do
        JID=$(sbatch --parsable --dependency=afternotok:${JID} \
            -A "${ACCOUNT}" -p "${PARTITION}" \
            --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
            --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
            --export=ALL "${SCRIPT_FILE}")
        echo "  retry ${RETRY}: job ${JID} (afternotok)"
    done
    ALL_SHARD_JIDS="${ALL_SHARD_JIDS:+${ALL_SHARD_JIDS} }${JID}"
done

echo ""
echo "All ${NUM_SHARDS} shard(s) submitted ($((NUM_SHARDS*4)) nodes; 4 per shard, TP=4 DP=4)."
echo "Run dir: ${RUN_DIR}"
echo "Pinned commit: ${PIN_SHA}  (snapshot: ${SNAPSHOT_DIR})"

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
