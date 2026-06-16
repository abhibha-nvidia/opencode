#!/bin/bash
# =========================================================================
#  run_glm51_opencode_source_sharded_exa.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  EXA VARIANT of run_glm51_opencode_source_sharded.sh.
#
#  Difference from the tavily script: there is NO tavily MCP. We use opencode's
#  OWN built-in `websearch` + `webfetch` tools (https://opencode.ai/docs/tools/).
#  `websearch` defaults to the Exa provider (https://mcp.exa.ai/mcp), keyed off
#  $EXA_API_KEY; we pin the provider explicitly with OPENCODE_WEBSEARCH_PROVIDER=exa.
#  `webfetch` is a plain HTTP GET (no key). So the shard opencode.json drops the
#  whole "mcp" block and flips the build agent's websearch/webfetch perms from
#  "deny" to "allow" (covered by "*":"allow").
#
#  Runs the opencode harness FROM SOURCE (via Bun); edits under
#  packages/opencode/src/** take effect immediately -- no rebuild needed.
#  OPENCODE_BIN points at the `opencode-src` wrapper (next to this file), which
#  execs `bun run packages/opencode/src/index.ts "$@"`.
#
#  One-time setup (already done once; redo on a fresh clone):
#    export BUN_INSTALL=<repo>/bun
#    curl -fsSL https://bun.sh/install | bash -s bun-v1.3.14
#    cd <repo> && <repo>/bun/bin/bun install
#
#  Hosts GLM-5.1-FP8 (TP=16, 2 nodes x 8 GPUs) and runs opencode evals.
#  Defaults to NUM_SHARDS=1 (single 2-node job). Scale up via env:
#    NUM_SHARDS=4 bash run_glm51_opencode_source_sharded_exa.sh
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (env-overridable) ===================
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/hiteshis/models/GLM-5.1-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
DATASET="${DATASET:-/lustre/fsw/portfolios/llmservice/users/abhibhag/Gym/benchmarks/browsecomp/data/browsecomp_benchmark_400.jsonl}"

NUM_SHARDS="${NUM_SHARDS:-60}"
PARALLEL="${PARALLEL:-16}"
KEY_OFFSET="${KEY_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-nemotron_agents_dev}"
JOB_PREFIX="${JOB_PREFIX:-glm51_ocsrc_exa}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# Compaction-trigger override (ABLATION KNOB). Default EMPTY = stock opencode:
# no limit.input emitted, so compaction fires at the natural context-max_output (~99072).
# Set MODEL_INPUT_LIMIT=<N> to set model.limit.input=N. We do NOT set compaction.reserved,
# so opencode uses its own default buffer (min(20000, max_output) = 20000), and compaction
# triggers at (N - 20000). e.g. MODEL_INPUT_LIMIT=90000 => compaction triggers at ~70000.
MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-}"
# System-prompt knob. EMPTY = stock opencode source selection (system.ts routes
# GLM-5.1 to its frankenstein fallback). Set SYSTEM_PROMPT_FILE=<abs path to a
# .txt> and the opencode source override (OPENCODE_SYSTEM_PROMPT_FILE) uses that
# file's contents verbatim as the system prompt for EVERY rollout in this run,
# overriding all model-id routing. Path must be readable from inside the job.
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}"
# --- EXA web-search config (replaces all TAVILY_* knobs) ---
# Force opencode's websearch tool onto the Exa provider (otherwise it picks
# exa/parallel pseudo-randomly per-session via the session checksum).
WEBSEARCH_PROVIDER="${WEBSEARCH_PROVIDER:-exa}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"

# --- RUN-FROM-SOURCE config (this is the only real difference vs the binary script) ---
OPENCODE_SRC_REPO="${OPENCODE_SRC_REPO:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode}"
BUN_BIN="${BUN_BIN:-${OPENCODE_SRC_REPO}/bun/bin/bun}"
OPENCODE_SRC_ENTRY="${OPENCODE_SRC_ENTRY:-${OPENCODE_SRC_REPO}/packages/opencode/src/index.ts}"
# The wrapper that execs `bun run <entry>`; used in place of the OOB binary.
OPENCODE_BIN="${OPENCODE_BIN:-${SCRIPT_DIR}/opencode-src}"

NODE_BIN="${NODE_BIN:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/node/node-v22.11.0-linux-x64/bin}"
DRIVER_PY="${DRIVER_PY:-${SCRIPT_DIR}/launch_opencode_evals.py}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/.env/bin/python3}"

GRADE_SCRIPT="${GRADE_SCRIPT:-${SCRIPT_DIR}/grade_with_glm5.sh}"
RUN_GRADING="${RUN_GRADING:-true}"
EXA_KEYS_FILE="${EXA_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/exa_keys.txt}"
WORK_BASE="${WORK_BASE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/runs}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/lvega/sqsh/vllm-openai-v0.20.1.sqsh}"

RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# =========================================================================

# --- sanity checks ---
[[ -f "${DATASET}" ]]      || { echo "ERROR: DATASET not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DRIVER_PY}" ]]    || { echo "ERROR: DRIVER_PY not found: ${DRIVER_PY}" >&2; exit 1; }
[[ -x "${VENV_PY}" ]]      || { echo "ERROR: VENV_PY not found: ${VENV_PY}" >&2; exit 1; }
[[ -f "${EXA_KEYS_FILE}" ]] || { echo "ERROR: EXA_KEYS_FILE not found: ${EXA_KEYS_FILE}" >&2; exit 1; }
[[ -z "${SYSTEM_PROMPT_FILE}" || -f "${SYSTEM_PROMPT_FILE}" ]] || { echo "ERROR: SYSTEM_PROMPT_FILE set but not found: ${SYSTEM_PROMPT_FILE}" >&2; exit 1; }
# --- run-from-source sanity checks ---
[[ -x "${BUN_BIN}" ]]      || { echo "ERROR: BUN_BIN not found/executable: ${BUN_BIN} (run: curl -fsSL https://bun.sh/install | BUN_INSTALL=${OPENCODE_SRC_REPO}/bun bash -s bun-v1.3.14)" >&2; exit 1; }
[[ -f "${OPENCODE_SRC_ENTRY}" ]] || { echo "ERROR: opencode source entry not found: ${OPENCODE_SRC_ENTRY}" >&2; exit 1; }
[[ -x "${OPENCODE_BIN}" ]] || { echo "ERROR: opencode-src wrapper not found/executable: ${OPENCODE_BIN} (chmod +x it)" >&2; exit 1; }
[[ -d "${OPENCODE_SRC_REPO}/node_modules" ]] || { echo "ERROR: deps not installed: ${OPENCODE_SRC_REPO}/node_modules missing (run: cd ${OPENCODE_SRC_REPO} && ${BUN_BIN} install)" >&2; exit 1; }

# --- load EXA key pool (one UUID-style key per non-empty, non-comment line) ---
mapfile -t EXA_KEYS < <(grep -oE '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' "${EXA_KEYS_FILE}")
NKEYS=${#EXA_KEYS[@]}
(( NKEYS > 0 )) || { echo "ERROR: no exa keys parsed from ${EXA_KEYS_FILE}" >&2; exit 1; }
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
echo "  GLM-5.1 opencode rollout [FROM SOURCE | EXA websearch]$( ((RESUMING==1)) && echo '  [RESUME]' )"
echo "  Dataset:   ${DATASET}"
echo "  Shards:    ${NUM_SHARDS}  (x2 nodes = $((NUM_SHARDS*2)) nodes, TP=16)"
echo "  Parallel:  ${PARALLEL} rollouts/shard"
echo "  Search:    opencode built-in websearch (provider=${WEBSEARCH_PROVIDER}) + webfetch"
echo "  Sys prompt:${SYSTEM_PROMPT_FILE:-<stock source default (frankenstein)>}"
echo "  Exa keys:  ${NKEYS} (offset ${KEY_OFFSET})"
echo "  Retries:   ${MAX_RETRIES}"
echo "  Bun:       ${BUN_BIN}"
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
    EXA_KEY="${EXA_KEYS[$KEY_IDX]}"
    SHARD_DIR="${RUN_DIR}/shard${SHARD_ID}"
    SHARD_JSONL="${RUN_DIR}/shards/shard${SHARD_ID}.jsonl"
    SCRIPT_FILE="${RUN_DIR}/shard_scripts/shard${SHARD_ID}.sh"
    mkdir -p "${SHARD_DIR}"

    cat > "${SCRIPT_FILE}" <<'SHARD_EOF'
#!/bin/bash
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

echo "JOB ${SLURM_JOB_ID}  shard ${SHARD_ID}/${NUM_SHARDS}  exa key ...${EXA_KEY: -6}"
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
            pip install ray -q
            ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=8 --block
        " &
done

# --- Ray head + vLLM serve ---
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" ${CONTAINER_ARGS} -o "${VLLM_LOG}" bash -c "
    set -euo pipefail
    pip install ray -q
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL}
    ray start --head --port=${RAY_PORT} --num-gpus=8 --dashboard-host=0.0.0.0
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
        --tensor-parallel-size 16 \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --enable-log-requests \
        --chat-template-content-format string \
        --distributed-executor-backend ray \
        --max-model-len ${MAX_MODEL_LEN} \
        --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 32}' \
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
#
# EXA VARIANT: no "mcp" block at all. We rely on opencode's built-in websearch
# (Exa provider) + webfetch tools, so the build agent's perms just need "*":"allow"
# (the tavily script explicitly DENIED websearch/webfetch -- we do the opposite).
if [[ -n "${MODEL_INPUT_LIMIT}" && "${MODEL_INPUT_LIMIT}" != "0" ]]; then
    INPUT_LIMIT_FRAG=", \"input\": ${MODEL_INPUT_LIMIT}"
else
    INPUT_LIMIT_FRAG=""
fi
cat > "${SHARD_DIR}/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
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
  "agent": {
    "build": {
      "permission": {
        "*": "allow"
      }
    }
  }
}
JSON

echo "Wrote ${SHARD_DIR}/opencode.json"
# Put bun, node, and the opencode-src wrapper on PATH for the driver + opencode subprocesses.
export PATH="$(dirname "${BUN_BIN}"):${NODE_BIN}:$(dirname "${OPENCODE_BIN}"):${PATH}"
# EXA web-search wiring. Three distinct things are needed:
#  1. OPENCODE_ENABLE_EXA=true  -> REGISTERS the websearch tool at all. Without
#     it, webSearchEnabled() is false for a non-"opencode" provider (we use
#     "local"), so the model is never even offered websearch and falls back to
#     scraping search engines via webfetch. THIS is the critical switch.
#  2. EXA_API_KEY               -> the key the websearch tool sends to mcp.exa.ai.
#  3. OPENCODE_WEBSEARCH_PROVIDER=exa -> pins the provider to exa at call time
#     (otherwise it's chosen exa/parallel pseudo-randomly per session).
export OPENCODE_ENABLE_EXA=true
export EXA_API_KEY="${EXA_KEY}"
export OPENCODE_WEBSEARCH_PROVIDER="${WEBSEARCH_PROVIDER}"
# --- System prompt: resolve, export (only when overridden), and DUMP the full
# prompt to this shard's log so every run records exactly which prompt it used. ---
if [[ -n "${SYSTEM_PROMPT_FILE}" ]]; then
    export OPENCODE_SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    EFFECTIVE_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    PROMPT_SOURCE="launcher override (OPENCODE_SYSTEM_PROMPT_FILE)"
else
    # No override -> opencode's system.ts routing picks it. For glm-5.1 that is
    # the frankenstein fallback; surface that file so the log isn't blind.
    EFFECTIVE_PROMPT_FILE="${OPENCODE_SRC_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt.txt"
    PROMPT_SOURCE="stock source default (system.ts -> frankenstein fallback for glm-5.1)"
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
echo "bun: $(command -v bun || echo MISSING)  node: $(command -v node || echo MISSING)"
echo "opencode-src -> $(${OPENCODE_BIN} --version 2>/dev/null || echo MISSING)"
echo "websearch: ENABLE_EXA=${OPENCODE_ENABLE_EXA} provider=${OPENCODE_WEBSEARCH_PROVIDER} exa key ...${EXA_API_KEY: -6}"

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
    |& tee "${RUN_DIR}/slurm_logs/driver_shard${SHARD_ID}_${SLURM_JOB_ID}.log"

echo "shard ${SHARD_ID} done -> ${SHARD_DIR}/trajectories_shard${SHARD_ID}.jsonl"
SHARD_EOF

    export SHARD_ID NUM_SHARDS SHARD_DIR SHARD_JSONL EXA_KEY RUN_DIR
    export MODEL_PATH SERVED_MODEL_NAME MODEL_API_KEY MAX_MODEL_LEN MODEL_INPUT_LIMIT VLLM_LOG_LEVEL
    export CONTAINER SERVER_PORT RAY_PORT WEBSEARCH_PROVIDER SYSTEM_PROMPT_FILE
    export OPENCODE_BIN NODE_BIN DRIVER_PY VENV_PY PARALLEL AGENT
    export BUN_BIN OPENCODE_SRC_REPO OPENCODE_SRC_ENTRY

    JID=$(sbatch --parsable \
        -A "${ACCOUNT}" -p "${PARTITION}" \
        --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
        --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
        --export=ALL "${SCRIPT_FILE}")
    echo "shard ${SHARD_ID}: submitted job ${JID} (key idx ${KEY_IDX}, ...${EXA_KEY: -6})"

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
