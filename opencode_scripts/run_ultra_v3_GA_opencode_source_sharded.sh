#!/bin/bash
# =========================================================================
#  run_ultra_v3_GA_opencode_source_sharded.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  Same as run_ultra_v3_GA_opencode_sharded.sh, but runs the opencode harness
#  FROM SOURCE (via Bun) instead of the OOB precompiled binary -- so harness
#  edits (e.g. the [COMPACTION_DEBUG] logs) and the MODEL_INPUT_LIMIT compaction
#  knob take effect. See run_glm51_opencode_source_sharded.sh for the glm twin.
#
#  Serves Nemotron-Ultra (TP=8, 2 nodes x 4 GPUs) per shard and drives opencode.
#  Defaults to NUM_SHARDS=15. Scale/override via env, e.g.:
#    MODEL_INPUT_LIMIT=95000 NUM_SHARDS=15 bash run_ultra_v3_GA_opencode_source_sharded.sh
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (env-overridable) ===================
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ygalron/checkpoints/green_ultra_step42_with_mtp_boosted_iter5000_hf/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ultra-v3-step42-green-mtp-boosted}"
DATASET="${DATASET:-/lustre/fsw/portfolios/llmservice/users/abhibhag/Gym/benchmarks/browsecomp/data/browsecomp_benchmark_400.jsonl}"

NUM_SHARDS="${NUM_SHARDS:-15}"
PARALLEL="${PARALLEL:-16}"
KEY_OFFSET="${KEY_OFFSET:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
QOS="${QOS:-interactive}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-llmservice_nemotron_ultra}"
JOB_PREFIX="${JOB_PREFIX:-ultra_ocsrc}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# Compaction-trigger override (ABLATION KNOB). Default EMPTY = stock opencode:
# no limit.input emitted, so compaction fires at the natural context-max_output (~99072).
# Set MODEL_INPUT_LIMIT=<N> to set model.limit.input=N. compaction.reserved is NOT set,
# so opencode uses its own default buffer (min(20000, max_output)=20000); trigger = N-20000.
MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}"
TAVILY_MAX_RESULTS="${TAVILY_MAX_RESULTS:-5}"
TAVILY_SEARCH_DEPTH="${TAVILY_SEARCH_DEPTH:-advanced}"
TAVILY_INCLUDE_RAW_CONTENT="${TAVILY_INCLUDE_RAW_CONTENT:-true}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"

# --- RUN-FROM-SOURCE config (the difference vs the OOB-binary ultra script) ---
OPENCODE_SRC_REPO="${OPENCODE_SRC_REPO:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode}"
BUN_BIN="${BUN_BIN:-${OPENCODE_SRC_REPO}/bun/bin/bun}"
OPENCODE_SRC_ENTRY="${OPENCODE_SRC_ENTRY:-${OPENCODE_SRC_REPO}/packages/opencode/src/index.ts}"
OPENCODE_BIN="${OPENCODE_BIN:-${SCRIPT_DIR}/opencode-src}"
NODE_BIN="${NODE_BIN:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/node/node-v22.11.0-linux-x64/bin}"
TAVILY_MCP_ENTRY="${TAVILY_MCP_ENTRY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/mcp/node_modules/tavily-mcp/build/index.js}"

DRIVER_PY="${DRIVER_PY:-${SCRIPT_DIR}/launch_opencode_evals.py}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"

GRADE_SCRIPT="${GRADE_SCRIPT:-${SCRIPT_DIR}/grade_with_glm5.sh}"
RUN_GRADING="${RUN_GRADING:-true}"
EXCLUDE_DOMAINS_FILE="${EXCLUDE_DOMAINS_FILE:-/lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json}"
TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/tavily_keys.txt}"
WORK_BASE="${WORK_BASE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/runs}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/containers/vllm-hsg-0.17.0.sqsh}"

RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# =========================================================================

# --- sanity checks ---
[[ -f "${DATASET}" ]]      || { echo "ERROR: DATASET not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DRIVER_PY}" ]]    || { echo "ERROR: DRIVER_PY not found: ${DRIVER_PY}" >&2; exit 1; }
[[ -x "${VENV_PY}" ]]      || { echo "ERROR: VENV_PY not found: ${VENV_PY}" >&2; exit 1; }
[[ -f "${TAVILY_KEYS_FILE}" ]] || { echo "ERROR: TAVILY_KEYS_FILE not found: ${TAVILY_KEYS_FILE}" >&2; exit 1; }
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
echo "  Bun:       ${BUN_BIN}"
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
#SBATCH -N 4
#SBATCH --gpus-per-node=8
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
            pip install ray -q
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
    pip install ray -q
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
        --pipeline-parallel-size 4 \
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
cat > "${SHARD_DIR}/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
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
      "permission": {
        "*": "allow",
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
    export MODEL_PATH SERVED_MODEL_NAME MODEL_API_KEY MAX_MODEL_LEN MODEL_INPUT_LIMIT VLLM_LOG_LEVEL
    export CONTAINER SERVER_PORT RAY_PORT EXCLUDE_DOMAINS_FILE
    export TAVILY_MAX_RESULTS TAVILY_SEARCH_DEPTH TAVILY_INCLUDE_RAW_CONTENT
    export OPENCODE_BIN NODE_BIN TAVILY_MCP_ENTRY DRIVER_PY VENV_PY PARALLEL AGENT
    export BUN_BIN OPENCODE_SRC_REPO OPENCODE_SRC_ENTRY

    JID=$(sbatch --parsable \
        -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" \
        --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
        --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
        --export=ALL "${SCRIPT_FILE}")
    echo "shard ${SHARD_ID}: submitted job ${JID} (key idx ${KEY_IDX}, ...${TAVILY_KEY: -6})"

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
