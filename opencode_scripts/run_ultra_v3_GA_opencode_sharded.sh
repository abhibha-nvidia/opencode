#!/bin/bash
# =========================================================================
#  run_ultra_opencode_sharded.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  Shards an input jsonl (default browsecomp_benchmark_400.jsonl) across
#  NUM_SHARDS, and submits ONE 2-node vLLM+opencode sbatch job per shard.
#  Each shard:
#    * serves its own Nemotron-Ultra vLLM instance (TP=8),
#    * gets its OWN Tavily key from tavily_keys.txt
#      (KEY_IDX = (SHARD_ID + KEY_OFFSET) % num_keys),
#    * runs opencode over its questions via launch_opencode_evals.py with
#      PARALLEL rollouts in flight (bounded pool),
#    * writes ${RUN_DIR}/shard${i}/trajectories_shard${i}.jsonl.
#
#  Design choices (per request):
#    - Bash driver wraps opencode `run` (opencode is single-prompt; the driver
#      provides the dataset loop + parallelism that browsecomp_eval.py has).
#    - Strided shard split: record idx % NUM_SHARDS == shard_id.
#    - The dataset's `tools` are NEVER passed to opencode -> the MCP shim's
#      tavily tools are the only tools. System prompt = build agent's own.
#    - Per-shard trajectory files (cross-node append is unsafe on lustre);
#      collate into a single trajectories.jsonl after all shards finish.
#
#  Sharded run, everything overridable via env, e.g.:
#    NUM_SHARDS=8 PARALLEL=16 bash run_ultra_opencode_sharded.sh
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================ CONFIG (env-overridable) ===================
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/ygalron/checkpoints/green_ultra_step42_with_mtp_boosted_iter5000_hf/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ultra-v3-step42-green-mtp-boosted}"
DATASET="${DATASET:-/lustre/fsw/portfolios/llmservice/users/abhibhag/Gym/benchmarks/browsecomp/data/browsecomp_benchmark_400.jsonl}"

NUM_SHARDS="${NUM_SHARDS:-15}"     # set to 15      # one 2-node sbatch job per shard
PARALLEL="${PARALLEL:-16}"             # concurrent opencode rollouts per shard
KEY_OFFSET="${KEY_OFFSET:-0}"          # rotate the starting key
MAX_RETRIES="${MAX_RETRIES:-5}"   # set to 4       # afternotok retry chain per shard

TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
QOS="${QOS:-interactive}" # nemotron-priority
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-llmservice_nemotron_ultra}"
JOB_PREFIX="${JOB_PREFIX:-ultra_oc}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-DEBUG}"
TAVILY_MAX_RESULTS="${TAVILY_MAX_RESULTS:-5}"
TAVILY_SEARCH_DEPTH="${TAVILY_SEARCH_DEPTH:-advanced}"
TAVILY_INCLUDE_RAW_CONTENT="${TAVILY_INCLUDE_RAW_CONTENT:-true}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"

OPENCODE_BIN="${OPENCODE_BIN:-/home/abhibhag/.opencode/bin/opencode}"
DRIVER_PY="${DRIVER_PY:-${SCRIPT_DIR}/launch_opencode_evals.py}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"

# Grading: spin up GLM-5-FP8 after all shards finish and grade trajectories.
# Set to empty to skip grading.
GRADE_SCRIPT="${GRADE_SCRIPT:-${SCRIPT_DIR}/grade_with_glm5.sh}"
RUN_GRADING="${RUN_GRADING:-true}"
EXCLUDE_DOMAINS_FILE="${EXCLUDE_DOMAINS_FILE:-/lustre/fsw/portfolios/llmservice/users/rgala/frozen/2025_12_15_nv_tdm_opt_out_registry.json}"
TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"
WORK_BASE="${WORK_BASE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode_scripts/runs}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/containers/vllm-hsg-0.17.0.sqsh}"

# Resume: point at an existing run dir to continue it. The existing shards are
# reused (NUM_SHARDS is derived from them, NOT re-sharded), and the driver skips
# every sample that already has a valid trajectory.json -- only the unfinished /
# broken samples are re-run. Empty -> a fresh timestamped run dir is created.
#   RESUME_RUN_DIR=/lustre/.../runs/sharded/20260604_003646_... bash run_ultra_opencode_sharded.sh
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
# =========================================================================

# --- sanity checks ---
[[ -f "${DATASET}" ]]      || { echo "ERROR: DATASET not found: ${DATASET}" >&2; exit 1; }
[[ -f "${DRIVER_PY}" ]]    || { echo "ERROR: DRIVER_PY not found: ${DRIVER_PY}" >&2; exit 1; }
[[ -x "${VENV_PY}" ]]      || { echo "ERROR: VENV_PY not found: ${VENV_PY}" >&2; exit 1; }
[[ -f "${TAVILY_KEYS_FILE}" ]] || { echo "ERROR: TAVILY_KEYS_FILE not found: ${TAVILY_KEYS_FILE}" >&2; exit 1; }

# --- load tavily key pool (strip quotes/commas/blank lines) ---
mapfile -t TAVILY_KEYS < <(grep -oE 'tvly-[A-Za-z0-9_-]+' "${TAVILY_KEYS_FILE}")
NKEYS=${#TAVILY_KEYS[@]}
(( NKEYS > 0 )) || { echo "ERROR: no tavily keys parsed from ${TAVILY_KEYS_FILE}" >&2; exit 1; }
(( NKEYS >= NUM_SHARDS )) || echo "WARN: only ${NKEYS} keys for ${NUM_SHARDS} shards -> keys will be reused" >&2

# --- run dir (new, or resume an existing one) ---
RESUMING=0
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    [[ -d "${RESUME_RUN_DIR}" ]] || { echo "ERROR: RESUME_RUN_DIR not found: ${RESUME_RUN_DIR}" >&2; exit 1; }
    RUN_DIR="${RESUME_RUN_DIR%/}"
    RESUMING=1
else
    TS="$(date +%Y%m%d_%H%M%S)"
    SLUG="${SERVED_MODEL_NAME//\//__}"
    RUN_DIR="${WORK_BASE}/sharded/${TS}_${SLUG}"
fi
mkdir -p "${RUN_DIR}"/{shards,shard_scripts,slurm_logs,vllm_logs}

# --- shard the dataset (strided: idx % NUM_SHARDS == shard_id), tag _sample_id ---
# On resume, reuse the existing shard files (so sample->shard assignment and
# _sample_id stay identical) and derive NUM_SHARDS from them. Only shard fresh.
# NOTE: use find (exits 0 on no-match); a glob+ls fails under `set -o pipefail`
# on a FRESH dir and would abort the whole script before sharding/submitting.
EXISTING_SHARDS=$(find "${RUN_DIR}/shards" -maxdepth 1 -name 'shard*.jsonl' 2>/dev/null | wc -l)
if (( RESUMING == 1 )) && (( EXISTING_SHARDS > 0 )); then
    NUM_SHARDS=${EXISTING_SHARDS}
    DONE_CNT=$(cat "${RUN_DIR}"/shard*/work/q*/trajectory.json 2>/dev/null | grep -c . || true)
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
    rec["_sample_id"]=idx           # stable global id (position in the 400-file)
    s=idx % ns
    outs[s].write(json.dumps(rec)+"\n"); counts[s]+=1
for o in outs: o.close()
print(f"sharded {len(recs)} records into {ns} shards (strided): {counts}")
PY
fi

echo "============================================"
echo "  Sharded opencode rollout$( ((RESUMING==1)) && echo '  [RESUME]' )"
echo "  Dataset:   ${DATASET}"
echo "  Shards:    ${NUM_SHARDS}  (x2 nodes = $((NUM_SHARDS*2)) nodes)"
echo "  Parallel:  ${PARALLEL} rollouts/shard"
echo "  Keys:      ${NKEYS} (offset ${KEY_OFFSET})"
echo "  Retries:   ${MAX_RETRIES}"
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

    # Per-shard sbatch script. QUOTED heredoc => fully literal; every ${VAR}
    # below is resolved at SHARD RUNTIME from the exported env (--export=ALL),
    # exactly like the reference spinup script.
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

# --- per-shard opencode.json (THIS shard's Tavily key; MCP owns the tools) ---
cat > "${SHARD_DIR}/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Ultra (vLLM)",
      "options": { "baseURL": "${MODEL_BASE_URL}", "apiKey": "${MODEL_API_KEY}" },
      "models": {
        "${SERVED_MODEL_NAME}": { "name": "Nemotron Ultra (local)", "limit": { "context": ${MAX_MODEL_LEN}, "output": 32768 } }
      }
    }
  },
  "mcp": {
    "tavily": {
      "type": "local",
      "command": ["npx", "-y", "tavily-mcp@latest"],
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
export PATH="$(dirname "${OPENCODE_BIN}"):${PATH}"

# --- drive opencode over this shard's questions (PARALLEL rollouts) ---
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

    # Export everything the (literal) shard script references at runtime.
    export SHARD_ID NUM_SHARDS SHARD_DIR SHARD_JSONL TAVILY_KEY RUN_DIR
    export MODEL_PATH SERVED_MODEL_NAME MODEL_API_KEY MAX_MODEL_LEN VLLM_LOG_LEVEL
    export CONTAINER SERVER_PORT RAY_PORT EXCLUDE_DOMAINS_FILE
    export TAVILY_MAX_RESULTS TAVILY_SEARCH_DEPTH TAVILY_INCLUDE_RAW_CONTENT
    export OPENCODE_BIN DRIVER_PY VENV_PY PARALLEL AGENT

    JID=$(sbatch --parsable \
        -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" \
        --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
        --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
        --export=ALL "${SCRIPT_FILE}")
    echo "shard ${SHARD_ID}: submitted job ${JID} (key idx ${KEY_IDX}, ...${TAVILY_KEY: -6})"

    # retry chain (each runs only if the previous attempt failed)
    for RETRY in $(seq 1 "${MAX_RETRIES}"); do
        JID=$(sbatch --parsable --dependency=afternotok:${JID} \
            -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" \
            --time "${TIME_LIMIT}" --job-name "${JOB_PREFIX}_sh${SHARD_ID}" \
            --output "${RUN_DIR}/slurm_logs/%j_%x.out" \
            --export=ALL "${SCRIPT_FILE}")
        echo "  retry ${RETRY}: job ${JID} (afternotok)"
    done
    # track the final JID of each shard chain for the grading dependency
    ALL_SHARD_JIDS="${ALL_SHARD_JIDS:+${ALL_SHARD_JIDS} }${JID}"
done

echo ""
echo "All ${NUM_SHARDS} shards submitted ($((NUM_SHARDS*2)) nodes)."
echo "Run dir: ${RUN_DIR}"
echo "Trajectories stream to: ${RUN_DIR}/trajectories.jsonl as rollouts complete."

# --- Submit grading job after all primary shard jobs finish ---
if [[ "${RUN_GRADING}" == "true" ]] && [[ -f "${GRADE_SCRIPT}" ]]; then
    # Collect the last JID of each shard's chain (the final retry) to depend on
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
