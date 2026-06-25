#!/bin/bash
# =========================================================================
#  spinup_ultra_compaction_endpoint.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  Brings up a STANDALONE, PERSISTENT Nemotron-Ultra vLLM endpoint to be used
#  ONLY as opencode's compaction (summarization) model. The GLM rollout runs
#  separately (run_glm51_ultra_compaction.sh) and points agent.compaction.model
#  at THIS endpoint, so:
#     * GLM does every normal turn,
#     * when context crosses the ~45k compaction trigger, opencode hands the
#       history to THIS Ultra endpoint to write the anchored summary,
#     * GLM resumes on [summary + retained tail + continue].
#
#  WHY STANDALONE (not co-located per shard): compaction fires infrequently
#  (only at the overflow trigger), so a SINGLE shared Ultra endpoint serves all
#  rollout shards' occasional summary calls. Co-locating Ultra inside every
#  shard job would ~double the node count for almost no utilization.
#
#  Uses the SAME vLLM serve recipe as spinup_sftmix0604_oc_source_sharded_hsg_n4post.sh
#  (TP=8, 2 nodes x 4 GPUs, HSG arm64, qwen3_coder / deepseek_r1 parsers).
#
#  IMPORTANT (cross-cluster): the GLM rollout shards must be able to reach this
#  endpoint's node IP over the network -- i.e. run BOTH on the SAME cluster
#  (same ACCOUNT/partition family). This recipe defaults to the HSG
#  llmservice_modelalignment_ppo cluster; keep the GLM launcher on the same one.
#
#  USAGE:
#    bash spinup_ultra_compaction_endpoint.sh
#  then read the endpoint file it prints (…/runs/_compaction_endpoints/<job>.env)
#  and pass it to the launcher:
#    ENDPOINT_FILE=<that file> bash ../launcher_scripts/run_glm51_ultra_compaction.sh
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../opencode_scripts/spinup_scripts
OC_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"             # .../opencode_scripts

# ============================ CONFIG (env-overridable) ===================
# Default "Ultra" = venkats' sft_mix_0604_v2 step-0000485 (the checkpoint the
# existing run_ultra_v2_hsg_* scripts label "Nemotron Ultra"). Override freely.
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/runs/checkpoints/sft_mix_0604_v2_only_192k_128n_mem900g_20260605/eval/0000485/hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ultra_compaction}"

# Cluster / scheduling (must match the GLM rollout's cluster for IP reachability).
NODES_PER_JOB="${NODES_PER_JOB:-2}"
TP_SIZE="${TP_SIZE:-8}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"     # keep up for the whole rollout (+ retries)
QOS="${QOS:-interactive}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-llmservice_modelalignment_ppo}"
JOB_PREFIX="${JOB_PREFIX:-ultra_compaction_ep}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-INFO}"
SERVER_PORT="${SERVER_PORT:-12952}"      # distinct from the GLM rollout's 12951
RAY_PORT="${RAY_PORT:-6380}"             # distinct from the GLM rollout's 6379
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/dmosallanezh/containers/vllm-hsg-0.17.0.sqsh}"

# Where the discovered endpoint URL is written for the launcher to consume.
ENDPOINT_DIR="${ENDPOINT_DIR:-${OC_SCRIPTS_DIR}/runs/_compaction_endpoints}"
# =========================================================================

[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${ENDPOINT_DIR}"
JOB_SCRIPT="${ENDPOINT_DIR}/_job_$(date +%Y%m%d_%H%M%S).sh"

cat > "${JOB_SCRIPT}" <<SBATCH_EOF
#!/bin/bash
#SBATCH -N ${NODES_PER_JOB}
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --qos ${QOS}
#SBATCH --time ${TIME_LIMIT}
#SBATCH --job-name ${JOB_PREFIX}
#SBATCH --output ${ENDPOINT_DIR}/%j_${JOB_PREFIX}.out
set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

MOUNTS="/lustre:/lustre"
CONTAINER_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=\${MOUNTS}"
VLLM_LOG="${ENDPOINT_DIR}/vllm_\${SLURM_JOB_ID}.log"

NUM_NODES=\$SLURM_JOB_NUM_NODES
NODES=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
HEAD_NODE=\${NODES[0]}
HEAD_NODE_IP=\$(srun --nodes=1 --ntasks=1 -w "\$HEAD_NODE" hostname --ip-address)

# --- Ray worker(s) ---
for ((i=1; i<NUM_NODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "\${NODES[\$i]}" \${CONTAINER_ARGS} \\
        -o "${ENDPOINT_DIR}/worker_\${SLURM_JOB_ID}_\${i}.log" \\
        bash -c "
            export FLASHINFER_WORKSPACE_BASE=/tmp
            export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            ray start --address=\${HEAD_NODE_IP}:${RAY_PORT} --block
        " &
done

# --- Ray head + vLLM serve (same Ultra recipe as the sft_mix HSG spinup) ---
srun --nodes=1 --ntasks=1 -w "\$HEAD_NODE" \${CONTAINER_ARGS} -o "\${VLLM_LOG}" bash -c "
    set -euo pipefail
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL}
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
    EXPECTED=\${NUM_NODES}
    for attempt in \\\$(seq 1 120); do
        COUNT=\\\$(ray status 2>/dev/null | grep -c 'node_' || true); COUNT=\\\${COUNT:-0}
        if [ \\\"\\\${COUNT}\\\" -ge \\\"\\\${EXPECTED}\\\" ] 2>/dev/null; then echo 'All workers connected'; break; fi
        sleep 10
    done
    ray status
    echo 'Starting vLLM serve (Ultra compaction endpoint)...'
    vllm serve ${MODEL_PATH} \\
        --served-model-name ${SERVED_MODEL_NAME} \\
        --tensor-parallel-size ${TP_SIZE} \\
        --trust-remote-code \\
        --dtype bfloat16 \\
        --kv-cache-dtype fp8 \\
        --gpu-memory-utilization 0.95 \\
        --enable-expert-parallel \\
        --mamba-ssm-cache-dtype float32 \\
        --distributed-executor-backend ray \\
        --no-enable-prefix-caching \\
        --enable-auto-tool-choice \\
        --tool-call-parser qwen3_coder \\
        --reasoning-parser deepseek_r1 \\
        --enable-log-requests \\
        --compilation-config '{\\\"pass_config\\\": {\\\"fuse_allreduce_rms\\\": false}}' \\
        --max-model-len ${MAX_MODEL_LEN} \\
        --max-num-seqs 32 \\
        --chat-template ${MODEL_PATH}/chat_template.jinja \\
        --host 0.0.0.0 \\
        --port ${SERVER_PORT}
" &
VLLM_PID=\$!

# --- wait for readiness, then publish the endpoint for the launcher ---
MODEL_BASE_URL="http://\${HEAD_NODE_IP}:${SERVER_PORT}/v1"
echo "Waiting for Ultra compaction endpoint at \${MODEL_BASE_URL} ..."
for i in \$(seq 1 60); do
    if curl -s "\${MODEL_BASE_URL}/models" >/dev/null 2>&1; then echo "Ultra endpoint ready"; break; fi
    if [[ \$i -eq 60 ]]; then echo "ERROR: Ultra endpoint not ready within 30 min" >&2; exit 1; fi
    sleep 30
done

ENDPOINT_ENV="${ENDPOINT_DIR}/\${SLURM_JOB_ID}.env"
cat > "\${ENDPOINT_ENV}" <<ENV
# Source this from the GLM launcher (or pass COMPACTION_BASE_URL/COMPACTION_MODEL directly).
COMPACTION_BASE_URL="\${MODEL_BASE_URL}"
COMPACTION_MODEL="${SERVED_MODEL_NAME}"
COMPACTION_API_KEY="${MODEL_API_KEY}"
COMPACTION_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
COMPACTION_JOB_ID="\${SLURM_JOB_ID}"
ENV
ln -sfn "\${ENDPOINT_ENV}" "${ENDPOINT_DIR}/latest.env"
echo "================================================================"
echo "  Ultra compaction endpoint LIVE"
echo "    URL:        \${MODEL_BASE_URL}"
echo "    model:      ${SERVED_MODEL_NAME}"
echo "    endpoint:   \${ENDPOINT_ENV}"
echo "    (also symlinked: ${ENDPOINT_DIR}/latest.env)"
echo "  Launch the rollout with:"
echo "    ENDPOINT_FILE=\${ENDPOINT_ENV} bash ${OC_SCRIPTS_DIR}/launcher_scripts/run_glm51_ultra_compaction.sh"
echo "  This job stays up until it hits its time limit (${TIME_LIMIT}) or is cancelled."
echo "================================================================"

# Keep the job alive as long as vLLM serves.
wait \${VLLM_PID}
SBATCH_EOF

echo "Submitting Ultra compaction endpoint job (${NODES_PER_JOB} nodes, TP=${TP_SIZE})..."
JID=$(sbatch --parsable "${JOB_SCRIPT}")
echo "============================================================"
echo "  Submitted job ${JID}"
echo "  Job script:    ${JOB_SCRIPT}"
echo "  Endpoint .env: ${ENDPOINT_DIR}/${JID}.env   (written once vLLM is ready)"
echo "  Watch:         tail -f ${ENDPOINT_DIR}/${JID}_${JOB_PREFIX}.out"
echo ""
echo "  Once ready, launch the rollout:"
echo "    ENDPOINT_FILE=${ENDPOINT_DIR}/${JID}.env bash ${OC_SCRIPTS_DIR}/launcher_scripts/run_glm51_ultra_compaction.sh"
echo "============================================================"
