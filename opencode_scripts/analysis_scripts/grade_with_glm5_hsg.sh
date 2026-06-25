#!/bin/bash
# =========================================================================
#  grade_with_glm5_hsg.sh  -- HSG-cluster (gpu:4/node) port of grade_with_glm5.sh
#
#  Serves GLM-5.1-FP8 as the judge (TP=16) and runs grade_trajectories.py against
#  the local endpoint, then summarize_results.py -> results.txt (accuracy etc).
#
#  Differences vs grade_with_glm5.sh (which assumed 8-GPU nodes / DFW paths):
#    * TP=16 -> 4 nodes x 4 GPUs here (was 2 nodes x 8). Ray uses --num-gpus=4.
#    * Account nemotron_n4_post; GLM-5.1-FP8 + vllm-openai-glm51 container that
#      actually exist on this cluster; frankie_evals venv as the grader python.
#
#  Submit with (defaults already point at the current sftmix0604 n4post run):
#    TRAJECTORIES=/path/to/trajectories.jsonl \
#    sbatch -A nemotron_n4_post --qos normal \
#      --output=/path/to/grading/slurm_%j.out grade_with_glm5_hsg.sh
# =========================================================================
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -A nemotron_n4_post
#SBATCH -t 04:00:00
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=glm51_judge_hsg

set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

# ============================ CONFIG (env-overridable) ===================
RUN_DIR_DEFAULT="/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode-hsg/opencode_scripts/runs/sharded_source/sft_mix_0604_v2_n4post_frankprompt/20260619_001114_sft_mix_0604_v2_step0000485"
TRAJECTORIES="${TRAJECTORIES:-${RUN_DIR_DEFAULT}/trajectories.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "${TRAJECTORIES}")/grading/graded_results_glm5.jsonl}"
NUM_PARALLEL="${NUM_PARALLEL:-32}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-400}"

# Judge model = GLM-5.1-FP8 (paths that exist on THIS cluster)
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/boxinw/models/GLM-5.1-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
SERVER_PORT="${SERVER_PORT:-12961}"
RAY_PORT="${RAY_PORT:-6380}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/lvega/sqsh/vllm-openai-glm51.sqsh}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$(dirname "${OUTPUT}")}"
GRADE_PY="${GRADE_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode-hsg/opencode_scripts/grade_trajectories.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode-hsg/opencode_scripts/summarize_results.py}"
RESULTS_TXT="${RESULTS_TXT:-$(dirname "${OUTPUT}")/results.txt}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"
# =========================================================================

mkdir -p "${VLLM_LOG_DIR}" "$(dirname "${OUTPUT}")"
VLLM_LOG="${VLLM_LOG_DIR}/vllm_${SLURM_JOB_ID}.log"

MOUNTS="/lustre:/lustre"
CONTAINER_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS}"

echo "========================================="
echo "  GLM-5.1-FP8 Judge (TP=16, 4 nodes x 4 GPUs)"
echo "  Job:          ${SLURM_JOB_ID}"
echo "  Trajectories: ${TRAJECTORIES}"
echo "  Output:       ${OUTPUT}"
echo "  Results:      ${RESULTS_TXT}"
echo "========================================="

NUM_NODES=$SLURM_JOB_NUM_NODES
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

# --- Ray worker(s): one per non-head node, 4 GPUs each ---
for ((i=1; i<NUM_NODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" ${CONTAINER_ARGS} \
        -o "${VLLM_LOG_DIR}/worker_${SLURM_JOB_ID}_${i}.log" \
        bash -c "
            pip install ray -q
            ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=4 --block
        " &
done

# --- Ray head + vLLM serve (GLM-5.1, TP=16 across 16 GPUs) ---
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" ${CONTAINER_ARGS} -o "${VLLM_LOG}" bash -c "
    set -euo pipefail
    pip install ray -q
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL}
    ray start --head --port=${RAY_PORT} --num-gpus=4 --dashboard-host=0.0.0.0
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
        --trust-remote-code \
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

# --- run grader against the local GLM-5.1 endpoint ---
echo "Running grade_trajectories.py against ${SERVED_MODEL_NAME} at ${MODEL_BASE_URL}"
"${VENV_PY}" "${GRADE_PY}" \
    --trajectories  "${TRAJECTORIES}" \
    --output        "${OUTPUT}" \
    --model         "${SERVED_MODEL_NAME}" \
    --base-url      "${MODEL_BASE_URL}" \
    --api-key       "${MODEL_API_KEY}" \
    --num-parallel  "${NUM_PARALLEL}" \
    --total-samples "${TOTAL_SAMPLES}" \
    --resume \
    |& tee "${VLLM_LOG_DIR}/grading_${SLURM_JOB_ID}.log"

# --- write results.txt (accuracy + avg tool calls + avg context resets) ---
if [[ -f "${SUMMARIZE_PY}" ]]; then
    "${VENV_PY}" "${SUMMARIZE_PY}" \
        --trajectories  "${TRAJECTORIES}" \
        --graded        "${OUTPUT}" \
        --output        "${RESULTS_TXT}" \
        --total-samples "${TOTAL_SAMPLES}" \
        |& tee "${VLLM_LOG_DIR}/results_summary_${SLURM_JOB_ID}.log" || echo "WARN: summarize failed"
fi

echo "Grading complete. Results: ${OUTPUT}"
echo "Summary: ${RESULTS_TXT}"
exit 0
