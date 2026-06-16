#!/bin/bash
# =========================================================================
#  grade_with_glm5.sh
#
#  Spins up GLM-5.1-FP8 as the judge using the SAME proven serving logic as
#  run_glm51_opencode_sharded.sh (TP=16 across 2 nodes, Ray head+worker),
#  waits for readiness, runs grade_trajectories.py against the local endpoint,
#  writes results.txt (accuracy + avg tool calls + avg context resets), then
#  the job exits (Slurm tears down the vLLM server).
#
#  Submit with:
#    TRAJECTORIES=/path/to/trajectories.jsonl \
#    OUTPUT=/path/to/grading/graded_results_glm5.jsonl \
#    VLLM_LOG_DIR=/path/to/grading \
#    sbatch --output=/path/to/grading/slurm_%j.out grade_with_glm5.sh
#
#  Uses full 8-GPU nodes (TP=16 = all 16 GPUs, no stranded subset -> compliant
#  with the Idle GPU monitor).
# =========================================================================
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -A nemotron_agents_dev
#SBATCH -t 04:00:00
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=glm5_judge

set -euo pipefail
set -x
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

# ============================ CONFIG (env-overridable) ===================
TRAJECTORIES="${TRAJECTORIES:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/runs/sharded/minimax_2_7/20260607_012016_minimax-m2.7/trajectories.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "${TRAJECTORIES}")/grading/graded_results_glm5.jsonl}"
NUM_PARALLEL="${NUM_PARALLEL:-32}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-400}"

# Judge model = GLM-5.1-FP8 (same model/container as the eval serving script)
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/hiteshis/models/GLM-5.1-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
MODEL_API_KEY="${MODEL_API_KEY:-token-abc123}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
SERVER_PORT="${SERVER_PORT:-12951}"
RAY_PORT="${RAY_PORT:-6379}"
VLLM_LOG_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/lvega/sqsh/vllm-openai-v0.20.1.sqsh}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$(dirname "${OUTPUT}")}"
GRADE_PY="${GRADE_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/grade_trajectories.py}"
SUMMARIZE_PY="${SUMMARIZE_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/opencode_scripts/summarize_results.py}"
RESULTS_TXT="${RESULTS_TXT:-$(dirname "${OUTPUT}")/results.txt}"
VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/browsecomp-interleaved-reasoning/.envs/bin/python3}"
# =========================================================================

mkdir -p "${VLLM_LOG_DIR}" "$(dirname "${OUTPUT}")"
VLLM_LOG="${VLLM_LOG_DIR}/vllm_${SLURM_JOB_ID}.log"

MOUNTS="/lustre:/lustre"
CONTAINER_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS}"

echo "========================================="
echo "  GLM-5.1-FP8 Judge (TP=16, 2 nodes)"
echo "  Job:          ${SLURM_JOB_ID}"
echo "  Trajectories: ${TRAJECTORIES}"
echo "  Output:       ${OUTPUT}"
echo "  Results:      ${RESULTS_TXT}"
echo "========================================="

NUM_NODES=$SLURM_JOB_NUM_NODES
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
# Robust head-node IP (real cluster fabric IP, not link-local)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

# --- Ray worker(s) ---
for ((i=1; i<NUM_NODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" ${CONTAINER_ARGS} \
        -o "${VLLM_LOG_DIR}/worker_${SLURM_JOB_ID}_${i}.log" \
        bash -c "
            pip install ray -q
            ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=8 --block
        " &
done

# --- Ray head + vLLM serve (GLM-5.1, TP=16) ---
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

# --- run grader against the local GLM-5.1 endpoint ---
echo "========================================="
echo "  Running grade_trajectories.py"
echo "  Model:  ${SERVED_MODEL_NAME}  at  ${MODEL_BASE_URL}"
echo "========================================="
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
    echo "Writing results summary -> ${RESULTS_TXT}"
    "${VENV_PY}" "${SUMMARIZE_PY}" \
        --trajectories  "${TRAJECTORIES}" \
        --graded        "${OUTPUT}" \
        --output        "${RESULTS_TXT}" \
        --total-samples "${TOTAL_SAMPLES}" \
        |& tee "${VLLM_LOG_DIR}/results_summary_${SLURM_JOB_ID}.log" || echo "WARN: summarize failed"
else
    echo "WARN: SUMMARIZE_PY not found: ${SUMMARIZE_PY}"
fi

echo "Grading complete. Results: ${OUTPUT}"
echo "Summary: ${RESULTS_TXT}"
# Job exit releases the allocation and tears down the vLLM server.
exit 0
