#!/bin/bash
# =========================================================================
#  ULTRA NVFP4 (Nemotron-3-Ultra-550B-A55B-NVFP4) ABLATION ARM:
#    model.limit.input = 65000  ->  compaction (context reset) triggers at ~45000
#    (opencode default reserved = min(20000, max_output) = 20000; 65000 - 20000 = 45000)
#
#  NOTE: input=65000 triggers at 45k, NOT 65k. For a 65k trigger, set MODEL_INPUT_LIMIT=85000.
#
#  Thin wrapper around run_ultra_nvfp4_opencode_source_sharded.sh (original NOT modified).
#  Mirrors run_ultra_sftmix0604_trig45k.sh, but points at the NVFP4-quantized Ultra
#  checkpoint + vLLM 0.20.0 container (single-node TP=8) used by jkyi's
#  serve_ultra_ga_nvfp4.slurm, and overrides the base script defaults for cw-dfw.
#
#  Runs the opencode harness FROM SOURCE (via Bun) so harness edits take effect.
#  Override anything via env, e.g.:
#    NUM_SHARDS=1 MAX_RETRIES=1 PARTITION=interactive QOS=normal bash run_ultra_nvfp4_trig45k.sh
# =========================================================================
set -euo pipefail

# --- the ablation knob: input limit 65000 => context reset at ~45k ---
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-65000}"

# --- NVFP4 model + container (jkyi's serve_ultra_ga_nvfp4 checkpoint/image) ---
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/jkyi/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/snapshots/688671def6031a28f31635804c7856497db7f6a1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron-ultra-550b-nvfp4}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/sgunasekar/images/vllm-0.20.0-latest.sqsh}"

# --- cw-dfw cluster-correct defaults ---
export VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/.env/bin/python3}"
export ACCOUNT="${ACCOUNT:-nemotron_agents_dev}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-normal}"   # batch wants QOS=normal (base default 'interactive' is rejected on batch)

# --- run labels + tavily key window (disjoint from other arms) ---
export MAX_RETRIES="${MAX_RETRIES:-3}"
export KEY_OFFSET="${KEY_OFFSET:-105}"
export JOB_PREFIX="${JOB_PREFIX:-nvfp4_trig45k}"
export MODEL_GROUP="${MODEL_GROUP:-nemotron_ultra_550b_nvfp4_trig45k}"  # run dir: runs/sharded_source/<group>/<ts>_<slug>

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ultra_nvfp4_opencode_source_sharded.sh" "$@"
