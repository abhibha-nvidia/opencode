#!/bin/bash
# =========================================================================
#  ULTRA (sft_mix_0604 v2, step 0000485) ABLATION ARM:
#    model.limit.input = 65000  ->  compaction (context reset) triggers at ~45000
#    (opencode default reserved = min(20000, max_output) = 20000; 65000 - 20000 = 45000)
#
#  NOTE: input=65000 triggers at 45k, NOT 65k. For a 65k trigger, set MODEL_INPUT_LIMIT=85000.
#
#  Thin wrapper around run_ultra_v3_GA_opencode_source_sharded.sh (original NOT modified).
#  Mirrors run_ultra_oc_ablate_input65k.sh / run_glm51_oc_ablate_input65k.sh (-> glm_5_1_trig45k),
#  but points at the locally-cached sft_mix_0604 checkpoint + transferred container, and
#  overrides the base script's defaults that are wrong on cw-dfw (venv / account / container).
#
#  Runs the opencode harness FROM SOURCE (via Bun) so harness edits take effect.
#  Defaults to a massively-parallel sweep: 60 shards x 4 nodes/shard = 240 nodes,
#  3 retries each, batch partition (QOS=normal). Override anything via env, e.g.:
#    NUM_SHARDS=1 MAX_RETRIES=1 PARTITION=interactive QOS=normal bash run_ultra_sftmix0604_trig45k.sh
# =========================================================================
set -euo pipefail

# --- the ablation knob: input limit 65000 => context reset at ~45k ---
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-65000}"

# --- model + container for this checkpoint (the session's sft_mix ckpt) ---
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/local_models/sft_mix_0604_v2_only_192k_128n_mem900g_20260605_step0000485}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sft_mix_0604_v2_step0000485}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/containers/vllm-hsg-0.17.0.sqsh}"

# --- cw-dfw cluster-correct defaults (base script's defaults are for the oci box) ---
export VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/.env/bin/python3}"
export ACCOUNT="${ACCOUNT:-nemotron_agents_dev}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-normal}"   # batch wants QOS=normal (base default 'interactive' is rejected on batch)

# --- massively-parallel sweep: 60 shards (x4 nodes/shard = 240 nodes), 3 retries each ---
export NUM_SHARDS="${NUM_SHARDS:-60}"

# --- run labels + tavily key window: 60 shards from offset 90 => keys[90..149], all distinct
#     (disjoint from other arms: glm45k[30..44], ultra45k[75..89]; 161 keys total) ---
export MAX_RETRIES="${MAX_RETRIES:-3}"
export KEY_OFFSET="${KEY_OFFSET:-90}"
export JOB_PREFIX="${JOB_PREFIX:-sftmix_trig45k}"
export MODEL_GROUP="${MODEL_GROUP:-sft_mix_0604_v2_trig45k}"  # run dir: runs/sharded_source/sft_mix_0604_v2_trig45k/<ts>_<slug>

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ultra_v3_GA_opencode_source_sharded.sh" "$@"
