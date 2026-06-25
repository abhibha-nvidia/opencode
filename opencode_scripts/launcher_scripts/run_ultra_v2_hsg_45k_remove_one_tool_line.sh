#!/bin/bash
# =========================================================================
#  ULTRA (sft_mix_0604 v2, step 0000485) -- opencode rollout on nemotron_n4_post,
#  HSG cluster (gpu:4 per node), frankenstein (remove-one-tool-line) prompt arm.
#
#  Thin wrapper around run_sftmix0604_oc_source_sharded_hsg_n4post.sh (NOT modified).
#  Runs the opencode harness FROM SOURCE out of THIS repo (opencode-hsg, the
#  ablation branch) via Bun, so local harness edits take effect. Serves venkats'
#  bf16 sft_mix_0604_v2 step-0000485 HF checkpoint with the HSG-correct vLLM serve
#  recipe (TP=8 across 2 nodes x 4 GPUs, Ray) -- mirrors the bf16 spinup, NOT the
#  DFW single-node / pipeline-parallel layout.
#
#  Configurable knobs (mirrors run_glm51_oc_ablate_frankenstein_prompt_*_maxstep400.sh):
#    SYSTEM_PROMPT_FILE  abs path to a .txt; exported as OPENCODE_SYSTEM_PROMPT_FILE
#                        and used verbatim as the system prompt for EVERY rollout
#                        (overrides system.ts model-id routing). Default: the
#                        frankenstein "remove one tool line" prompt in THIS repo.
#    MODEL_INPUT_LIMIT   model.limit.input=N; compaction triggers at ~N-20000.
#                        Default 65000 -> compaction at ~45k (the trig45k arm).
#    MAX_STEPS           agent.build.steps=N; hard step cap per rollout. Default 400.
#  Override any of them (and NUM_SHARDS / MAX_RETRIES / QOS / KEY_OFFSET / ...) via env.
#
#  PREREQUISITE (once): opencode-hsg deps differ from the sibling `opencode`
#  checkout, so install them before first run:
#    cd /lustre/fsw/portfolios/llmservice/users/abhibhag/opencode-hsg && bun/bin/bun install
#
#  SMOKE TEST (1 shard, 1 retry):
#    NUM_SHARDS=1 MAX_RETRIES=1 bash run_ultra_sftmix0604_n4post.sh
#
#  Override anything via env, e.g.:
#    NUM_SHARDS=15 SYSTEM_PROMPT_FILE=/abs/other_prompt.txt bash run_ultra_sftmix0604_n4post.sh
# =========================================================================
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_HSG_REPO="$(cd "${THIS_DIR}/.." && pwd)"

# --- model + container for this checkpoint (venkats' sft_mix_0604 v2 step 485) ---
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/runs/checkpoints/sft_mix_0604_v2_only_192k_128n_mem900g_20260605/eval/0000485/hf}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sft_mix_0604_v2_step0000485}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/dmosallanezh/containers/vllm-hsg-0.17.0.sqsh}"

# --- this-cluster-correct defaults (account nemotron_n4_post) ---
export VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"
export ACCOUNT="${ACCOUNT:-nemotron_n4_post}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-interactive}"   # valid for nemotron_n4_post: interactive/normal (NOT nemotron-priority)

# --- ablation knobs (configurable; defaults = frankenstein remove-one-tool-line arm) ---
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${OPENCODE_HSG_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt_remove_one_tool_line.txt}"
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-65000}"   # compaction triggers at ~45k
export MAX_STEPS="${MAX_STEPS:-400}"                     # hard step cap per rollout

# --- tavily keys (this cluster) ---
export TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"

# --- run labels + tavily key window (disjoint from other arms) ---
export MAX_RETRIES="${MAX_RETRIES:-3}"
export KEY_OFFSET="${KEY_OFFSET:-120}"
export JOB_PREFIX="${JOB_PREFIX:-sftmix0604_n4post_frank}"
export MODEL_GROUP="${MODEL_GROUP:-sft_mix_0604_v2_n4post_frankprompt}"  # run dir: runs/sharded_source/<group>/<ts>_<slug>

exec "${THIS_DIR}/run_sftmix0604_oc_source_sharded_hsg_n4post.sh" "$@"
