#!/bin/bash
# =========================================================================
#  ULTRA (sft_mix_0604 v2, step 0000485) ABLATION ARM: NO COMPACTION
#
#  Runs the step-485 bf16 checkpoint from opencode source (opencode-hsg branch,
#  Bun), tavily MCP enabled, the "remove one tool line" frankenstein system
#  prompt, and compaction FULLY DISABLED. HSG-correct serve (TP=8, 2 nodes x 4
#  GPUs, Ray). Account nemotron_n4_post.
#
#  How "no compaction" is enforced: DISABLE_COMPACTION=1 makes
#  run_sftmix0604_oc_source_sharded_hsg_n4post.sh emit "compaction":{"auto":false}
#  into each shard's opencode.json. overflow.ts:isOverflow() then returns false
#  unconditionally (packages/opencode/src/session/overflow.ts:30), so NO automatic
#  compaction ever fires -- both the token-threshold paths AND a server-side
#  ContextOverflowError. Only an explicit /compact (unreachable in non-interactive
#  rollouts) is not gated.
#
#  IMPLICATION: any rollout that would have overflowed the model context now
#  ERRORS OUT (finish="error") rather than continuing. Grading / summarize must
#  treat those as errored runs. MODEL_INPUT_LIMIT is intentionally left UNSET --
#  with compaction off, the threshold knob is a no-op.
#
#  Branch: runs opencode FROM SOURCE out of THIS repo (opencode-hsg). Ensure deps
#  are installed (cd <opencode-hsg> && bun/bin/bun install) before first run.
#
#  Launches 15 shards. KEY_OFFSET=135 -> owns tavily keys [135..149] (disjoint
#  from the frankenstein-compaction arm's [120..134]). Run with `bash`, NOT sbatch:
#    bash run_ultra_sftmix0604_n4post_nocompaction.sh
#  Override shard count etc. via env, e.g. NUM_SHARDS=2 bash run_..._nocompaction.sh
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

# --- ablation knobs ---
export DISABLE_COMPACTION="${DISABLE_COMPACTION:-1}"            # this arm's defining choice: compaction OFF
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${OPENCODE_HSG_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt_remove_one_tool_line.txt}"
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-}"               # no-op with compaction off; left unset on purpose
# MAX_STEPS intentionally left UNSET -> unbounded (overflowing rollouts error out)

# --- tavily keys (this cluster) ---
export TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"

# --- run labels + tavily key window (disjoint: compaction arm owns [120..134]) ---
export MAX_RETRIES="${MAX_RETRIES:-3}"
export KEY_OFFSET="${KEY_OFFSET:-135}"
export JOB_PREFIX="${JOB_PREFIX:-sftmix0604_n4post_nocompact}"
export MODEL_GROUP="${MODEL_GROUP:-sft_mix_0604_v2_n4post_nocompaction_frankprompt}"

exec "${THIS_DIR}/run_sftmix0604_oc_source_sharded_hsg_n4post.sh" "$@"
