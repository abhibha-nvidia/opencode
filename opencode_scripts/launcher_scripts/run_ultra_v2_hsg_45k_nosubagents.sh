#!/bin/bash
# =========================================================================
#  ULTRA (sft_mix_0604 v2, step 0000485) ABLATION ARM: NO SUBAGENTS
#
#  Same checkpoint / serve recipe / frankenstein "remove one tool line" prompt /
#  ~45k overflow trigger as run_ultra_v2_hsg_45k_remove_one_tool_line.sh, but
#  DISABLES subagents: the `task` tool is denied so the model cannot spawn
#  explore/general subagents.
#
#  HOW it is wired (knob threaded by spinup_sftmix0604_oc_source_sharded_hsg_n4post.sh):
#    DISABLE_SUBAGENTS=1  -> agent.build.permission gets "task": "deny"
#                           -> opencode Permission.disabled() strips `task` from the
#                              request schema entirely (it will NOT appear in the
#                              model's tools / vLLM <tools> block, and cannot be
#                              invoked). Default (0) keeps subagents ENABLED.
#
#  SMOKE TEST (1 shard, 1 retry) -- this is the default invocation:
#    bash run_ultra_v2_hsg_45k_nosubagents.sh
#  Scale up via env, e.g.:
#    NUM_SHARDS=15 MAX_RETRIES=3 bash run_ultra_v2_hsg_45k_nosubagents.sh
#
#  Verify after it runs:
#    * `task` NOT in the vLLM <tools> block:
#        grep -c 'name>task</name>' <run>/vllm_logs/vllm_shard0_*.log   # expect 0
#    * `task` never invoked in the trajectory:
#        python check on trajectories.jsonl: no tool part with tool=="task"
#
#  Tavily key window KEY_OFFSET=180 (disjoint from summary [120..], nocompaction
#  [135..], discard [150..]).
# =========================================================================
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../opencode_scripts/launcher_scripts
OPENCODE_HSG_REPO="$(cd "${THIS_DIR}/../.." && pwd)"             # .../opencode-hsg (repo root)
SPINUP="${OPENCODE_HSG_REPO}/opencode_scripts/spinup_scripts/spinup_sftmix0604_oc_source_sharded_hsg_n4post.sh"

# --- model + container for this checkpoint (venkats' sft_mix_0604 v2 step 485) ---
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/training_actual_0603/runs/checkpoints/sft_mix_0604_v2_only_192k_128n_mem900g_20260605/eval/0000485/hf}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-sft_mix_0604_v2_step0000485}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/dmosallanezh/containers/vllm-hsg-0.17.0.sqsh}"

# --- this-cluster-correct defaults (account nemotron_n4_post) ---
export VENV_PY="${VENV_PY:-/lustre/fsw/portfolios/llmservice/users/abhibhag/frankie_evals/.venv/bin/python3}"
export ACCOUNT="${ACCOUNT:-nemotron_n4_post}"
export PARTITION="${PARTITION:-batch}"
export QOS="${QOS:-interactive}"   # valid for nemotron_n4_post: interactive/normal

# --- unified-format driver (this arm only): writes <run>/complete_trajectories/
# (root trajectory.json + subagents/ + tree.json) per sample AND a root-session
# trajectories.jsonl (fixes the legacy 'newest session' export bug). Other arms
# keep the stock driver. ---
export DRIVER_PY="${DRIVER_PY:-${OPENCODE_HSG_REPO}/opencode_scripts/analysis_scripts/launch_opencode_evals_unified.py}"

# --- ablation knobs (this arm: subagents OFF, otherwise = frankenstein 45k arm) ---
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${OPENCODE_HSG_REPO}/packages/opencode/src/session/prompt/frankenstein_system_prompt_remove_one_tool_line.txt}"
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-65000}"     # overflow trigger at ~45k
export MAX_STEPS="${MAX_STEPS:-400}"                       # hard step cap per rollout
export DISABLE_SUBAGENTS="${DISABLE_SUBAGENTS:-1}"         # this arm's defining choice: deny the `task` tool

# --- tavily keys (this cluster) ---
export TAVILY_KEYS_FILE="${TAVILY_KEYS_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/.cache/tavily_keys.txt}"

# --- smoke-test defaults (1 shard / 1 retry); override via env to scale ---
export NUM_SHARDS="${NUM_SHARDS:-1}"
export MAX_RETRIES="${MAX_RETRIES:-1}"
export KEY_OFFSET="${KEY_OFFSET:-180}"
export JOB_PREFIX="${JOB_PREFIX:-sftmix0604_n4post_nosub}"
export MODEL_GROUP="${MODEL_GROUP:-sft_mix_0604_v2_n4post_nosubagents_frankprompt}"

# --- harness version (optional): pin to a git commit for reproducibility ---
export OPENCODE_PIN_COMMIT="${OPENCODE_PIN_COMMIT:-}"

[[ -f "${SPINUP}" ]] || { echo "ERROR: spinup script not found: ${SPINUP}" >&2; exit 1; }
exec "${SPINUP}" "$@"
