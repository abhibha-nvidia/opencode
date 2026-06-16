#!/bin/bash
# =========================================================================
#  ABLATION ARM: frankenstein system prompt + MAX_STEPS=400 cap
#
#  Identical to run_glm51_oc_ablate_frankenstein_prompt.sh (same frankenstein
#  system prompt, same model.limit.input=65000 -> compaction triggers at
#  ~45000) EXCEPT each rollout is hard-capped at 400 agent steps. This is a
#  clean A/B against the uncapped frankenstein arm — the ONLY difference is the
#  step cap.
#
#  How the cap is applied: MAX_STEPS=400 is propagated by
#  run_glm51_opencode_source_sharded.sh into each shard's opencode.json as
#  agent.build.steps=400. The opencode source enforces it in
#  packages/opencode/src/session/prompt.ts (maxSteps = agent.steps ?? Infinity;
#  the run stops once step >= maxSteps).
#
#  Launches 15 shards. KEY_OFFSET=0 -> owns tavily keys [0..14] (no overlap with
#  the input65k arm's [30..44] or the frankenstein arm's [45..59]). Run with
#  `bash`, NOT sbatch:
#    bash run_glm51_oc_ablate_frankenstein_prompt_maxstep400.sh
#  Override shard count etc. via env, e.g. NUM_SHARDS=2 bash run_..._maxstep400.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=65000
# Step cap (this arm's defining choice). Each rollout stops after 400 agent steps.
export MAX_STEPS="${MAX_STEPS:-400}"
# System-prompt knob (same as the frankenstein arm).
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/packages/opencode/src/session/prompt/frankenstein_system_prompt_remove_one_tool_line.txt}"
export NUM_SHARDS="${NUM_SHARDS:-15}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-0}"                          # this arm owns tavily keys [0..14]
export JOB_PREFIX="${JOB_PREFIX:-glm51_frankprompt_s400}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_frankprompt_trig45k_maxstep400}"  # run dir: runs/sharded_source/<group>/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded.sh" "$@"
