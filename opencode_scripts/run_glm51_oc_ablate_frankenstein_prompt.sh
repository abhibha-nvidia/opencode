#!/bin/bash
# =========================================================================
#  ABLATION ARM: NEW SYSTEM PROMPT ("frankenstein" research prompt)
#
#  Identical harness config to run_glm51_oc_ablate_input65k.sh
#  (model.limit.input = 65000 -> compaction triggers at ~45000), so this is a
#  clean A/B against the glm_5_1_trig45k baseline run 20260610_013303 — the
#  ONLY difference is the system prompt.
#
#  The prompt swap itself lives in the opencode source:
#    packages/opencode/src/session/system.ts  -> returns PROMPT_FRANKENSTEIN
#    packages/opencode/src/session/prompt/frankenstein_system_prompt.txt
#  Because this runs opencode FROM SOURCE (Bun) via
#  run_glm51_opencode_source_sharded.sh, that change takes effect automatically.
#
#  Launches 15 shards. KEY_OFFSET=45 -> owns tavily keys [45..59] (no overlap
#  with the input65k arm's [30..44]). Run with `bash`, NOT sbatch:
#    bash run_glm51_oc_ablate_frankenstein_prompt.sh
#  Override shard count etc. via env, e.g. NUM_SHARDS=2 bash run_..._prompt.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=65000
# System-prompt knob (this is the arm's defining choice). Points at the prompt
# file used for every rollout in this run; the spinup script propagates it as
# OPENCODE_SYSTEM_PROMPT_FILE into each shard, and the opencode source uses its
# contents verbatim (overriding system.ts model-id routing). Swap this path to
# run a different prompt arm without touching source. Empty => stock default.
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/packages/opencode/src/session/prompt/frankenstein_system_prompt.txt}"
export NUM_SHARDS="${NUM_SHARDS:-15}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-45}"                          # this arm owns tavily keys [45..59]
export JOB_PREFIX="${JOB_PREFIX:-glm51_frankprompt}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_frankprompt_trig45k}"  # run dir: runs/sharded_source/glm_5_1_frankprompt_trig45k/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded.sh" "$@"
