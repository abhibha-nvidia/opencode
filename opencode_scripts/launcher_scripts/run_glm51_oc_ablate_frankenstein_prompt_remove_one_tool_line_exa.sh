#!/bin/bash
# =========================================================================
#  ABLATION ARM: NEW SYSTEM PROMPT ("frankenstein" research prompt) + EXA SEARCH
#
#  Same ablation as run_glm51_oc_ablate_frankenstein_prompt.sh (model.limit.input
#  = 65000 -> compaction triggers at ~45000), but the search backend is opencode's
#  built-in websearch (Exa provider) + webfetch tools instead of the tavily MCP.
#  It therefore execs the EXA spinup script:
#    run_glm51_opencode_source_sharded_exa.sh
#  (no "mcp" block in opencode.json; websearch/webfetch are "allow", not "deny").
#
#  The prompt swap itself lives in the opencode source:
#    packages/opencode/src/session/system.ts  -> returns PROMPT_FRANKENSTEIN
#    packages/opencode/src/session/prompt/frankenstein_system_prompt.txt
#  Because this runs opencode FROM SOURCE (Bun), that change takes effect automatically.
#
#  Launches 15 shards. KEY_OFFSET rotates the Exa key pool (exa_keys.txt). NOTE:
#  the pool currently has only ~10 keys, so with 15 shards keys WILL be reused
#  regardless of offset (the spinup script warns about this). Run with `bash`:
#    bash run_glm51_oc_ablate_frankenstein_prompt_exa.sh
#  Override shard count etc. via env, e.g. NUM_SHARDS=2 bash run_..._exa.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=65000
# System-prompt knob (this is the arm's defining choice). Points at the prompt
# file used for every rollout in this run; the spinup script propagates it as
# OPENCODE_SYSTEM_PROMPT_FILE into each shard, and the opencode source uses its
# contents verbatim (overriding system.ts model-id routing). Swap this path to
# run a different prompt arm without touching source. Empty => stock default.
export SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode/packages/opencode/src/session/prompt/frankenstein_system_prompt_remove_one_tool_line.txt}"
export NUM_SHARDS="${NUM_SHARDS:-15}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-0}"                              # rotates the exa key pool
export JOB_PREFIX="${JOB_PREFIX:-glm51_frankprompt_exa}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_frankprompt_trig45k_frankenstein_system_prompt_remove_one_tool_line_exa}"  # run dir: runs/sharded_source/glm_5_1_frankprompt_trig45k_exa/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded_exa.sh" "$@"
