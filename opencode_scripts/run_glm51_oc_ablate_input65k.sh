#!/bin/bash
# =========================================================================
#  ABLATION ARM: model.limit.input = 65000  ->  compaction triggers at ~45000
#    (opencode default reserved = min(20000, max_output) = 20000; 65000 - 20000 = 45000)
#
#  NOTE: input=65000 triggers at 45k, NOT 65k. For a 65k trigger, set
#        MODEL_INPUT_LIMIT=85000 instead.
#
#  Thin wrapper around run_glm51_opencode_source_sharded.sh. The original script
#  is NOT modified — this only exports the ablation knob + labels, then exec's it.
#  All other env vars / args pass through, e.g.:
#    NUM_SHARDS=2 bash run_glm51_oc_ablate_input65k.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=65000
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-30}"                  # this arm owns tavily keys [30..44]
export JOB_PREFIX="${JOB_PREFIX:-glm51_trig45k}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_trig45k}"   # run dir: runs/sharded_source/glm_5_1_trig45k/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded.sh" "$@"
