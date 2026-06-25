#!/bin/bash
# =========================================================================
#  ABLATION ARM: model.limit.input = 95000  ->  compaction triggers at ~75000
#    (opencode default reserved = min(20000, max_output) = 20000; 95000 - 20000 = 75000)
#
#  Thin wrapper around run_glm51_opencode_source_sharded.sh. The original script
#  is NOT modified — this only exports the ablation knob + labels, then exec's it.
#  All other env vars / args pass through, e.g.:
#    NUM_SHARDS=2 bash run_glm51_oc_ablate_input95k.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=95000
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-0}"                   # this arm owns tavily keys [0..14]
export JOB_PREFIX="${JOB_PREFIX:-glm51_trig75k}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_trig75k}"   # run dir: runs/sharded_source/glm_5_1_trig75k/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded.sh" "$@"
