#!/bin/bash
# =========================================================================
#  ABLATION ARM: model.limit.input = 80000  ->  compaction triggers at ~60000
#    (opencode default reserved = min(20000, max_output) = 20000; 80000 - 20000 = 60000)
#
#  Thin wrapper around run_glm51_opencode_source_sharded.sh. The original script
#  is NOT modified — this only exports the ablation knob + labels, then exec's it.
#  All other env vars / args pass through, e.g.:
#    NUM_SHARDS=2 bash run_glm51_oc_ablate_input80k.sh
# =========================================================================
set -euo pipefail
export MODEL_INPUT_LIMIT=80000
export MAX_RETRIES="${MAX_RETRIES:-3}"
export PARTITION="${PARTITION:-batch}"
export KEY_OFFSET="${KEY_OFFSET:-15}"                  # this arm owns tavily keys [15..29]
export JOB_PREFIX="${JOB_PREFIX:-glm51_trig60k}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_trig60k}"   # run dir: runs/sharded_source/glm_5_1_trig60k/<ts>_<slug>
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_glm51_opencode_source_sharded.sh" "$@"
