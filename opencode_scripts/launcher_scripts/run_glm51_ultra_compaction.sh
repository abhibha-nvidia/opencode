#!/bin/bash
# =========================================================================
#  run_glm51_ultra_compaction.sh   --   RUN WITH `bash`, NOT sbatch.
#
#  EXPERIMENT: GLM rolls out from the input prompt; at the ~45k compaction
#  trigger, opencode hands the history to a SEPARATE live Ultra endpoint to
#  write the anchored summary; GLM then resumes on [summary + retained tail +
#  continue]. This is opencode's NATIVE post-compaction behavior -- the only
#  change is that the `compaction` agent uses Ultra instead of GLM.
#
#  HOW (no harness or driver edits): this is a thin wrapper around the UNMODIFIED
#  run_glm51_opencode_source_sharded.sh. It only:
#    1. sets MODEL_INPUT_LIMIT=65000  -> compaction fires at ~45k (limit - 20000),
#    2. writes an OPENCODE_CONFIG overlay that ADDS a second provider ("ultra")
#       + sets agent.compaction.model="ultra/<model>",
#    3. exports OPENCODE_CONFIG so opencode DEEP-MERGES the overlay on top of the
#       per-shard opencode.json (GLM provider). Disjoint keys (provider.ultra,
#       agent.compaction) coexist with provider.local / agent.build, confirmed in
#       config.ts (OPENCODE_CONFIG merged at load, then project config merged).
#
#  PREREQUISITE: a live Ultra endpoint from spinup_ultra_compaction_endpoint.sh,
#  on the SAME cluster as the GLM rollout (shards must reach its node IP).
#
#  USAGE (point at the endpoint .env the spinup wrote):
#    ENDPOINT_FILE=.../runs/_compaction_endpoints/<jobid>.env \
#      bash run_glm51_ultra_compaction.sh
#  OR pass the endpoint pieces directly:
#    COMPACTION_BASE_URL=http://<ip>:12952/v1 COMPACTION_MODEL=ultra_compaction \
#      bash run_glm51_ultra_compaction.sh
#  Any other env (NUM_SHARDS, ACCOUNT, PARTITION, ...) passes through to the driver.
# =========================================================================
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../launcher_scripts
OC_SCRIPTS_DIR="$(cd "${THIS_DIR}/.." && pwd)"             # .../opencode_scripts
INNER_DRIVER="${INNER_DRIVER:-${OC_SCRIPTS_DIR}/run_glm51_opencode_source_sharded.sh}"

# --- resolve the Ultra compaction endpoint ---
# Prefer an endpoint .env (written by the spinup) if given; else require the
# pieces directly. ENDPOINT_FILE may also be the spinup's latest.env symlink.
if [[ -n "${ENDPOINT_FILE:-}" ]]; then
    [[ -f "${ENDPOINT_FILE}" ]] || { echo "ERROR: ENDPOINT_FILE not found: ${ENDPOINT_FILE}" >&2; exit 1; }
    # shellcheck disable=SC1090
    source "${ENDPOINT_FILE}"
fi
: "${COMPACTION_BASE_URL:?ERROR: set COMPACTION_BASE_URL (or ENDPOINT_FILE) -- the live Ultra endpoint, e.g. http://<ip>:12952/v1}"
: "${COMPACTION_MODEL:?ERROR: set COMPACTION_MODEL (or ENDPOINT_FILE) -- the Ultra served-model name}"
COMPACTION_API_KEY="${COMPACTION_API_KEY:-token-abc123}"
COMPACTION_MAX_MODEL_LEN="${COMPACTION_MAX_MODEL_LEN:-131072}"

# --- experiment knob: compaction trigger point ---
# MODEL_INPUT_LIMIT -> model.limit.input on the GLM (rollout) model. Compaction
# fires at (limit - 20000) since compaction.reserved is left at the default.
export MODEL_INPUT_LIMIT="${MODEL_INPUT_LIMIT:-65000}"   # -> first compaction at ~45k

# --- compaction knobs: KEPT AT OPENCODE DEFAULTS (intentionally NOT set) ---
#   tail_turns (=2), preserve_recent_tokens (~25% usable, clamp 2k-8k),
#   reserved (=min(20000,maxOut)=20000), prune (=false), auto (=true).
# To pin any of them later, add a "compaction": {...} block to the overlay below.

# --- run labels / tavily key window (disjoint from other arms) ---
export KEY_OFFSET="${KEY_OFFSET:-150}"
export JOB_PREFIX="${JOB_PREFIX:-glm51_ultracompact}"
export MODEL_GROUP="${MODEL_GROUP:-glm_5_1_ultra_compaction}"

# --- write the OPENCODE_CONFIG overlay (adds Ultra provider + compaction agent) ---
OVERLAY_DIR="${OVERLAY_DIR:-${OC_SCRIPTS_DIR}/runs/_compaction_overlays}"
mkdir -p "${OVERLAY_DIR}"
OVERLAY="${OVERLAY_DIR}/ultra_compaction_$(date +%Y%m%d_%H%M%S)_$$.json"
cat > "${OVERLAY}" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ultra": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ultra compaction endpoint (vLLM)",
      "options": { "baseURL": "${COMPACTION_BASE_URL}", "apiKey": "${COMPACTION_API_KEY}" },
      "models": {
        "${COMPACTION_MODEL}": { "name": "Nemotron Ultra (compaction)", "limit": { "context": ${COMPACTION_MAX_MODEL_LEN}, "output": 32768 } }
      }
    }
  },
  "agent": {
    "compaction": { "model": "ultra/${COMPACTION_MODEL}" }
  }
}
JSON
export OPENCODE_CONFIG="${OVERLAY}"

echo "============================================================"
echo "  GLM rollout + ULTRA compaction"
echo "    GLM compaction trigger:  MODEL_INPUT_LIMIT=${MODEL_INPUT_LIMIT} (fires at ~$((MODEL_INPUT_LIMIT-20000)))"
echo "    Ultra (compaction) URL:  ${COMPACTION_BASE_URL}"
echo "    Ultra model:             ultra/${COMPACTION_MODEL}"
echo "    OPENCODE_CONFIG overlay: ${OPENCODE_CONFIG}"
echo "    compaction knobs:        opencode defaults (tail_turns=2, preserve_recent~8k, reserved=20k, prune=off)"
echo "    inner driver:            ${INNER_DRIVER}"
echo "============================================================"
echo "  NOTE: ensure the Ultra endpoint and the GLM shards are on the SAME"
echo "        cluster (the shards open ${COMPACTION_BASE_URL} over the network)."
echo "============================================================"

[[ -f "${INNER_DRIVER}" ]] || { echo "ERROR: inner driver not found: ${INNER_DRIVER}" >&2; exit 1; }
exec "${INNER_DRIVER}" "$@"
