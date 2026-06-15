import type { Config } from "@/config/config"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import type { MessageV2 } from "./message-v2"
import { Log } from "@opencode-ai/core/util/log"

const COMPACTION_BUFFER = 20_000
const log = Log.create({ service: "session.overflow" })

export function usable(input: { cfg: ConfigV1.Info; model: Provider.Model; outputTokenMax?: number }) {
  const context = input.model.limit.context
  if (context === 0) return 0

  const reserved =
    input.cfg.compaction?.reserved ??
    Math.min(COMPACTION_BUFFER, ProviderTransform.maxOutputTokens(input.model, input.outputTokenMax))
  return input.model.limit.input
    ? Math.max(0, input.model.limit.input - reserved)
    : Math.max(0, context - ProviderTransform.maxOutputTokens(input.model, input.outputTokenMax))
}

export function isOverflow(input: {
  cfg: ConfigV1.Info
  tokens: SessionV1.Assistant["tokens"]
  model: Provider.Model
  outputTokenMax?: number
}) {
  if (input.cfg.compaction?.auto === false) return false
  if (input.model.limit.context === 0) return false

  const count =
    input.tokens.total || input.tokens.input + input.tokens.output + input.tokens.cache.read + input.tokens.cache.write
  const threshold = usable(input)
  const triggered = count >= threshold
  // [COMPACTION_DEBUG] verify limit.input is honored and see when compaction fires.
  // threshold should equal (limit.input - reserved); e.g. input=50000, reserved=20000 -> 30000.
  log.info("[COMPACTION_DEBUG] overflow check", {
    count,
    threshold,
    limitInput: input.model.limit.input ?? null,
    limitContext: input.model.limit.context,
    reserved: input.cfg.compaction?.reserved ?? null,
    triggered,
  })
  return triggered
}
