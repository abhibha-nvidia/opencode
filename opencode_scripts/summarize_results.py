#!/usr/bin/env python3
"""
summarize_results.py -- compute single-run (absolute, no A/B) metrics for a
graded opencode run and write a Markdown report.

This mirrors the metric set that the BrowseComp harness computes per-model in
`ritu_temp/analysis/deep_compare/single_run_metrics.py` (consumed by
compare_2_models.py), adapted to the opencode trajectory format. Only ABSOLUTE
single-run values are emitted — no model-vs-model comparison.

Section parity with single_run_metrics
---------------------------------------
  * Overall Metrics      : accuracy, failed-to-answer, tool calls, context resets
  * Tool Call Breakdown  : search / browse-fetch / other invocations
  * Tool-Call Distribution by tool name
  * Hit Max Turns        : steps >= --max-steps (+ of which correct)
  * Tool Calls per Context-Reset Segment
  * Reasoning            : char proxy (avg reasoning chars per assistant msg)
  * Accuracy by Context-Reset Bucket
  * Search Query Stats   : dups, quoted, parallel-search proxies
  * Tool Output Lengths  : chars (or tokens with --tokenizer), per-tool
  * Malformed Tool Calls : opencode-native (invalid / error-status / raw tag)

opencode-format mapping
-----------------------
  * A trajectory is `record["messages"]`; each message has `parts`.
  * Tool call  = part with `type=="tool"` (`tool` name, `state.status`,
    `state.input`, `state.output`). "Tool calls" counts individual invocations
    (parts), which the breakdown partitions.
  * Context reset = assistant message with `summary is True` (opencode
    compaction). It delimits a reset segment; R resets -> R+1 segments.
  * Step = a `step-start` part (one LLM generation step).
  * Reasoning = `reasoning` parts' text.
  * Tool taxonomy:
      - search : tool name contains "search" or ends with "research"
      - browse : webfetch / *extract* / *crawl* / *map* (URL fetch/extract)
      - other  : bash, read, grep, glob, todowrite, ...
      - invalid: malformed call opencode could not build (counted as malformed)
  * Malformed = part with `tool=="invalid"`, or `state.status=="error"`, or a
    literal `<tool_call>` string leaking into assistant text/reasoning. These
    are opencode-native (BrowseComp's schema-specific categories don't apply).

is_correct / extracted answer are read from each record (top-level `is_correct`,
`grading.extracted_answer`); --graded can override accuracy from a separate
graded_results JSONL. The file is streamed so the large graded_trajectories.jsonl
is never fully materialised.
"""
import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Empty-answer sentinels (case-insensitive) for the "failed to produce answer" metric.
NO_ANSWER_SENTINELS = {"", "none", "null", "n/a", "na"}


# ── Tool taxonomy ──────────────────────────────────────────────────────────────

def tool_category(name):
    n = (name or "").lower()
    if n == "invalid":
        return "invalid"
    if "search" in n or n.endswith("research"):
        return "search"
    if n == "webfetch" or "extract" in n or "crawl" in n or "map" in n:
        return "browse"
    return "other"


# ── IO ──────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def iter_records(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ── Per-trajectory helpers ───────────────────────────────────────────────────────

def iter_tool_parts(messages):
    for m in (messages or []):
        for p in m.get("parts", []):
            if p.get("type") == "tool":
                yield p


def count_context_resets(messages):
    return sum(1 for m in (messages or []) if m.get("summary") is True)


def count_steps(messages):
    return sum(1 for m in (messages or [])
               for p in m.get("parts", []) if p.get("type") == "step-start")


def category_counts(messages):
    """Counter of tool-category -> invocations for one trajectory."""
    c = Counter()
    for p in iter_tool_parts(messages):
        c[tool_category(p.get("tool"))] += 1
    return c


def tool_name_counts(messages):
    return Counter(p.get("tool") or "<unknown>" for p in iter_tool_parts(messages))


def reasoning_chars(messages):
    """(total reasoning chars, number of assistant messages) for one trajectory."""
    total = 0
    assistant_msgs = 0
    for m in (messages or []):
        if m.get("role") == "assistant":
            assistant_msgs += 1
        for p in m.get("parts", []):
            if p.get("type") == "reasoning":
                total += len(p.get("text", "") or "")
    return total, assistant_msgs


def no_answer(record):
    g = record.get("grading")
    if isinstance(g, dict):
        ea = g.get("extracted_answer")
        return str(ea).strip().lower() in NO_ANSWER_SENTINELS if ea is not None else True
    # fall back to last assistant text
    for m in reversed(record.get("messages", []) or []):
        if m.get("role") == "assistant":
            txt = "".join(p.get("text", "") for p in m.get("parts", [])
                          if p.get("type") == "text")
            return txt.strip().lower() in NO_ANSWER_SENTINELS
    return True


def search_query_stats(messages):
    """Per-trajectory search-query stats over search-category tool calls.

    opencode's search tool takes a single `query` string; a `queries` list is
    also accepted. Returns counts: search_calls, queries, multi_q (calls issuing
    >1 query), dups (exact repeat within trajectory), quoted (contains a `"`).
    """
    seen = set()
    search_calls = queries = multi_q = dups = quoted = 0
    for p in iter_tool_parts(messages):
        if tool_category(p.get("tool")) != "search":
            continue
        search_calls += 1
        inp = p.get("state", {}).get("input")
        if not isinstance(inp, dict):
            continue
        q = inp.get("query")
        qlist = [q] if isinstance(q, str) else (inp.get("queries") or [])
        if isinstance(qlist, str):
            qlist = [qlist]
        qlist = [str(x).strip() for x in qlist if str(x).strip()]
        if len(qlist) > 1:
            multi_q += 1
        for qs in qlist:
            queries += 1
            if '"' in qs:
                quoted += 1
            if qs in seen:
                dups += 1
            seen.add(qs)
    return {"search_calls": search_calls, "queries": queries, "multi_q": multi_q,
            "dups": dups, "quoted": quoted}


def malformed_stats(messages):
    """opencode-native malformed/errored tool-call stats for one trajectory."""
    invalid = error = raw_tag = 0
    intended = Counter()
    error_by_tool = Counter()
    for m in (messages or []):
        for p in m.get("parts", []):
            t = p.get("type")
            if t == "tool":
                name = p.get("tool") or ""
                st = p.get("state", {})
                if name == "invalid":
                    invalid += 1
                    inp = st.get("input")
                    if isinstance(inp, dict):
                        intended[inp.get("tool") or "<unknown>"] += 1
                if st.get("status") == "error":
                    error += 1
                    error_by_tool[name] += 1
            elif t in ("text", "reasoning"):
                txt = p.get("text", "") or ""
                if "<tool_call>" in txt:
                    raw_tag += txt.count("<tool_call>")
    return {"invalid": invalid, "error": error, "raw_tag": raw_tag,
            "total": invalid + error + raw_tag,
            "intended": intended, "error_by_tool": error_by_tool}


# ── Preemptive-stop detection ───────────────────────────────────────────────────
# The agent loop ends when the model emits a turn with no tool calls (the last
# assistant message's finish == "stop"). A "preemptive stop" is when it ended
# VOLUNTARILY but produced no extractable answer — it summarized, asked the user a
# question, or emitted nothing. Phrase lists were mined from the actual final
# messages of these cases in the nemotron-ultra run.
_PREEMPT_ASK = (
    "would you like", "could you clarify", "can you clarify", "what would you like",
    "is there something specific", "do you want me to", "shall i ", "let me know if you'd like",
    "let me know which", "what task you'd like", "how would you like", "which of these would you like",
    "should i continue", "would you prefer", "what would you like me to do",
)
_PREEMPT_SUMMARY = (
    "next step", "next priorit", "current blocker", "blocker", "blocked", "conclusion",
    "recommendation", "current status", "to summarize", "in summary", "summary:", "next:",
    "## next", "remaining", "still need", "needs verification", "unverified",
    # false-complete closers: model claims it finished but emitted no extractable answer
    "task is complete", "task complete", "no further step", "final answer has been written",
    "answer has been provided", "provided the final answer", "the research is complete",
    "has been written to", "has been regenerated", "has been saved",
)


def last_assistant_finish_and_text(messages):
    """(finish reason, concatenated text) of the LAST assistant message, or (None, '')."""
    asst = [m for m in (messages or []) if m.get("role") == "assistant"]
    if not asst:
        return (None, "")
    last = asst[-1]
    txt = "".join(p.get("text", "") for p in last.get("parts", []) if p.get("type") == "text")
    return (last.get("finish"), txt)


# Output-format adherence: the frankenstein/ultra research prompt mandates a final
# response with "Explanation:", "Exact Answer:", "Confidence:" labels. Models also
# satisfy this with markdown headers ("## Exact Answer", "**Confidence**"), so the
# matchers are line-anchored and tolerant of markdown (#, *, >, _, -) + optional
# colon. We require the two load-bearing labels: Exact Answer + Confidence.
_RE_EXACT_ANSWER = re.compile(r"(?im)^[\s#*>_-]*exact\s*answer\b\s*[:*_]*")
_RE_CONFIDENCE = re.compile(r"(?im)^[\s#*>_-]*confidence\b\s*[:*_]*")


def adheres_to_output_format(final_text):
    """True if the final response uses the required answer format (Exact Answer +
    Confidence labels), markdown-tolerant."""
    t = final_text or ""
    return bool(_RE_EXACT_ANSWER.search(t)) and bool(_RE_CONFIDENCE.search(t))


def parallel_tool_stats(messages):
    """Per-trajectory parallel-tool-call stats. An assistant turn = one step
    (… up to a step-finish). A turn issues PARALLEL tool calls when it emits ≥2
    tool parts in that single generation (before any tool result returns).
    Returns: tool_steps (turns with ≥1 tool), parallel_steps (turns with ≥2 tools),
    max_tools (most tools in any one turn)."""
    tool_steps = parallel_steps = max_tools = 0
    cur = 0
    for m in (messages or []):
        if m.get("role") != "assistant":
            continue
        for p in m.get("parts", []):
            t = p.get("type")
            if t == "tool":
                cur += 1
            elif t == "step-finish":
                if cur >= 1:
                    tool_steps += 1
                if cur >= 2:
                    parallel_steps += 1
                max_tools = max(max_tools, cur)
                cur = 0
        if cur >= 1:  # trailing step with no step-finish (incomplete generation)
            tool_steps += 1
            if cur >= 2:
                parallel_steps += 1
            max_tools = max(max_tools, cur)
            cur = 0
    return {"tool_steps": tool_steps, "parallel_steps": parallel_steps, "max_tools": max_tools}


def classify_preempt_stop(final_text):
    """Sub-classify a preemptive stop by its final assistant text:
    empty / proactive_ask / premature_summary / other."""
    t = (final_text or "").strip()
    if not t:
        return "empty"
    low = t.lower()
    if any(p in low for p in _PREEMPT_ASK) or low.rstrip().endswith("?"):
        return "proactive_ask"
    if any(p in low for p in _PREEMPT_SUMMARY):
        return "premature_summary"
    return "other"


def vllm_rejection_stats(logs_dir):
    """Scan vLLM server logs for HTTP 400 rejections and context-length overflows.

    A context-overflow 400 means the request opencode sent (input prompt + the
    reserved output budget) exceeded the model's max context — i.e. compaction did
    NOT shrink the prompt before it hit the ceiling. vLLM logs each such rejection
    as an INFO access line (`"POST /v1/... HTTP/1.1" 400`) plus a VLLMValidationError
    ("maximum context length is N tokens. However, you requested ..."). We count the
    access lines (authoritative, always logged) and classify how many are context
    overflows, and record the largest input-token figure seen as evidence.
    """
    import os, re, glob
    s = {"scanned": False, "files": 0, "http_400": 0, "ctx_overflow": 0,
         "max_input_tokens": 0, "model_max_ctx": None}
    if not logs_dir or not os.path.isdir(logs_dir):
        return s
    files = sorted(glob.glob(os.path.join(logs_dir, "*.log")))
    if not files:
        return s
    s["scanned"] = True
    re_400 = re.compile(r'"(?:POST|GET) /v1/[a-z/]+ HTTP/1\.1" 400')
    re_input = re.compile(r"prompt contains at least (\d+) input tokens")
    re_maxctx = re.compile(r"maximum context length is (\d+) tokens")
    for fp in files:
        s["files"] += 1
        try:
            with open(fp, errors="ignore") as fh:
                for line in fh:
                    if '" 400' in line and re_400.search(line):
                        s["http_400"] += 1
                    # one create_error_response line per rejection (avoids the
                    # 2x double-count from the traceback copy of the message)
                    if "create_error_response called with" in line and "maximum context length" in line:
                        s["ctx_overflow"] += 1
                    if "input tokens" in line:
                        m = re_input.search(line)
                        if m:
                            s["max_input_tokens"] = max(s["max_input_tokens"], int(m.group(1)))
                    if s["model_max_ctx"] is None and "maximum context length is" in line:
                        mc = re_maxctx.search(line)
                        if mc:
                            s["model_max_ctx"] = int(mc.group(1))
        except Exception:
            pass
    return s


def tool_output_lengths(messages, tokenizer=None):
    out = []
    for p in iter_tool_parts(messages):
        name = p.get("tool") or "<unknown>"
        o = p.get("state", {}).get("output")
        if o is None:
            continue
        s = o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)
        length = (len(tokenizer(s, add_special_tokens=False)["input_ids"])
                  if tokenizer is not None else len(s))
        out.append((name, length))
    return out


# ── Reset buckets ─────────────────────────────────────────────────────────────

def build_reset_buckets(max_resets):
    candidates = [
        (0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"),
        (16, 20, "16-20"), (21, 40, "21-40"), (41, 60, "41-60"),
        (61, 80, "61-80"), (81, 100, "81-100"), (101, 150, "101-150"),
        (151, 200, "151-200"), (201, 300, "201-300"), (301, None, "301+"),
    ]
    buckets = []
    for lo, hi, label in candidates:
        if lo > max_resets:
            break
        if hi is None or hi >= max_resets:
            lbl = label if hi is not None else (f"{lo}-{max_resets}" if max_resets > lo else f"{lo}+")
            buckets.append((lo, max(hi or max_resets, max_resets), lbl))
            break
        buckets.append((lo, hi, label))
    return buckets


def get_bucket(value, buckets):
    for lo, hi, label in buckets:
        if lo <= value <= hi:
            return label
    return buckets[-1][2]


# ── Formatting ────────────────────────────────────────────────────────────────

def pct_str(n, d):
    return f"{100 * n / d:.1f}%" if d else "N/A"


def stats_rows(vals):
    vs = sorted(vals)
    n = len(vs)

    def pct(p):
        return vs[min(int(n * p / 100), n - 1)]

    rows = [("n", n), ("total", sum(vs)), ("min", min(vs)), ("max", max(vs)),
            ("mean", round(statistics.mean(vs), 2)),
            ("median", round(statistics.median(vs), 1))]
    if n > 1:
        rows.append(("stdev", round(statistics.stdev(vs), 2)))
    rows += [("p25", pct(25)), ("p75", pct(75)), ("p90", pct(90)), ("p99", pct(99))]
    return rows


def kv_table(rows):
    out = ["| Metric | Value |", "|--------|-------|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True,
                    help="graded_trajectories JSONL (messages + is_correct + grading)")
    ap.add_argument("--output", required=True, help="Markdown report path")
    ap.add_argument("--graded", default=None,
                    help="optional graded_results JSONL to source is_correct from "
                         "(by sample_id); defaults to per-record is_correct")
    ap.add_argument("--total-samples", type=int, default=400,
                    help="worst-case accuracy denominator (launched sample count)")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="step threshold for 'hit max turns' (step-start parts)")
    ap.add_argument("--tokenizer", default=None,
                    help="optional HF tokenizer path; tool-output lengths in tokens "
                         "instead of characters")
    ap.add_argument("--vllm-logs-dir", default=None,
                    help="dir of vLLM server logs to scan for HTTP 400 rejections "
                         "(context-length overflows). If omitted, auto-inferred as "
                         "<run>/vllm_logs relative to --output.")
    args = ap.parse_args()

    # Locate vLLM logs: explicit flag, else infer <run>/vllm_logs from the output
    # path (output is typically <run>/grading*/results.txt).
    vllm_logs_dir = args.vllm_logs_dir
    if not vllm_logs_dir:
        cand = Path(args.output).resolve().parent.parent / "vllm_logs"
        if cand.is_dir():
            vllm_logs_dir = str(cand)
    vllm_rej = vllm_rejection_stats(vllm_logs_dir)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    graded_correct = None
    if args.graded and Path(args.graded).exists():
        graded_correct = {g.get("sample_id"): bool(g.get("is_correct"))
                          for g in load_jsonl(args.graded)}

    # accumulators
    n_traj = n_correct = n_no_answer = 0
    tool_calls = []          # individual tool invocations / traj
    resets = []
    steps = []
    cat_per_traj = {"search": [], "browse": [], "other": []}
    cat_totals = Counter()
    tool_name_total = Counter()
    tool_name_in_traj = Counter()
    # search-query aggregates
    sq_agg = Counter()  # search_calls, queries, multi_q, dups, quoted
    samples_with_dups = samples_with_quoted = 0
    # reasoning
    reason_total_chars = 0
    reason_assistant_msgs = 0
    # malformed
    mal_invalid = mal_error = mal_raw = 0
    mal_intended = Counter()
    mal_error_by_tool = Counter()
    samples_with_malformed = 0
    malformed_sample_ids = []
    # preemptive stops (voluntary finish=="stop" but no extractable answer)
    preempt_counts = Counter()
    preempt_sample_ids = defaultdict(list)
    # output-format adherence (among answered trajectories)
    n_answered = n_format_ok = n_format_bad = 0
    format_bad_ids = []
    # parallel tool calls
    total_tool_steps = total_parallel_steps = samples_with_parallel = 0
    # tool outputs
    per_tool_output_lens = defaultdict(list)
    out_unit = "tokens" if tokenizer else "chars"
    # hit max
    hit_max = 0
    # buckets
    by_reset = defaultdict(lambda: {"n": 0, "correct": 0})
    max_resets_seen = 0

    for rec in iter_records(args.trajectories):
        n_traj += 1
        msgs = rec.get("messages")
        sid = rec.get("sample_id")

        correct = (graded_correct.get(sid, bool(rec.get("is_correct")))
                   if graded_correct is not None else bool(rec.get("is_correct")))
        n_correct += int(correct)
        rec_no_answer = no_answer(rec)
        if rec_no_answer:
            n_no_answer += 1

        # Preemptive stop: model ended its turn voluntarily (last finish=="stop")
        # but produced no extractable answer. Sub-classify by the final text.
        fin, final_txt = last_assistant_finish_and_text(msgs)
        if fin == "stop" and rec_no_answer:
            bucket = classify_preempt_stop(final_txt)
            preempt_counts[bucket] += 1
            preempt_sample_ids[bucket].append(sid)

        # Output-format adherence: only meaningful for trajectories that produced an
        # answer; counts how many of those failed to use the required answer format.
        if not rec_no_answer:
            n_answered += 1
            if adheres_to_output_format(final_txt):
                n_format_ok += 1
            else:
                n_format_bad += 1
                format_bad_ids.append(sid)

        # Parallel tool calls: turns that issued ≥2 tools in one generation.
        pts = parallel_tool_stats(msgs)
        total_tool_steps += pts["tool_steps"]
        total_parallel_steps += pts["parallel_steps"]
        if pts["parallel_steps"] > 0:
            samples_with_parallel += 1

        cc = category_counts(msgs)
        tool_calls.append(sum(cc.values()))
        for cat in ("search", "browse", "other"):
            cat_per_traj[cat].append(cc.get(cat, 0))
        cat_totals.update(cc)

        r = count_context_resets(msgs)
        resets.append(r)
        max_resets_seen = max(max_resets_seen, r)
        by_reset[r]["n"] += 1
        by_reset[r]["correct"] += int(correct)

        s = count_steps(msgs)
        steps.append(s)
        if s >= args.max_steps:
            hit_max += 1

        tnc = tool_name_counts(msgs)
        tool_name_total.update(tnc)
        for name in tnc:
            tool_name_in_traj[name] += 1

        sq = search_query_stats(msgs)
        for k in ("search_calls", "queries", "multi_q", "dups", "quoted"):
            sq_agg[k] += sq[k]
        if sq["dups"] > 0:
            samples_with_dups += 1
        if sq["quoted"] > 0:
            samples_with_quoted += 1

        rc, am = reasoning_chars(msgs)
        reason_total_chars += rc
        reason_assistant_msgs += am

        mal = malformed_stats(msgs)
        mal_invalid += mal["invalid"]
        mal_error += mal["error"]
        mal_raw += mal["raw_tag"]
        mal_intended.update(mal["intended"])
        mal_error_by_tool.update(mal["error_by_tool"])
        if mal["total"] > 0:
            samples_with_malformed += 1
            malformed_sample_ids.append(sid)

        for name, length in tool_output_lengths(msgs, tokenizer):
            per_tool_output_lens[name].append(length)

    # ── derived ──
    acc_finished = n_correct / n_traj if n_traj else 0.0
    acc_worst = n_correct / args.total_samples if args.total_samples else 0.0
    reset_buckets = build_reset_buckets(max_resets_seen)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    L = []
    L.append("# opencode Run — Single-Run Trajectory Metrics\n")
    L.append(f"- **Trajectories file:** `{args.trajectories}`")
    if args.graded:
        L.append(f"- **Graded file (accuracy):** `{args.graded}`")
    L.append(f"- **Trajectories:** {n_traj}")
    L.append(f"- **Worst-case denominator:** {args.total_samples} "
             f"(missing: {args.total_samples - n_traj})")
    L.append("")

    # Overall
    L.append("## Overall Metrics\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Accuracy (finished) | {acc_finished:.2%} ({n_correct}/{n_traj}) |")
    L.append(f"| Accuracy (worst-case) | {acc_worst:.2%} "
             f"({n_correct}/{args.total_samples}, unfinished=incorrect) |")
    L.append(f"| Failed to produce answer (`extracted_answer` empty/none/null) | "
             f"{n_no_answer} ({pct_str(n_no_answer, n_traj)}) |")
    L.append(f"| Avg tool calls (individual invocations) | {avg(tool_calls):.1f} |")
    L.append(f"| Min / Max tool calls | {min(tool_calls)} / {max(tool_calls)} |")
    L.append(f"| Avg context resets | {avg(resets):.1f} |")
    L.append(f"| Min / Max context resets | {min(resets)} / {max(resets)} |")
    L.append("")

    # Tool call breakdown
    L.append("## Tool Call Breakdown (individual invocations)\n")
    L.append("_search = name contains `search`/ends `research`; browse = "
             "`webfetch`/`*extract*`/`*crawl*`/`*map*`; other = bash/read/grep/glob/todowrite. "
             "`invalid` calls are excluded here and counted under Malformed._\n")
    L.append("| Category | Avg / sample | Min / Max | Total |")
    L.append("|----------|--------------|-----------|-------|")
    for cat in ("search", "browse", "other"):
        v = cat_per_traj[cat]
        L.append(f"| {cat} | {avg(v):.1f} | {min(v)} / {max(v)} | {cat_totals.get(cat, 0)} |")
    L.append("")

    # Tool-call distribution by name
    L.append("## Tool-Call Distribution by Tool Name\n")
    grand = sum(tool_name_total.values())
    L.append(f"_{len(tool_name_total)} distinct tools, {grand} total invocations._\n")
    L.append("| Tool | Category | Calls | Share | Trajectories using |")
    L.append("|------|----------|-------|-------|--------------------|")
    for name, cnt in tool_name_total.most_common():
        L.append(f"| `{name}` | {tool_category(name)} | {cnt} | {pct_str(cnt, grand)} "
                 f"| {tool_name_in_traj[name]}/{n_traj} |")
    L.append("")

    # Hit max turns
    L.append(f"## Hit Max Turns (steps >= {args.max_steps})\n")
    L.append("_A step = one `step-start` part (LLM generation step)._\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Samples that hit max | {hit_max} ({pct_str(hit_max, n_traj)}) |")
    L.append(f"| Samples that did NOT hit max | {n_traj - hit_max} "
             f"({pct_str(n_traj - hit_max, n_traj)}) |")
    L.append("")

    # Reasoning
    L.append("## Reasoning (char proxy for tokens)\n")
    avg_reason = reason_total_chars / reason_assistant_msgs if reason_assistant_msgs else 0
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total reasoning chars | {reason_total_chars} |")
    L.append(f"| Assistant messages | {reason_assistant_msgs} |")
    L.append(f"| Avg reasoning chars / assistant msg | {avg_reason:.0f} |")
    L.append("")

    # Accuracy by reset bucket
    L.append("## Accuracy by Context-Reset Bucket\n")
    L.append("| Bucket (resets) | n | Correct | Accuracy |")
    L.append("|-----------------|---|---------|----------|")
    bucket_agg = defaultdict(lambda: {"n": 0, "correct": 0})
    for r, d in by_reset.items():
        lbl = get_bucket(r, reset_buckets)
        bucket_agg[lbl]["n"] += d["n"]
        bucket_agg[lbl]["correct"] += d["correct"]
    for _, _, lbl in reset_buckets:
        b = bucket_agg.get(lbl)
        if not b or b["n"] == 0:
            continue
        L.append(f"| {lbl} | {b['n']} | {b['correct']} | {pct_str(b['correct'], b['n'])} |")
    L.append("")

    # vLLM rejected requests (HTTP 400 / context overflow)
    L.append("## vLLM Rejected Requests (HTTP 400)\n")
    if not vllm_rej["scanned"]:
        L.append("_No vLLM logs scanned (none found at the inferred `<run>/vllm_logs`; "
                 "pass `--vllm-logs-dir` to enable)._\n")
    else:
        mc = vllm_rej["model_max_ctx"]
        L.append(f"_Scanned {vllm_rej['files']} vLLM log file(s). A **context-overflow 400** "
                 f"means the request (input prompt + reserved output budget) exceeded the model's "
                 f"max context{f' ({mc} tokens)' if mc else ''} — i.e. **compaction did NOT shrink "
                 f"the prompt before it hit the ceiling**, so vLLM rejected the request._\n")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| Total HTTP 400 rejections (request-level, all shards) | {vllm_rej['http_400']} |")
        L.append(f"| of which context-length overflow | {vllm_rej['ctx_overflow']} |")
        if mc:
            L.append(f"| Model max context (tokens) | {mc} |")
        if vllm_rej["max_input_tokens"]:
            L.append(f"| Largest input prompt seen in an overflow (tokens) | {vllm_rej['max_input_tokens']} |")
        L.append("")
        L.append("_Counts are **request-level** (every rejected call across all shards, including "
                 "retries), scanned from the vLLM server logs._\n")
        other = vllm_rej["http_400"] - vllm_rej["ctx_overflow"]
        if vllm_rej["http_400"] == 0:
            L.append("_No 400s — every request fit within the context window._\n")
        elif other > 0:
            L.append(f"_{other} of the 400s were NOT context overflows (other bad-request causes)._\n")
    L.append("")

    # Preemptive stops (model ended without an answer)
    preempt_total = sum(preempt_counts.values())
    L.append("## Preemptive Stops (model ended its turn without an answer)\n")
    L.append("_Trajectories where the model **voluntarily ended** its turn (last assistant "
             "`finish == \"stop\"`, i.e. it chose not to call another tool) **but produced no "
             "extractable answer**. This is a subset of the no-answer trajectories; it EXCLUDES "
             "interrupted/incomplete sessions (no finish recorded) and length-truncated ones, "
             "which are counted elsewhere. Sub-buckets are a keyword heuristic over the final "
             "assistant message, so treat them as approximate._\n")
    L.append("| Reason | Samples | What it counts |")
    L.append("|--------|---------|----------------|")
    L.append(f"| `proactive_ask` | {preempt_counts.get('proactive_ask', 0)} | "
             "Asked the user a clarifying question instead of answering "
             "(\"Would you like…\", \"Could you clarify…\", or the final text ends in `?`). |")
    L.append(f"| `premature_summary` | {preempt_counts.get('premature_summary', 0)} | "
             "Ended on a progress/status summary — \"Next Steps\", \"Blockers\", \"Conclusion\", "
             "\"Recommendation\", or a false \"task complete\" — rather than giving the answer. |")
    L.append(f"| `empty` | {preempt_counts.get('empty', 0)} | "
             "The final assistant turn contained **no text at all** (stopped emitting nothing). |")
    L.append(f"| `other` | {preempt_counts.get('other', 0)} | "
             "Voluntarily stopped with non-empty text matching none of the above. |")
    L.append(f"| **TOTAL** | **{preempt_total}/{n_traj}** | "
             "All preemptive stops (voluntary `stop` + no answer). |")
    L.append("")
    for b in ("proactive_ask", "premature_summary", "empty", "other"):
        ids = preempt_sample_ids.get(b, [])
        if ids:
            ex = ", ".join(f"`{s}`" for s in ids[:10])
            L.append(f"- **{b}** example sample IDs (first 10): {ex}")
    L.append("")

    # Output-format adherence
    L.append("## Output-Format Adherence\n")
    L.append("_The research prompt requires the final response to use the labels "
             "**`Explanation:` / `Exact Answer:` / `Confidence:`**. Adherence here = the final "
             "message contains the two load-bearing labels (**Exact Answer** + **Confidence**), "
             "matched markdown-tolerantly (so `## Exact Answer` / `**Confidence**` count too). "
             "Computed only over **answered** trajectories (those with an extractable answer), so "
             "it isolates pure formatting failures from no-answer/preemptive-stop cases._\n")
    L.append("| Metric | Value | What it counts |")
    L.append("|--------|-------|----------------|")
    L.append(f"| Answered trajectories | {n_answered} | Produced an extractable final answer (denominator). |")
    L.append(f"| Adhering to format | {n_format_ok} | Final response used the Exact Answer + Confidence labels. |")
    L.append(f"| **NOT adhering to format** | **{n_format_bad}** | "
             "Gave an answer but in prose/markdown without the required labels. |")
    L.append(f"| Adherence rate | {pct_str(n_format_ok, n_answered)} | Adhering / answered. |")
    L.append("")
    if format_bad_ids:
        ex = ", ".join(f"`{s}`" for s in format_bad_ids[:10])
        L.append(f"_Non-adhering example sample IDs (first 10): {ex}_\n")

    # Parallel tool calls
    L.append("## Parallel Tool Calls\n")
    L.append("_An assistant **turn** = one generation step (up to its `step-finish`). A turn "
             "issues **parallel** tool calls when it emits **≥2 tool calls in that single "
             "generation** (the model fires multiple tools at once, before any tool result "
             "returns); a turn with exactly 1 tool is serial. Reported at two granularities._\n")
    L.append("| Metric | Value | What it counts |")
    L.append("|--------|-------|----------------|")
    L.append(f"| Tool-calling turns | {total_tool_steps} | Assistant turns that issued ≥1 tool (denominator). |")
    L.append(f"| **Parallel turns (≥2 tools)** | **{total_parallel_steps}** "
             f"({pct_str(total_parallel_steps, total_tool_steps)}) | "
             "Turns that issued ≥2 tool calls in one generation. |")
    L.append(f"| **Samples with ≥1 parallel turn** | **{samples_with_parallel}/{n_traj}** "
             f"({pct_str(samples_with_parallel, n_traj)}) | "
             "Trajectories where parallel tool calls happened at least once. |")
    L.append("")

    # Search query stats
    L.append("## Search Query Stats\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total search calls | {sq_agg['search_calls']} |")
    L.append(f"| Total queries | {sq_agg['queries']} |")
    L.append(f"| Duplicate queries (exact, within trajectory) | {sq_agg['dups']} |")
    L.append(f"| Samples with ≥1 duplicate | {samples_with_dups}/{n_traj} |")
    L.append(f"| Avg duplicates / sample | {sq_agg['dups'] / n_traj if n_traj else 0:.2f} |")
    L.append("")

    # Tool output lengths
    L.append(f"## Tool Output Lengths ({out_unit})\n")
    all_lens = [x for lens in per_tool_output_lens.values() for x in lens]
    if all_lens:
        L.append(f"_{len(all_lens)} tool outputs measured in {out_unit}._\n")
        L.append(f"- **Avg {out_unit}/output:** {avg(all_lens):.1f}")
        L.append(f"- **Median:** {statistics.median(all_lens):.1f}  **Max:** {max(all_lens)}\n")
        L.append(f"| Tool | Outputs | Avg {out_unit} | Max {out_unit} |")
        L.append("|------|---------|----------|----------|")
        for name in sorted(per_tool_output_lens, key=lambda k: -sum(per_tool_output_lens[k])):
            lens = per_tool_output_lens[name]
            L.append(f"| `{name}` | {len(lens)} | {avg(lens):.1f} | {max(lens)} |")
        L.append("")
    else:
        L.append("_(no tool outputs)_\n")

    # Malformed
    L.append("## Malformed / Errored Tool Calls\n")
    L.append("_opencode-native categories (BrowseComp's schema-specific ones don't apply). "
             "Counts are over all trajectories; the three categories are disjoint._\n")
    L.append("**What each category means:**")
    L.append("- **`invalid` tool parts** — the model emitted a tool call opencode "
             "**could not even build** into a real invocation (unparseable JSON args, "
             "unknown tool name, etc.). A model-side formatting failure. opencode files "
             "these under a synthetic tool named `invalid`.")
    L.append("- **error-status tool parts** — the call was well-formed and **ran, but the "
             "tool returned an error** (dead URL / fetch timeout, a `bash` command that "
             "failed, a permission denial, a search that errored). The call was fine; the "
             "execution failed. On web-research benchmarks this is largely expected (the open web).")
    L.append("- **raw `<tool_call>` in text/reasoning** — a literal `<tool_call>` tag "
             "**leaked into the assistant's visible text/reasoning** instead of being parsed "
             "as an actual tool call. A tool-call format/parser glitch.")
    L.append("- **Samples with ≥1** — how many distinct trajectories hit any of the above "
             "at least once (the rest had none). **Example sample IDs** point at affected "
             "rollouts for inspection.\n")
    mal_total = mal_invalid + mal_error + mal_raw
    L.append("| Category | Count |")
    L.append("|----------|-------|")
    L.append(f"| `invalid` tool parts | {mal_invalid} |")
    L.append(f"| error-status tool parts | {mal_error} |")
    L.append(f"| raw `<tool_call>` in text/reasoning | {mal_raw} |")
    L.append(f"| **Total malformed/errored** | **{mal_total}** |")
    L.append(f"| Samples with ≥1 malformed/errored | {samples_with_malformed}/{n_traj} |")
    if malformed_sample_ids:
        ex = ", ".join(f"`{s}`" for s in malformed_sample_ids[:5])
        L.append(f"| Example sample IDs (first 5) | {ex} |")
    L.append("")
    if mal_intended:
        L.append("**`invalid` parts by intended tool:**\n")
        L.append("_For each call opencode could not build, the tool the model was **trying** "
                 "to call (read from the attempted call's `input.tool` field). Clean tool names "
                 "(e.g. `bash`, `tavily_tavily_search`) = a real call that was malformed. "
                 "Garbage names containing reasoning text or `</think>` = the model's "
                 "chain-of-thought **leaked into the tool-call payload**, so the parser captured "
                 "CoT text where the tool name should be — a tool-call/reasoning boundary failure._\n")
        L.append("| Intended tool | Count |")
        L.append("|---------------|-------|")
        for name, cnt in mal_intended.most_common():
            L.append(f"| `{name}` | {cnt} |")
        L.append("")
    if mal_error_by_tool:
        L.append("**error-status parts by tool:**\n")
        L.append("_Of the calls that were well-formed and **ran but returned an error**, which "
                 "tool errored. `webfetch` dominating is usually benign (dead URLs, timeouts, "
                 "bot-blocks, size-limit); other tools = file-not-found, failed shell commands, "
                 "search-API errors, etc._\n")
        L.append("| Tool | Count |")
        L.append("|------|-------|")
        for name, cnt in mal_error_by_tool.most_common():
            L.append(f"| `{name}` | {cnt} |")
        L.append("")

    text = "\n".join(L) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)
    print(text)
    print(f"[written] {args.output}")


if __name__ == "__main__":
    main()
