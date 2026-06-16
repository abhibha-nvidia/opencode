#!/usr/bin/env python3
"""
Analyze agent trajectory compaction events for BrowseComp/OpenCode-style runs.

This script is intentionally dependency-free and schema-tolerant. It extracts one
row per compaction event, computes deterministic approximations for retention
metrics, applies optional manual/LLM label overrides, and writes funnel/failure
summary tables.

It natively understands the OpenCode graded-trajectory schema:
  - top-level record: sample_id, shard_id, question, ground_truth, session,
    messages, is_correct, grading{extracted_answer,...}
  - messages[].parts[] with part types: text, reasoning, tool, step-start,
    step-finish, compaction
  - a compaction event is the assistant message with `summary == true`, whose
    text part(s) hold the compacted summary; it directly follows a user message
    carrying a `compaction` part.
The generic text-marker heuristics remain as a fallback for other schemas.

Expected usage:
  python analyze_compaction_trajectories.py \
    --input /path/to/glm.jsonl /path/to/ultra.jsonl \
    --matched-only \
    --output-dir /tmp/compaction_out
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TEXT_EXTENSIONS = {".json", ".jsonl", ".txt", ".md", ".log"}
URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
CAPITALIZED_SPAN_RE = re.compile(r"\b(?:[A-Z][\w'&.-]+(?:\s+|$)){1,6}")
NEGATIVE_TERMS = (
    "not found", "no results", "failed", "irrelevant", "wrong", "ruled out",
    "does not contain", "didn't contain", "did not contain", "not the same",
    "different person", "different entity", "dead end", "inaccessible",
    "404", "timeout", "contradict", "mismatch", "not useful", "unhelpful",
)
NEXT_STEP_TERMS = (
    "next", "continue", "need to", "should", "try", "look for", "search for",
    "verify", "check", "open", "follow up", "remaining", "unresolved",
)
COMPACTION_MARKERS = (
    "compact", "compaction", "summarize conversation", "summarize the conversation",
    "summary of previous", "conversation summary", "condensed", "context summary",
    "state summary", "handoff summary", "memory update",
)
TOOL_MARKERS = (
    "tool", "browser", "browse", "search", "open", "navigate", "visit", "fetch",
    "read", "grep", "bash", "terminal", "command", "url",
)
SEARCH_MARKERS = (
    "search", "query", "web_search", "browser.search", "google", "bing",
)
# tool-name substrings that indicate a page fetch/open (vs a search listing)
PAGE_TOOL_MARKERS = (
    "extract", "crawl", "fetch", "scrape", "read", "open", "visit", "navigate", "goto",
)
PAGE_MARKERS = (
    "open", "navigate", "browser.open", "visit", "url", "page", "fetch",
)
UNKNOWN = "unknown"


def stable_id(*parts: str, n: int = 12) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return h[:n]


def safe_str(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        out = ""
    elif isinstance(value, str):
        out = value
    else:
        try:
            out = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            out = str(value)
    out = out.replace("\x00", "")
    if max_chars is not None and len(out) > max_chars:
        return out[:max_chars] + "...[truncated]"
    return out


def normalize_text(text: Any) -> str:
    s = safe_str(text).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    # English/browser traces are often close to 4 chars per token. Bound by words.
    char_est = len(text) / 4.0
    word_est = len(text.split()) * 1.25
    return int(max(1, round((char_est + word_est) / 2)))


def flatten_strings(obj: Any, max_depth: int = 8, _depth: int = 0) -> List[str]:
    if _depth > max_depth:
        return []
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (int, float, bool)):
        return [str(obj)]
    out: List[str] = []
    if isinstance(obj, list):
        for item in obj:
            out.extend(flatten_strings(item, max_depth, _depth + 1))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                out.extend(flatten_strings(value, max_depth, _depth + 1))
            elif value is not None:
                out.append(f"{key}: {value}")
    return out


def render_part(part: Any) -> str:
    """Render a single OpenCode message part to text."""
    if not isinstance(part, dict):
        return safe_str(part)
    ptype = part.get("type")
    if ptype in ("step-start", "step-finish"):
        return ""
    if ptype == "text":
        return safe_str(part.get("text"))
    if ptype == "reasoning":
        t = safe_str(part.get("text"))
        return f"reasoning: {t}" if t else ""
    if ptype == "compaction":
        return "[compaction]"
    if ptype == "tool":
        name = part.get("tool") or part.get("tool_name") or "tool"
        st = part.get("state") if isinstance(part.get("state"), dict) else {}
        bits = [f"tool: {name}"]
        title = st.get("title")
        if title:
            bits.append(f"title: {safe_str(title)}")
        if st.get("input") is not None:
            bits.append(f"input: {safe_str(st.get('input'))}")
        if st.get("output") is not None:
            bits.append(f"output: {safe_str(st.get('output'))}")
        status = st.get("status")
        if status:
            bits.append(f"status: {safe_str(status)}")
        return "\n".join(bits)
    for key in ("text", "content", "summary"):
        if part.get(key) not in (None, ""):
            return safe_str(part.get(key))
    return ""


def event_to_text(event: Any) -> str:
    if isinstance(event, str):
        return event
    if not isinstance(event, dict):
        return safe_str(event)

    pieces: List[str] = []
    role = event.get("role") or event.get("type") or event.get("name") or event.get("event")
    if role:
        pieces.append(f"[{role}]")

    # OpenCode-style messages carry their content in `parts`.
    parts = event.get("parts")
    if isinstance(parts, list):
        for p in parts:
            rendered = render_part(p)
            if rendered:
                pieces.append(rendered)
        return "\n".join(pieces)

    preferred_keys = [
        "content", "text", "message", "messages", "summary", "output", "result",
        "observation", "tool_result", "tool_input", "arguments", "input", "response",
        "delta", "reasoning", "title", "url", "query",
    ]
    for key in preferred_keys:
        if key in event and event[key] is not None:
            pieces.append(f"{key}: {safe_str(event[key])}")
    if not pieces or (len(pieces) == 1 and role):
        extra = flatten_strings(event)
        if extra:
            pieces = ([f"[{role}]"] if role else []) + extra
    return "\n".join(pieces)


def event_label(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    keys = [
        "type", "event", "event_type", "name", "role", "action", "subtype", "kind",
        "tool", "tool_name", "function", "function_name",
    ]
    vals = []
    for key in keys:
        if key in event and event[key] is not None:
            vals.append(str(event[key]))
    return " ".join(vals).lower()


def find_first(obj: Any, keys: Sequence[str]) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
        for value in obj.values():
            found = find_first(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first(item, keys)
            if found not in (None, ""):
                return found
    return None


def as_bool(value: Any) -> Any:
    if value is None or value == "":
        return UNKNOWN
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "1", "correct", "pass", "passed", "success"}:
        return True
    if s in {"false", "f", "no", "n", "0", "incorrect", "fail", "failed", "wrong"}:
        return False
    return UNKNOWN


def parse_json_or_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"jsonl parse error line {line_no}: {exc}") from exc
        return rows
    if path.suffix.lower() == ".json":
        return json.loads(text)
    # Plain transcript fallback.
    return {"events": [{"type": "text", "content": text}], "trajectory_path": str(path)}


def discover_files(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS:
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                    files.append(child)
        else:
            # Allow glob patterns.
            for child in sorted(Path().glob(raw)):
                if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS:
                    files.append(child.resolve())
    return sorted(dict.fromkeys(files))


def coerce_runs(obj: Any, path: Path) -> List[Dict[str, Any]]:
    """Return likely run objects from an arbitrary JSON object."""
    if isinstance(obj, list):
        if all(isinstance(x, dict) for x in obj):
            # JSONL may be a list of independent runs or a list of events. Decide by keys.
            if any(any(k in x for k in ("events", "messages", "steps", "trajectory")) for x in obj):
                return [dict(x) for x in obj]
            return [{"events": obj, "trajectory_path": str(path)}]
        return [{"events": obj, "trajectory_path": str(path)}]
    if not isinstance(obj, dict):
        return [{"events": [{"type": "text", "content": safe_str(obj)}], "trajectory_path": str(path)}]

    for key in ("runs", "trajectories", "rollouts", "episodes", "results"):
        val = obj.get(key)
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            runs = []
            for run in val:
                merged = {k: v for k, v in obj.items() if k != key and not isinstance(v, (list, dict))}
                merged.update(run)
                merged["trajectory_path"] = str(path)
                runs.append(merged)
            return runs
    obj = dict(obj)
    obj["trajectory_path"] = str(path)
    return [obj]


def extract_events(run: Dict[str, Any]) -> List[Any]:
    for key in ("events", "messages", "steps", "trajectory", "turns", "log", "history", "items"):
        val = run.get(key)
        if isinstance(val, list):
            return val
    # Fallback: nested list with event-looking dicts.
    for value in run.values():
        if isinstance(value, list) and value and all(isinstance(x, (dict, str)) for x in value):
            return value
    return [{"type": "run", "content": run}]


def has_part_type(event: Any, ptype: str) -> bool:
    if not isinstance(event, dict):
        return False
    parts = event.get("parts")
    if not isinstance(parts, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == ptype for p in parts)


def is_compaction_event(event: Any) -> bool:
    # OpenCode: the assistant message carrying the compacted summary is flagged
    # `summary == True`. Its text part holds the summary. Anchor on it so the
    # pre/summary/post split is clean.
    if isinstance(event, dict):
        if event.get("summary") is True:
            return True
        # A `compaction` part marks the trigger boundary on the preceding user
        # message; the summary text lives in the following summary=True message.
        # Don't count the trigger itself, to avoid double-counting.
        if has_part_type(event, "compaction"):
            return False
    label = event_label(event)
    text = normalize_text(event_to_text(event))
    combined = f"{label}\n{text}"
    if any(marker in combined for marker in COMPACTION_MARKERS):
        return True
    return False


def extract_compacted_summary(event: Any) -> str:
    if isinstance(event, dict):
        # OpenCode: the summary text lives in the message's text part(s).
        parts = event.get("parts")
        if isinstance(parts, list):
            texts = [
                safe_str(p.get("text"))
                for p in parts
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text") not in (None, "")
            ]
            if texts:
                return "\n".join(texts)
        for key in (
            "compacted_summary", "compact_summary", "new_context",
            "context", "output", "result", "content", "message", "text",
        ):
            value = event.get(key)
            if value not in (None, "") and not isinstance(value, bool):
                return safe_str(value)
    return event_to_text(event)


def join_events(events: Sequence[Any]) -> str:
    return "\n\n".join(f"--- event {i} ---\n{event_to_text(ev)}" for i, ev in enumerate(events))


def _tool_parts(events: Sequence[Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for ev in events:
        if isinstance(ev, dict) and isinstance(ev.get("parts"), list):
            for p in ev["parts"]:
                if isinstance(p, dict) and p.get("type") == "tool":
                    st = p.get("state") if isinstance(p.get("state"), dict) else {}
                    yield str(p.get("tool") or "").lower(), st


def count_tool_events(events: Sequence[Any]) -> int:
    structured = list(_tool_parts(events))
    if structured:
        return len(structured)
    n = 0
    for ev in events:
        blob = f"{event_label(ev)}\n{normalize_text(event_to_text(ev))}"
        if any(m in blob for m in TOOL_MARKERS):
            n += 1
    return n


def count_search_events(events: Sequence[Any]) -> int:
    structured = list(_tool_parts(events))
    if structured:
        n = 0
        for name, st in structured:
            inp = normalize_text(safe_str(st.get("input")))
            if any(m in name for m in SEARCH_MARKERS) or "query" in inp:
                n += 1
        return n
    n = 0
    for ev in events:
        blob = f"{event_label(ev)}\n{normalize_text(event_to_text(ev))}"
        if any(m in blob for m in SEARCH_MARKERS):
            n += 1
    return n


def count_page_events(events: Sequence[Any]) -> int:
    structured = list(_tool_parts(events))
    if structured:
        n = 0
        for name, st in structured:
            if any(m in name for m in PAGE_TOOL_MARKERS):
                n += 1
        return n
    n = 0
    for ev in events:
        text = event_to_text(ev)
        blob = f"{event_label(ev)}\n{normalize_text(text)}"
        if URL_RE.search(text) or any(m in blob for m in PAGE_MARKERS):
            n += 1
    return n


def extract_urls(text: str) -> List[str]:
    urls = []
    for match in URL_RE.findall(text or ""):
        urls.append(match.rstrip(".,;:)]}"))
    return sorted(dict.fromkeys(urls))


def extract_search_queries(text: str) -> List[str]:
    queries: List[str] = []
    patterns = [
        r"(?:query|search(?:_query)?|search for|searched for)[\"']?\s*[:=]\s*[\"']?([^\n\"'}\]]{3,200})",
        r"web_search\s*\([^)]*?query[\"']?\s*[:=]\s*[\"']([^\"']{3,200})",
        r"browser\.search[^\n]*?[:=]\s*[\"']?([^\n\"'}\]]{3,200})",
    ]
    for pat in patterns:
        for match in re.findall(pat, text or "", flags=re.IGNORECASE):
            q = re.sub(r"\s+", " ", match).strip().strip(",.;")
            if len(q) >= 3:
                queries.append(q[:200])
    return sorted(dict.fromkeys(queries))


def extract_capitalized_spans(text: str, limit: int = 60) -> List[str]:
    spans: List[str] = []
    stop = {"I", "The", "A", "An", "This", "That", "It", "We", "You", "Search", "Result", "Page", "URL"}
    for match in CAPITALIZED_SPAN_RE.findall(text or ""):
        s = re.sub(r"\s+", " ", match).strip()
        if not s or s in stop or len(s) < 3:
            continue
        if sum(ch.isalpha() for ch in s) < 3:
            continue
        spans.append(s)
        if len(spans) >= limit:
            break
    return sorted(dict.fromkeys(spans))


def contains_any_term(text: str, terms: Sequence[str]) -> bool:
    t = normalize_text(text)
    return any(term in t for term in terms)


def answer_aliases(gold_answer: Any) -> List[str]:
    if gold_answer in (None, ""):
        return []
    if isinstance(gold_answer, list):
        vals = gold_answer
    else:
        vals = [gold_answer]
    aliases: List[str] = []
    for val in vals:
        s = safe_str(val).strip()
        if not s:
            continue
        aliases.append(s)
        # Simple normalization variants.
        aliases.append(re.sub(r"\s+", " ", s))
        aliases.append(s.replace("-", " "))
    return [a for a in sorted(dict.fromkeys(aliases), key=len, reverse=True) if len(a) >= 2]


def text_contains_answer(text: str, gold_answer: Any) -> Any:
    aliases = answer_aliases(gold_answer)
    if not aliases:
        return UNKNOWN
    norm = normalize_text(text)
    for alias in aliases:
        if normalize_text(alias) in norm:
            return True
    return False


def extract_gold_map(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"gold file not found: {p}")
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix.lower() == ".jsonl":
        with p.open(encoding="utf-8", errors="replace") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            # Either {task_id: answer} or a dataset object.
            if any(isinstance(v, (str, list, int, float)) for v in data.values()) and not any(k in data for k in ("rows", "examples", "data")):
                return {str(k): v for k, v in data.items()}
            for key in ("rows", "examples", "data", "items"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
        elif isinstance(data, list):
            rows = data
    out: Dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = first_nonempty(row, ["task_id", "id", "question_id", "example_id", "problem_id", "sample_id"])
        ans = first_nonempty(row, ["gold_answer", "ground_truth", "answer", "target", "label", "expected", "reference_answer"])
        if task_id not in (None, "") and ans not in (None, ""):
            out[str(task_id)] = ans
    return out


def load_label_overrides(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"labels file not found: {p}")
    rows: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    elif p.suffix.lower() == ".jsonl":
        with p.open(encoding="utf-8", errors="replace") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        rows = data if isinstance(data, list) else list(data.values())
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_id = row.get("event_id")
        if not event_id:
            continue
        clean = {k: parse_label_value(v) for k, v in row.items() if k and k != "event_id" and v not in (None, "")}
        out[str(event_id)] = clean
    return out


def parse_label_value(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in {"true", "false", "yes", "no", "unknown"}:
            return as_bool(s)
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return s
        return s
    return value


def first_nonempty(obj: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return None


def _model_id(value: Any) -> Optional[str]:
    """Extract a clean model id from a dict, a JSON-encoded string, or a plain string."""
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        m = value.get("id") or value.get("model") or value.get("name")
        return safe_str(m) if m not in (None, "") else None
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
                if isinstance(d, dict):
                    m = d.get("id") or d.get("model") or d.get("name")
                    if m not in (None, ""):
                        return safe_str(m)
            except Exception:
                pass
        return s
    return safe_str(value)


def detect_model(run: Dict[str, Any], path: Path) -> str:
    value = find_first(run, ["model", "model_name", "input_model", "agent_model", "compactor_model", "modelID"])
    mid = _model_id(value)
    if mid:
        return mid
    p = str(path).lower()
    if "glm" in p:
        return "glm"
    if "ultra" in p or "nemotron" in p:
        return "ultra"
    if "kimi" in p:
        return "kimi"
    return UNKNOWN


def detect_task_id(run: Dict[str, Any], path: Path, run_idx: int) -> str:
    value = find_first(run, ["sample_id", "task_id", "question_id", "example_id", "problem_id", "benchmark_id"])
    if value not in (None, ""):
        return safe_str(value)
    return f"{path.stem}:{run_idx}"


def detect_run_id(run: Dict[str, Any], path: Path, run_idx: int) -> str:
    value = find_first(run, ["run_id", "trajectory_id", "rollout_id", "session_id", "uuid", "trace_id"])
    if value not in (None, ""):
        return safe_str(value)
    # OpenCode session id is a stable per-run identifier.
    sess = run.get("session")
    if isinstance(sess, dict) and sess.get("id"):
        return safe_str(sess["id"])
    return stable_id(str(path), str(run_idx), n=10)


def detect_final_answer(run: Dict[str, Any], events: Sequence[Any]) -> str:
    value = find_first(run, ["extracted_answer", "final_answer", "prediction", "model_answer", "output_answer"])
    if value not in (None, "") and not isinstance(value, (dict, list)):
        return safe_str(value)
    grading = run.get("grading")
    if isinstance(grading, dict) and grading.get("extracted_answer") not in (None, ""):
        return safe_str(grading.get("extracted_answer"))
    # Heuristic: scan last few events for final answer markers.
    tail = "\n".join(event_to_text(ev) for ev in events[-8:])
    patterns = [
        r"final answer\s*[:\-]\s*(.+)",
        r"exact answer\s*[:\-]\s*(.+)",
        r"answer\s*[:\-]\s*(.+)",
        r"FINAL\s*[:\-]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:1000]
    return ""


def detect_final_correct(run: Dict[str, Any]) -> Any:
    value = find_first(run, ["is_correct", "final_correct", "correct", "success", "passed", "score", "accuracy"])
    if value in (None, ""):
        return UNKNOWN
    if isinstance(value, (int, float)) and value not in (0, 1):
        return value
    return as_bool(value)


def detect_harness_version(run: Dict[str, Any]) -> str:
    value = find_first(run, ["harness_version", "opencode_version", "benchmark_version", "version", "commit"])
    return safe_str(value) if value not in (None, "") else UNKNOWN


def detect_budget(run: Dict[str, Any]) -> str:
    value = find_first(run, ["wall_clock_or_step_budget", "wall_clock", "duration", "elapsed", "step_budget", "max_steps", "budget"])
    return safe_str(value) if value not in (None, "") else ""


@dataclass
class Layer1Event:
    event_id: str
    task_id: str
    run_id: str
    model: str
    harness_version: str
    trajectory_path: str
    compaction_index: int
    compaction_event_index: int
    num_compactions_total: int
    is_first_compaction: bool
    tokens_before_compaction: int
    tokens_after_compaction: int
    pre_compaction_token_length: int
    post_compaction_token_length: int
    summary_token_length: int
    compression_ratio: float
    pre_compaction_transcript: str
    compacted_summary: str
    post_compaction_continuation: str
    final_answer: str
    gold_answer: Any
    final_correct: Any
    num_tool_calls_before_compaction: int
    num_tool_calls_after_compaction: int
    num_searches_before_compaction: int
    num_searches_after_compaction: int
    num_pages_opened_before_compaction: int
    num_pages_opened_after_compaction: int
    wall_clock_or_step_budget: str
    trajectory_status: str


def _prefix_sums(values: Sequence[int]) -> List[int]:
    """prefix[i] = sum(values[:i]); length = len(values)+1."""
    out = [0]
    acc = 0
    for v in values:
        acc += v
        out.append(acc)
    return out


def extract_layer1_events(
    path: Path,
    gold_map: Dict[str, Any],
    max_transcript_chars: int = 20000,
) -> Tuple[List[Layer1Event], List[Dict[str, Any]], List[str]]:
    """Extract compaction events.

    Layer-2 metrics are computed inline on the full (untruncated) pre/summary/post
    text, while only a bounded preview of each transcript is retained in the
    Layer-1 rows. Per-event text is rendered exactly once per run and tool/search/
    page counts use prefix sums, so cost is roughly linear in trajectory size
    instead of the quadratic blow-up of re-rendering every prefix per compaction.
    """
    errors: List[str] = []
    try:
        obj = parse_json_or_text(path)
    except Exception as exc:
        return [], [], [f"{path}: parse failed: {exc}"]
    events_out: List[Layer1Event] = []
    layer2_out: List[Dict[str, Any]] = []
    cap = max_transcript_chars if max_transcript_chars and max_transcript_chars > 0 else None
    runs = coerce_runs(obj, path)
    for run_idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        events = extract_events(run)
        compaction_idxs = [i for i, ev in enumerate(events) if is_compaction_event(ev)]
        if not compaction_idxs:
            continue
        task_id = detect_task_id(run, path, run_idx)
        run_id = detect_run_id(run, path, run_idx)
        model = detect_model(run, path)
        harness_version = detect_harness_version(run)
        final_answer = detect_final_answer(run, events)
        final_correct = detect_final_correct(run)
        gold_answer = first_nonempty(run, ["ground_truth", "gold_answer", "gold", "target", "expected", "reference_answer"])
        if gold_answer in (None, ""):
            gold_answer = gold_map.get(str(task_id), "")
        status = safe_str(first_nonempty(run, ["trajectory_status", "status", "failure_reason", "error"]), max_chars=500)
        budget = detect_budget(run)

        # Render each event once; build prefix sums for tool/search/page counts.
        rendered = [event_to_text(ev) for ev in events]
        tool_pe = [count_tool_events([ev]) for ev in events]
        search_pe = [count_search_events([ev]) for ev in events]
        page_pe = [count_page_events([ev]) for ev in events]
        tool_ps, search_ps, page_ps = _prefix_sums(tool_pe), _prefix_sums(search_pe), _prefix_sums(page_pe)
        n = len(events)

        def join_slice(lo: int, hi: int) -> str:
            return "\n\n".join(f"--- event {i} ---\n{rendered[i]}" for i in range(lo, hi))

        for compaction_number, idx in enumerate(compaction_idxs, start=1):
            pre_text = join_slice(0, idx)
            post_text = join_slice(idx + 1, n)
            summary = extract_compacted_summary(events[idx])
            pre_tokens = approx_tokens(pre_text)
            post_tokens = approx_tokens(post_text)
            summary_tokens = approx_tokens(summary)
            compression = round(summary_tokens / pre_tokens, 6) if pre_tokens else 0.0
            event_id = stable_id(str(path), run_id, str(idx), str(compaction_number), n=16)

            # Layer-2 metrics on FULL text before any truncation.
            l2 = compute_layer2_fields(
                pre=pre_text,
                summary=summary,
                post=post_text,
                final_answer=final_answer,
                gold=gold_answer,
                final_correct=final_correct,
            )
            l2.update({
                "event_id": event_id,
                "task_id": task_id,
                "run_id": run_id,
                "model": model,
                "compaction_index": compaction_number,
                "final_correct": final_correct,
                "summary_token_length": summary_tokens,
                "compression_ratio": compression,
            })
            layer2_out.append(l2)

            events_out.append(
                Layer1Event(
                    event_id=event_id,
                    task_id=task_id,
                    run_id=run_id,
                    model=model,
                    harness_version=harness_version,
                    trajectory_path=str(path),
                    compaction_index=compaction_number,
                    compaction_event_index=idx,
                    num_compactions_total=len(compaction_idxs),
                    is_first_compaction=(compaction_number == 1),
                    tokens_before_compaction=pre_tokens,
                    tokens_after_compaction=post_tokens,
                    pre_compaction_token_length=pre_tokens,
                    post_compaction_token_length=post_tokens,
                    summary_token_length=summary_tokens,
                    compression_ratio=compression,
                    pre_compaction_transcript=safe_str(pre_text, max_chars=cap),
                    compacted_summary=safe_str(summary, max_chars=cap),
                    post_compaction_continuation=safe_str(post_text, max_chars=cap),
                    final_answer=final_answer,
                    gold_answer=gold_answer,
                    final_correct=final_correct,
                    num_tool_calls_before_compaction=tool_ps[idx],
                    num_tool_calls_after_compaction=tool_ps[n] - tool_ps[idx + 1],
                    num_searches_before_compaction=search_ps[idx],
                    num_searches_after_compaction=search_ps[n] - search_ps[idx + 1],
                    num_pages_opened_before_compaction=page_ps[idx],
                    num_pages_opened_after_compaction=page_ps[n] - page_ps[idx + 1],
                    wall_clock_or_step_budget=budget,
                    trajectory_status=status,
                )
            )
    return events_out, layer2_out, errors


def boolish(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def bool_to_cell(value: Any) -> Any:
    if value is True or value is False:
        return value
    return UNKNOWN


def compute_layer2(event: Layer1Event) -> Dict[str, Any]:
    fields = compute_layer2_fields(
        pre=event.pre_compaction_transcript,
        summary=event.compacted_summary,
        post=event.post_compaction_continuation,
        final_answer=event.final_answer,
        gold=event.gold_answer,
        final_correct=event.final_correct,
    )
    fields.update({
        "event_id": event.event_id,
        "task_id": event.task_id,
        "run_id": event.run_id,
        "model": event.model,
        "compaction_index": event.compaction_index,
        "final_correct": event.final_correct,
        "summary_token_length": event.summary_token_length,
        "compression_ratio": event.compression_ratio,
    })
    return fields


def compute_layer2_fields(
    pre: str,
    summary: str,
    post: str,
    final_answer: str,
    gold: Any,
    final_correct: Any,
) -> Dict[str, Any]:
    answer_seen = text_contains_answer(pre, gold)
    answer_preserved = text_contains_answer(summary, gold)
    answer_used_post = text_contains_answer(post + "\n" + (final_answer or ""), gold)

    pre_urls = extract_urls(pre)
    summary_urls = extract_urls(summary)
    post_urls = extract_urls(post)
    source_url_preserved = bool(set(pre_urls) & set(summary_urls)) if pre_urls else UNKNOWN
    new_urls_in_summary = sorted(set(summary_urls) - set(pre_urls))

    pre_queries = extract_search_queries(pre)
    summary_queries = extract_search_queries(summary)
    query_norm_pre = {normalize_text(q) for q in pre_queries}
    query_norm_summary = {normalize_text(q) for q in summary_queries}
    search_queries_preserved = bool(query_norm_pre & query_norm_summary) if pre_queries else UNKNOWN

    negative_in_pre = contains_any_term(pre, NEGATIVE_TERMS)
    negative_in_summary = contains_any_term(summary, NEGATIVE_TERMS)
    negative_evidence_preserved = negative_in_summary if negative_in_pre else UNKNOWN

    pre_candidates = extract_capitalized_spans(pre)
    summary_candidates = extract_capitalized_spans(summary)
    cand_pre_norm = {normalize_text(c) for c in pre_candidates}
    cand_sum_norm = {normalize_text(c) for c in summary_candidates}
    candidate_entities_preserved = bool(cand_pre_norm & cand_sum_norm) if pre_candidates else UNKNOWN

    open_next_steps_preserved = contains_any_term(summary, NEXT_STEP_TERMS)

    hallucination_suspect = False
    hallucination_reasons = []
    if new_urls_in_summary:
        hallucination_suspect = True
        hallucination_reasons.append("summary contains urls not seen before compaction")
    if answer_seen is False and answer_preserved is True:
        hallucination_suspect = True
        hallucination_reasons.append("gold answer appears in summary but not pre-compaction transcript")

    post_repeats_query = False
    if pre_queries:
        post_norm = normalize_text(post)
        post_repeats_query = any(normalize_text(q) in post_norm for q in pre_queries)
    post_reopens_url = bool(set(pre_urls) & set(post_urls)) if pre_urls else False

    evidence_level = 0
    if answer_seen is True and pre_urls:
        evidence_level = 4
    elif answer_seen is True:
        evidence_level = 3
    elif pre_candidates or pre_queries or pre_urls:
        evidence_level = 2 if (pre_candidates and (pre_queries or pre_urls)) else 1

    retention_level = 0
    if answer_preserved is True and source_url_preserved is True:
        retention_level = 4 if negative_evidence_preserved is True or open_next_steps_preserved else 3
    elif answer_preserved is True or candidate_entities_preserved is True:
        retention_level = 2
    elif source_url_preserved is True or search_queries_preserved is True or open_next_steps_preserved:
        retention_level = 1

    failure_category = classify_failure(
        final_correct=final_correct,
        evidence_level=evidence_level,
        answer_seen=answer_seen,
        answer_preserved=answer_preserved,
        answer_used_post=answer_used_post,
        source_preserved=source_url_preserved,
        hallucination_suspect=hallucination_suspect,
        negative_preserved=negative_evidence_preserved,
        post_repeats_query=post_repeats_query,
        post_reopens_url=post_reopens_url,
    )

    return {
        "answer_seen_before_compaction": bool_to_cell(answer_seen),
        "answer_preserved_in_summary": bool_to_cell(answer_preserved),
        "answer_used_after_compaction": bool_to_cell(answer_used_post),
        "source_url_preserved_in_summary": bool_to_cell(source_url_preserved),
        "negative_evidence_preserved": bool_to_cell(negative_evidence_preserved),
        "candidate_entities_preserved": bool_to_cell(candidate_entities_preserved),
        "search_queries_preserved": bool_to_cell(search_queries_preserved),
        "open_next_steps_preserved": bool_to_cell(open_next_steps_preserved),
        "summary_hallucinated_new_facts": "suspect" if hallucination_suspect else False,
        "hallucination_reasons": "; ".join(hallucination_reasons),
        "pre_compaction_evidence_level": evidence_level,
        "summary_retention_level": retention_level,
        "pre_url_count": len(pre_urls),
        "summary_url_count": len(summary_urls),
        "pre_search_query_count": len(pre_queries),
        "summary_search_query_count": len(summary_queries),
        "pre_candidate_count": len(pre_candidates),
        "summary_candidate_count": len(summary_candidates),
        "post_compaction_repeated_query": post_repeats_query,
        "post_compaction_reopened_url": post_reopens_url,
        "failure_category": failure_category,
        "needs_semantic_review": needs_semantic_review(answer_seen, answer_preserved, source_url_preserved, gold),
    }


def needs_semantic_review(answer_seen: Any, answer_preserved: Any, source_preserved: Any, gold_answer: Any) -> bool:
    if gold_answer in (None, ""):
        return True
    return UNKNOWN in {answer_seen, answer_preserved, source_preserved}


def classify_failure(
    final_correct: Any,
    evidence_level: int,
    answer_seen: Any,
    answer_preserved: Any,
    answer_used_post: Any,
    source_preserved: Any,
    hallucination_suspect: bool,
    negative_preserved: Any,
    post_repeats_query: bool,
    post_reopens_url: bool,
) -> str:
    if hallucination_suspect:
        return "summary_invented_unsupported_answer_or_fact"
    if evidence_level < 2:
        return "answer_not_found_before_compaction"
    if answer_seen is True and answer_preserved is False:
        return "answer_or_candidate_dropped_in_summary"
    if answer_preserved is True and source_preserved is False:
        return "answer_preserved_but_source_lost"
    if answer_preserved is True and answer_used_post is False and final_correct is not True:
        return "answer_preserved_but_ignored_later"
    if negative_preserved is False and (post_repeats_query or post_reopens_url):
        return "negative_evidence_lost_repeated_dead_end"
    if evidence_level >= 2 and answer_seen is not True and final_correct is not True:
        return "wrong_candidate_preserved"
    if final_correct is False:
        return "good_compaction_final_wrong"
    if final_correct is True:
        return "success"
    return "unknown_needs_review"


def apply_overrides(metrics: List[Dict[str, Any]], overrides: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not overrides:
        return metrics
    out = []
    for row in metrics:
        row = dict(row)
        override = overrides.get(str(row.get("event_id")))
        if override:
            row.update(override)
            if "failure_category" not in override:
                row["failure_category"] = classify_from_overridden(row)
        out.append(row)
    return out


def classify_from_overridden(row: Dict[str, Any]) -> str:
    hallucinated = row.get("summary_hallucinated_new_facts")
    if hallucinated is True or str(hallucinated).lower() in {"true", "suspect"}:
        return "summary_invented_unsupported_answer_or_fact"
    evidence_level = int(row.get("pre_compaction_evidence_level") or 0)
    if evidence_level < 2:
        return "answer_not_found_before_compaction"
    if row.get("answer_seen_before_compaction") is True and row.get("answer_preserved_in_summary") is False:
        return "answer_or_candidate_dropped_in_summary"
    if row.get("answer_preserved_in_summary") is True and row.get("source_url_preserved_in_summary") is False:
        return "answer_preserved_but_source_lost"
    if row.get("answer_preserved_in_summary") is True and row.get("answer_used_after_compaction") is False and row.get("final_correct") is not True:
        return "answer_preserved_but_ignored_later"
    if row.get("negative_evidence_preserved") is False and (row.get("post_compaction_repeated_query") is True or row.get("post_compaction_reopened_url") is True):
        return "negative_evidence_lost_repeated_dead_end"
    if row.get("final_correct") is False:
        return "good_compaction_final_wrong"
    if row.get("final_correct") is True:
        return "success"
    return "unknown_needs_review"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flatten_for_csv(row: Dict[str, Any], max_chars: int = 2000) -> Dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        out[key] = safe_str(value, max_chars=max_chars)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], max_chars: int = 2000) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row, max_chars=max_chars))


def group_by_model(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("model") or UNKNOWN)].append(row)
    return dict(groups)


def count_true(rows: Sequence[Dict[str, Any]], field: str) -> int:
    return sum(1 for r in rows if r.get(field) is True or str(r.get(field)).lower() == "true")


def compute_funnel(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for model, group in sorted(group_by_model(metrics).items()):
        total = len(group)
        evidence = [r for r in group if int(r.get("pre_compaction_evidence_level") or 0) >= 2 or r.get("answer_seen_before_compaction") is True]
        preserved = [r for r in evidence if r.get("answer_preserved_in_summary") is True or int(r.get("summary_retention_level") or 0) >= 2]
        used = [r for r in preserved if r.get("answer_used_after_compaction") is True]
        correct = [r for r in used if r.get("final_correct") is True]
        steps = [
            ("compacted_trajectories", group, total),
            ("had_answer_or_strong_candidate_before_compaction", evidence, total),
            ("preserved_answer_or_candidate_in_summary", preserved, len(evidence)),
            ("used_preserved_answer_or_candidate_after_compaction", used, len(preserved)),
            ("final_answer_correct", correct, len(used)),
        ]
        for step, subset, denom in steps:
            rows.append({
                "model": model,
                "funnel_step": step,
                "count": len(subset),
                "denominator": denom,
                "pct_of_prior_step": round(100.0 * len(subset) / denom, 2) if denom else 0.0,
                "pct_of_all_compacted": round(100.0 * len(subset) / total, 2) if total else 0.0,
            })
    return rows


def compute_failure_counts(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = sorted(group_by_model(metrics).keys())
    cats = sorted(set(str(r.get("failure_category") or UNKNOWN) for r in metrics))
    rows = []
    totals = {m: len(group_by_model(metrics)[m]) for m in models}
    by_model_cat: Dict[Tuple[str, str], int] = Counter((str(r.get("model") or UNKNOWN), str(r.get("failure_category") or UNKNOWN)) for r in metrics)
    for cat in cats:
        row = {"failure_category": cat}
        for model in models:
            count = by_model_cat[(model, cat)]
            row[f"{model}_count"] = count
            row[f"{model}_pct"] = round(100.0 * count / totals[model], 2) if totals[model] else 0.0
        if len(models) == 2:
            a, b = models[0], models[1]
            row["pct_delta_{}_minus_{}".format(b, a)] = round(row[f"{b}_pct"] - row[f"{a}_pct"], 2)
        rows.append(row)
    return rows


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def compute_summary_by_model(layer1: Sequence[Dict[str, Any]], metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    layer1_by_id = {r["event_id"]: r for r in layer1}
    rows = []
    for model, group in sorted(group_by_model(metrics).items()):
        l1 = [layer1_by_id.get(r["event_id"], {}) for r in group]
        rows.append({
            "model": model,
            "compaction_events": len(group),
            "unique_tasks": len(set(r.get("task_id") for r in group)),
            "final_correct_count": count_true(group, "final_correct"),
            "final_correct_pct": round(100.0 * count_true(group, "final_correct") / len(group), 2) if group else 0.0,
            "answer_seen_before_count": count_true(group, "answer_seen_before_compaction"),
            "answer_preserved_count": count_true(group, "answer_preserved_in_summary"),
            "answer_used_after_count": count_true(group, "answer_used_after_compaction"),
            "source_url_preserved_count": count_true(group, "source_url_preserved_in_summary"),
            "negative_evidence_preserved_count": count_true(group, "negative_evidence_preserved"),
            "mean_summary_tokens": mean([r.get("summary_token_length") for r in group]),
            "mean_compression_ratio": mean([r.get("compression_ratio") for r in group]),
            "mean_tool_calls_before": mean([r.get("num_tool_calls_before_compaction") for r in l1]),
            "mean_tool_calls_after": mean([r.get("num_tool_calls_after_compaction") for r in l1]),
            "mean_searches_before": mean([r.get("num_searches_before_compaction") for r in l1]),
            "mean_searches_after": mean([r.get("num_searches_after_compaction") for r in l1]),
            "mean_pages_opened_before": mean([r.get("num_pages_opened_before_compaction") for r in l1]),
            "mean_pages_opened_after": mean([r.get("num_pages_opened_after_compaction") for r in l1]),
        })
    return rows


def compute_matched_task_slice(metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_task: Dict[str, set] = defaultdict(set)
    for r in metrics:
        by_task[str(r.get("task_id"))].add(str(r.get("model") or UNKNOWN))
    if not by_task:
        return []
    all_models = set(str(r.get("model") or UNKNOWN) for r in metrics)
    matched_tasks = {task for task, models in by_task.items() if all_models.issubset(models)}
    return [r for r in metrics if str(r.get("task_id")) in matched_tasks]


def write_labeling_template(path: Path, layer1: Sequence[Dict[str, Any]], metrics: Sequence[Dict[str, Any]]) -> None:
    layer1_by_id = {r["event_id"]: r for r in layer1}
    rows = []
    for r in metrics:
        if not (r.get("needs_semantic_review") or r.get("failure_category") in {"unknown_needs_review", "wrong_candidate_preserved"}):
            continue
        l1 = layer1_by_id.get(r["event_id"], {})
        rows.append({
            "event_id": r.get("event_id"),
            "task_id": r.get("task_id"),
            "run_id": r.get("run_id"),
            "model": r.get("model"),
            "failure_category": r.get("failure_category"),
            "pre_compaction_evidence_level": r.get("pre_compaction_evidence_level"),
            "summary_retention_level": r.get("summary_retention_level"),
            "answer_seen_before_compaction": r.get("answer_seen_before_compaction"),
            "answer_preserved_in_summary": r.get("answer_preserved_in_summary"),
            "answer_used_after_compaction": r.get("answer_used_after_compaction"),
            "source_url_preserved_in_summary": r.get("source_url_preserved_in_summary"),
            "negative_evidence_preserved": r.get("negative_evidence_preserved"),
            "candidate_entities_preserved": r.get("candidate_entities_preserved"),
            "search_queries_preserved": r.get("search_queries_preserved"),
            "open_next_steps_preserved": r.get("open_next_steps_preserved"),
            "summary_hallucinated_new_facts": r.get("summary_hallucinated_new_facts"),
            "answer_seen_basis": "",
            "secondary_failure_tags": "",
            "review_notes": "",
            "gold_answer_preview": safe_str(l1.get("gold_answer"), max_chars=200),
            "summary_preview": safe_str(l1.get("compacted_summary"), max_chars=800),
        })
    write_csv(path, rows, max_chars=1000)


def render_markdown_report(
    output_dir: Path,
    layer1: Sequence[Dict[str, Any]],
    metrics: Sequence[Dict[str, Any]],
    funnel: Sequence[Dict[str, Any]],
    failure_counts: Sequence[Dict[str, Any]],
    summary: Sequence[Dict[str, Any]],
    errors: Sequence[str],
    matched_only: bool = False,
) -> None:
    def md_table(rows: Sequence[Dict[str, Any]], max_rows: int = 20) -> str:
        if not rows:
            return "_No rows._\n"
        fields = list(rows[0].keys())
        lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
        for row in rows[:max_rows]:
            lines.append("| " + " | ".join(safe_str(row.get(f, ""), max_chars=120).replace("|", "\\|") for f in fields) + " |")
        if len(rows) > max_rows:
            lines.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
        return "\n".join(lines) + "\n"

    report = []
    report.append("# Compaction Failure Analysis Report\n")
    if matched_only:
        report.append("\n_Scope: matched-only — restricted to tasks that compacted under every observed model (common-instance comparison)._\n")
    report.append("\n## Dataset Summary\n")
    report.append(f"- Compaction events: {len(layer1)}\n")
    report.append(f"- Unique tasks: {len(set(r.get('task_id') for r in layer1))}\n")
    report.append(f"- Models: {', '.join(sorted(set(str(r.get('model')) for r in layer1)))}\n")
    if errors:
        report.append(f"- Parse/extraction warnings: {len(errors)}. See `extraction_errors.txt`.\n")

    report.append("\n## Metrics Summary by Model\n")
    report.append(md_table(summary))

    report.append("\n## Retention Funnel by Model\n")
    report.append(md_table(funnel, max_rows=50))

    report.append("\n## Failure Categories by Model\n")
    report.append(md_table(failure_counts, max_rows=50))

    matched = compute_matched_task_slice(metrics)
    report.append("\n## Matched-Task Slice\n")
    report.append(f"Matched-task events across all observed models: {len(matched)}\n")
    if matched:
        matched_failure = compute_failure_counts(matched)
        report.append(md_table(matched_failure, max_rows=50))

    report.append("\n## Interpretation Notes\n")
    report.append("- GLM-self vs Ultra-self trajectories are observational full-stack comparisons; they mix pre-compaction search, compaction, and post-compaction continuation.\n")
    report.append("- Causal claims about compactor quality require same-prefix/same-executor replay.\n")
    report.append("- Treat deterministic semantic labels as draft labels. Use `labeling_template.csv` and the rubric for manual/LLM adjudication of ambiguous cases.\n")
    report.append("- Exclude hallucinated or gold-leaking summaries from any candidate SFT data.\n")
    (output_dir / "analysis_report.md").write_text("".join(report), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze compaction events in agent trajectories.")
    parser.add_argument("--input", nargs="+", required=True, help="trajectory files, directories, or glob patterns")
    parser.add_argument("--gold-file", help="optional JSON/JSONL/CSV file mapping task_id to gold answer")
    parser.add_argument("--labels", help="optional CSV/JSONL manual label overrides keyed by event_id")
    parser.add_argument("--output-dir", required=True, help="directory to write outputs")
    parser.add_argument("--matched-only", action="store_true", help="restrict all tables to tasks that compacted under every observed model (common-instance comparison)")
    parser.add_argument("--csv-max-chars", type=int, default=2000, help="max characters per CSV cell for raw text fields")
    parser.add_argument("--max-transcript-chars", type=int, default=20000, help="cap on stored pre/post transcript chars in layer1 outputs (metrics still computed on full text). 0 = no cap")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_map = extract_gold_map(args.gold_file)
    overrides = load_label_overrides(args.labels)
    files = discover_files(args.input)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    all_events: List[Layer1Event] = []
    all_layer2: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path in files:
        events, l2, errs = extract_layer1_events(path, gold_map, max_transcript_chars=args.max_transcript_chars)
        all_events.extend(events)
        all_layer2.extend(l2)
        errors.extend(errs)
        print(f"parsed {path.name}: {len(events)} compaction events", file=sys.stderr)

    if args.matched_only:
        by_task: Dict[str, set] = defaultdict(set)
        for e in all_events:
            by_task[e.task_id].add(e.model)
        all_models = set(e.model for e in all_events)
        keep = {t for t, ms in by_task.items() if all_models.issubset(ms)}
        before = len(all_events)
        all_events = [e for e in all_events if e.task_id in keep]
        all_layer2 = [r for r in all_layer2 if str(r.get("task_id")) in keep]
        print(f"matched-only: kept {len(all_events)}/{before} compaction events across {len(keep)} common tasks (models: {sorted(all_models)})")

    layer1_rows = [asdict(e) for e in all_events]
    # Layer-2 metrics were computed inline on full text during extraction; the
    # stored Layer-1 transcripts are already truncated to --max-transcript-chars.
    layer2_rows = apply_overrides(all_layer2, overrides)

    funnel = compute_funnel(layer2_rows)
    failure_counts = compute_failure_counts(layer2_rows)
    summary = compute_summary_by_model(layer1_rows, layer2_rows)
    matched = compute_matched_task_slice(layer2_rows)

    write_jsonl(output_dir / "layer1_compaction_events.jsonl", layer1_rows)
    write_csv(output_dir / "layer1_compaction_events.csv", layer1_rows, max_chars=args.csv_max_chars)
    write_jsonl(output_dir / "layer2_metrics.jsonl", layer2_rows)
    write_csv(output_dir / "layer2_metrics.csv", layer2_rows, max_chars=args.csv_max_chars)
    write_csv(output_dir / "retention_funnel_by_model.csv", funnel)
    write_csv(output_dir / "failure_categories_by_model.csv", failure_counts)
    write_csv(output_dir / "metrics_summary_by_model.csv", summary)
    write_csv(output_dir / "matched_task_layer2_metrics.csv", matched)
    write_labeling_template(output_dir / "labeling_template.csv", layer1_rows, layer2_rows)
    if errors:
        (output_dir / "extraction_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    else:
        (output_dir / "extraction_errors.txt").write_text("", encoding="utf-8")
    render_markdown_report(output_dir, layer1_rows, layer2_rows, funnel, failure_counts, summary, errors, matched_only=args.matched_only)

    print(f"Wrote {len(layer1_rows)} compaction events to {output_dir}")
    if errors:
        print(f"Warnings/errors: {len(errors)}; see extraction_errors.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
