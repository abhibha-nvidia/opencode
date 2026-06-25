#!/usr/bin/env python3
"""
launch_opencode_evals.py

Per-shard driver: called by run_ultra_opencode_sharded.sh after vLLM is up
and the shard's opencode.json has been written.

For each record in the shard JSONL it:
  1. Extracts ONLY the last user message from responses_create_params.input
     (system prompt and dataset tools are dropped — opencode's researcher
     agent provides the system prompt; MCP provides the tools).
  2. Runs `opencode run --agent researcher` in an isolated working dir
     with its own XDG_DATA_HOME (separate sqlite DB per rollout, no lock
     contention under parallel execution).
  3. Exports the trajectory from that rollout's sqlite DB, wraps it with
     sample metadata, and atomically writes trajectory.json — only if the
     session is non-null and messages are non-empty (otherwise the sample
     is left without a trajectory.json so resume will retry it).
  4. After all questions finish, collates work/qNNNNN/trajectory.json into
     ${SHARD_DIR}/trajectories_shard${SHARD_ID}.jsonl.

Resume: samples whose trajectory.json already passes the completeness check
are skipped; everything else (missing / empty / crashed) is re-run after
wiping stale partial state.

Usage (called by the shard sbatch script):
    python3 launch_opencode_evals.py \\
        --shard-jsonl  /path/to/shard0.jsonl \\
        --shard-dir    /path/to/shard0/ \\
        --shard-id     0 \\
        --opencode-bin /home/.../.opencode/bin/opencode \\
        --model        local/ultra-v3-step42-green-mtp-boosted \\
        [--agent       researcher] \\
        [--parallel    16]
"""

import argparse
import fcntl
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trajectory SQL — same query as run_ultra_opencode.sh
# ---------------------------------------------------------------------------
TRAJ_SQL = """
SELECT json_object(
  'session_id', s.id,
  'session', json_object(
    'id', s.id, 'slug', s.slug, 'directory', s.directory, 'title', s.title,
    'version', s.version, 'agent', s.agent, 'model', s.model, 'cost', s.cost,
    'tokens_input', s.tokens_input, 'tokens_output', s.tokens_output,
    'tokens_reasoning', s.tokens_reasoning,
    'time_created', s.time_created, 'time_updated', s.time_updated),
  'messages', (
    SELECT json_group_array(json_set(m.data, '$.parts',
      (SELECT json_group_array(json(p.data))
         FROM (SELECT data FROM part WHERE message_id = m.id
               ORDER BY time_created) p)))
    FROM (SELECT id, data FROM message
          WHERE session_id = s.id ORDER BY time_created) m))
FROM session s ORDER BY s.time_created DESC LIMIT 1;
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_user_question(record: dict) -> str:
    """Return the last user message content from responses_create_params.input.
    Returns empty string if not found."""
    inp = record.get("responses_create_params", {}).get("input", [])
    users = [m for m in inp if isinstance(m, dict) and m.get("role") == "user"]
    return users[-1].get("content", "") if users else ""


def is_complete(traj_path: Path) -> bool:
    """True only if trajectory.json exists, parses, has a non-null session
    and a non-empty messages list."""
    if not traj_path.is_file() or traj_path.stat().st_size == 0:
        return False
    try:
        d = json.loads(traj_path.read_text())
        msgs = d.get("messages")
        return d.get("session") is not None and isinstance(msgs, list) and len(msgs) > 0
    except Exception:
        return False


def export_trajectory(db_path: Path) -> dict:
    """Query opencode's sqlite DB and return the raw session dict, or {}."""
    if not db_path.is_file():
        return {}
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(TRAJ_SQL).fetchone()
        con.close()
        return json.loads(row[0]) if row else {}
    except Exception as e:
        log.warning("sqlite export failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Unified multi-session export (main agent + every subagent)
# ---------------------------------------------------------------------------
# Same {session, messages} shape as TRAJ_SQL but for ONE session id, so the format
# is identical to the legacy export -- just selected per-session instead of "newest".
SESSION_SQL = """
SELECT json_object(
  'session', json_object(
    'id', s.id, 'slug', s.slug, 'directory', s.directory, 'title', s.title,
    'version', s.version, 'agent', s.agent, 'model', s.model, 'cost', s.cost,
    'tokens_input', s.tokens_input, 'tokens_output', s.tokens_output,
    'tokens_reasoning', s.tokens_reasoning,
    'time_created', s.time_created, 'time_updated', s.time_updated),
  'messages', (
    SELECT json_group_array(json_set(m.data, '$.parts',
      (SELECT json_group_array(json(p.data))
         FROM (SELECT data FROM part WHERE message_id = m.id ORDER BY time_created) p)))
    FROM (SELECT id, data FROM message WHERE session_id = s.id ORDER BY time_created) m))
FROM session s WHERE s.id = ?;
""".strip()


def _wjson(obj, path: Path) -> None:
    """Write pretty JSON, tolerating lone surrogates stored by opencode."""
    with open(path, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def export_all_sessions(db_path: Path):
    """Return (root_export, subagents, tree). root_export is the {session,messages}
    dict for the ROOT (parent_id IS NULL, earliest) session -- the MAIN agent thread,
    fixing the legacy 'newest session' (DESC LIMIT 1) bug. subagents is a list of
    (parent_id, session_id, agent, {session,messages}) for every non-root session.
    tree is a per-session manifest. Returns ({}, [], []) on failure."""
    if not db_path.is_file():
        return {}, [], []
    try:
        con = sqlite3.connect(str(db_path))
        sessions = con.execute(
            "SELECT id, parent_id, agent, time_created FROM session ORDER BY time_created"
        ).fetchall()
        if not sessions:
            con.close()
            return {}, [], []

        def export_one(sid):
            row = con.execute(SESSION_SQL, (sid,)).fetchone()
            return json.loads(row[0]) if row and row[0] else {"session": None, "messages": []}

        roots = [s for s in sessions if not s[1]]
        root = sorted(roots, key=lambda s: s[3])[0] if roots else sorted(sessions, key=lambda s: s[3])[0]
        root_id = root[0]
        root_export = export_one(root_id)

        subagents = []
        tree = []
        for sid, parent_id, agent, t in sessions:
            mc = con.execute("SELECT COUNT(*) FROM message WHERE session_id=?", (sid,)).fetchone()[0]
            tree.append({"session_id": sid, "parent_id": parent_id, "agent": agent,
                         "time_created": t, "messages": mc, "is_root": (sid == root_id)})
            if sid != root_id:
                subagents.append((parent_id, sid, agent, export_one(sid)))
        con.close()
        return root_export, subagents, tree
    except Exception as e:
        log.warning("multi-session export failed: %s", e)
        return {}, [], []


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------

def run_one(idx: int, record: dict, args: argparse.Namespace,
            shard_dir: Path, run_trajectories: Path | None) -> bool:
    """Run one question. Returns True if trajectory produced, False otherwise."""
    work_dir = shard_dir / "work" / f"q{idx:05d}"
    traj_path = work_dir / "trajectory.json"
    shard_id = args.shard_id

    # Resume: skip already-complete rollouts
    if is_complete(traj_path):
        log.info("[shard %s] q%05d already complete, skip", shard_id, idx)
        return True

    # Wipe stale partial state
    for stale in [".xdg", "_traj_raw.json", "_traj_wrapped.json", "trajectory.json"]:
        p = work_dir / stale
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write record for debugging / resume
    (work_dir / "record.json").write_text(json.dumps(record))

    # Extract the user question; drop system prompt + dataset tools
    question = extract_user_question(record)
    if not question.strip():
        log.warning("[shard %s] q%05d: no user message, skipping", shard_id, idx)
        return False

    # Isolated XDG so each rollout has its own sqlite DB
    xdg = work_dir / ".xdg"
    xdg.mkdir(parents=True, exist_ok=True)

    # opencode reads opencode.json from CWD — copy the shard config in
    shutil.copy2(shard_dir / "opencode.json", work_dir / "opencode.json")

    env = {**os.environ, "XDG_DATA_HOME": str(xdg)}

    # `--dir` pins the project root to the per-rollout work dir so opencode
    # doesn't walk up to the git root and lose the shard's opencode.json.
    # `--` ends flag parsing so questions starting with `-` aren't treated as flags.
    cmd = [
        args.opencode_bin, "run",
        "--agent", args.agent,
        "--model", args.model,
        "--dir", str(work_dir),
        "--print-logs", "--log-level", "DEBUG",
        "--", question,
    ]
    log_path = work_dir / "opencode_run.log"
    try:
        with open(log_path, "w") as lf:
            subprocess.run(cmd, cwd=str(work_dir), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    except Exception as e:
        log.warning("[shard %s] q%05d: opencode error: %s", shard_id, idx, e)

    # Export trajectory from isolated DB. Newer opencode (run-from-source)
    # names the DB "opencode-local.db"; the older OOB binary used "opencode.db".
    # Prefer whichever exists so both harnesses work.
    db_dir = xdg / "opencode"
    db_path = next((db_dir / n for n in ("opencode-local.db", "opencode.db")
                    if (db_dir / n).is_file()), db_dir / "opencode.db")
    # UNIFIED export: pull EVERY session (root + subagents), not just the newest.
    root_export, subagents, tree = export_all_sessions(db_path)
    raw = root_export  # the ROOT/main session drives completeness + collation

    # Human-readable RAW transcript with discard-all boundaries marked inline
    # (prefix -> tool calls -> DISCARD ALL [last-k retained] -> ... -> next discard).
    # Written per-sample regardless of completion; best-effort.
    try:
        if db_path.is_file():
            render_py = Path(__file__).parent / "render_discard_trajectory.py"
            render_out = work_dir / "discard_snapshots" / "trajectory_rendered.txt"
            render_out.parent.mkdir(parents=True, exist_ok=True)
            with open(render_out, "w") as rf:
                subprocess.run([sys.executable, str(render_py), str(db_path), "--max-chars", "400"],
                               stdout=rf, stderr=subprocess.STDOUT, timeout=120)
    except Exception as e:
        log.warning("[shard %s] q%05d: trajectory render failed: %s", shard_id, idx, e)

    msgs = raw.get("messages")
    complete = raw.get("session") is not None and isinstance(msgs, list) and len(msgs) > 0

    wrapped = {
        "sample_id":    record.get("_sample_id", idx),
        "shard_id":     shard_id,
        "question":     record.get("question", ""),
        "ground_truth": record.get("ground_truth", ""),
        "session":      raw.get("session"),
        "messages":     msgs,
    }

    # Atomic write: only promote to trajectory.json if export is complete
    tmp = work_dir / "_traj_wrapped.json"
    tmp.write_text(json.dumps(wrapped))
    if complete:
        tmp.rename(traj_path)
        log.info("[shard %s] q%05d done", shard_id, idx)
        if run_trajectories:
            append_to_trajectories(traj_path, run_trajectories)
        # UNIFIED FORMAT: write the per-sample folder (root + subagents + tree)
        # under <run-dir>/complete_trajectories/sample_<id>/. Friendly to both
        # agent and subagent traces; root trajectory.json == the main thread.
        try:
            if args.run_dir:
                sid = record.get("_sample_id", idx)
                folder = Path(args.run_dir) / "complete_trajectories" / f"sample_{sid}"
                folder.mkdir(parents=True, exist_ok=True)
                _wjson(wrapped, folder / "trajectory.json")
                if subagents:
                    subdir = folder / "subagents"
                    subdir.mkdir(exist_ok=True)
                    for parent_id, sess_id, agent, exp in subagents:
                        _wjson({"sample_id": sid, "shard_id": shard_id,
                                "parent_session_id": parent_id, "agent": agent,
                                "session": exp.get("session"), "messages": exp.get("messages")},
                               subdir / f"{parent_id}__{sess_id}.json")
                _wjson({"sample_id": sid, "shard_id": shard_id,
                        "num_sessions": len(tree), "sessions": tree},
                       folder / "tree.json")
        except Exception as e:
            log.warning("[shard %s] q%05d: unified-format write failed: %s", shard_id, idx, e)
        return True
    else:
        tmp.unlink(missing_ok=True)
        log.warning("[shard %s] q%05d: export incomplete -> left for resume retry", shard_id, idx)
        return False


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def append_to_trajectories(traj_path: Path, run_trajectories: Path) -> None:
    """Thread-safe append of one trajectory to the shared run-level trajectories.jsonl.
    Uses an exclusive flock so concurrent shard workers don't interleave writes."""
    try:
        data = json.loads(traj_path.read_text())
        line = json.dumps(data) + "\n"
        with open(run_trajectories, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        log.warning("Failed to append %s to trajectories.jsonl: %s", traj_path, e)


def collate(shard_dir: Path, shard_id: int) -> int:
    """Write all completed per-question trajectories into trajectories_shard{i}.jsonl."""
    out_path = shard_dir / f"trajectories_shard{shard_id}.jsonl"
    work_dir = shard_dir / "work"
    written = 0
    with open(out_path, "w") as out:
        for traj in sorted(work_dir.glob("q*/trajectory.json")):
            try:
                out.write(json.dumps(json.loads(traj.read_text())) + "\n")
                written += 1
            except Exception as e:
                log.warning("Skipping %s: %s", traj, e)
    log.info("[shard %s] wrote %d trajectories -> %s", shard_id, written, out_path)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenCode per-shard eval driver")
    parser.add_argument("--shard-jsonl",  required=True, help="Path to this shard's JSONL")
    parser.add_argument("--shard-dir",    required=True, help="Per-shard working directory")
    parser.add_argument("--shard-id",     required=True, type=int)
    parser.add_argument("--opencode-bin", required=True, help="Path to opencode binary")
    parser.add_argument("--model",        required=True, help="local/<served-model-name>")
    parser.add_argument("--run-dir",      default=None,
                        help="Top-level run dir. If set, each completed trajectory "
                             "is appended to <run-dir>/trajectories.jsonl as it finishes.")
    parser.add_argument("--agent",        default="build")
    parser.add_argument("--parallel",     default=16, type=int,
                        help="Max concurrent opencode rollouts")
    # Tavily params — passed for logging/audit; actual values are baked into
    # the shard's opencode.json DEFAULT_PARAMETERS by the bash orchestrator.
    parser.add_argument("--tavily-search-depth",         default="advanced")
    parser.add_argument("--tavily-max-results",          default="5")
    parser.add_argument("--tavily-include-raw-content",  default="true")
    parser.add_argument("--tavily-exclude-domains-file", default="")
    args = parser.parse_args()

    shard_dir  = Path(args.shard_dir)
    shard_jsonl = Path(args.shard_jsonl)

    if not shard_jsonl.is_file():
        sys.exit(f"ERROR: shard JSONL not found: {shard_jsonl}")
    if not (shard_dir / "opencode.json").is_file():
        sys.exit(f"ERROR: opencode.json not found in {shard_dir}")

    run_trajectories = Path(args.run_dir) / "trajectories.jsonl" if args.run_dir else None

    records = [json.loads(l) for l in shard_jsonl.read_text().splitlines() if l.strip()]
    log.info("[shard %d] %d records | parallel=%d | agent=%s | model=%s",
             args.shard_id, len(records), args.parallel, args.agent, args.model)
    log.info("[shard %d] tavily params: search_depth=%s max_results=%s "
             "include_raw_content=%s exclude_domains_file=%s",
             args.shard_id, args.tavily_search_depth, args.tavily_max_results,
             args.tavily_include_raw_content,
             args.tavily_exclude_domains_file or "<none>")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(run_one, idx, rec, args, shard_dir, run_trajectories): idx
            for idx, rec in enumerate(records)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                if fut.result():
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                log.error("q%05d raised: %s", idx, e)
                failed += 1

    log.info("[shard %d] finished: %d done, %d failed/skipped", args.shard_id, done, failed)
    collate(shard_dir, args.shard_id)


if __name__ == "__main__":
    main()
