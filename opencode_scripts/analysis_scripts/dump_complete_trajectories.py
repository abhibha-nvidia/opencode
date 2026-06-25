#!/usr/bin/env python3
"""Dump COMPLETE trajectories (main agent + all subagents) per sample.

For each per-question opencode-local.db it exports EVERY session using the same
{session, messages} shape as launch_opencode_evals.py (format unchanged), then
writes:

  complete_trajectories/
    sample_<id>/
      trajectory.json              # root/main session (parent_id IS NULL)
      subagents/
        <parent_id>__<session_id>.json   # each non-root session
      tree.json                    # session-tree manifest (ids/parents/agents/counts)
    index.json                     # run-level summary

Format of session+messages is identical to the existing export (same per-session SQL).
"""
import json, sqlite3, glob, re, os, sys

# Run dir is the first CLI arg; falls back to the original subagent-enabled run.
RUN = sys.argv[1] if len(sys.argv) > 1 else "/lustre/fsw/portfolios/llmservice/users/abhibhag/opencode-hsg/opencode_scripts/runs/sharded_source/sft_mix_0604_v2_n4post_frankprompt/20260619_001114_sft_mix_0604_v2_step0000485"
OUT = os.path.join(RUN, "complete_trajectories")

# Per-session export — identical session/messages shape to launch_opencode_evals.TRAJ_SQL
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


def build_sample_map():
    """(shard, qidx) -> {sample_id, question, ground_truth} from shards/shardN.jsonl."""
    m = {}
    for fp in glob.glob(os.path.join(RUN, "shards", "shard*.jsonl")):
        shard = int(re.search(r"shard(\d+)\.jsonl", fp).group(1))
        with open(fp) as f:
            for i, line in enumerate(f):  # line i (0-based) -> q{i:05d}
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                m[(shard, i)] = {
                    "sample_id": d.get("_sample_id"),
                    "question": d.get("question"),
                    "ground_truth": d.get("ground_truth"),
                }
    return m


def export_session(con, sid):
    row = con.execute(SESSION_SQL, (sid,)).fetchone()
    return json.loads(row[0]) if row and row[0] else {"session": None, "messages": []}


def main():
    smap = build_sample_map()
    dbs = glob.glob(os.path.join(RUN, "*/work/q*/.xdg/opencode/opencode-local.db"))
    print(f"sample-map entries: {len(smap)} | DBs found: {len(dbs)}")
    os.makedirs(OUT, exist_ok=True)
    index = []
    n_main = n_sub = n_nomain = 0
    for db in sorted(dbs):
        mm = re.search(r"/shard(\d+)/work/q(\d+)/", db)
        if not mm:
            continue
        shard, qidx = int(mm.group(1)), int(mm.group(2))
        meta = smap.get((shard, qidx), {})
        sid = meta.get("sample_id")
        folder = os.path.join(OUT, f"sample_{sid}" if sid is not None else f"shard{shard}_q{qidx:05d}")
        try:
            con = sqlite3.connect(db)
            sessions = con.execute(
                "SELECT id, parent_id, agent, time_created FROM session ORDER BY time_created"
            ).fetchall()
        except Exception as e:
            print(f"  ! {db}: {e}")
            continue
        if not sessions:
            con.close()
            continue
        roots = [s for s in sessions if not s[1]]
        subs = [s for s in sessions if s[1]]
        os.makedirs(folder, exist_ok=True)
        # root / main
        if roots:
            root = sorted(roots, key=lambda s: s[3])[0]
            exp = export_session(con, root[0])
            main_obj = {
                "sample_id": sid, "shard_id": shard, "q_index": qidx,
                "question": meta.get("question"), "ground_truth": meta.get("ground_truth"),
                "session": exp.get("session"), "messages": exp.get("messages"),
            }
            json.dump(main_obj, open(os.path.join(folder, "trajectory.json"), "w", encoding="utf-8", errors="surrogatepass"),
                      indent=2, ensure_ascii=False)
            root_id = root[0]
            n_main += 1
        else:
            root_id = None
            n_nomain += 1
        # subagents
        if subs:
            subdir = os.path.join(folder, "subagents")
            os.makedirs(subdir, exist_ok=True)
            for s in subs:
                exp = export_session(con, s[0])
                sub_obj = {
                    "sample_id": sid, "shard_id": shard,
                    "parent_session_id": s[1], "root_session_id": root_id,
                    "agent": s[2],
                    "session": exp.get("session"), "messages": exp.get("messages"),
                }
                fname = f"{s[1]}__{s[0]}.json"  # <parent_id>__<session_id>.json
                json.dump(sub_obj, open(os.path.join(subdir, fname), "w", encoding="utf-8", errors="surrogatepass"),
                          indent=2, ensure_ascii=False)
                n_sub += 1
        # per-sample session-tree manifest
        tree = []
        for s in sessions:
            mc = con.execute("SELECT COUNT(*) FROM message WHERE session_id=?", (s[0],)).fetchone()[0]
            tree.append({"session_id": s[0], "parent_id": s[1], "agent": s[2],
                         "time_created": s[3], "messages": mc,
                         "is_root": not s[1]})
        json.dump({"sample_id": sid, "shard_id": shard, "q_index": qidx,
                   "num_sessions": len(sessions), "sessions": tree},
                  open(os.path.join(folder, "tree.json"), "w", encoding="utf-8", errors="surrogatepass"), indent=2, ensure_ascii=False)
        index.append({"sample_id": sid, "shard_id": shard, "folder": os.path.basename(folder),
                      "num_sessions": len(sessions), "num_subagents": len(subs),
                      "has_main": bool(roots)})
        con.close()
    json.dump({"run": os.path.basename(RUN), "samples": len(index),
               "main_trajectories": n_main, "subagent_trajectories": n_sub,
               "samples_without_main_session": n_nomain,
               "index": sorted(index, key=lambda x: (x["sample_id"] is None, x["sample_id"]))},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8", errors="surrogatepass"), indent=2, ensure_ascii=False)
    print(f"main={n_main} subagents={n_sub} no-main={n_nomain} -> {OUT}")


if __name__ == "__main__":
    main()
