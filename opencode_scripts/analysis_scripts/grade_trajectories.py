#!/usr/bin/env python3
"""
grade_trajectories.py  --  judge opencode trajectories using the same
grading logic as frankie_evals/browsecomp_eval.py, against the NV Inference
endpoint, then print accuracy + trajectory stats.

Usage:
    python3 grade_trajectories.py \\
        --trajectories /path/to/trajectories.jsonl \\
        --output       /path/to/graded_results.jsonl \\
        [--model       nvidia/zai-org/glm-5.1] \\
        [--base-url    https://inference-api.nvidia.com/v1] \\
        [--api-key     sk-...] \\
        [--num-parallel 32] \\
        [--total-samples 400]   # denominator for worst-case accuracy

The grader re-uses the EXACT grader template, _parse_grading, and
grade_sample logic from browsecomp_eval.py (copied verbatim).

--total-samples: expected total dataset size (default 400). Used to compute
  worst-case accuracy where unfinished samples count as incorrect.
"""

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Judge fields kept under the "grading" key in graded_trajectories.jsonl
JUDGE_KEYS = [
    "is_correct", "extracted_answer", "judge_reasoning", "judge_think",
    "grading_text", "grading_attempts", "judge_model", "judge_base_url",
    "response", "tool_calls", "num_messages",
]

# ---------------------------------------------------------------------------
# Grader template + parser (verbatim from browsecomp_eval.py)
# ---------------------------------------------------------------------------
GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to \
extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect \
based on [correct_answer], focusing only on if there are meaningful \
differences between [correct_answer] and the extracted_final_answer. Do not \
comment on any background to the problem, do not attempt to solve the \
problem, do not argue for any answer different than [correct_answer], focus \
only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] \
given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, \
non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from \
[response]. Put 100 if there is no confidence score available."""


def _parse_grading(text):
    """Parse grading output. Returns (is_correct, extracted, parsed_ok).
    Uses the LAST 'correct: yes/no' match to avoid picking up template
    echoes or reasoning inside <think> blocks.
    """
    matches = list(re.finditer(r"correct:\s*(yes|no)\b", text, re.IGNORECASE))
    if not matches:
        return False, None, False
    is_correct = matches[-1].group(1).lower() == "yes"
    ans_matches = list(re.finditer(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text))
    extracted = ans_matches[-1].group(1).strip() if ans_matches else None
    if extracted and "The final exact answer extracted from the [response]" in extracted:
        return False, None, False
    return is_correct, extracted, True


def _parse_grading_details(text):
    """Extract judge's <think> reasoning (if any) and the 'reasoning:' field
    from the structured output section."""
    # <think>...</think> block — present when the judge model emits chain-of-thought
    think_match = re.search(r"<think>(.*?)</think>", text, re.S | re.IGNORECASE)
    judge_think = think_match.group(1).strip() if think_match else None

    # strip the think block to isolate the structured output
    structured = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.IGNORECASE).strip()

    # 'reasoning: <text up to the next labelled field>'
    reasoning_match = re.search(
        r"reasoning:\s*(.*?)(?=\ncorrect:|\nconfidence:|\nextracted_final_answer:|$)",
        structured, re.S | re.IGNORECASE,
    )
    judge_reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    return judge_think, judge_reasoning


def grade_sample(grader_client, grader_model, question, correct_answer, response):
    """Grade one sample. Retries up to 10 times with escalating temperature."""
    prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    max_attempts = 10
    last_text = ""
    for attempt in range(max_attempts):
        try:
            temp = 0.0 if attempt == 0 else min(0.3 + 0.1 * attempt, 1.0)
            completion = grader_client.chat.completions.create(
                model=grader_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=temp,
            )
            text = completion.choices[0].message.content or ""
            last_text = text
            is_correct, extracted, parsed_ok = _parse_grading(text)
            if parsed_ok:
                return {
                    "is_correct": is_correct,
                    "extracted_answer": extracted,
                    "grading_text": text,
                    "grading_attempts": attempt + 1,
                }
            logger.warning("attempt %d/%d unparseable (temp=%.1f), retrying",
                           attempt + 1, max_attempts, temp)
        except Exception as e:
            logger.warning("grading attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2 ** attempt, 30))
    logger.warning("grading failed after %d attempts, marking incorrect", max_attempts)
    return {"is_correct": False, "extracted_answer": None,
            "grading_text": last_text, "grading_attempts": max_attempts}


# ---------------------------------------------------------------------------
# Response extraction from opencode trajectory messages
# ---------------------------------------------------------------------------
def extract_response(messages):
    """Return the last non-empty assistant text across all message parts."""
    final_texts = [
        p.get("text", "")
        for m in (messages or [])
        if m.get("role") == "assistant"
        for p in m.get("parts", [])
        if p.get("type") == "text" and p.get("text", "").strip()
    ]
    return final_texts[-1].strip() if final_texts else ""


def count_tool_calls(messages):
    return sum(
        1 for m in (messages or [])
        for p in m.get("parts", [])
        if p.get("type") == "tool"
    )


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------
def print_stats(results, total_samples):
    import statistics

    n = len(results)
    n_correct = sum(1 for r in results if r.get("is_correct"))
    n_no_answer = sum(1 for r in results if not r.get("response", "").strip())
    missing = total_samples - n

    # accuracy on finished samples
    acc_finished = n_correct / n if n else 0.0
    # worst-case: unfinished = incorrect
    acc_worst = n_correct / total_samples

    tool_counts = [r["tool_calls"] for r in results]
    tool_counts_sorted = sorted(tool_counts)

    def pct(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    print("\n" + "=" * 60)
    print("  TRAJECTORY STATS")
    print("=" * 60)
    print(f"  Finished samples       : {n}  (missing: {missing}/{total_samples})")
    print(f"  Samples with no answer : {n_no_answer}")
    print(f"  Correct                : {n_correct}")
    print(f"  Accuracy (finished)    : {acc_finished:.1%}  ({n_correct}/{n})")
    print(f"  Accuracy (worst-case)  : {acc_worst:.1%}  "
          f"({n_correct}/{total_samples}, unfinished=incorrect)")
    print()
    print("  Tool calls per trajectory")
    print(f"    min    : {min(tool_counts)}")
    print(f"    max    : {max(tool_counts)}")
    print(f"    mean   : {statistics.mean(tool_counts):.1f}")
    print(f"    median : {statistics.median(tool_counts):.1f}")
    print(f"    stdev  : {statistics.stdev(tool_counts):.1f}" if n > 1 else "")
    print(f"    p25    : {pct(tool_counts_sorted, 25)}")
    print(f"    p75    : {pct(tool_counts_sorted, 75)}")
    print(f"    p90    : {pct(tool_counts_sorted, 90)}")
    print(f"    p99    : {pct(tool_counts_sorted, 99)}")
    print()
    print("  Distribution of tool calls")
    buckets = [(0,4),(5,9),(10,14),(15,19),(20,29),(30,49),(50,999)]
    for lo, hi in buckets:
        cnt = sum(1 for t in tool_counts if lo <= t <= hi)
        bar = "█" * (cnt * 30 // max(n, 1))
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"    {label:>6}  {bar:<30}  {cnt}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grade opencode trajectories")
    parser.add_argument("--trajectories", required=True,
                        help="Path to trajectories.jsonl")
    parser.add_argument("--output", default=None,
                        help="Path for graded results JSONL (default: <traj_dir>/graded_results.jsonl)")
    parser.add_argument("--model",    required=True,
                        help="Judge model name. For local vLLM use the --served-model-name "
                             "(e.g. 'GLM-5-FP8'). For remote use the full model id.")
    parser.add_argument("--base-url", required=True,
                        help="Judge API base URL. For local vLLM: http://<host>:<port>/v1")
    parser.add_argument("--api-key",
                        default=os.environ.get("JUDGE_API_KEY", "token-abc123"),
                        help="Judge API key. For local vLLM any non-empty string works. "
                             "Also read from JUDGE_API_KEY env var.")
    parser.add_argument("--num-parallel", type=int, default=32)
    parser.add_argument("--total-samples", type=int, default=400,
                        help="Expected total dataset size (for worst-case accuracy)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in --output")
    parser.add_argument("--graded-trajectories", default=None,
                        help="Path for full graded trajectories JSONL (entire original "
                             "trajectory + reward + judge output/metadata). "
                             "Default: <output_dir>/graded_trajectories.jsonl")
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    out_path = Path(args.output) if args.output else traj_path.parent / "graded_results.jsonl"
    graded_traj_path = (Path(args.graded_trajectories) if args.graded_trajectories
                        else out_path.parent / "graded_trajectories.jsonl")

    if not args.api_key:
        parser.error("--api-key or INFERENCE_API_KEY env var required")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # load already-graded sample_ids for resume
    graded_ids = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    graded_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        logger.info("Resume: skipping %d already-graded samples", len(graded_ids))

    # load trajectories
    records = []
    with open(traj_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info("Loaded %d trajectories from %s", len(records), traj_path)

    to_grade = [r for r in records if r["sample_id"] not in graded_ids]
    logger.info("Grading %d samples | parallel=%d | judge_model=%s | judge_base_url=%s",
                len(to_grade), args.num_parallel, args.model, args.base_url)

    results = []
    done = 0

    def grade_one(rec):
        messages = rec.get("messages") or []
        response = extract_response(messages)
        grade = grade_sample(client, args.model,
                             rec["question"], rec["ground_truth"], response)
        grading_text = grade["grading_text"] or ""
        judge_think, judge_reasoning = _parse_grading_details(grading_text)
        return {
            "sample_id":        rec["sample_id"],
            "shard_id":         rec.get("shard_id"),
            "question":         rec["question"],
            "ground_truth":     rec["ground_truth"],
            "response":         response,
            "tool_calls":       count_tool_calls(messages),
            "num_messages":     len(messages),
            "judge_model":      args.model,
            "judge_base_url":   args.base_url,
            "is_correct":       grade["is_correct"],
            "extracted_answer": grade["extracted_answer"],
            "judge_think":      judge_think,      # <think>...</think> chain-of-thought
            "judge_reasoning":  judge_reasoning,  # 'reasoning:' field from structured output
            "grading_text":     grading_text,     # full raw judge response
            "grading_attempts": grade["grading_attempts"],
        }

    with open(out_path, "a") as out_f, open(graded_traj_path, "a") as traj_f:
        with ThreadPoolExecutor(max_workers=args.num_parallel) as pool:
            futures = {pool.submit(grade_one, r): r for r in to_grade}
            for fut in as_completed(futures):
                rec = futures[fut]  # original trajectory record
                try:
                    result = fut.result()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    # full record: entire original trajectory + reward + judge output
                    grading = {k: result.get(k) for k in JUDGE_KEYS}
                    merged = {**rec, "is_correct": result.get("is_correct"), "grading": grading}
                    traj_f.write(json.dumps(merged) + "\n")
                    traj_f.flush()
                    results.append(result)
                    done += 1
                    if done % 10 == 0 or done == len(to_grade):
                        correct_so_far = sum(1 for r in results if r["is_correct"])
                        logger.info("Progress: %d/%d graded | correct so far: %d (%.1f%%)",
                                    done, len(to_grade), correct_so_far,
                                    correct_so_far / done * 100)
                except Exception as e:
                    logger.error("Failed to grade sample: %s", e)

    # merge with any previously graded results (for resume)
    if graded_ids:
        with open(out_path) as f:
            all_results = [json.loads(l) for l in f if l.strip()]
        results = all_results

    logger.info("Grading complete. Results written to %s", out_path)
    logger.info("Full graded trajectories written to %s", graded_traj_path)
    print_stats(results, args.total_samples)


if __name__ == "__main__":
    main()
