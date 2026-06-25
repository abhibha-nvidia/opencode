#!/usr/bin/env python3
"""Serial grading script using raw HTTP requests - avoids OpenAI/jiter segfault issues"""
import json
import re
import logging
import os
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

def _parse_grading(text):
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
    think_match = re.search(r"<tool_call>(.*?)\*/", text, re.S | re.IGNORECASE)
    judge_think = think_match.group(1).strip() if think_match else None
    structured = re.sub(r"<tool_call>.*?\*/", "", text, flags=re.S | re.IGNORECASE).strip()
    reasoning_match = re.search(
        r"reasoning:\s*(.*?)(?=\ncorrect:|\nconfidence:|\nextracted_final_answer:|$)",
        structured, re.S | re.IGNORECASE,
    )
    judge_reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    return judge_think, judge_reasoning

def extract_response(messages):
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

def grade_sample(base_url, api_key, grader_model, question, correct_answer, response):
    """Grade one sample using raw HTTP requests"""
    prompt = GRADER_TEMPLATE.format(question=question, correct_answer=correct_answer, response=response)
    max_attempts = 10
    last_text = ""
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_attempts):
        try:
            temp = 0.0 if attempt == 0 else min(0.3 + 0.1 * attempt, 1.0)
            payload = {
                "model": grader_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": temp,
            }
            
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            last_text = text
            
            is_correct, extracted, parsed_ok = _parse_grading(text)
            if parsed_ok:
                return {
                    "is_correct": is_correct,
                    "extracted_answer": extracted,
                    "grading_text": text,
                    "grading_attempts": attempt + 1,
                }
            logger.warning("attempt %d/%d unparseable (temp=%.1f), retrying", attempt + 1, max_attempts, temp)
        except Exception as e:
            logger.warning("grading attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2 ** attempt, 30))
    logger.warning("grading failed after %d attempts, marking incorrect", max_attempts)
    return {"is_correct": False, "extracted_answer": None, "grading_text": last_text, "grading_attempts": max_attempts}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="nvidia/zai-org/glm-5.1")
    parser.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    parser.add_argument("--api-key", default="sk-XZDJokqgVdqZ6UhAmKHGTA")
    parser.add_argument("--total-samples", type=int, default=400)
    args = parser.parse_args()

    out_path = args.output
    graded_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    graded_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        logger.info("Resume: skipping %d already-graded samples", len(graded_ids))

    records = []
    with open(args.trajectories) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info("Loaded %d trajectories", len(records))

    to_grade = [r for r in records if r["sample_id"] not in graded_ids]
    logger.info("Grading %d samples | judge_model=%s", len(to_grade), args.model)

    results = []
    with open(out_path, "a") as out_f:
        for i, rec in enumerate(to_grade):
            messages = rec.get("messages") or []
            response = extract_response(messages)
            grade = grade_sample(args.base_url, args.api_key, args.model, rec["question"], rec["ground_truth"], response)
            grading_text = grade["grading_text"] or ""
            judge_think, judge_reasoning = _parse_grading_details(grading_text)
            
            result = {
                "sample_id": rec["sample_id"],
                "shard_id": rec.get("shard_id"),
                "question": rec["question"],
                "ground_truth": rec["ground_truth"],
                "response": response,
                "tool_calls": count_tool_calls(messages),
                "num_messages": len(messages),
                "judge_model": args.model,
                "judge_base_url": args.base_url,
                "is_correct": grade["is_correct"],
                "extracted_answer": grade["extracted_answer"],
                "judge_think": judge_think,
                "judge_reasoning": judge_reasoning,
                "grading_text": grading_text,
                "grading_attempts": grade["grading_attempts"],
            }
            
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            results.append(result)
            
            correct_so_far = sum(1 for r in results if r["is_correct"])
            logger.info("Progress: %d/%d graded | correct: %d (%.1f%%)", 
                        i+1, len(to_grade), correct_so_far, correct_so_far/(i+1)*100)

    # Print stats
    n = len(results)
    n_correct = sum(1 for r in results if r.get("is_correct"))
    n_no_answer = sum(1 for r in results if not r.get("response", "").strip())
    missing = args.total_samples - n
    acc_finished = n_correct / n if n else 0.0
    acc_worst = n_correct / args.total_samples

    logger.info("Grading complete. Results written to %s", out_path)
    logger.info(f"Samples graded: {n}, correct: {n_correct}")
    logger.info(f"Accuracy (finished): {acc_finished:.3f} ({acc_finished*100:.1f}%)")
    logger.info(f"Accuracy (worst-case, {args.total_samples} total): {acc_worst:.3f} ({acc_worst*100:.1f}%)")

if __name__ == "__main__":
    main()
