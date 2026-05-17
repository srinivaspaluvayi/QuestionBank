"""
LLM-as-Judge for dbms_dataset_mixed.json answers.

Reads from:
  dbms_dataset_mixed.json          – questions, question types, choices, images, answers
  verify/<candidate>/dbms/<id>.txt – LLM-generated answer to evaluate

Gold answer is the inline `answer` field.
Entries with an empty `answer` field are skipped (the 25 Multiple Choice entries
that have no explicit gold answer).

Scores are written to:
  verify/judge/dbms/<candidate>/<id>.json
  verify/judge/dbms/<candidate>/summary.json

An optional per-type breakdown is printed in the summary.

Requires OLLAMA_API_KEY in verify/.env.

Usage:
    python judge_dbms.py --candidate gemma4:31b-cloud
    python judge_dbms.py --candidate gemma4:31b-cloud --ids dbms_001 dbms_005
    python judge_dbms.py --candidate gemma4:31b-cloud --last 10
    python judge_dbms.py --candidate gemma4:31b-cloud --type "Short Answer"
    python judge_dbms.py --candidate gemma4:31b-cloud --model qwen3.5:cloud
    python judge_dbms.py --list-candidates
    python judge_dbms.py --json /path/to/dbms_dataset_mixed.json --candidate gemma4:31b-cloud
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv
from ollama import Client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

VERIFY_DIR          = os.path.dirname(__file__)
DEFAULT_JSON        = os.path.join(VERIFY_DIR, "..", "dbms_dataset_mixed.json")
OLLAMA_CLOUD_HOST   = "https://ollama.com"
POST_CALL_DELAY_SEC = 5
REQUEST_TIMEOUT_SEC = 600

JUDGE_SYSTEM = (
    "You are an expert evaluator assessing the quality of a candidate answer "
    "to a given problem. Your task is to grade the answer across five dimensions "
    "using the provided 1–5 rubric scales. Evaluate carefully, consistently, and "
    "objectively. You MUST respond with valid JSON only — no markdown fences, "
    "no extra text."
)

JUDGE_TEMPLATE = """\
EVALUATION INSTRUCTIONS:
1. Read the problem carefully.
2. Compare the candidate answer with the gold reference.
3. Score each rubric independently.
4. Provide brief but specific justification for each score.
5. Be strict but fair; avoid inflating scores.
6. Focus on objective quality, not style preference.
7. If the answer is factually wrong, correctness should be scored low even if clarity is high.
8. If no reasoning is shown, reasoning quality should be scored low even if the final answer is correct.

EVALUATION RUBRICS:

A. Correctness (1-5): Measures whether the solution is mathematically, logically, or algorithmically right.
5 - Fully correct; all computations, logic steps, and final answers are accurate.
4 - Mostly correct; minor errors that do not change the overall result or core method.
3 - Partially correct; key ideas are right but major errors appear, or the final answer is correct with a flawed derivation.
2 - Limited correctness; shows some understanding but contains major conceptual or computational mistakes.
1 - Largely incorrect; little to no evidence of correct understanding.

B. Reasoning Quality (1-5): Measures the depth, rigor, and coherence of the explanation.
5 - Fully justified reasoning; every step is explained clearly and demonstrates strong conceptual understanding.
4 - Mostly solid reasoning; minor gaps exist but the overall logic is coherent and well-supported.
3 - Adequate reasoning; key steps are present but explanations are shallow, incomplete, or procedural.
2 - Weak reasoning; major steps lack justification or the logical flow is difficult to follow.
1 - Minimal reasoning; little to no explanation of the steps taken.

C. Clarity (1-5): Measures how clearly the solution is communicated.
5 - Exceptionally clear; well-organized, easy to follow, and uses precise notation throughout.
4 - Mostly clear; minor ambiguities or inconsistencies but overall understandable.
3 - Adequately clear; readable but somewhat disorganized or ambiguous in places.
2 - Hard to follow; poor structure or unclear notation impedes understanding.
1 - Very unclear; the solution is confusing or difficult to interpret.

D. Conciseness (1-5): Measures the efficiency of the explanation.
5 - Highly concise; includes all necessary information without redundancy or filler.
4 - Mostly concise; a bit wordy or slightly incomplete but overall efficient.
3 - Moderately concise; acceptable but could be noticeably shorter or clearer.
2 - Not concise; overly verbose or too brief in ways that hinder understanding.
1 - Very inefficient or incomplete; contains substantial irrelevant content or omits key details.

E. Completeness (1-5): Measures whether the solution addresses all parts of the problem fully.
5 - Fully complete; every component of the problem is addressed with no omissions.
4 - Mostly complete; minor missing details but core requirements are met.
3 - Partially complete; some parts handled well but others missing or underdeveloped.
2 - Incomplete; large portions of the problem are unaddressed.
1 - Very incomplete; only a small fraction of the required work is present.

---

PROBLEM TYPE: {q_type}

PROBLEM:
{question}

GOLD REFERENCE ANSWER:
{gold_answer}

CANDIDATE ANSWER:
{candidate_answer}

---

OUTPUT FORMAT (JSON only, no markdown):
{{
  "correctness": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "reasoning_quality": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "clarity": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "conciseness": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "completeness": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }}
}}
"""

DIMENSIONS = ["correctness", "reasoning_quality", "clarity", "conciseness", "completeness"]


def make_client(local: bool = False) -> Client:
    if local:
        return Client(host="http://localhost:11434", timeout=REQUEST_TIMEOUT_SEC)
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("OLLAMA_API_KEY is not set. Add it to verify/.env.", file=sys.stderr)
        sys.exit(1)
    return Client(host=OLLAMA_CLOUD_HOST, timeout=REQUEST_TIMEOUT_SEC)


def load_entries(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_question_prompt(entry: dict) -> str:
    lines = [entry["question"].strip()]
    choices = entry.get("choices") or {}
    if choices:
        lines.append("")
        lines.append("Choices:")
        for letter, text in sorted(choices.items()):
            lines.append(f"  {letter}. {text}")
    return "\n".join(lines)


def resolve_image(entry: dict, json_dir: str) -> str | None:
    rel = entry.get("image", "")
    if not rel:
        return None
    full = os.path.normpath(os.path.join(json_dir, rel))
    return full if os.path.exists(full) else None


def read_candidate_answer(candidate_dir: str, entry_id: str) -> str:
    path = os.path.join(candidate_dir, f"{entry_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def extract_json(text: str) -> dict:
    text  = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


def judge(client: Client, entry: dict, judge_model: str,
          candidate_dir: str, out_dir: str, json_dir: str) -> dict | None:
    entry_id = str(entry["id"])
    out_path = os.path.join(out_dir, f"{entry_id}.json")

    if os.path.exists(out_path):
        print(f"  [SKIP] #{entry_id} — already judged")
        return None

    gold_answer = (entry.get("answer") or "").strip()
    if not gold_answer:
        print(f"  [SKIP] #{entry_id} — no gold answer (skipping {entry.get('question_type', '')})")
        return None

    candidate_answer = read_candidate_answer(candidate_dir, entry_id)
    if not candidate_answer:
        print(f"  [SKIP] #{entry_id} — no candidate answer found")
        return None

    question_text = build_question_prompt(entry)
    q_type        = entry.get("question_type", "")

    prompt   = JUDGE_TEMPLATE.format(
        q_type=q_type,
        question=question_text,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer,
    )
    user_msg: dict = {"role": "user", "content": prompt}

    img_path = resolve_image(entry, json_dir)
    if img_path:
        user_msg["images"] = [img_path]

    response = client.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            user_msg,
        ],
        options={"temperature": 0},
    )

    raw = response["message"]["content"].strip()
    try:
        scores = extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [ERR]  #{entry_id} — JSON parse failed: {exc}")
        print(f"         Raw: {raw[:200]}")
        return None

    result = {
        "id":            entry_id,
        "question_type": q_type,
        "candidate":     os.path.basename(candidate_dir),
        "judge":         judge_model,
        "scores":        scores,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    dim_str = "  ".join(
        f"{d[:3].upper()}:{scores.get(d, {}).get('score', '?')}"
        for d in DIMENSIONS
    )
    print(f"  [OK]   #{entry_id} [{q_type}]  [{dim_str}]")
    return result


def write_summary(out_dir: str, results: list[dict]) -> None:
    if not results:
        return

    # Overall averages
    totals: dict[str, list] = {d: [] for d in DIMENSIONS}
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: {d: [] for d in DIMENSIONS})

    for r in results:
        qt = r.get("question_type", "unknown")
        for d in DIMENSIONS:
            score = r.get("scores", {}).get(d, {}).get("score")
            if isinstance(score, (int, float)):
                totals[d].append(score)
                by_type[qt][d].append(score)

    averages = {d: (sum(v) / len(v) if v else 0.0) for d, v in totals.items()}
    overall  = sum(averages.values()) / len(averages) if averages else 0.0

    type_summary = {}
    for qt, dims in by_type.items():
        avgs    = {d: round(sum(v) / len(v), 2) if v else 0.0 for d, v in dims.items()}
        ov      = round(sum(avgs.values()) / len(avgs), 2) if avgs else 0.0
        type_summary[qt] = {"averages": avgs, "overall": ov, "n": len(list(dims.values())[0])}

    summary = {
        "n_judged":     len(results),
        "averages":     {d: round(avg, 2) for d, avg in averages.items()},
        "overall":      round(overall, 2),
        "by_type":      type_summary,
    }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n── Overall Summary ──────────────────────")
    for d, avg in averages.items():
        bar = "█" * int(round(avg)) + "░" * (5 - int(round(avg)))
        print(f"  {d:<20s} {bar}  {avg:.2f}")
    print(f"  {'Overall':<20s}            {overall:.2f}")

    print("\n── By Question Type ─────────────────────")
    for qt, ts in sorted(type_summary.items()):
        print(f"  {qt} (n={ts['n']})  overall={ts['overall']:.2f}")

    print(f"\nSaved: {path}")


def list_candidates() -> None:
    base = os.path.join(VERIFY_DIR, "judge", "dbms")
    if not os.path.isdir(base):
        print("No judged dbms results yet.")
        return
    print("Judged candidates (dbms):")
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.endswith(".json") and f != "summary.json"])
            print(f"  {name}  ({count} judged)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge for dbms_dataset_mixed.json answers."
    )
    parser.add_argument("--candidate", metavar="MODEL",
                        help="Model subfolder in verify/<model>/dbms/ to judge")
    parser.add_argument("--list-candidates", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids",  nargs="+", metavar="ID")
    group.add_argument("--last", type=int,  metavar="N")
    parser.add_argument("--type",  dest="q_type", metavar="TYPE",
                        help='Filter by question type, e.g. "Short Answer"')
    parser.add_argument("--model", default="gemma4:31b-cloud",
                        help="Ollama Cloud judge model (default: gemma4:31b-cloud)")
    parser.add_argument("--local", action="store_true",
                        help="Use local Ollama instead of Cloud")
    parser.add_argument("--json",  dest="json_path", default=DEFAULT_JSON,
                        help="Path to dbms_dataset_mixed.json")
    args = parser.parse_args()

    if args.list_candidates:
        list_candidates()
        return

    if not args.candidate:
        parser.error("--candidate is required")

    json_path = os.path.abspath(args.json_path)
    if not os.path.isfile(json_path):
        print(f"JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    json_dir      = os.path.dirname(json_path)
    candidate_dir = os.path.join(VERIFY_DIR, args.candidate, "dbms")
    if not os.path.isdir(candidate_dir):
        print(f"Candidate answer directory not found: {candidate_dir}", file=sys.stderr)
        sys.exit(1)

    client      = make_client(local=args.local)
    all_entries = load_entries(json_path)

    if args.ids:
        id_set  = set(args.ids)
        entries = [e for e in all_entries if str(e["id"]) in id_set]
        if not entries:
            print(f"No entries found for ids: {args.ids}", file=sys.stderr)
            sys.exit(1)
    elif args.last:
        entries = all_entries[-args.last:]
    else:
        entries = [
            e for e in all_entries
            if os.path.exists(os.path.join(candidate_dir, f"{e['id']}.txt"))
        ]

    if args.q_type:
        entries = [e for e in entries if e.get("question_type", "").lower() == args.q_type.lower()]
        if not entries:
            print(f"No entries with question_type '{args.q_type}'", file=sys.stderr)
            sys.exit(1)

    out_dir = os.path.join(VERIFY_DIR, "judge", "dbms", args.candidate)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Judge model : {args.model}")
    print(f"Candidate   : {args.candidate}/dbms  ({len(entries)} answer files)")
    print(f"Output      : {out_dir}")
    print(f"Running {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}…\n")

    collected = []
    for i, entry in enumerate(entries):
        try:
            result = judge(client, entry, args.model, candidate_dir, out_dir, json_dir)
            if result:
                collected.append(result)
        except Exception as exc:
            print(f"  [ERR]  #{entry['id']} — {exc}")
        if i < len(entries) - 1:
            print(f"  Waiting {POST_CALL_DELAY_SEC}s…")
            time.sleep(POST_CALL_DELAY_SEC)

    write_summary(out_dir, collected)
    print("\nDone.")


if __name__ == "__main__":
    main()
