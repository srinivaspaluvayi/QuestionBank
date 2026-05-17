"""
LLM-as-Judge: evaluate existing model answers against gold solutions.

Reads from:
  Data/questions/<id>.txt   – question text
  Data/solutions/<id>.txt   – gold / expected answer
  verify/<candidate>/<id>.txt – LLM-generated answer to evaluate

Scores are written to:
  verify/judge/<candidate>/<id>.json   – per-entry scores (JSON)
  verify/judge/<candidate>/summary.json – aggregate scores

Requires OLLAMA_API_KEY in verify/.env.

Usage:
    python judge.py --candidate gemma4:31b-cloud              # judge all entries
    python judge.py --candidate gemma4:31b-cloud --ids 1 5 12
    python judge.py --candidate gemma4:31b-cloud --last 10
    python judge.py --candidate gemma4:31b-cloud --model qwen3.5:cloud
    python judge.py --list-candidates                          # show available models
"""

import argparse
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from ollama import Client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR            = os.path.join(os.path.dirname(__file__), "..", "Data")
JSON_FILE           = os.path.join(DATA_DIR, "questions.json")
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

B. Reasoning Quality (1-5): Measures the depth, rigor, and coherence of the explanation and whether each step logically follows.
5 - Fully justified reasoning; every step is explained clearly and demonstrates strong conceptual understanding.
4 - Mostly solid reasoning; minor gaps exist but the overall logic is coherent and well-supported.
3 - Adequate reasoning; key steps are present but explanations are shallow, incomplete, or procedural.
2 - Weak reasoning; major steps lack justification or the logical flow is difficult to follow.
1 - Minimal reasoning; little to no explanation of the steps taken.

C. Clarity (1-5): Measures how clearly the solution is communicated, including structure, notation, and readability.
5 - Exceptionally clear; well-organized, easy to follow, and uses precise notation throughout.
4 - Mostly clear; minor ambiguities or inconsistencies but overall understandable.
3 - Adequately clear; readable but somewhat disorganized or ambiguous in places.
2 - Hard to follow; poor structure or unclear notation impedes understanding.
1 - Very unclear; the solution is confusing or difficult to interpret.

D. Conciseness (1-5): Measures the efficiency of the explanation — avoids unnecessary verbosity while including essential details.
5 - Highly concise; includes all necessary information without redundancy or filler.
4 - Mostly concise; a bit wordy or slightly incomplete but overall efficient.
3 - Moderately concise; acceptable but could be noticeably shorter or clearer.
2 - Not concise; overly verbose or too brief in ways that hinder understanding.
1 - Very inefficient or incomplete; contains substantial irrelevant content or omits key details.

E. Completeness (1-5): Measures whether the solution addresses all parts of the problem fully and thoroughly.
5 - Fully complete; every component of the problem is addressed with no omissions.
4 - Mostly complete; minor missing details but core requirements are met.
3 - Partially complete; some parts handled well but others missing or underdeveloped.
2 - Incomplete; large portions of the problem are unaddressed.
1 - Very incomplete; only a small fraction of the required work is present.

---

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


# ── Client ────────────────────────────────────────────────────────────────────

def make_client(local: bool = False) -> Client:
    if local:
        return Client(host="http://localhost:11434", timeout=REQUEST_TIMEOUT_SEC)
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print(
            "OLLAMA_API_KEY is not set. Add it to verify/.env:\n"
            "  OLLAMA_API_KEY=your_api_key",
            file=sys.stderr,
        )
        sys.exit(1)
    return Client(host=OLLAMA_CLOUD_HOST, timeout=REQUEST_TIMEOUT_SEC)


# ── File helpers ──────────────────────────────────────────────────────────────

def read_file(rel_path: str) -> str:
    full = os.path.join(DATA_DIR, rel_path.lstrip("./"))
    if not os.path.exists(full):
        return ""
    with open(full, encoding="utf-8") as f:
        return f.read().strip()


def read_candidate_answer(candidate_dir: str, entry_id: str) -> str:
    path = os.path.join(candidate_dir, f"{entry_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def load_entries() -> list[dict]:
    with open(JSON_FILE, encoding="utf-8") as f:
        return json.load(f)


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Extract JSON object from model response, tolerating markdown fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the outermost { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end])


# ── Judge one entry ───────────────────────────────────────────────────────────

def judge(client: Client, entry: dict, judge_model: str,
          candidate_dir: str, out_dir: str) -> dict | None:
    entry_id = str(entry["id"])
    out_path = os.path.join(out_dir, f"{entry_id}.json")

    if os.path.exists(out_path):
        print(f"  [SKIP] #{entry_id} — already judged")
        return None

    question_text = read_file(entry["question"])
    if not question_text:
        print(f"  [SKIP] #{entry_id} — question file missing")
        return None

    gold_answer = read_file(entry.get("solution", ""))
    if not gold_answer:
        print(f"  [SKIP] #{entry_id} — solution file missing")
        return None

    candidate_answer = read_candidate_answer(candidate_dir, entry_id)
    if not candidate_answer:
        print(f"  [SKIP] #{entry_id} — no candidate answer found")
        return None

    # Attach images so the judge has full context
    image_paths = []
    for rel in entry.get("images", []):
        full = os.path.join(DATA_DIR, rel.lstrip("./"))
        if os.path.exists(full):
            image_paths.append(full)

    prompt = JUDGE_TEMPLATE.format(
        question=question_text,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer,
    )

    user_msg: dict = {"role": "user", "content": prompt}
    if image_paths:
        user_msg["images"] = image_paths

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
        print(f"         Raw response: {raw[:200]}")
        return None

    # Attach metadata
    result = {
        "id":        entry_id,
        "topic":     entry.get("topic", ""),
        "subtopic":  entry.get("subtopic", ""),
        "candidate": os.path.basename(candidate_dir),
        "judge":     judge_model,
        "scores":    scores,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    dim_str = "  ".join(
        f"{d[:3].upper()}:{scores.get(d, {}).get('score', '?')}"
        for d in DIMENSIONS
    )
    print(f"  [OK]   #{entry_id} — {entry.get('topic', '')}  [{dim_str}]")
    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(out_dir: str, results: list[dict]) -> None:
    if not results:
        return

    totals: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    for r in results:
        for d in DIMENSIONS:
            score = r.get("scores", {}).get(d, {}).get("score")
            if isinstance(score, (int, float)):
                totals[d].append(score)

    averages = {d: (sum(v) / len(v) if v else 0.0) for d, v in totals.items()}
    overall  = sum(averages.values()) / len(averages) if averages else 0.0

    summary = {
        "n_judged":  len(results),
        "averages":  {d: round(avg, 2) for d, avg in averages.items()},
        "overall":   round(overall, 2),
    }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n── Summary ──────────────────────────────")
    for d, avg in averages.items():
        bar = "█" * int(round(avg)) + "░" * (5 - int(round(avg)))
        print(f"  {d:<20s} {bar}  {avg:.2f}")
    print(f"  {'Overall':<20s}            {overall:.2f}")
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def list_candidates() -> None:
    verify_dir = os.path.dirname(__file__)
    candidates = [
        d for d in os.listdir(verify_dir)
        if os.path.isdir(os.path.join(verify_dir, d)) and d != "judge"
        and not d.startswith(".")
    ]
    print("Available candidate model directories:")
    for c in sorted(candidates):
        count = len([
            f for f in os.listdir(os.path.join(verify_dir, c))
            if f.endswith(".txt")
        ])
        print(f"  {c}  ({count} answer files)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: score existing model answers against gold solutions."
    )
    parser.add_argument(
        "--candidate", metavar="MODEL_DIR",
        help="Model whose answers to judge (name of subfolder in verify/), e.g. gemma4:31b-cloud",
    )
    parser.add_argument(
        "--list-candidates", action="store_true",
        help="List available candidate model directories and exit",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids",  nargs="+", metavar="ID", help="Entry IDs to judge")
    group.add_argument("--last", type=int,  metavar="N",  help="Judge latest N entries")
    parser.add_argument(
        "--model", default="gemini-3-flash-preview:cloud",
        help="Ollama Cloud judge model (default: gemini-3-flash-preview:cloud)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama instead of Ollama Cloud",
    )
    args = parser.parse_args()

    if args.list_candidates:
        list_candidates()
        return

    if not args.candidate:
        parser.error("--candidate is required (use --list-candidates to see options)")

    verify_dir    = os.path.dirname(__file__)
    candidate_dir = os.path.join(verify_dir, args.candidate)

    if not os.path.isdir(candidate_dir):
        print(f"Candidate directory not found: {candidate_dir}", file=sys.stderr)
        print("Use --list-candidates to see available options.", file=sys.stderr)
        sys.exit(1)

    client      = make_client(local=args.local)
    all_entries = load_entries()

    if args.ids:
        id_set  = set(args.ids)
        entries = [e for e in all_entries if str(e["id"]) in id_set]
        if not entries:
            print(f"No entries found for ids: {args.ids}", file=sys.stderr)
            sys.exit(1)
    elif args.last:
        entries = all_entries[-args.last:]
    else:
        # Only judge entries that have a candidate answer
        entries = [
            e for e in all_entries
            if os.path.exists(os.path.join(candidate_dir, f"{e['id']}.txt"))
        ]

    out_dir = os.path.join(verify_dir, "judge", args.candidate)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Judge model : {args.model}")
    print(f"Candidate   : {args.candidate}  ({len(entries)} answer files)")
    print(f"Output      : {out_dir}")
    print(f"Running {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}…\n")

    collected = []
    for i, entry in enumerate(entries):
        try:
            result = judge(client, entry, args.model, candidate_dir, out_dir)
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
