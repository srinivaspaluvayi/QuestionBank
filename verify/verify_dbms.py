"""
Send each question from dbms_dataset_mixed.json to an Ollama Cloud model and save the answer.

JSON structure per entry:
  id            – string (e.g. "dbms_001")
  image_id      – string
  image         – relative path to image (e.g. "images/image_001.png")
  question_type – Multiple Choice | Short Answer | Explanation | Analysis |
                  SQL Interpretation | Normalization Reasoning |
                  Concurrency Reasoning | ER Mapping
  question      – question text
  choices       – dict {A: ..., B: ..., C: ..., D: ...}  (MC only, else absent)
  answer        – NOT used

Requires OLLAMA_API_KEY in verify/.env

Answers are written to:  verify/<model>/dbms/<id>.txt

Usage:
    python verify_dbms.py                          # run all entries
    python verify_dbms.py --ids dbms_001 dbms_005  # specific ids
    python verify_dbms.py --last 10                # latest N entries
    python verify_dbms.py --type "Multiple Choice" # filter by question type
    python verify_dbms.py --model gemma4:31b-cloud
    python verify_dbms.py --json /path/to/dbms_dataset_mixed.json
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from ollama import Client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

OLLAMA_CLOUD_HOST   = "https://ollama.com"
POST_CALL_DELAY_SEC = 5
REQUEST_TIMEOUT_SEC = 600
DEFAULT_JSON        = os.path.join(os.path.dirname(__file__), "..", "dbms_dataset_mixed.json")
SYSTEM_PROMPT = (
    "You are an expert database systems tutor. "
    "Analyze any provided diagram or image carefully before answering. "
    "Answer step by step, showing your reasoning clearly. "
    "For multiple-choice questions, state the correct option letter and explain why."
)


def make_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print(
            "OLLAMA_API_KEY is not set. Add it to verify/.env:\n"
            "  OLLAMA_API_KEY=your_api_key\n"
            "Create a key at https://ollama.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return Client(host=OLLAMA_CLOUD_HOST, timeout=REQUEST_TIMEOUT_SEC)


def load_entries(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(entry: dict) -> str:
    """Build a prompt from question text, type label, and choices (if any)."""
    q_type = entry.get("question_type", "")
    lines  = []

    if q_type:
        lines.append(f"[Question Type: {q_type}]")
        lines.append("")

    lines.append(entry["question"].strip())

    choices: dict = entry.get("choices") or {}
    if choices:
        lines.append("")
        lines.append("Choices:")
        for letter, text in sorted(choices.items()):
            lines.append(f"  {letter}. {text}")

    return "\n".join(lines)


def resolve_image(entry: dict, json_dir: str) -> str | None:
    """Return the absolute image path, or None if the file is missing."""
    rel = entry.get("image", "")
    if not rel:
        return None
    # Images are relative to the JSON file's directory
    full = os.path.normpath(os.path.join(json_dir, rel))
    if not os.path.exists(full):
        return None
    return full


def ask(client: Client, entry: dict, model: str, out_dir: str, json_dir: str) -> bool:
    out_path = os.path.join(out_dir, f"{entry['id']}.txt")
    if os.path.exists(out_path):
        print(f"  [SKIP] #{entry['id']} — already answered")
        return False

    prompt = build_prompt(entry)

    user_message: dict = {"role": "user", "content": prompt}

    img_path = resolve_image(entry, json_dir)
    if img_path:
        user_message["images"] = [img_path]
    else:
        img_rel = entry.get("image", "")
        if img_rel:
            print(f"  [WARN] #{entry['id']} — image not found: {img_rel}")

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)

    q_type = entry.get("question_type", "")
    print(f"  [OK]   #{entry['id']} [{q_type}] → {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Ollama Cloud questions from dbms_dataset_mixed.json."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids",  nargs="+", metavar="ID", help="Entry IDs to process")
    group.add_argument("--last", type=int,  metavar="N",  help="Process latest N entries")
    parser.add_argument(
        "--type",
        dest="q_type",
        metavar="TYPE",
        help='Filter by question type, e.g. "Multiple Choice"',
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b-cloud",
        help="Ollama Cloud model (default: gemma4:31b-cloud)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=DEFAULT_JSON,
        help="Path to dbms_dataset_mixed.json",
    )
    args = parser.parse_args()

    json_path = os.path.abspath(args.json_path)
    if not os.path.isfile(json_path):
        print(f"JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    json_dir    = os.path.dirname(json_path)
    client      = make_client()
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
        entries = all_entries

    if args.q_type:
        entries = [e for e in entries if e.get("question_type", "").lower() == args.q_type.lower()]
        if not entries:
            print(f"No entries with question_type '{args.q_type}'", file=sys.stderr)
            sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), args.model, "dbms")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Host   : {OLLAMA_CLOUD_HOST}")
    print(f"Model  : {args.model}")
    print(f"Source : {json_path}")
    print(f"Output : {out_dir}")
    print(f"Running {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}…\n")

    for i, entry in enumerate(entries):
        try:
            ask(client, entry, args.model, out_dir, json_dir)
        except Exception as exc:
            print(f"  [ERR]  #{entry['id']} — {exc}")
        if i < len(entries) - 1:
            print(f"  Waiting {POST_CALL_DELAY_SEC}s…")
            time.sleep(POST_CALL_DELAY_SEC)

    print("\nDone.")


if __name__ == "__main__":
    main()
