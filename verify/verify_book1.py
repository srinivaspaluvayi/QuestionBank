"""
Send each question from book1.json to an Ollama Cloud model and save the answer.

book1.json structure per entry:
  id            – string
  question      – main question text
  sub_questions – list of {id, sub_question}
  images        – list of {media_type, base64, sub_figure_label}
  answer_key    – NOT used (answers are excluded from the prompt)

Requires OLLAMA_API_KEY in verify/.env

Answers are written to:  verify/<model>/book1/<id>.txt

Usage:
    python verify_book1.py                          # run all entries
    python verify_book1.py --ids 1 3 5              # specific entry ids
    python verify_book1.py --last 10                # latest N entries
    python verify_book1.py --model gemma4:31b-cloud
    python verify_book1.py --json /path/to/book1.json
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from ollama import Client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

OLLAMA_CLOUD_HOST  = "https://ollama.com"
POST_CALL_DELAY_SEC = 10
REQUEST_TIMEOUT_SEC = 120
DEFAULT_JSON        = os.path.join(
    os.path.dirname(__file__), "..", "..", "Downloads", "book1.json"
)
SYSTEM_PROMPT = (
    "You are an expert computer science and networking tutor. "
    "Answer each sub-question step by step. "
    "Label your answers clearly (e.g. (a), (b), (c)). "
    "Show your reasoning before giving the final answer."
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
    """Combine main question + sub-questions into a single prompt string."""
    lines: list[str] = [entry["question"].strip()]

    sub_qs = entry.get("sub_questions", [])
    if sub_qs:
        lines.append("")
        lines.append("Sub-questions:")
        for sq in sub_qs:
            label = sq.get("id", "")
            text  = sq.get("sub_question", "").strip()
            lines.append(f"  ({label}) {text}")

    return "\n".join(lines)


def ask(client: Client, entry: dict, model: str, out_dir: str) -> bool:
    prompt = build_prompt(entry)
    if not prompt.strip():
        print(f"  [SKIP] #{entry['id']} — empty question")
        return False

    # Images are already base64-encoded in the JSON; pass them directly to Ollama
    images = [img["base64"] for img in entry.get("images", []) if img.get("base64")]

    user_message: dict = {"role": "user", "content": prompt}
    if images:
        user_message["images"] = images

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    out_path = os.path.join(out_dir, f"{entry['id']}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"  [OK]   #{entry['id']} → {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Ollama Cloud questions from book1.json and save answers."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids",  nargs="+", metavar="ID", help="Entry IDs to process")
    group.add_argument("--last", type=int,  metavar="N",  help="Process latest N entries")
    parser.add_argument(
        "--model",
        default="gemma4:31b-cloud",
        help="Ollama Cloud model (default: gemma4:31b-cloud)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=DEFAULT_JSON,
        help="Path to book1.json (default: ~/Downloads/book1.json)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.json_path):
        print(f"JSON file not found: {args.json_path}", file=sys.stderr)
        print("Pass the path with --json /path/to/book1.json", file=sys.stderr)
        sys.exit(1)

    client      = make_client()
    all_entries = load_entries(args.json_path)

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

    out_dir = os.path.join(os.path.dirname(__file__), args.model, "book1")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Host   : {OLLAMA_CLOUD_HOST}")
    print(f"Model  : {args.model}")
    print(f"Source : {args.json_path}")
    print(f"Output : {out_dir}")
    print(f"Running {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}…\n")

    for i, entry in enumerate(entries):
        try:
            ask(client, entry, args.model, out_dir)
        except Exception as exc:
            print(f"  [ERR]  #{entry['id']} — {exc}")
        if i < len(entries) - 1:
            print(f"  Waiting {POST_CALL_DELAY_SEC}s…")
            time.sleep(POST_CALL_DELAY_SEC)

    print("\nDone.")


if __name__ == "__main__":
    main()
