"""
Send each question to an Ollama Cloud model and save the answer.

Requires OLLAMA_API_KEY in the environment or in verify/.env
(create a key at https://ollama.com/settings/keys).

Answers are written to:  verify/<model>/<id>.txt

Usage:
    # verify/.env:  OLLAMA_API_KEY=your_api_key
    python verify.py                     # run all entries
    python verify.py --ids 1 5 12        # specific entries by id
    python verify.py --last 10           # latest N entries
    python verify.py --model gpt-oss:120b
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from ollama import Client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
JSON_FILE = os.path.join(DATA_DIR, "questions.json")
OLLAMA_CLOUD_HOST = "https://ollama.com"
POST_CALL_DELAY_SEC = 10   # pause between calls to avoid rate-limit hangs
REQUEST_TIMEOUT_SEC = 120  # per-request timeout; raises error instead of hanging
SYSTEM_PROMPT = "You are expert computer science tutor. Answer the the given question step by step. Begin by explaining your reasoning process clearly. Think step by step before answering the question."


def make_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print(
            "OLLAMA_API_KEY is not set. Add it to verify/.env or export it:\n"
            "  OLLAMA_API_KEY=your_api_key\n"
            "Create a key at https://ollama.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    return Client(host=OLLAMA_CLOUD_HOST, timeout=REQUEST_TIMEOUT_SEC)


def read_file(rel_path: str) -> str:
    full = os.path.join(DATA_DIR, rel_path.lstrip("./"))
    if not os.path.exists(full):
        return ""
    with open(full, encoding="utf-8") as f:
        return f.read().strip()


def load_entries() -> list[dict]:
    with open(JSON_FILE, encoding="utf-8") as f:
        return json.load(f)



def ask(client: Client, entry: dict, model: str, out_dir: str) -> bool:
    question_text = read_file(entry["question"])
    if not question_text:
        print(f"  [SKIP] #{entry['id']} — question file missing")
        return False

    image_paths = []
    for rel in entry.get("images", []):
        full = os.path.join(DATA_DIR, rel.lstrip("./"))
        if os.path.exists(full):
            image_paths.append(full)

    user_message = {"role": "user", "content": question_text}
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    if image_paths:
        user_message["images"] = image_paths

    response = client.chat(
        model=model,
        messages=[
            system_message,
            user_message,
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    out_path = os.path.join(out_dir, f"{entry['id']}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"  [OK]   #{entry['id']} — {entry.get('topic', '')} → {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Ollama Cloud and save answers.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+", metavar="ID", help="Entry IDs to process")
    group.add_argument("--last", type=int, metavar="N", help="Process latest N entries")
    parser.add_argument(
        "--model",
        default="gemma4:31b-cloud",
        help="Ollama Cloud model name (default: gemma4:31b-cloud)",
    )
    args = parser.parse_args()

    client = make_client()

    all_entries = load_entries()

    if args.ids:
        id_set = set(args.ids)
        entries = [e for e in all_entries if e["id"] in id_set]
        if not entries:
            print(f"No entries found for ids: {args.ids}", file=sys.stderr)
            sys.exit(1)
    elif args.last:
        entries = all_entries[-args.last:]
    else:
        entries = all_entries

    out_dir = os.path.join(os.path.dirname(__file__), args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Host   : {OLLAMA_CLOUD_HOST}")
    print(f"Model  : {args.model}")
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
