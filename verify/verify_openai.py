"""
Send each question to an OpenAI model and save the answer.

Answers are written to:  verify/<model>/<id>.txt

Usage:
    python verify_openai.py                        # run all entries
    python verify_openai.py --ids 1 5 12           # specific entries by id
    python verify_openai.py --last 10              # latest N entries
    python verify_openai.py --model gpt-4o         # choose model (default: gpt-4o)

Set your API key before running:
    export OPENAI_API_KEY=sk-...
"""

import argparse
import base64
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
JSON_FILE = os.path.join(DATA_DIR, "questions.json")


def read_file(rel_path: str) -> str:
    full = os.path.join(DATA_DIR, rel_path.lstrip("./"))
    if not os.path.exists(full):
        return ""
    with open(full, encoding="utf-8") as f:
        return f.read().strip()


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def load_entries() -> list[dict]:
    with open(JSON_FILE, encoding="utf-8") as f:
        return json.load(f)


def ask(client: OpenAI, entry: dict, model: str, out_dir: str) -> None:
    question_text = read_file(entry["question"])
    if not question_text:
        print(f"  [SKIP] #{entry['id']} — question file missing")
        return

    content: list = [{"type": "text", "text": question_text}]

    for rel in entry.get("images", []):
        full = os.path.join(DATA_DIR, rel.lstrip("./"))
        if os.path.exists(full):
            b64 = encode_image(full)
            media = image_media_type(full)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{b64}"},
            })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()

    out_path = os.path.join(out_dir, f"{entry['id']}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer)

    print(f"  [OK]   #{entry['id']} — {entry.get('topic', '')} → {out_path}")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Ask OpenAI and save answers.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ids",  nargs="+", metavar="ID", help="Entry IDs to process")
    group.add_argument("--last", type=int,  metavar="N",  help="Process latest N entries")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    args = parser.parse_args()

    client = OpenAI(api_key=api_key)
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

    print(f"Model  : {args.model}")
    print(f"Output : {out_dir}")
    print(f"Running {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}…\n")

    for entry in entries:
        try:
            ask(client, entry, args.model, out_dir)
        except Exception as exc:
            print(f"  [ERR]  #{entry['id']} — {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
