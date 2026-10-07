"""Course demo: "Creating a Custom Model" - preparing the fine-tuning dataset.

Use case: take a human request ("ask my aunt if she can go to the JDRF walk with me on October 6")
and rephrase it into the utterance a virtual assistant should say ("Can you go to the JDRF walk with me
on October 6?").  The course dataset came from the paper "Sound Control: Natural Rephrasing in Dialog
Systems"; only two columns were kept: the human request and the virtual-assistant utterance.

OCI Generative AI requires **JSONL**:
  * one JSON object per line (each line is its own document, not one big JSON array)
  * each object has the properties  "prompt"  and  "completion"
  * UTF-8 encoded
Breaking any of these makes custom-model creation fail - so validate before uploading to Object Storage.

    python .../05_prepare_finetune_dataset.py build    # CSV -> train/test JSONL
    python .../05_prepare_finetune_dataset.py validate data/rephrasing_train.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List

DATA = Path(__file__).parent / "data"


def csv_to_records(csv_path: Path, prompt_col: str, completion_col: str) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {"prompt": r[prompt_col].strip(), "completion": r[completion_col].strip()}
            for r in csv.DictReader(f)
            if r[prompt_col].strip() and r[completion_col].strip()
        ]


def write_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # one object per line


def validate_jsonl(path: Path) -> List[str]:
    """Return a list of human-readable problems (empty list == file is valid)."""
    problems: List[str] = []
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return [f"file is not valid UTF-8: {e}"]
    if text.lstrip().startswith("["):
        problems.append("looks like a JSON array - JSONL needs one object per line, not a list")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ["file is empty"]
    for n, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e.msg})")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {n}: expected a JSON object")
            continue
        for key in ("prompt", "completion"):
            if key not in obj:
                problems.append(f"line {n}: missing '{key}'")
            elif not isinstance(obj[key], str) or not obj[key].strip():
                problems.append(f"line {n}: '{key}' must be a non-empty string")
    return problems


def split_records(records: List[dict], test_fraction: float = 0.2, seed: int = 42):
    """Hold out a test set the model never sees - the course evaluated on prompts NOT in training."""
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--csv", type=Path, default=DATA / "rephrasing_sample.csv")
    b.add_argument("--prompt-col", default="human_request")
    b.add_argument("--completion-col", default="assistant_utterance")
    v = sub.add_parser("validate")
    v.add_argument("path", type=Path)
    args = ap.parse_args()

    if args.cmd == "build":
        recs = csv_to_records(args.csv, args.prompt_col, args.completion_col)
        train, test = split_records(recs)
        write_jsonl(train, DATA / "rephrasing_train.jsonl")
        write_jsonl(test, DATA / "rephrasing_test.jsonl")
        print(f"{len(recs)} records -> {len(train)} train / {len(test)} test in {DATA}")
        for name in ("rephrasing_train.jsonl", "rephrasing_test.jsonl"):
            print(name, "->", validate_jsonl(DATA / name) or "valid")
    else:
        problems = validate_jsonl(args.path)
        print("valid" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)
