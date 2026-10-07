"""Terminal quiz runner for the two practice exams.

    python practice-exams/quiz.py --exam genai --n 20                 # 20 random questions
    python practice-exams/quiz.py --exam foundations --section LLM    # filter by section keyword
    python practice-exams/quiz.py --exam genai --n 10 --seed 1 --review

Options are SHUFFLED every run because the source answer key is heavily skewed to "B".
The key itself is unverified - if you disagree with an answer, check the docs/notes and open an issue.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
EXAMS = {"foundations": "foundations_175.json", "genai": "genai_professional_140.json"}


def load(exam: str) -> list[dict]:
    return json.loads((HERE / EXAMS[exam]).read_text(encoding="utf-8"))


def pick(questions: list[dict], n: int | None, section: str | None, seed: int | None) -> list[dict]:
    pool = [q for q in questions if not section or section.lower() in q["section"].lower()]
    random.Random(seed).shuffle(pool)
    return pool[:n] if n else pool


def shuffled_options(q: dict, rng: random.Random):
    """Return ([(label, text)], correct_label) with options in a new random order."""
    items = list(q["options"].items())
    rng.shuffle(items)
    labels = "ABCD"
    shuffled = [(labels[i], text) for i, (_, text) in enumerate(items)]
    correct = next(labels[i] for i, (orig, _) in enumerate(items) if orig == q["answer"])
    return shuffled, correct


def run(questions: list[dict], seed: int | None, review: bool, input_fn=input, print_fn=print) -> float:
    rng = random.Random(seed)
    wrong, score = [], 0
    for n, q in enumerate(questions, 1):
        opts, correct = shuffled_options(q, rng)
        print_fn(f"\nQ{n}/{len(questions)} [{q['section']}]\n{q['question']}")
        for label, text in opts:
            print_fn(f"  {label}. {text}")
        ans = ""
        while ans not in list("ABCD"):
            ans = input_fn("Your answer (A-D, q to quit): ").strip().upper()
            if ans == "Q":
                return score / max(n - 1, 1)
        if ans == correct:
            score += 1
            print_fn("  correct")
        else:
            print_fn(f"  wrong - answer: {correct}. {dict(opts)[correct]}")
            wrong.append((q, correct, dict(opts)[correct]))
    pct = score / len(questions) * 100
    print_fn(f"\nScore: {score}/{len(questions)} ({pct:.0f}%)")
    if review and wrong:
        print_fn("\nReview:")
        for q, label, text in wrong:
            print_fn(f"- {q['question']}\n    -> {label}. {text}")
    return pct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", choices=EXAMS, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--section", help="keyword to filter sections")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--review", action="store_true", help="list missed questions at the end")
    a = ap.parse_args()
    qs = pick(load(a.exam), a.n, a.section, a.seed)
    if not qs:
        raise SystemExit("no questions match that section filter")
    run(qs, a.seed, a.review)
