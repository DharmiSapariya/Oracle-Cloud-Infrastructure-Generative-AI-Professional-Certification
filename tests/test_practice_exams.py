import json
import random
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[1] / "practice-exams"
spec = importlib.util.spec_from_file_location("quiz", ROOT / "quiz.py")
quiz = importlib.util.module_from_spec(spec); spec.loader.exec_module(quiz)


def test_exam_files_are_complete():
    for name, expected in (("foundations", 175), ("genai", 140)):
        qs = quiz.load(name)
        assert len(qs) == expected and [q["id"] for q in qs] == list(range(1, expected + 1))
        for q in qs:
            assert set(q["options"]) == set("ABCD") and q["answer"] in "ABCD" and q["question"].strip()


def test_shuffle_preserves_the_correct_answer():
    q = quiz.load("genai")[0]
    right_text = q["options"][q["answer"]]
    for seed in range(20):
        opts, correct = quiz.shuffled_options(q, random.Random(seed))
        assert dict(opts)[correct] == right_text and sorted(t for _, t in opts) == sorted(q["options"].values())


def test_section_filter_and_scoring_flow():
    qs = quiz.pick(quiz.load("genai"), 3, "RAG", 1)
    assert qs and all("rag" in q["section"].lower() for q in qs)
    answers = iter("A" * 10)
    pct = quiz.run(qs, 1, False, input_fn=lambda _: next(answers), print_fn=lambda *a: None)
    assert 0 <= pct <= 100
