"""Module 1 / Lesson 3 + Module 2 / Prompt-engineering lesson - prompting patterns as code.

Builders for every strategy covered in the course. By default the script just *prints* the prompts so
you can study them offline; add --live to send them to OCI Generative AI (needs .env, see README).

    python .../prompting_patterns.py            # print the prompts
    python .../prompting_patterns.py --live     # also call the chat model

Patterns:
  zero-shot            task description only
  k-shot (few-shot)    k demonstrations in the prompt (in-context learning: model weights do NOT change)
  chain-of-thought     demonstrations that include the reasoning steps
  zero-shot CoT        just append "Let's think step by step"
  least-to-most        solve simple sub-problems first, feed the answers into harder ones
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Example = Tuple[str, str]


def zero_shot(task: str, item: str) -> str:
    return f"{task}\n\nInput: {item}\nOutput:"


def few_shot(task: str, examples: Sequence[Example], item: str) -> str:
    """k-shot prompt where k = len(examples). k=0 degenerates to zero-shot."""
    demos = "\n\n".join(f"Input: {i}\nOutput: {o}" for i, o in examples)
    return f"{task}\n\n{demos}\n\nInput: {item}\nOutput:" if examples else zero_shot(task, item)


def chain_of_thought(demos: Iterable[Tuple[str, str, str]], question: str) -> str:
    """Each demo is (question, reasoning, answer) - the reasoning is what teaches the model to decompose."""
    blocks = [f"Q: {q}\nA: {why} The answer is {a}." for q, why, a in demos]
    return "\n\n".join(blocks) + f"\n\nQ: {question}\nA:"


def zero_shot_cot(question: str) -> str:
    return f"Q: {question}\nA: Let's think step by step."


def least_to_most_prompts(words: Sequence[str]) -> list[str]:
    """Prompts for the course's 'concatenate the last letter of each word' task, built incrementally."""
    prompts = []
    for n in range(2, len(words) + 1):
        prefix = " ".join(words[:n])
        if n == 2:
            prompts.append(f'Q: Take the last letters of the words in "{prefix}" and concatenate them.\nA:')
        else:
            prev = " ".join(words[: n - 1])
            prompts.append(
                f'Q: "{prev}" gives <previous answer>. Now take the last letters of the words in '
                f'"{prefix}" and concatenate them, reusing the previous answer.\nA:'
            )
    return prompts


def least_to_most_reference(words: Sequence[str]) -> list[str]:
    """Ground-truth for the same task, solved incrementally exactly the way the prompts ask the model to."""
    answers, acc = [], ""
    for i, w in enumerate(words):
        acc += w[-1]
        if i >= 1:
            answers.append(acc)
    return answers


# ----------------------------------------------------------------------------- course examples
SENTIMENT_TASK = "Classify the sentiment of each message as positive, negative or neutral."
SENTIMENT_EXAMPLES = [
    ("I can't stand waiting in long lines at the grocery store.", "negative"),
    ("The new update made the app so much faster!", "positive"),
    ("The meeting is at 3 pm on Thursday.", "neutral"),
]
TRANSLATE_EXAMPLES = [("sea otter", "loutre de mer"), ("peppermint", "menthe poivree"), ("plush giraffe", "girafe peluche")]
TENNIS_DEMO = [
    (
        "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many balls does he have now?",
        "Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11.",
        "11",
    )
]


def all_demo_prompts() -> dict[str, str]:
    return {
        "zero-shot sentiment": zero_shot(SENTIMENT_TASK, "Learning a new language has been challenging but rewarding."),
        "3-shot sentiment": few_shot(
            SENTIMENT_TASK, SENTIMENT_EXAMPLES, "Learning a new language has been challenging but rewarding."
        ),
        "3-shot translation (GPT-3 paper style)": few_shot("Translate English to French.", TRANSLATE_EXAMPLES, "cheese"),
        "chain-of-thought": chain_of_thought(
            TENNIS_DEMO, "The cafeteria had 23 apples. They used 20 and bought 6 more. How many apples are there?"
        ),
        "zero-shot chain-of-thought": zero_shot_cot(
            "The cafeteria had 23 apples. They used 20 and bought 6 more. How many apples are there?"
        ),
        "least-to-most (step 1)": least_to_most_prompts(["think", "machine", "learning", "reasoning"])[0],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="send the prompts to OCI Generative AI")
    args = ap.parse_args()

    prompts = all_demo_prompts()
    client = settings = None
    if args.live:
        from common.genai_common import get_inference_client, load_settings, require

        settings = load_settings()
        require(settings, "compartment_id")
        client = get_inference_client(settings)

    for name, p in prompts.items():
        print(f"\n=== {name} ===\n{p}")
        if client:
            from common.genai_common import chat

            print("--- model ---\n" + chat(client, settings, p, max_tokens=200, temperature=0))

    print("\nleast-to-most reference answers:", least_to_most_reference(["think", "machine", "learning", "reasoning"]))
