"""Course lesson: "Fine-Tuning Configuration" - accuracy vs loss, on toy numbers.

Accuracy = correct tokens / total tokens x 100 (positional token match against the annotated answer).
  truth      : "the cat sat on the mat"
  prediction : "the cat slept on the rug"  -> 4/6 = 67%

Loss measures how *wrong* the predicted probability distribution is (cross-entropy). A near-miss
("slept" for "sat", "rug" for "mat") is penalised far less than total nonsense ("the airplane flew at
midnight") even if token accuracy looks similar - which is why loss is usually the better guide for
generative tasks. The probabilities below are made-up numbers to illustrate the idea, NOT real model output.

Also shows the console tooltip: accuracy 0.9 means 90% of output tokens match the training data.
"""
from __future__ import annotations

import math
from typing import Sequence


def token_accuracy(truth: Sequence[str], prediction: Sequence[str]) -> float:
    n = max(len(truth), len(prediction))
    matches = sum(t == p for t, p in zip(truth, prediction))
    return matches / n


def cross_entropy(prob_of_true_token: Sequence[float]) -> float:
    """Mean negative log-probability the model assigned to each TRUE token. 0 == perfect."""
    return -sum(math.log(max(p, 1e-12)) for p in prob_of_true_token) / len(prob_of_true_token) + 0.0


if __name__ == "__main__":
    truth = "the cat sat on the mat".split()
    near_miss = "the cat slept on the rug".split()
    nonsense = "the airplane flew at midnight".split()

    print("accuracy near-miss :", f"{token_accuracy(truth, near_miss):.0%}")
    print("accuracy nonsense  :", f"{token_accuracy(truth, nonsense):.0%}")

    # Probability each model gave to the TRUE token at each position (illustrative):
    near_miss_probs = [0.95, 0.90, 0.30, 0.92, 0.94, 0.25]  # 'sat' and 'mat' were plausible alternatives
    nonsense_probs = [0.60, 0.02, 0.01, 0.02, 0.01, 0.01]  # true tokens were considered very unlikely
    print("loss near-miss     :", round(cross_entropy(near_miss_probs), 3), "(low: mistakes are semantically close)")
    print("loss nonsense      :", round(cross_entropy(nonsense_probs), 3), "(high: semantically unrelated)")
    print("loss perfect       :", round(cross_entropy([1.0] * 6), 3))
