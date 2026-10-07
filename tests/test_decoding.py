import random

import numpy as np
from decoding_playground import (ZOO_NEXT_WORD, apply_penalties, apply_temperature, beam_search, generate, greedy,
                                 top_k_filter, top_p_filter)
from self_attention_numpy import self_attention


def test_distributions_sum_to_one():
    for d in (apply_temperature(ZOO_NEXT_WORD, 0.5), top_k_filter(ZOO_NEXT_WORD, 3), top_p_filter(ZOO_NEXT_WORD, 0.7)):
        assert abs(sum(d.values()) - 1) < 1e-9


def test_temperature_never_changes_ranking():
    base = sorted(ZOO_NEXT_WORD, key=ZOO_NEXT_WORD.get, reverse=True)
    for t in (0.3, 1, 3):
        d = apply_temperature(ZOO_NEXT_WORD, t)
        assert sorted(d, key=d.get, reverse=True) == base


def test_low_temperature_peaks_high_temperature_flattens():
    low, high = apply_temperature(ZOO_NEXT_WORD, 0.2), apply_temperature(ZOO_NEXT_WORD, 3)
    assert low["dog"] > ZOO_NEXT_WORD["dog"] > high["dog"]


def test_greedy_and_zero_temperature_are_deterministic():
    assert greedy(ZOO_NEXT_WORD) == "dog"
    assert generate(temperature=0, seed=1) == generate(temperature=0, seed=99) == ["a", "dog"]


def test_top_k_and_top_p_restrict_vocabulary():
    assert set(top_k_filter(ZOO_NEXT_WORD, 2)) == {"dog", "cat"}
    assert "lion" not in top_p_filter(ZOO_NEXT_WORD, 0.75)


def test_sampling_is_reproducible_with_seed():
    assert generate(temperature=1.5, seed=7) == generate(temperature=1.5, seed=7)


def test_beam_search_returns_best_sequences_first():
    beams = beam_search(beam_width=3)
    probs = [p for _, p in beams]
    assert probs == sorted(probs, reverse=True) and len(beams) == 3


def test_penalties_reduce_repeated_token():
    hist = ["dog"] * 3
    assert apply_penalties(ZOO_NEXT_WORD, hist, frequency_penalty=0.1)["dog"] < ZOO_NEXT_WORD["dog"]
    # presence penalty is flat: one appearance and three appearances are penalised identically
    a = apply_penalties(ZOO_NEXT_WORD, ["dog"], presence_penalty=0.1)["dog"]
    b = apply_penalties(ZOO_NEXT_WORD, hist, presence_penalty=0.1)["dog"]
    assert abs(a - b) < 1e-12


def test_self_attention_rows_sum_to_one_and_causal_mask_works():
    X = np.random.default_rng(0).normal(size=(6, 16))
    out, w = self_attention(X)
    assert out.shape == (6, 8) and np.allclose(w.sum(axis=1), 1)
    _, wc = self_attention(X, causal=True)
    assert np.allclose(np.triu(wc, 1), 0)
