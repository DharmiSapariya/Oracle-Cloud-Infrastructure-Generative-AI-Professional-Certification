"""Module 1 / Lesson 6 - Decoding, made tangible (no cloud account needed).

An LLM outputs a probability distribution over its vocabulary at every step. *Decoding* turns that
distribution into text, one token at a time. This file implements, on a toy distribution:

* greedy decoding            - always take the most probable token (deterministic)
* temperature                - reshapes the distribution (never changes the ranking)
* top-k / top-p (nucleus)    - restrict which tokens may be sampled
* sampling                   - draw randomly from the (reshaped / truncated) distribution
* beam search                - keep several candidate sequences, prune low-probability ones
* frequency vs presence penalty (Module 2 / chat parameters)

Run:  python part2-genai-professional/m1_llm_fundamentals/decoding_playground.py
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

Distribution = Dict[str, float]

# The course example: "I wrote to the zoo to send me a pet. They sent me a ___"
ZOO_NEXT_WORD: Distribution = {
    "dog": 0.45, "lion": 0.03, "elephant": 0.02, "panda": 0.10, "cat": 0.20, "small": 0.12, "<eos>": 0.08,
}


def normalize(dist: Distribution) -> Distribution:
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


# ----------------------------------------------------------------------------- reshaping
def apply_temperature(dist: Distribution, temperature: float) -> Distribution:
    """p_i^(1/T) renormalised (equivalent to softmax(logits / T)).

    T < 1  -> distribution gets peaked (closer to greedy, more predictable)
    T > 1  -> distribution flattens (rarer words become more likely, more creative)
    T -> 0 -> collapses to greedy.  The ranking of words NEVER changes.
    """
    if temperature <= 0:
        best = max(dist, key=dist.get)
        return {k: (1.0 if k == best else 0.0) for k in dist}
    scaled = {k: math.exp(math.log(max(v, 1e-12)) / temperature) for k, v in dist.items()}
    return normalize(scaled)


def top_k_filter(dist: Distribution, k: int) -> Distribution:
    """Keep only the k most probable tokens."""
    keep = sorted(dist, key=dist.get, reverse=True)[:k]
    return normalize({t: dist[t] for t in keep})


def top_p_filter(dist: Distribution, p: float) -> Distribution:
    """Nucleus sampling: smallest set of top tokens whose cumulative probability reaches p."""
    kept, cum = {}, 0.0
    for tok in sorted(dist, key=dist.get, reverse=True):
        kept[tok] = dist[tok]
        cum += dist[tok]
        if cum >= p:
            break
    return normalize(kept)


# ----------------------------------------------------------------------------- selection
def greedy(dist: Distribution) -> str:
    return max(dist, key=dist.get)


def sample(dist: Distribution, rng: random.Random) -> str:
    r, cum = rng.random(), 0.0
    for tok, p in dist.items():
        cum += p
        if r <= cum:
            return tok
    return tok  # numerical safety


def decode_step(
    dist: Distribution, *, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
    rng: random.Random | None = None,
) -> str:
    """One generation step with the usual knobs. temperature == 0 means greedy."""
    if temperature == 0:
        return greedy(dist)
    d = apply_temperature(dist, temperature)
    if top_k:
        d = top_k_filter(d, top_k)
    if top_p:
        d = top_p_filter(d, top_p)
    return sample(d, rng or random.Random())


# ----------------------------------------------------------------------------- penalties
def apply_penalties(
    dist: Distribution, history: Sequence[str], frequency_penalty: float = 0.0, presence_penalty: float = 0.0
) -> Distribution:
    """Frequency penalty scales with how many times a token already appeared;
    presence penalty is a flat penalty once it has appeared at least once."""
    out = {}
    for tok, p in dist.items():
        n = history.count(tok)
        adjusted = p - frequency_penalty * n - (presence_penalty if n > 0 else 0.0)
        out[tok] = max(adjusted, 1e-9)
    return normalize(out)


# ----------------------------------------------------------------------------- full sequences
NextTokenFn = Callable[[Tuple[str, ...]], Distribution]


def toy_language_model(context: Tuple[str, ...]) -> Distribution:
    """A tiny hand-written 'LLM' over the zoo example so we can decode whole sentences."""
    last = context[-1] if context else "<start>"
    table: Dict[str, Distribution] = {
        "<start>": {"a": 1.0},
        "a": {"dog": 0.45, "cat": 0.2, "small": 0.25, "lion": 0.1},
        "small": {"red": 0.5, "dog": 0.3, "cat": 0.2},
        "red": {"panda": 0.8, "dog": 0.2},
        "dog": {"<eos>": 0.9, "named": 0.1},
        "cat": {"<eos>": 0.85, "named": 0.15},
        "lion": {"<eos>": 1.0},
        "panda": {"<eos>": 1.0},
        "named": {"Rex": 0.6, "Tom": 0.4},
        "Rex": {"<eos>": 1.0},
        "Tom": {"<eos>": 1.0},
    }
    return normalize(table[last])


def generate(
    next_token: NextTokenFn = toy_language_model, *, max_tokens: int = 10, temperature: float = 0.0,
    top_k: int | None = None, top_p: float | None = None, seed: int | None = None,
) -> List[str]:
    """Iterative decoding: distribution -> select -> append -> feed back, until <eos>."""
    rng = random.Random(seed)
    out: List[str] = []
    for _ in range(max_tokens):
        tok = decode_step(next_token(tuple(out)), temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
        if tok == "<eos>":
            break
        out.append(tok)
    return out


def beam_search(next_token: NextTokenFn = toy_language_model, *, beam_width: int = 2, max_tokens: int = 10):
    """Keep `beam_width` best partial sequences by total log-probability, pruning the rest.

    Not greedy: it can find sequences with higher *joint* probability than greedy decoding.
    """
    beams: List[Tuple[Tuple[str, ...], float, bool]] = [((), 0.0, False)]
    for _ in range(max_tokens):
        candidates = []
        for seq, score, done in beams:
            if done:
                candidates.append((seq, score, True))
                continue
            for tok, p in next_token(seq).items():
                if tok == "<eos>":
                    candidates.append((seq, score + math.log(p), True))
                else:
                    candidates.append((seq + (tok,), score + math.log(p), False))
        beams = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]
        if all(done for _, _, done in beams):
            break
    return [(list(seq), math.exp(score)) for seq, score, _ in beams]


# ----------------------------------------------------------------------------- demo
def _fmt(d: Distribution) -> str:
    return ", ".join(f"{k}:{v:.2f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    print("Base distribution :", _fmt(ZOO_NEXT_WORD))
    for t in (0.2, 1.0, 2.0):
        print(f"temperature={t:<3}   :", _fmt(apply_temperature(ZOO_NEXT_WORD, t)))
    print("top_k=3           :", _fmt(top_k_filter(ZOO_NEXT_WORD, 3)))
    print("top_p=0.75        :", _fmt(top_p_filter(ZOO_NEXT_WORD, 0.75)))
    print("freq penalty 0.1  :", _fmt(apply_penalties(ZOO_NEXT_WORD, ["dog", "dog", "dog"], frequency_penalty=0.1)))
    print("presence pen 0.1  :", _fmt(apply_penalties(ZOO_NEXT_WORD, ["dog", "dog", "dog"], presence_penalty=0.1)))
    print()
    print("greedy sentence    :", "They sent me " + " ".join(generate(temperature=0)))
    for seed in range(3):
        print(f"sampled (T=1.2) #{seed}:", "They sent me " + " ".join(generate(temperature=1.2, seed=seed)))
    print("beam search (w=3)  :", [(" ".join(s), round(p, 3)) for s, p in beam_search(beam_width=3)])
