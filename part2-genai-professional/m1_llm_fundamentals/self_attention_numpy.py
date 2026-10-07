"""Module 1 / Lessons 2 + 19-20 - Self-attention from scratch with NumPy.

Transformers look at *all* tokens at once and let every token weigh every other token ("bird's-eye
view"). That is how "it" in  "Jane threw the Frisbee and her dog fetched it"  can be linked to
"Frisbee". This is the scaled dot-product attention at the core of the transformer
("Attention Is All You Need", 2017):

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Run:  python part2-genai-professional/m1_llm_fundamentals/self_attention_numpy.py
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Returns (output, attention_weights). Each row of the weights sums to 1."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # e.g. causal mask for decoder-only models
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def self_attention(X: np.ndarray, d_k: int = 8, seed: int = 0, causal: bool = False):
    """Project embeddings X (n_tokens x d_model) into Q, K, V with random weights and attend."""
    rng = np.random.default_rng(seed)
    d_model = X.shape[1]
    Wq, Wk, Wv = (rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_k)) for _ in range(3))
    mask = np.tril(np.ones((X.shape[0], X.shape[0]), dtype=bool)) if causal else None
    return scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv, mask)


if __name__ == "__main__":
    tokens = "Jane threw the Frisbee and her dog fetched it".split()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(len(tokens), 16))  # stand-in embeddings (a real model learns these)
    out, w = self_attention(X)
    print("tokens          :", tokens)
    print("output shape    :", out.shape, "(one context-aware vector per token)")
    print("weights rows sum:", np.allclose(w.sum(axis=1), 1.0))
    print("\nattention paid BY 'it' TO each token (random embeddings, so illustrative only):")
    for tok, a in zip(tokens, w[tokens.index("it")]):
        print(f"  {tok:<8} {a:.3f}")
    _, wc = self_attention(X, causal=True)
    print("\ncausal (decoder-only) mask -> upper triangle is zero:", np.allclose(np.triu(wc, 1), 0))
