"""Course demo: "Embedding Models in Action".

Embed a file of sentences about country capitals plus one unrelated question, then check that
semantically similar sentences are numerically close:
  * capital questions cluster together, "smallest state in the US" is an outlier
  * "smallest state in India" lands next to "smallest state in the US"
  * "largest state in the US" sits between the two groups

    python .../04_embeddings_and_similarity.py            # real OCI embeddings (needs .env)
    python .../04_embeddings_and_similarity.py --offline  # TF-IDF vectors: no cloud, but LEXICAL not semantic

Outputs a 2-D PCA scatter (real embeddings have 1,024+ dims; 2-D loses information but builds intuition)
to ../outputs/m2_embeddings_scatter.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SENTENCES = [
    "What is the capital of France?",
    "What is the capital of Sweden?",
    "What is the capital of Canada?",
    "What is the capital of the United Kingdom?",
    "Which city is the capital of Japan?",
    "What is the smallest state in the United States?",
    "What is the smallest state in India?",
    "What is the largest state in the United States?",
]


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return v @ v.T


def nearest_neighbours(sim: np.ndarray, labels: list[str]) -> list[tuple[str, str, float]]:
    out = []
    for i, label in enumerate(labels):
        s = sim[i].copy()
        s[i] = -1
        j = int(s.argmax())
        out.append((label, labels[j], float(s[j])))
    return out


def tfidf_vectors(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer().fit_transform(texts).toarray()


def oci_vectors(texts: list[str]) -> np.ndarray:
    from common.genai_common import embed, get_inference_client, load_settings, require

    s = load_settings()
    require(s, "compartment_id")
    return np.array(embed(get_inference_client(s), s, texts))


def plot_2d(vectors: np.ndarray, labels: list[str], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pts = PCA(n_components=2).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(pts[:, 0], pts[:, 1])
    for (x, y), label in zip(pts, labels):
        ax.annotate(label.replace("What is the ", "")[:38], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    vecs = tfidf_vectors(SENTENCES) if args.offline else oci_vectors(SENTENCES)
    kind = "TF-IDF (lexical)" if args.offline else "OCI embeddings"
    print(f"{kind}: {vecs.shape[0]} sentences x {vecs.shape[1]} dims")
    for a, b, score in nearest_neighbours(cosine_matrix(vecs), SENTENCES):
        print(f"{score:.3f}  {a}\n       -> {b}")
    out = Path(__file__).resolve().parents[1] / "outputs" / "m2_embeddings_scatter.png"
    plot_2d(vecs, SENTENCES, out, f"2-D projection of sentence vectors - {kind}")
    print("saved", out)
