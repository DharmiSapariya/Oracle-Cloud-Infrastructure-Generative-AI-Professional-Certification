"""The whole RAG pipeline with ZERO cloud dependencies - great for understanding and for CI.

  Ingestion : load -> chunk -> embed -> index
  Retrieval : embed query -> top-K by cosine similarity
  Generation: put retrieved chunks + question in a prompt -> "LLM"

Embeddings here are TF-IDF vectors (lexical, not semantic - 'canine' will NOT match 'dog') and the "LLM" is an extractive stub that
returns the best-matching sentence. Swap `embed_fn` / `generate_fn` for OCI Generative AI and you have
the real thing - the structure is identical to scripts 02 and 03.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_common import PROMPT_TEMPLATE, read_text, split_pages  # noqa: E402

DEFAULT_FILE = Path(__file__).parent / "data" / "oci_ai_foundations_overview.txt"


@dataclass
class LocalRAG:
    chunks: List[str]
    embed_fn: Callable[[List[str]], np.ndarray]
    generate_fn: Callable[[str, List[str]], str]

    def __post_init__(self):
        v = self.embed_fn(self.chunks)
        self.matrix = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def retrieve(self, question: str, k: int = 3) -> List[str]:
        q = self.embed_fn([question])[0]
        q = q / max(np.linalg.norm(q), 1e-12)
        scores = self.matrix @ q
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]

    def prompt(self, question: str, k: int = 3) -> str:
        return PROMPT_TEMPLATE.format(context="\n\n".join(self.retrieve(question, k)), question=question)

    def ask(self, question: str, k: int = 3) -> str:
        return self.generate_fn(question, self.retrieve(question, k))


def build_tfidf_rag(path: Path = DEFAULT_FILE, chunk_size: int = 300, chunk_overlap: int = 40) -> LocalRAG:
    from sklearn.feature_extraction.text import TfidfVectorizer

    chunks = [c for _, c in split_pages(read_text(path), chunk_size, chunk_overlap)]
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b", sublinear_tf=True).fit(chunks)  # "ingestion": index built once

    def embed_fn(texts: List[str]) -> np.ndarray:
        return vec.transform(texts).toarray()

    def generate_fn(question: str, context: List[str]) -> str:
        """Extractive stub: return the retrieved sentence most similar (TF-IDF cosine) to the question."""
        sentences = [s.strip() for c in context for s in re.split(r"(?<=[.!?])\s+", c) if s.strip()]
        m = embed_fn(sentences)
        q = embed_fn([question])[0]
        sims = (m @ q) / np.maximum(np.linalg.norm(m, axis=1) * max(np.linalg.norm(q), 1e-12), 1e-12)
        return f"According to the provided context: {sentences[int(np.argmax(sims))]}"

    return LocalRAG(chunks, embed_fn, generate_fn)


if __name__ == "__main__":
    rag = build_tfidf_rag()
    print(f"{len(rag.chunks)} chunks indexed\n")
    for q in ["Tell us about Module 4 of the AI Foundations course.", "Why does RAG reduce hallucination?", "What is T-Few?"]:
        print("Q:", q)
        print("A:", rag.ask(q), "\n")
