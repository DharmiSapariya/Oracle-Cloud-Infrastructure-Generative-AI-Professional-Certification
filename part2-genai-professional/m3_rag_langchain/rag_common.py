"""Pieces shared by the RAG scripts: PDF/text loading, chunking, Document wrapping, DB connection."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

PROMPT_TEMPLATE = """Answer the question based only on the following context.
If the context does not contain the answer, say you do not know.

Context:
{context}

Question: {question}
"""


def read_text(path: Path) -> List[tuple[int, str]]:
    """Return [(page_number, text)]. PDFs are read page by page; .txt/.md count as a single page."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8"))]


def split_pages(pages: List[tuple[int, str]], chunk_size: int = 800, chunk_overlap: int = 100, separator: str = ". "):
    """CharacterTextSplitter as in the demo.

    * separator     - preferred split point (a full stop keeps sentences whole); if none is found before
                      chunk_size the splitter cuts there instead
    * chunk_size    - bounded by the model's context window; too small loses meaning, too large loses specificity
    * chunk_overlap - part of the previous chunk repeated in the next one to preserve continuity
    Returns [(page_number, chunk_text)].
    """
    splitter = CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for page_no, text in pages:
        for chunk in splitter.split_text(text):
            out.append((page_no, chunk))
    return out


def chunks_to_docs_wrapper(row: dict) -> Document:
    """Dictionary -> Document (metadata + page_content), as in the demo's wrapper function."""
    metadata = {"id": row["id"], "link": row["link"]}
    return Document(page_content=row["text"], metadata=metadata)


def chunks_to_documents(chunks: List[tuple[int, str]], source: str) -> List[Document]:
    docs = []
    for n, (page_no, text) in enumerate(chunks):
        docs.append(chunks_to_docs_wrapper({"id": f"{Path(source).stem}-{n}", "link": f"{Path(source).name}#page={page_no}", "text": text}))
    return docs


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def connect_oracle(settings):
    """oracledb.connect(user, password, dsn) - the DSN comes from Autonomous DB -> Database connection."""
    import oracledb

    return oracledb.connect(user=settings.db_user, password=settings.db_password, dsn=settings.db_dsn)
