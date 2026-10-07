"""Course demo: "Full RAG Implementation with Oracle 23ai" - Part 1: ingestion.

  document -> split into chunks -> Document objects -> embed with OCI GenAI -> store in Oracle Vector Store

Steps (matching the demo):
  1. connect with oracledb.connect(user, password, dsn)
  2. read the PDF (PdfReader) and extract text from all pages
  3. CharacterTextSplitter(separator, chunk_size, chunk_overlap)
  4. chunks_to_docs_wrapper -> list of Document(metadata, page_content)
  5. embedding model (OCI Generative AI)
  6. OracleVS.from_documents(docs, embedding, client=connection, table_name=..., distance_strategy=...)
Result: a table with 4 columns - primary key, text, metadata, embedding (check it in SQL Developer).

Prerequisite (see docs/runbooks/03-autonomous-database-23ai.md): an Autonomous Database 23ai and .env filled in.

    python .../02_ingest_pdf_to_oracle_vs.py                         # sample corpus
    python .../02_ingest_pdf_to_oracle_vs.py --file my_course.pdf    # your own PDF
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_common import chunks_to_documents, connect_oracle, read_text, split_pages  # noqa: E402

DEFAULT_FILE = Path(__file__).parent / "data" / "oci_ai_foundations_overview.txt"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="only show the chunks (no cloud/database needed)")
    args = ap.parse_args()

    chunks = split_pages(read_text(args.file), args.chunk_size, args.chunk_overlap)
    docs = chunks_to_documents(chunks, str(args.file))
    print(f"{args.file.name}: {len(docs)} chunks")
    for d in docs[:3]:
        print(f"  [{d.metadata['id']}] {d.page_content[:90]!r}...")
    if args.dry_run:
        sys.exit(0)

    from langchain_community.vectorstores.oraclevs import OracleVS
    from langchain_community.vectorstores.utils import DistanceStrategy

    from common.genai_common import load_settings, require
    from common.lc_oci import make_embeddings

    s = load_settings()
    require(s, "compartment_id", "db_user", "db_password", "db_dsn")
    conn = connect_oracle(s)
    vs = OracleVS.from_documents(
        docs,
        make_embeddings(s),
        client=conn,
        table_name=s.table_name,
        distance_strategy=DistanceStrategy.DOT_PRODUCT,  # cosine / euclidean / dot product are all available
    )
    print(f"Stored {len(docs)} embedded chunks in table {s.table_name}.")
    conn.close()
