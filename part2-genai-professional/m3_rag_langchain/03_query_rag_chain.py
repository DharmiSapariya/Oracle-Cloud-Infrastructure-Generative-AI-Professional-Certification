"""Course demo: "Full RAG Implementation with Oracle 23ai" - Part 2: retrieval + generation.

  question -> retriever (top 3 similar chunks from Oracle Vector Store) -> prompt (context + question)
           -> LLM -> answer

The chain (LCEL):
    {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

Demo question + expected grounded answer:
    "Tell us about Module 4 of AI Foundation Certification Course."
    -> "According to the provided context, Module 4 ... is about Generative AI and LLMs"

    python .../03_query_rag_chain.py
    python .../03_query_rag_chain.py --question "How long is the exam?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_common import PROMPT_TEMPLATE, connect_oracle, format_docs  # noqa: E402

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def build_rag_chain(retriever, llm):
    """Retriever supplies the context; RunnablePassthrough forwards the user question unchanged."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Tell us about Module 4 of AI Foundation Certification Course.")
    ap.add_argument("--k", type=int, default=3, help="top-k chunks to retrieve")
    args = ap.parse_args()

    from langchain_community.vectorstores.oraclevs import OracleVS
    from langchain_community.vectorstores.utils import DistanceStrategy

    from common.genai_common import load_settings, require
    from common.lc_oci import make_chat_llm, make_embeddings

    s = load_settings()
    require(s, "compartment_id", "db_user", "db_password", "db_dsn")
    conn = connect_oracle(s)
    vs = OracleVS(
        client=conn,
        embedding_function=make_embeddings(s),
        table_name=s.table_name,  # already populated by 02_ingest_pdf_to_oracle_vs.py
        distance_strategy=DistanceStrategy.DOT_PRODUCT,  # must match the strategy used at ingestion
    )
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": args.k})
    chain = build_rag_chain(retriever, make_chat_llm(s, max_tokens=300, temperature=0))

    print("Question:", args.question)
    print("Answer  :", chain.invoke(args.question))
    conn.close()
