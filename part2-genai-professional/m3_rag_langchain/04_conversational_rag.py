"""Course lesson: "RAG for Conversational Chatbots" - memory + retrieval combined.

A chat is a series of Q&A turns. Follow-ups lean on earlier turns:
    Q1 "Tell me about Las Vegas"  ->  Q2 "Tell me about its typical temperature throughout the year"
("its" = Las Vegas). So the chain passes BOTH (a) newly retrieved documents and (b) the conversation
history from memory to the LLM; memory is updated after every answer.

LangChain support: ConversationBufferMemory + ConversationalRetrievalChain
(it first condenses "follow-up + history" into a standalone question, retrieves, then answers).

    python .../04_conversational_rag.py            # needs Oracle DB + OCI (after running script 02)
    python .../04_conversational_rag.py --offline  # in-memory vector store + fake models
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_common import PROMPT_TEMPLATE, chunks_to_documents, connect_oracle, read_text, split_pages  # noqa: E402

import warnings

from langchain_core._api import LangChainDeprecationWarning

# ConversationChain / ConversationBufferMemory are what the course teaches; they are deprecated in
# LangChain 0.3 and removed in 1.0. Script 06 shows the modern replacement (RunnableWithMessageHistory).
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

SAMPLE = Path(__file__).parent / "data" / "oci_ai_foundations_overview.txt"


def build_chain(llm, retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=True), memory


def offline_components():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.vectorstores import InMemoryVectorStore

    docs = chunks_to_documents(split_pages(read_text(SAMPLE)), str(SAMPLE))
    store = InMemoryVectorStore.from_documents(docs, DeterministicFakeEmbedding(size=64))
    llm = FakeListChatModel(responses=["(fake answer 1)", "(fake standalone question)", "(fake answer 2)"])
    return llm, store.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.offline:
        llm, retriever = offline_components()
    else:
        from langchain_community.vectorstores.oraclevs import OracleVS
        from langchain_community.vectorstores.utils import DistanceStrategy

        from common.genai_common import load_settings, require
        from common.lc_oci import make_chat_llm, make_embeddings

        s = load_settings()
        require(s, "compartment_id", "db_user", "db_password", "db_dsn")
        conn = connect_oracle(s)
        vs = OracleVS(client=conn, embedding_function=make_embeddings(s), table_name=s.table_name,
                      distance_strategy=DistanceStrategy.DOT_PRODUCT)
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        llm = make_chat_llm(s, max_tokens=300, temperature=0)

    chain, memory = build_chain(llm, retriever)
    for q in ["What is Module 4 about?", "And how long is the exam for it?"]:
        result = chain.invoke({"question": q})
        print(f"\nQ: {q}\nA: {result['answer']}\n   sources: {[d.metadata.get('id') for d in result['source_documents']]}")
    print("\nmessages held in memory:", len(memory.chat_memory.messages))
