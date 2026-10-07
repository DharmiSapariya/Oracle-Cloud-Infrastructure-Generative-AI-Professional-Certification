"""Modern-ish replacement for the course's ConversationBufferMemory + ConversationalRetrievalChain.

Same idea (history + retrieval), built from current LangChain primitives that also exist in LangChain 1.x:

  follow-up question + history --(condense)--> standalone question --> retriever --> prompt --> LLM
  RunnableWithMessageHistory stores every turn per session id, so the memory is updated automatically.

Note: LangChain now steers *new* projects toward LangGraph's built-in persistence (it emits a pending
 deprecation warning for RunnableWithMessageHistory). The retrieval + history pattern is the same either way.

    python .../06_conversational_rag_runnable_history.py --offline
    python .../06_conversational_rag_runnable_history.py            # Oracle DB + OCI (after script 02)
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_common import chunks_to_documents, connect_oracle, format_docs, read_text, split_pages  # noqa: E402

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

SAMPLE = Path(__file__).parent / "data" / "oci_ai_foundations_overview.txt"

CONDENSE = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's follow-up as a standalone question using the chat history. Return only the question."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
ANSWER = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only this context. If it is not in the context, say you do not know.\n\nContext:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def build_conversational_rag(llm, retriever):
    condense_chain = CONDENSE | llm | StrOutputParser()

    def standalone(inputs: dict) -> str:
        return condense_chain.invoke(inputs) if inputs.get("history") else inputs["question"]

    chain = (
        RunnablePassthrough.assign(standalone=RunnableLambda(standalone))
        | RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["standalone"])))
        | ANSWER
        | llm
        | StrOutputParser()
    )
    sessions: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        return sessions.setdefault(session_id, InMemoryChatMessageHistory())

    return RunnableWithMessageHistory(chain, get_history, input_messages_key="question", history_messages_key="history"), sessions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.offline:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from langchain_core.vectorstores import InMemoryVectorStore

        docs = chunks_to_documents(split_pages(read_text(SAMPLE)), str(SAMPLE))
        retriever = InMemoryVectorStore.from_documents(docs, DeterministicFakeEmbedding(size=64)).as_retriever(search_kwargs={"k": 3})
        llm = FakeListChatModel(responses=["(fake answer 1)", "(fake standalone question)", "(fake answer 2)"])
    else:
        from langchain_community.vectorstores.oraclevs import OracleVS
        from langchain_community.vectorstores.utils import DistanceStrategy

        from common.genai_common import load_settings, require
        from common.lc_oci import make_chat_llm, make_embeddings

        s = load_settings()
        require(s, "compartment_id", "db_user", "db_password", "db_dsn")
        vs = OracleVS(client=connect_oracle(s), embedding_function=make_embeddings(s), table_name=s.table_name,
                      distance_strategy=DistanceStrategy.DOT_PRODUCT)
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        llm = make_chat_llm(s, max_tokens=300, temperature=0)

    rag, sessions = build_conversational_rag(llm, retriever)
    cfg = {"configurable": {"session_id": "demo"}}
    for q in ["What is Module 4 about?", "And how long is the exam?"]:
        print(f"\nQ: {q}\nA: {rag.invoke({'question': q}, config=cfg)}")
    print("\nturns stored for session 'demo':", len(sessions["demo"].messages))
