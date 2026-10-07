import pytest

pytest.importorskip("langchain")
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore

import importlib.util
from pathlib import Path

M3 = Path(__file__).resolve().parents[1] / "part2-genai-professional" / "m3_rag_langchain"


def load(name):
    from conftest import load_module

    return load_module(name, M3 / f"{name}.py")


rag_common = load("rag_common")
local_rag = load("05_local_rag_no_cloud")
query = load("03_query_rag_chain")
conv = load("06_conversational_rag_runnable_history")


def test_chunking_respects_size_and_overlap_and_tracks_pages():
    text = ". ".join(f"Sentence number {i} is here" for i in range(60))
    chunks = rag_common.split_pages([(3, text)], chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 3 and all(p == 3 for p, _ in chunks) and all(len(c) <= 150 for _, c in chunks)


def test_docs_wrapper_keeps_metadata():
    d = rag_common.chunks_to_docs_wrapper({"id": "x-1", "link": "f.pdf#page=2", "text": "hello"})
    assert d.page_content == "hello" and d.metadata == {"id": "x-1", "link": "f.pdf#page=2"}


def test_local_rag_answers_course_questions():
    rag = local_rag.build_tfidf_rag()
    assert "Generative AI" in rag.ask("Tell us about Module 4 of the AI Foundations course.")
    assert "T-Few" in rag.ask("What is T-Few?")
    assert "grounded" in rag.ask("Why does RAG reduce hallucination?")
    assert "Context:" in rag.prompt("anything") and len(rag.retrieve("exam", 2)) == 2


def test_lcel_rag_chain_wires_retriever_prompt_and_llm():
    docs = [Document(page_content="Module 4 covers Generative AI.", metadata={})]
    retriever = InMemoryVectorStore.from_documents(docs, DeterministicFakeEmbedding(size=32)).as_retriever(search_kwargs={"k": 1})
    seen = {}

    class Spy(FakeListChatModel):
        def _generate(self, messages, *a, **k):
            seen["prompt"] = messages[0].content
            return super()._generate(messages, *a, **k)

    out = query.build_rag_chain(retriever, Spy(responses=["grounded answer"])).invoke("What is Module 4?")
    assert out == "grounded answer" and "Module 4 covers Generative AI." in seen["prompt"] and "What is Module 4?" in seen["prompt"]


def test_conversational_rag_stores_history_and_condenses_followups():
    docs = [Document(page_content="Module 4 covers Generative AI.", metadata={})]
    retriever = InMemoryVectorStore.from_documents(docs, DeterministicFakeEmbedding(size=32)).as_retriever(search_kwargs={"k": 1})
    llm = FakeListChatModel(responses=["answer 1", "standalone question", "answer 2"])
    rag, sessions = conv.build_conversational_rag(llm, retriever)
    cfg = {"configurable": {"session_id": "t"}}
    assert rag.invoke({"question": "What is Module 4?"}, config=cfg) == "answer 1"
    assert rag.invoke({"question": "And its exam?"}, config=cfg) == "answer 2"  # 1 condense call + 1 answer call
    assert len(sessions["t"].messages) == 4


def test_split_messages_maps_langchain_roles():
    lc = pytest.importorskip("common.lc_oci")
    pre, hist, last = lc.split_messages([SystemMessage(content="be brief"), HumanMessage(content="q1"),
                                         AIMessage(content="a1"), HumanMessage(content="q2")])
    assert pre == "be brief" and last == "q2" and [h["role"] for h in hist] == ["USER", "CHATBOT"]


def test_shim_calls_oci_chat_and_embeddings(monkeypatch):
    lc = pytest.importorskip("common.lc_oci")
    from common.genai_common import Settings

    calls = {}
    monkeypatch.setattr(lc, "chat", lambda client, s, msg, **kw: calls.update(msg=msg, **kw) or "pong")
    monkeypatch.setattr(lc, "embed", lambda client, s, texts, input_type: [[float(len(t)), 1.0] for t in texts] if calls.setdefault("it", []).append(input_type) is None else None)
    s = Settings(compartment_id="x")
    llm = lc.OCIGenAIChatShim(settings=s, client=object(), max_tokens=50, temperature=0.3)
    assert llm.invoke([HumanMessage(content="ping")]).content == "pong" and calls["msg"] == "ping" and calls["max_tokens"] == 50
    emb = lc.OCIGenAIEmbeddingsShim(s, client=object())
    assert emb.embed_documents(["ab", "c"]) == [[2.0, 1.0], [1.0, 1.0]] and emb.embed_query("abc") == [3.0, 1.0]
    assert calls["it"] == ["SEARCH_DOCUMENT", "SEARCH_QUERY"]
