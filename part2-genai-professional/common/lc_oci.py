"""LangChain adapters for OCI Generative AI that work with whichever model family is current.

Why this exists
---------------
The course uses langchain_community's `ChatOCIGenAI` / `OCIGenAIEmbeddings`. Those classes target the
older Cohere (v1) / Meta request shapes. Oracle has since moved on to newer models (e.g. Cohere Command A,
Embed v4) and a new official package (`langchain-oci`, which needs LangChain 1.x - and LangChain 1.x no
longer ships `ConversationChain` / `ConversationBufferMemory` that the course demo uses).

To keep the course code runnable on the LangChain 0.3 line *and* on current models, these two small
adapters reuse the request builders in `genai_common.py` (COHERE v1, COHEREV2 and GENERIC formats).
Set LANGCHAIN_USE_COMMUNITY=1 to use the original course classes instead.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .genai_common import Settings, chat, embed, get_inference_client


def split_messages(messages: List[BaseMessage]):
    """LangChain messages -> (preamble, history, final user message) as expected by build_chat_details."""
    preamble_parts, turns = [], []
    for m in messages:
        if isinstance(m, SystemMessage):
            preamble_parts.append(str(m.content))
        else:
            role = "CHATBOT" if isinstance(m, AIMessage) else "USER"
            turns.append({"role": role, "message": str(m.content)})
    if not turns:
        raise ValueError("at least one non-system message is required")
    final = turns.pop()
    # some providers need history to start with USER; drop any leading chatbot turn
    while turns and turns[0]["role"] == "CHATBOT":
        turns.pop(0)
    return ("\n".join(preamble_parts) or None), turns, final["message"]


class OCIGenAIChatShim(BaseChatModel):
    """A LangChain chat model backed by OCI Generative AI (COHERE / COHEREV2 / GENERIC aware)."""

    settings: Any
    client: Any = None
    max_tokens: int = 400
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "oci-genai-shim"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.client is None:
            self.client = get_inference_client(self.settings)
        preamble, history, message = split_messages(messages)
        text = chat(
            self.client,
            self.settings,
            message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            preamble=preamble,
            chat_history=history,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class OCIGenAIEmbeddingsShim(Embeddings):
    """LangChain `Embeddings` backed by OCI Generative AI (documents vs queries get the right input_type)."""

    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        self.client = client

    def _client(self):
        if self.client is None:
            self.client = get_inference_client(self.settings)
        return self.client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed(self._client(), self.settings, texts, input_type="SEARCH_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return embed(self._client(), self.settings, [text], input_type="SEARCH_QUERY")[0]


# --------------------------------------------------------------------------- factories
def _use_community() -> bool:
    return os.environ.get("LANGCHAIN_USE_COMMUNITY", "") == "1"


def make_chat_llm(settings: Settings, max_tokens: int = 200, temperature: float = 0.0):
    """Course version = ChatOCIGenAI (Model name, service endpoint, compartment id, max_tokens)."""
    if _use_community():
        from langchain_community.chat_models import ChatOCIGenAI

        return ChatOCIGenAI(
            model_id=settings.chat_model_id,
            service_endpoint=settings.endpoint,
            compartment_id=settings.compartment_id,
            auth_type="API_KEY",
            auth_profile=settings.profile,
            auth_file_location=settings.config_file,
            model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
        )
    return OCIGenAIChatShim(settings=settings, max_tokens=max_tokens, temperature=temperature)


def make_embeddings(settings: Settings):
    """Course version = OCIGenAIEmbeddings (model id, service endpoint, compartment id)."""
    if _use_community():
        from langchain_community.embeddings import OCIGenAIEmbeddings

        return OCIGenAIEmbeddings(
            model_id=settings.embed_model_id,
            service_endpoint=settings.endpoint,
            compartment_id=settings.compartment_id,
            auth_type="API_KEY",
            auth_profile=settings.profile,
            auth_file_location=settings.config_file,
        )
    return OCIGenAIEmbeddingsShim(settings)
