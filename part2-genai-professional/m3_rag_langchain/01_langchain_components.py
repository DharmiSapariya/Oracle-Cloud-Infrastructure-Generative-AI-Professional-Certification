"""Course demo: "LangChain Components in Action" - models, prompts, chains, memory.

LangChain = a framework of exchangeable building blocks (LLMs, prompts, memory, chains, vector stores,
document loaders, text splitters ...). This script walks through them in the same order as the demo:

 1. Models         - the LLM object (model name, service endpoint, compartment id, max_tokens=200)
 2. PromptTemplate - fixed text + runtime variables (user_input, city) -> a "prompt value"
 3. Chaining       - prompt | llm  (LCEL pipe operator): input -> prompt -> llm -> response
 4. ChatPromptTemplate - a list of role-tagged messages instead of one string
 5. Memory         - ConversationBufferMemory + ConversationChain: "Hello, my name is Hemant"
                     then "Can you tell me what my name is?" -> the chain remembers.

Needs OCI credentials (.env). `--offline` swaps in a fake LLM so you can see the wiring without a cloud account.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

from langchain_core._api import LangChainDeprecationWarning

# ConversationChain / ConversationBufferMemory are what the course teaches; they are deprecated in
# LangChain 0.3 and removed in 1.0. Script 06 shows the modern replacement (RunnableWithMessageHistory).
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def build_llm(offline: bool):
    if offline:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["(fake) Hello Hemant!", "(fake) Certainly, Hemant. Your name is Hemant."] * 3)
    from common.genai_common import load_settings, require
    from common.lc_oci import make_chat_llm

    s = load_settings()
    require(s, "compartment_id")
    return make_chat_llm(s, max_tokens=200, temperature=0)


def main(offline: bool) -> None:
    llm = build_llm(offline)

    print("### 1. Model: invoke directly")
    print(llm.invoke("Tell me one fact about the planet Mars."))

    print("\n### 2. PromptTemplate: fixed text + variables -> prompt value")
    prompt = PromptTemplate.from_template("{user_input} Keep it under 40 words and mention the city of {city}.")
    value = prompt.invoke({"user_input": "Tell us in an exciting tone about New York.", "city": "New York"})
    print(value.to_string())

    print("\n### 3. Chain: prompt | llm  (LCEL)")
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"user_input": "Tell us in an exciting tone about New York.", "city": "New York"}))

    print("\n### 4. ChatPromptTemplate: a list of messages")
    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a friendly travel guide for {city}."), ("human", "{question}")]
    )
    print(chat_prompt.invoke({"city": "New York", "question": "What's the New York culture like?"}).to_messages())
    print((chat_prompt | llm | StrOutputParser()).invoke({"city": "New York", "question": "What's the New York culture like?"}))

    print("\n### 5. Memory: ConversationBufferMemory + ConversationChain")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    print("Q1 ->", conversation.invoke({"input": "Hello, my name is Hemant"})["response"])
    print("memory after Q1:", memory.buffer)
    print("Q2 ->", conversation.invoke({"input": "Can you tell me what my name is?"})["response"])
    print("memory after Q2:\n" + memory.buffer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use a fake LLM (no cloud needed)")
    main(ap.parse_args().offline)
