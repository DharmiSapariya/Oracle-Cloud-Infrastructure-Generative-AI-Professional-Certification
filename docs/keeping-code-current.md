# Keeping the code current (read before running cloud labs)

The course material was recorded in 2024-2025. OCI Generative AI changes quickly, so some things shown in the videos no longer work as taught.

## Models
The course used `cohere.command-r-plus`, `cohere.command-r-16k`, `meta.llama-3-70b-instruct`, Cohere Embed v3 (English / Multilingual) and Cohere Command Light. Oracle's model-retirement schedule has since retired these on-demand (and Command R / R+ 08-2024 were scheduled for retirement on 2026-09-30, with Cohere Command A and Embed v4 as the replacements). **Always check the current list**: <https://docs.oracle.com/en-us/iaas/Content/generative-ai/deprecating.htm>.

How this repo copes:
- Model IDs are **settings**, not hard-coded: `OCI_CHAT_MODEL_ID`, `OCI_EMBED_MODEL_ID` in `.env`. Defaults are `cohere.command-a-03-2025` and `cohere.embed-v4.0`.
- `common/genai_common.py` builds the right request for each family: **COHERE** (v1, as in the course), **COHEREV2** (Command A family) and **GENERIC** (Meta Llama etc.). Override with `OCI_CHAT_API_FORMAT`.
- Embedding dimension is **not** assumed (v3 = 1,024, light = 384, v4 supports several sizes). Anything stored in a vector table must be queried with the *same* model.

## LangChain
The course uses `ChatOCIGenAI`, `OCIGenAIEmbeddings`, `ConversationChain` and `ConversationBufferMemory`.
- `requirements-oci.txt` pins **LangChain 0.3** because 1.x removed the classic memory/chain classes.
- `common/lc_oci.py` provides two small adapters (`OCIGenAIChatShim`, `OCIGenAIEmbeddingsShim`) that work with current models. `LANGCHAIN_USE_COMMUNITY=1` switches back to the original `langchain_community` classes.
- Oracle's newer packages `langchain-oci` and `langchain-oracledb` target LangChain 1.x; `06_conversational_rag_runnable_history.py` shows the history + retrieval pattern without the removed classes (LangChain now recommends LangGraph persistence for new projects).

## What has been verified
| Area | Status |
|---|---|
| Part 1 labs (scikit-learn), notebooks, decoding/attention, JSONL tools, cost calculator, quiz data, RAG pipeline wiring (fake models) | **run and unit-tested** (`pytest`) |
| OCI request builders (chat v1/v2/generic, embeddings, dedicated endpoint) | object construction **unit-tested offline**; live calls need your tenancy |
| Scripts that call OCI, Oracle DB or the Agents runtime; Oracle SQL; ADS lifecycle | **written from the course demos and SDK signatures, not executed here** |
Report anything that breaks as an issue - cloud APIs drift.
