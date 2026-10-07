# Part 2 / Module 3 - RAG with LangChain and Oracle Database 23ai

Code: [`m3_rag_langchain/`](../../part2-genai-professional/m3_rag_langchain/) - Runbook: [Autonomous DB 23ai](../runbooks/03-autonomous-database-23ai.md)

## LangChain
Framework of exchangeable components: LLMs, prompts, memory, chains, vector stores, document loaders, text splitters.
- **Model types**: *LLM* (string in, string out) vs *Chat model* (messages in, AI message out).
- **Prompts**: `PromptTemplate` (fixed text + runtime variables -> prompt value) and `ChatPromptTemplate` (role-tagged messages).
- **Chains**: **LCEL** (declarative, preferred; the `|` pipe) or classes such as `LLMChain`.
- **Memory**: the chain reads history by key, sends it with the new question, then writes query + answer back; memory types return the full history, a summary, or extracted entities.
- Oracle 23ai as a vector store (`OracleVS`). OCI GenAI integrates with 23ai three ways: embeddings generated outside the DB, **Select AI**, and LangChain/Python SDK.
Lab: [`01_langchain_components.py`](../../part2-genai-professional/m3_rag_langchain/01_langchain_components.py).

## RAG
Why: LLM answers rely on (possibly stale) training data. RAG adds up-to-date external context -> fewer errors/bias, works around token limits (only top-K chunks are sent), handles broader queries.
**Three phases**
1. **Ingestion** - load documents -> chunk -> embed -> index in a vector DB.
2. **Retrieval** - embed the query -> similarity search -> top-K chunks.
3. **Generation** - top-K chunks + question -> LLM.

**Loading**: loaders for PDF, CSV, HTML, JSON (single file or directory).
**Splitting** - three considerations: *chunk size* (bounded by the context window; too small loses meaning, too large loses specificity), *chunk overlap* (previous text repeated for continuity), *semantic splitting* (separator hierarchy: paragraph -> sentence -> word). Demo: `CharacterTextSplitter(separator, chunk_size, chunk_overlap)`.
**Embeddings + storage**: embeddings can be generated outside the DB or inside via an imported **ONNX** model; Oracle 23ai has the `VECTOR` type; store with `OracleVS.from_documents(docs, embedding, client, table_name, distance_strategy)`.

## Full demo (reproduced in code)
1. Create an Autonomous Database (Data Warehouse, Serverless, secure access from your IP, mTLS unchecked); copy the connection string.
2. `oracledb.connect(user, password, dsn)`; read the PDF with `PdfReader`; split; wrap each chunk in a `Document` via `chunks_to_docs_wrapper` (metadata: id, link); embed with OCI GenAI; `OracleVS.from_documents(...)`. Table columns: primary key, text, metadata, embedding. -> [`02_ingest_pdf_to_oracle_vs.py`](../../part2-genai-professional/m3_rag_langchain/02_ingest_pdf_to_oracle_vs.py)
3. Query: `ChatOCIGenAI` + embeddings + prompt template + `OracleVS` -> `as_retriever(search_type="similarity", search_kwargs={"k": 3})` -> chain with `RunnablePassthrough` for the question. Answer: *"Module 4 ... is about Generative AI and LLMs"*. -> [`03_query_rag_chain.py`](../../part2-genai-professional/m3_rag_langchain/03_query_rag_chain.py)

## Conversational RAG
A chat is Q&A turns ("Tell me about Las Vegas" -> "its typical temperature"). Memory holds all Q&A and is passed *alongside* newly retrieved documents; memory updates after every answer. LangChain provides memory + chain classes ([`04_conversational_rag.py`](../../part2-genai-professional/m3_rag_langchain/04_conversational_rag.py)); a current-API variant is in [`06_...runnable_history.py`](../../part2-genai-professional/m3_rag_langchain/06_conversational_rag_runnable_history.py). The whole pipeline also runs offline: [`05_local_rag_no_cloud.py`](../../part2-genai-professional/m3_rag_langchain/05_local_rag_no_cloud.py).
