# Part 2 / Module 4 - OCI Generative AI Agents

Code: [`m4_genai_agents/`](../../part2-genai-professional/m4_genai_agents/) - Runbooks: [Object Storage agent](../runbooks/04-genai-agents-object-storage.md), [Oracle 23ai agent](../runbooks/05-genai-agents-oracle-23ai.md)

## What it is
Fully managed service combining **LLMs with an intelligent retrieval system** to answer from your knowledge base - positioned as a **no-code RAG + chatbot** platform, usable via chat UI or API. An agent takes a request ("book a flight and a hotel"), plans, retrieves data and acts.

**Architecture**: interface -> LLM fed with short/long-term **memory**, **tools** (APIs, databases) and the **prompt**. The LLM performs **reasoning, acting, persona, planning**; the output can loop back into memory.

## Core concepts
- **RAG agent quality**: *answerability* (relevant answers to varied queries) and *groundedness* (traceable to sources).
- **Data hierarchy**: *data store* (where data lives) -> *data source* (connection details) -> *knowledge base* (vector storage that ingests from the source).
- **Data options**: Object Storage (managed ingestion), OpenSearch (bring your own index), Oracle Database 23ai vector store (bring your own embeddings).
- **Session** (conversation context; idle timeout default **3,600 s**, 1 h to 7 days) - **Agent endpoint** - **Trace** (prompt/response history) - **Citation** (title, path, doc id, pages) - **Content moderation** (input, output or both; default off).
- **Ingestion** = extract -> transform -> store in the knowledge base.

## Object Storage data source rules
One bucket per data source - **PDF and text only** - up to **100 MB** per file - embedded images/charts/tables must not exceed **8 MB** - charts must be 2-D with labelled axes - PDF hyperlinks appear as clickable links - an empty folder can be prepared in advance. **One data source per knowledge base.** New files or removals require a **new ingestion job**; a restarted job skips files already ingested; cancel only while in progress/waiting. Optional **hybrid search** (lexical + semantic) and **multimodal parsing** (charts/graphs).
Knowledge base deletion is permanent and requires deleting dependent agents first; deleting a data source used by an agent makes the agent stop answering from it.

## Oracle Database data source rules
You manage the database. Table needs **DOCID, body, vector** (optional CHUNKID, URL, title, page numbers). Provide a **retrieval function** (course: `retrieval_func_ai`) with inputs `p_query`, `top_k` that embeds the query, computes cosine/Euclidean distance, returns top-K by descending score as a `SYS_REFCURSOR` (DOCID, body, score). **The embedding model must match the one that produced the stored vectors.**

## Workflow
Knowledge base -> agent (welcome message, optional RAG instructions) -> endpoint (session, moderation, trace, citation) -> chat. Defaults on auto-created endpoints: trace, citation, session on; moderation off.
Demo questions: *"Tell me about Oracle free tier"*, *"How many modules ..."*, *"Who are the instructors ..."* - the third relied on **session memory** (it never named the course).
Code: SQL in [`m4_genai_agents/sql/`](../../part2-genai-professional/m4_genai_agents/sql/), API client in [`05_agent_chat_client.py`](../../part2-genai-professional/m4_genai_agents/05_agent_chat_client.py).
