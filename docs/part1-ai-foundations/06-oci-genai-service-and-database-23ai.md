# 06 - OCI Generative AI service, AI Vector Search and Select AI - Foundations view (Modules 29-32)

Deeper treatment in [Part 2 notes](../part2-genai-professional/02-oci-genai-service.md).

## OCI Generative AI service
Fully managed, **serverless**, **single API** to customizable LLMs. Three characteristics: pre-trained models (**Meta** and **Cohere**), flexible **fine-tuning** (T-Few), **dedicated AI clusters** (GPU + exclusive RDMA network, isolated per customer).
Course-time chat models: `command-r-plus` (128k tokens), `command-r-16k` (16k), `llama-3-70b-instruct` (8k in the demo). Embedding: Embed English / Multilingual (100+ languages; cross-language search). Playground = no-code testing + **View Code**; preamble override changes tone without fine-tuning.
> Those model IDs were later retired. See [keeping-code-current](../keeping-code-current.md).

## AI Vector Search in Oracle Database 23ai
- Converged DB: JSON, XML, graph, spatial, text, relational **and vectors** together.
- `VECTOR` datatype, `VECTOR_EMBEDDING` (load an ONNX model into the DB), `VECTOR_DISTANCE` (default **cosine**; smaller = more similar).
- Vector indexes: **IN MEMORY NEIGHBOR GRAPH** if it fits in memory, otherwise **NEIGHBOR PARTITIONS**; `TARGET ACCURACY` (e.g. 80% of top-5 means 4 of 5 correct); `FETCH APPROXIMATE` uses the index, else an exact search.
- Similarity search **over joins** (authors x books x pages) with the cost-based optimiser.
- GenAI pipeline: load -> transform (split/summarise) -> embed -> store -> similarity search / RAG. Integrates with **LangChain** and **LlamaIndex**.

## Select AI (Autonomous Database)
Natural language -> SQL: `SELECT AI <question>`; `SELECT AI showsql` reveals the generated SQL. An **AI profile** (`dbms_cloud_ai`) defines the LLM provider (OCI GenAI, Cohere, OpenAI, Llama...) and the schemas/tables. With OCI GenAI the data stays in your tenancy.
