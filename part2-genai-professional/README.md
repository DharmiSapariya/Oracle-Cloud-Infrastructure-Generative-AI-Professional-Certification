# Part 2 - Generative AI Professional labs

```
common/                 settings, OCI client, request builders (COHERE / COHEREV2 / GENERIC), LangChain adapters
m1_llm_fundamentals/    decoding_playground, self_attention_numpy, prompting_patterns, prompt_injection_lab   (offline)
m2_oci_genai_service/   00 setup check - 01 chat - 02 parameters - 03 use cases - 04 embeddings (--offline ok)
                        05 fine-tune dataset (JSONL build/validate) - 06 cluster cost calculator
                        07 accuracy vs loss - 08 custom-model endpoint inference
m3_rag_langchain/       01 components - 02 ingest to Oracle VS - 03 RAG chain - 04 conversational RAG
                        05 local RAG (no cloud) - 06 conversational RAG with current APIs
m4_genai_agents/        sql/ (ACL, credentials, chunking, vectors, retrieval function) - 05 agent chat client
```

| Needs | Scripts |
|---|---|
| nothing (offline) | everything in `m1_*`, `04 --offline`, `05`, `06`, `m3/05`, `m3/01 --offline`, `m3/04 --offline`, `m3/06 --offline` |
| OCI credentials | `m2/00-04`, `m2/08`, `m3/01` (online) |
| OCI + Autonomous DB 23ai | `m3/02-04`, `m3/06` (online) |
| OCI GenAI Agent endpoint | `m4/05` |

Setup: [`docs/runbooks/00-oci-setup.md`](../docs/runbooks/00-oci-setup.md). Model IDs and library versions have drifted since the course - read [`docs/keeping-code-current.md`](../docs/keeping-code-current.md) first.
