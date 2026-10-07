# Runbook 04 - Agent with an Object Storage knowledge base (Demo 1)

Console: *Analytics & AI -> Generative AI Agents*. Check regional availability first.

- [ ] Create a bucket and upload `part2-genai-professional/m4_genai_agents/data/faq.txt` (and optionally a course PDF). Only PDF/text, <= 100 MB each, one bucket per data source.
- [ ] **Knowledge bases -> Create**: name, compartment, data store type **Object Storage**; enable **hybrid search** (lexical + semantic) if you want it.
- [ ] **Data source**: name, optional multimodal parsing, choose the bucket/files; tick **start ingestion job**.
- [ ] Check the **ingestion job** status log; fix and restart on failure (already-ingested files are skipped). Re-run an ingestion job whenever files change.
- [ ] **Agents -> Create agent**: name (`demo-agent`), welcome message, RAG instructions (optional), pick the knowledge base, tick **auto-create endpoint**; accept the licence/acceptable use policy.
- [ ] Review the endpoint: session on (idle timeout default 3,600 s), trace on, citation on, moderation off.
- [ ] **Chat** -> select agent + endpoint and ask: (1) *Please tell me about Oracle free tier* (2) *How many modules are there in Oracle AI Foundations course?* (3) *Who are the instructors for this course?* (relies on session memory).
- [ ] Inspect **citations** (title, path, document id, source text) and **trace**.
- [ ] Programmatic chat: set `OCI_AGENT_ENDPOINT_ID` and run `python part2-genai-professional/m4_genai_agents/05_agent_chat_client.py`.
- [ ] Clean up: delete agent -> data source -> knowledge base (a knowledge base used by an agent cannot be deleted).
