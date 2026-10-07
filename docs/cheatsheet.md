# Exam cheat-sheet (one page)

**Stack**: AI > ML > DL > GenAI/LLM. Supervised = labels; unsupervised = clusters; reinforcement = reward.
**Regression** = continuous; **classification** = categorical; sigmoid -> probability, threshold 0.5.
**Architectures**: CNN images - RNN/LSTM sequences (vanishing gradient -> LSTM gates) - Transformer = self-attention, parallel.
**Transformer variants**: encoder-only = embeddings/RAG; decoder-only = generation; encoder-decoder = translation.
**Tokens**: ~1/word simple, 2-3/word complex.
**Decoding**: greedy deterministic; temperature never changes ranking; top-k = k best; top-p = cumulative mass; beam = several sequences.
**Penalties**: frequency scales with count; presence is flat.
**Prompting**: k-shot (weights unchanged), CoT (reasoning steps), zero-shot CoT ("Let's think step by step"), least-to-most.
**Customization order**: prompt engineering -> RAG (context) -> fine-tuning (behaviour). Additive.
**Hallucination**: not fully preventable; RAG + citations + groundedness checks help.
**OCI GenAI**: serverless, single API; Meta + Cohere; chat + embedding models; embeddings 1,024-d (v3), light 384-d, <=512 tokens, <=96 inputs/run.
**Fine-tuning**: JSONL `prompt`/`completion`, UTF-8. T-Few ~0.01% added layers, up to 50 endpoints per hosting cluster. Loss > accuracy for generative tasks.
**Clusters**: limits 0 by default; hosting = 744 h/month minimum; fine-tuning = 1 h minimum; no mixing Cohere/Meta units. Bob: 160 + 744 = 904 x $6.50.
**Security**: dedicated GPUs + RDMA, model/data isolation per tenancy, IAM, KMS + encrypted weights in Object Storage.
**RAG**: ingestion (load, chunk, embed, index) -> retrieval (top-K) -> generation. Chunk size / overlap / semantic splitting.
**Oracle 23ai**: VECTOR type, `VECTOR_DISTANCE` (cosine default), HNSW-style in-memory neighbor graph vs neighbor partitions, `TARGET ACCURACY`, `FETCH APPROXIMATE`.
**LangChain**: PromptTemplate/ChatPromptTemplate, LCEL `|`, memory, `OracleVS`, retriever `search_kwargs={"k":3}`.
**Agents**: data store -> data source -> knowledge base; Object Storage = PDF/text, 100 MB, one bucket, 8 MB for embedded images; Oracle DB = DOCID/body/vector + retrieval function with matching embedding model; session timeout 3,600 s default; moderation off by default.
