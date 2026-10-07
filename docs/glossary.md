# Glossary

| Term | Meaning |
|---|---|
| **Accuracy (fine-tuning)** | share of generated tokens matching the annotated tokens |
| **Agent endpoint** | access point through which applications talk to a GenAI agent |
| **AGI** | artificial general intelligence - any human capability, no intervention |
| **Answerability / Groundedness** | RAG-agent quality: relevant answers / answers traceable to sources |
| **Attention (self-attention)** | each token weighs every other token to build context |
| **Beam search** | decoding that keeps several candidate sequences |
| **Chain-of-thought** | prompting with intermediate reasoning steps |
| **Chunk (size / overlap)** | piece of a document; overlap repeats text for continuity |
| **Citation / Trace** | source of an answer / history of prompts and responses |
| **Custom model** | base model fine-tuned with your data |
| **Decoding** | turning the vocabulary distribution into text, one token at a time |
| **Dedicated AI cluster** | single-tenant GPU cluster for fine-tuning or hosting |
| **Embedding** | vector representing text; close vectors = similar meaning |
| **EOS** | end-of-sequence token that stops generation |
| **Few-shot / k-shot** | k examples in the prompt |
| **Fine-tuning (vanilla)** | updating all/most weights on custom data |
| **Frequency / presence penalty** | discourage repeats (scaled / flat) |
| **Groundedness** | generated text is supported by a document |
| **Hallucination** | fluent but ungrounded or false output |
| **In-context learning** | conditioning via the prompt; weights do not change |
| **Instruction tuning / RLHF** | aligning a model to follow instructions / human preferences |
| **JSONL** | one JSON object per line; UTF-8 (`prompt`, `completion` for fine-tuning) |
| **Knowledge base** | vector storage fed from a data source for an agent |
| **LCEL** | LangChain Expression Language (`a | b | c` pipes) |
| **LoRA / T-Few** | parameter-efficient fine-tuning methods |
| **Loss** | how wrong the predicted distribution is (0 = perfect) |
| **Multi-tenancy (hosting)** | many endpoints on one hosting cluster (up to 50 with T-Few) |
| **Nucleus (top-p) / top-k** | restrict sampling to a probability mass / k best tokens |
| **ONNX** | open model format loadable into Oracle Database 23ai |
| **Preamble** | initial instruction shaping chat behaviour (override to change tone) |
| **Prompt injection / leaking** | input that hijacks the task / reveals the hidden prompt |
| **RAG** | retrieval-augmented generation |
| **RDMA / RoCE** | CPU-bypass networking / RDMA over Converged Ethernet |
| **Select AI** | natural language to SQL in Autonomous Database |
| **Semantic vs lexical search** | meaning vs keyword matching |
| **Temperature** | flattens (high) or sharpens (low) the distribution; ranking unchanged |
| **Token** | word, sub-word or punctuation unit an LLM reads |
| **Transformer** | attention-based architecture behind LLMs |
| **VECTOR / VECTOR_DISTANCE** | Oracle 23ai datatype / similarity function (default cosine) |
