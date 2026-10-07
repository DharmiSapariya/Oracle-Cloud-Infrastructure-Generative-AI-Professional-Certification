# 04 - Generative AI, LLMs, transformers and prompting - the Foundations view (Modules 17-22)

Deeper treatment in [Part 2 notes](../part2-genai-professional/01-llm-fundamentals.md).

## Generative AI
Creates new content after learning patterns from training data (dog-drawing analogy: learns pointy ears/whiskers, then draws a *new* dog).
| | Traditional ML | Generative AI |
|---|---|---|
| Training data | features + labels | unstructured, unlabeled (pre-training) |
| Learns | data <-> label relationship | patterns in the content |
| Output | label / prediction | new text / media |
Types: text-based and **multimodal** models. A pre-trained model can later be **fine-tuned** on labeled data.

## LLMs
A language model is a **probabilistic model of text**: given previous words it assigns probabilities over its vocabulary ("They sent me a ___": dog 0.45, lion 0.03 ...). Generation: pick a word, append, repeat until an **EOS** token. "Large" = number of parameters (no agreed threshold). More parameters does not automatically mean better (overfitting risk).

## Transformers
- RNNs read one token at a time and forget distant context (vanishing gradient). Transformers look at **all tokens at once** with **self-attention**, linking "it" to "Frisbee" in *Jane threw the Frisbee and her dog fetched it*. Paper: *Attention Is All You Need*.
- **Tokens**: whole word, part of word or punctuation ("apple" = 1, "friendship" = 2). Simple text about 1 token/word; complex text 2-3.
- **Embeddings**: vectors representing text; support classification, vector DBs, **semantic search**.
| Variant | Does | Uses |
|---|---|---|
| Encoder-only | text -> vectors | RAG retrieval, vector DBs, semantic search |
| Decoder-only | emits next token, one at a time | text generation |
| Encoder-decoder | encode then decode | translation / seq2seq |
Lab: [self-attention in NumPy](../../part2-genai-professional/m1_llm_fundamentals/self_attention_numpy.py).

## Prompt engineering
- **Prompt** = input to the LLM; **prompt engineering** = iterating on it. Completion models just continue text; **instruction tuning** (Llama 2 Chat: ~28,000 prompt/response pairs) and **RLHF** (human comparisons -> reward model) make models follow instructions.
- **In-context / k-shot**: examples in the prompt (0-shot, 1-shot ... k-shot); *no weights change*.
- **Chain-of-thought**: include reasoning steps (Roger's tennis balls: 5 + 2 cans x 3 = 11).
- **Hallucination**: fluent but ungrounded or false text; RAG reduces it; nobody can fully prevent it yet.
Lab: [prompting patterns](../../part2-genai-professional/m1_llm_fundamentals/prompting_patterns.py).

## Customizing LLMs
| Method | Use when | Pros | Cons |
|---|---|---|---|
| **Prompt engineering** | the model already knows the topic | fastest, no training | prompt length |
| **RAG** | data changes fast / need grounding | latest data, fewer hallucinations, no fine-tune | needs good data source, more complex |
| **Fine-tuning** | model underperforms or must learn new style/domain | better performance and efficiency | labeled data + compute |
They are additive. **T-Few** (OCI): inserts new layers and updates a small fraction of weights = cheaper than vanilla fine-tuning.
Journey: prompt + evaluation baseline -> few-shot -> RAG -> fine-tune the RAG model -> optimise retrieval; iterate.
