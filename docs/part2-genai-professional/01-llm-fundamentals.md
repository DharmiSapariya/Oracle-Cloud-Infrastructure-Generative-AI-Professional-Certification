# Part 2 / Module 1 - Large Language Models

Code: [`m1_llm_fundamentals/`](../../part2-genai-professional/m1_llm_fundamentals/)

## Language models and "large"
A language model is a **probabilistic model of text**: given a word sequence it returns a probability for *every word in its vocabulary* (never for words outside it). "Large" refers to the **parameter count**; there is no agreed threshold, and an LLM is not conceptually different from a smaller LM (BERT is sometimes called one).

Three guiding questions: can we *affect* the distribution (prompting vs training)? how does that change which words appear? how do we *decode* text from it?

## Architectures
Everything is built on the **transformer** (*Attention Is All You Need*, 2017).
| Architecture | Capability | Examples |
|---|---|---|
| **Encoder** | embeddings (text -> vector per token + one for the whole sentence) | BERT |
| **Decoder** | generation, **one token at a time**, fed back until done | Cohere Command, GPT-4, Llama |
| **Encoder-decoder** | sequence-to-sequence (translation) | T5-style |
Model size = trainable parameters; decoders tend to be far larger than encoders (historical, not technical). Encoder use cases: classification features and **semantic / vector search** (embed corpus -> index -> embed query -> compare similarity). Decoder generation is expensive because it is invoked repeatedly.

## Prompting (parameters unchanged)
- Small input changes shift the output distribution (appending "little" favours small animals). It works because pre-training taught associations.
- **Prompt engineering** = iterative refinement; can be brittle (whitespace changes matter).
- **In-context learning / k-shot**: demonstrations in the prompt; **no parameter learning**. Terminology: k-shots = examples, task description = instruction, prompt = sometimes the whole input. Demonstrations usually beat 0-shot.
- **Chain-of-thought (2022)**: show steps (facts -> equation -> answer). Two hypotheses: similar worked problems existed in pre-training; decomposing makes sub-steps manageable. It *imitates* reasoning - the model still emits one word at a time.
- **Zero-shot CoT**: just add "Let's think step by step".
- **Least-to-most**: solve easy subproblems first and reuse the answers (last-letter concatenation: "think machine" -> "ke", then "keg" ...).
- **DeepMind physics/chemistry**: ask for first principles and equations before answering.
- Prompt format matters: Llama 2 expects its own instruction/system tags.
Code: [`prompting_patterns.py`](../../part2-genai-professional/m1_llm_fundamentals/prompting_patterns.py).

## Prompt injection
Crafting input to elicit behaviour the deployer did not intend: append "pwned" (mild) -> ignore the task -> output a destructive SQL statement (parallels SQL injection) -> **prompt leaking** (reveal the hidden prompt) -> private-data extraction. Risk appears whenever a third party reaches the model input; guardrails and vigilance are needed.
Code: [`prompt_injection_lab.py`](../../part2-genai-professional/m1_llm_fundamentals/prompt_injection_lab.py).

## Training (parameters change)
| Method | Parameters changed | Cost |
|---|---|---|
| **Fine-tuning** | all, labeled task data | high |
| **PEFT** | small subset or added params | lower |
| **LoRA** | original frozen, small added matrices trained | lower |
| **Soft prompting** | learned "virtual words" in the prompt | cheap |
| **Continual pre-training** | all, **no labels** (keep predicting next word, e.g. general -> scientific text) | very high |
Rough costs: a 7B model generates on one GPU; 150B+ may need 8-16 GPUs; PEFT needs a few GPUs for hours; full pre-training needs hundreds-thousands of GPUs for days.

## Decoding
Iterative: distribution -> select -> append -> repeat until **EOS**.
- **Greedy**: always the most probable word - deterministic.
- **Sampling**: draw from the distribution ("small" -> "red" -> "panda"); non-deterministic.
- **Temperature**: low = peaked/predictable, high = flat/creative. **It never changes the ranking.** Factoid Q&A -> greedy; creative writing -> higher temperature.
- **Nucleus (top-p)** sampling; **beam search** keeps several sequences and prunes low-probability ones (higher joint probability than greedy).
Code: [`decoding_playground.py`](../../part2-genai-professional/m1_llm_fundamentals/decoding_playground.py) (tested in `tests/test_decoding.py`).

## Hallucination
Text not grounded in training data or input - can be fluent and start correct; often subtle (one wrong adjective); most dangerous where the reader cannot fact-check. No method eliminates it. Mitigations: **RAG**, **groundedness checks** via Natural Language Inference (premise entails hypothesis? e.g. the "TRUE" model), grounded QA with **citations/attribution**.

## Applications of LLMs
- **RAG**: question -> query a corpus -> pass documents + question to the LLM. A *non-parametric* improvement: add documents, never retrain. Multi-doc QA, dialogue, fact-checking.
- **Code models** (Copilot, Codex, Code Llama): code is narrower and less ambiguous; best models patch real bugs correctly under 15% of the time.
- **Multimodal** + **diffusion** models (generate all pixels at once from noise; hard for text because length is unknown and words are discrete).
- **Language agents**: act in an environment toward a goal; **ReAct** (emit thoughts: goal, steps done, next steps); **tool use**; **reasoning** as high-level planning.
