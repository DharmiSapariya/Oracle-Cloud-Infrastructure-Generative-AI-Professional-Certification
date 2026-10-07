# Part 2 / Module 2 - OCI Generative AI service

Code: [`m2_oci_genai_service/`](../../part2-genai-professional/m2_oci_genai_service/) - Runbooks: [playground](../runbooks/01-playground-chat-and-embeddings.md), [fine-tuning](../runbooks/02-fine-tuning-and-endpoints.md)

## Overview
Fully managed, **serverless**, single-API service. Pre-trained models from **Meta** and **Cohere**; **flexible fine-tuning**; **dedicated AI clusters** (GPUs + exclusive RDMA network, single-tenant). Use cases: chat, text generation, information retrieval/extraction, classification, semantic search. Region availability is limited - check before you start.

## Chat models (as recorded in the course)
| Model | Prompt | Response | Notes |
|---|---|---|---|
| Command R+ | up to 128k tokens | up to 4k/run | most capable, most expensive |
| Command R (16k) | up to 16k | up to 4k/run | cheaper/faster, nearly as capable |
| Llama 3.1 70B | 128k (prompt + response) | - | Meta |
| Llama 3.1 405B | 128k | - | largest public model at the time |
Chat models are **instruction-tuned** and keep context across turns. *Token* intuition: ~1 token/word for simple text, 2-3 for rare words; punctuation counts.

## Chat parameters
Maximum output tokens - **preamble override** (changes behaviour/tone, *not* fine-tuning) - **temperature** (0 = deterministic) - **top-k** (sample only from k best) - **top-p** (smallest set reaching cumulative p; e.g. 0.75 drops the bottom 25%; combine with top-k) - **frequency penalty** (grows with each repeat) - **presence penalty** (flat once a token has appeared).
Demo use cases: chat, data/entity extraction, few-shot sentiment classification. Code: [`02_chat_parameter_experiments.py`](../../part2-genai-professional/m2_oci_genai_service/02_chat_parameter_experiments.py), [`03_use_cases_extraction_and_classification.py`](../../part2-genai-professional/m2_oci_genai_service/03_use_cases_extraction_and_classification.py).

## Inference API (SDK) flow
`import oci` -> compartment id + config profile -> load config (user OCID, fingerprint, tenancy, region, key file) -> service endpoint -> `GenerativeAiInferenceClient` (no retry, 10 s connect / 240 s read) -> `ChatDetails` -> chat request (Cohere: message, max_tokens=600, temperature=0 ...) -> **on-demand serving mode** + model id -> send. Response: status 200, API format, chat history (roles USER/CHATBOT), finish reason `COMPLETE`, text.
Auth failure demo: wrong/empty private key -> *"provided key is not a private key or the provided passphrase is incorrect"*; fix by regenerating the API key (PEM) and updating the config. Code: [`00_check_setup.py`](../../part2-genai-professional/m2_oci_genai_service/00_check_setup.py), [`01_chat_inference.py`](../../part2-genai-professional/m2_oci_genai_service/01_chat_inference.py).

## Embedding models
Numeric vector representations of words/sentences/documents; **numerically close = semantically close**; compare with **cosine** or **dot-product** similarity. Powers semantic (meaning) search vs lexical (keyword) search; multilingual models enable cross-language search.
Specs as recorded: English/Multilingual v3 = **1,024 dimensions**, light versions **384**, max **512 tokens** per input, **max 96 inputs per run**.
Demo: capital-city questions cluster, "smallest state in the US" is an outlier, "smallest state in India" lands beside it, "largest state in the US" sits between. Code: [`04_embeddings_and_similarity.py`](../../part2-genai-professional/m2_oci_genai_service/04_embeddings_and_similarity.py).

## Prompt engineering and customizing LLMs
See the [Foundations summary](../part1-ai-foundations/04-generative-ai-and-llm-basics.md). Why not train from scratch: ~$1M for a 10B-parameter model, 2 trillion tokens of data (Llama 2) plus annotation, and deep expertise. Few-shot limits: the context window. Framework: horizontal axis **context optimisation** (RAG), vertical axis **LLM optimisation** (fine-tuning); start with prompt engineering.

## Fine-tuning and inference
- **Custom model** = base model + your data. Workflow: fine-tuning dedicated cluster -> data -> fine-tune -> custom model. Inference: **hosting cluster** -> **endpoint** -> serve.
- **Vanilla** updates all/most layers. **T-Few** (PEFT) inserts layers about **0.01%** of the base size and updates only those -> far cheaper, and base + fine-tuned models **share most weights in GPU memory**, so one hosting cluster serves up to **50** endpoints (multi-tenancy) with little switching overhead.
- **LoRA** is the other PEFT option.
- Hyper-parameters: total training epochs, batch size, learning rate, early-stopping threshold, early-stopping patience, log-metrics interval (check current ranges in the docs).
- **Metrics**: *accuracy* = correct tokens / total (4/6 = 67% for "the cat slept on the rug"); *loss* = how wrong the distribution is (0 = perfect). Loss is generally better for generative tasks because near-synonyms are penalised lightly. Code: [`07_finetuning_metrics_explained.py`](../../part2-genai-professional/m2_oci_genai_service/07_finetuning_metrics_explained.py).

## Dedicated AI cluster sizing and pricing
Unit types: **Large Cohere**, **Small Cohere**, **Embed Cohere**, **Large Meta** (no mixing families). Service limits are **zero by default** - request an increase.
| Model | Fine-tuning | Hosting |
|---|---|---|
| Command R+ (08-2024) | not supported | 2 Large Cohere |
| Command R (08-2024) | 8 Small Cohere | 1 Small Cohere |
| Llama 3.x | 4 Large Meta | 1 Large Meta |
| Embedding | not supported | 1 Embed Cohere |
Commitments: **hosting = whole month (744 unit-hours)**; **fine-tuning = 1 hour minimum** in full-hour steps. **Bob's scenario**: 8 units x 5 h x 4 = 160 + 744 = **904 unit-hours x $6.50 = about $5,900**. Code: [`06_dedicated_cluster_cost_calculator.py`](../../part2-genai-professional/m2_oci_genai_service/06_dedicated_cluster_cost_calculator.py).

## Custom-model demo
Dataset from *Sound Control: Natural Rephrasing in Dialog Systems*: human request -> virtual-assistant utterance (~2,000 pairs). Training data must be **JSONL**: one object per line with `prompt` and `completion`, **UTF-8**. T-Few chosen; default hyper-parameters; upload JSONL to Object Storage (with IAM policies). Result: **accuracy 0.98**, loss near 0. Endpoint on the hosting cluster (capacity 50; optional content moderation). Evaluation = metrics **and** qualitative base-vs-custom comparison on *unseen* prompts - the custom model stayed consistent from temperature 0 to 5, the base (Command Light) did not. Code: [`05_prepare_finetune_dataset.py`](../../part2-genai-professional/m2_oci_genai_service/05_prepare_finetune_dataset.py), [`08_custom_model_inference.py`](../../part2-genai-professional/m2_oci_genai_service/08_custom_model_inference.py).

## Security architecture
**GPU isolation** (dedicated GPUs on a dedicated RDMA network, never shared) - **model isolation** (base and custom endpoints only on your GPUs) - **data access limited to your tenancy** - **IAM** for authN/authZ (e.g. App X -> custom model X only) - **Key Management + Object Storage**: model weights stored encrypted by default.
