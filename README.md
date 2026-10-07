# OCI Generative AI Professional - Certification Labs

> Hands-on labs, code, notes, runbooks and practice exams from the **Oracle Cloud Infrastructure 2025 Certified Generative AI Professional** path - from AI/ML/DL foundations to LLMs, the OCI Generative AI service, RAG with LangChain + Oracle Database 23ai, and GenAI Agents.

![OCI Generative AI Professional 2025](assets/oci-genai-professional-badge.jpg)

**Certified:** July 20, 2026 (valid until July 20, 2028) - see [`certificate/`](certificate/).

## What is inside
| Folder | Contents |
|---|---|
| [`part1-ai-foundations/`](part1-ai-foundations/) | 4 labs (+ notebooks): logistic regression on Iris, train/test + standardization pipeline, MLP on concentric circles, OCI Data Science ADS lifecycle |
| [`part2-genai-professional/`](part2-genai-professional/) | LLM fundamentals (decoding, attention, prompting, prompt injection) - OCI GenAI inference, parameters, embeddings, fine-tuning dataset tools, cluster cost calculator, custom endpoint - LangChain + Oracle 23ai RAG - GenAI Agents SQL + runtime client |
| [`docs/`](docs/) | structured notes for both courses, [runbooks](docs/runbooks/) for every console activity, glossary, one-page cheat-sheet |
| [`practice-exams/`](practice-exams/) | 175 + 140 questions with a shuffling terminal quiz |
| [`tests/`](tests/) | 60+ offline tests (no cloud account needed) + GitHub Actions CI |

## Quick start (no cloud account needed)
```bash
git clone https://github.com/<you>/oci-genai-professional-certification.git
cd oci-genai-professional-certification
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-oci.txt
python -m pytest -q                                        # everything offline should pass

python part1-ai-foundations/lab02_iris_pipeline_scaling_eval.py
python part2-genai-professional/m1_llm_fundamentals/decoding_playground.py
python part2-genai-professional/m3_rag_langchain/05_local_rag_no_cloud.py
python practice-exams/quiz.py --exam genai --n 10 --review
```

## Running the cloud labs
1. Follow [`docs/runbooks/00-oci-setup.md`](docs/runbooks/00-oci-setup.md) (API key, config, `.env`).
2. **Read [`docs/keeping-code-current.md`](docs/keeping-code-current.md)** - the course models (Command R / R+, Embed v3) have been retired; model IDs are settings here.
3. `python part2-genai-professional/m2_oci_genai_service/00_check_setup.py --ping`

## Course -> code map
| Course lesson | Where |
|---|---|
| Decoding, temperature, top-k/p, beam, penalties | `m1_llm_fundamentals/decoding_playground.py` |
| Self-attention | `m1_llm_fundamentals/self_attention_numpy.py` |
| k-shot, chain-of-thought, least-to-most | `m1_llm_fundamentals/prompting_patterns.py` |
| Prompt injection | `m1_llm_fundamentals/prompt_injection_lab.py` |
| Chat models, Inference API, config/auth troubleshooting | `m2_oci_genai_service/00-03` |
| Embedding models demo | `m2_oci_genai_service/04_embeddings_and_similarity.py` |
| Fine-tuning data (JSONL), metrics, sizing and pricing, custom endpoint | `m2_oci_genai_service/05-08` |
| LangChain components, RAG with Oracle 23ai, conversational RAG | `m3_rag_langchain/` |
| Agents: Object Storage and 23ai demos | `m4_genai_agents/`, runbooks 04-05 |

## Verification status
Offline code and tests pass; scripts that call OCI, Oracle DB, the Agents runtime, the Oracle SQL and the ADS lifecycle were written from the course demos and SDK signatures and **not executed against a live tenancy** in this repo. Details: [`docs/keeping-code-current.md`](docs/keeping-code-current.md).

## Safety
`.env`, private keys and OCI config files are git-ignored; `scripts/push_to_github.sh` refuses to commit them. Dedicated AI clusters are billed - use the cost calculator and delete resources after the labs.

## License
MIT - see [`LICENSE`](LICENSE). Course content belongs to Oracle University; notes here are personal study summaries.
