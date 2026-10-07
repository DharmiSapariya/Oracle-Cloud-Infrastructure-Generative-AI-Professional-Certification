# 05 - OCI AI services, Data Science and AI infrastructure (Modules 3, 23-27, 33-35)

## Oracle AI stack
Infrastructure and data at the base (Oracle Database, Autonomous DB, MySQL HeatWave) -> ML services -> AI services -> SaaS apps (ERP, HCM, CX) with embedded AI. AI services are **prebuilt models consumed through an API** with no infrastructure to manage.
Access: **Console**, **REST API**, **SDKs** (Java, Python, TypeScript/JS, .NET, Go, Ruby), **CLI**.

| Service | What it does |
|---|---|
| **Language** | language detection (**75 languages**), named entity recognition (14 entity types), sentiment at **aspect / sentence / document** level, key phrases, text classification (**600 categories**), PII detection, neural **translation**; custom NER + classification models |
| **Speech** | speech-to-text from Object Storage; timestamps, punctuation, normalisation ("100 percent" -> "100%"), per-word confidence, **SRT** captions, batch jobs, hours of audio in under 10 min (chunk + parallel), profanity filter: *remove* (asterisks), *mask* (keep first letter), *tag*; English, Spanish, Portuguese |
| **Vision** | image classification, object detection (bounding boxes + confidence), text detection, custom models |
| **Document Understanding** | OCR (also handwriting, tilted pages), document classification (10 types), language detection, table extraction, key-value extraction (13 receipt fields); PDF/JPEG/PNG/TIFF |
| **Digital Assistant** | build conversational interfaces; routes requests to *skills*, handles interruptions, disambiguation and exit |

## OCI Data Science (Module 24)
Managed service for the full ML lifecycle in JupyterLab. Principles: **Accelerated**, **Collaborative**, **Enterprise-grade**.
Key terms: **Project**, **Notebook session** (managed CPU/GPU compute), **Conda environment**, **ADS SDK** (Oracle's Python library), **Model catalog** (store/track/share models + provenance), **Model deployment** (HTTP endpoint), **Jobs**.
Demo lifecycle with ADS: `prepare()` -> `summary_status()` -> `verify()` -> `save()` -> `deploy()` -> `predict()`. Lab: [Lab 04](../../part1-ai-foundations/lab04_oci_data_science_ads_workflow.py).

## AI infrastructure (Modules 25, 27)
- **RDMA**: machine-to-machine transfer bypassing the CPU (low latency, high bandwidth). **RoCE** = RDMA over Converged Ethernet, OCI's fabric.
- **Supercluster**: up to 100,000+ GPUs in one RDMA network; each node (8 GPUs, NVLink) attaches at 1.6 Tbps (~200 Gbps/GPU); non-blocking, three-tier Clos. Latency: ~6.5 us inside a block, ~20 us worst case; still **lossless** via tuned buffers and congestion notification. Control-plane **placement** and **network-locality hints** keep traffic local (about 50% stays on the top-of-rack switch, 85% inside a block).
- **GPUs**: parallel cores suit repetitive tensor math. A100 (Ampere, tensor cores) - H100 (Hopper, transformer engine) - H200 (more memory) - Blackwell / GB200 (Grace CPU + Blackwell GPU). **AI Quick Actions** in Data Science deploy/fine-tune open LLMs (vLLM, NVIDIA NIM, Text Embedding Inference containers).

## Responsible AI (Module 28)
Trustworthy AI is **lawful, ethical, robust**. Ethics principles: help humans and allow oversight - never cause harm - be transparent, fair and explainable. Process: governance -> policies/procedures -> monitoring and evaluation. Healthcare challenges: bias from skewed training data, explainability, ongoing evaluation.
