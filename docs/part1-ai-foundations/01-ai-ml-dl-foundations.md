# 01 - AI, ML and DL foundations (course Modules 0-4)

Hands-on: [`part1-ai-foundations/`](../../part1-ai-foundations/) - Labs 01-04.

## Why AI skills matter (Microsoft/LinkedIn survey, 31,000 people in 31 countries)
- **Employees want AI now:** 75% of knowledge workers already use AI; 46% started under 6 months ago.
- **Aptitude over experience:** 66% of leaders would not hire someone without AI skills; 71% prefer a less experienced candidate *with* AI skills.
- **AI is mainstream:** business, software and creative roles are all skilling up.

## The AI stack
| Layer | Definition |
|---|---|
| **AI** | machines imitating the cognitive and problem-solving abilities of human intelligence |
| **ML** | subset of AI; algorithms learn from past data to predict outcomes / find trends |
| **DL** | subset of ML; learns from complex data with (deep) neural networks |
| **Generative AI** | AI that creates new content (text, images, audio, code); neural-network based |

> The course frames GenAI both as "a subset of AI" (intro) and "a subset of ML / deep learning" (later modules). For the exam: AI ⊃ ML ⊃ DL ⊃ GenAI.

**AGI vs AI.** AGI would replicate *any* human capability without intervention; today's AI applies AGI-like abilities to *specific, narrow* objectives.

**Why we need AI:** data volume exceeds human capacity; automate routine decisions (credit/loan approval, insurance claims, recommendations); act as a "smart friend" that writes stories, code, music.

## AI domains and examples
Language (translation) - Vision (image classification) - Speech (text-to-speech) - Recommendations (cross-selling) - Anomaly detection (fraud) - Reinforcement learning (self-driving cars) - Forecasting (weather) - Generative (image from text).

## AI tasks and data
**Language** - text is sequential. *Tokenization* turns words into numbers, *padding* equalises lengths, *similarity* is measured with dot or cosine similarity, *embeddings* place similar text close together. Architectures: **RNN** (sequential, hidden state) -> **LSTM** (gates keep context) -> **Transformer** (parallel, self-attention).
**Speech** - digitised as time samples; **sample rate** (44.1 kHz = 44,100 samples/s, CD standard) and **bit depth**. A single sample means little; correlated samples carry meaning.
**Vision** - images are pixels (a lone pixel is meaningless). **CNN** detects hierarchical patterns, **YOLO** detects objects, **GAN** generates realistic images. Facial recognition is a major use.
**Other:** anomaly detection and forecasting use time-series data; recommendations use similar users/products.

## AI vs ML vs DL (analogies)
AI = self-driving car deciding like a human - ML = spam filter that learns from your behaviour - DL = image recognition that finds cats in photos.

## Neural networks as function approximation
Neural networks are a supervised-ML algorithm best understood as estimating an unknown function from known examples. DL's core idea: **extract features directly from raw data** (pixels) instead of hand-written rules.
