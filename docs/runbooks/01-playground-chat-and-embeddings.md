# Runbook 01 - Playground: chat models, parameters and embeddings

Console: *burger menu -> Analytics & AI -> AI Services -> Generative AI*. Code equivalents are in `part2-genai-professional/m2_oci_genai_service/`.

## A. Chat activities
- [ ] Open **Playground -> Chat**, pick a chat model, open **Model details** (token limits, regions).
- [ ] **Use case 1 - chat**: ask for a course outline for an OCI GenAI course with max output tokens = 600 (output truncates). Follow up: *"Expand on number two above."* -> confirms context retention.
- [ ] **Temperature**: run the poem prompt *"Write a four-line poem about Cohere Embed V3 in the style of Rudyard Kipling"* with temperature 0 (repeat -> identical) and temperature 1 (varies).
- [ ] **Preamble override**: *"You are a travel advisor, answer in a pirate tone"* -> tone changes (this is not fine-tuning).
- [ ] Try **top-k**, **top-p**, **frequency** and **presence penalty**.
- [ ] **Use case 2 - extraction**: *"Extract the entities mentioned in the text below"* + a paragraph.
- [ ] **Use case 3 - classification**: few-shot sentiment prompt; expect *positive* for "Learning a new language has been challenging but rewarding."
- [ ] Click **View code -> Python**, paste into Jupyter/VS Code -> equals `01_chat_inference.py`.

## B. Embedding activities
- [ ] Playground -> **Embedding**; pick an embedding model.
- [ ] Add a file of capital-city questions plus *"What is the smallest state in the United States?"* -> Run -> 2-D plot: capitals cluster, the state question is an outlier.
- [ ] Add *"smallest state in India"* (joins the outlier), *"largest state in the US"* (sits between).
- [ ] View code and print one embedding; count its dimensions (1,024 for the course-era English/Multilingual v3 models; check the model you use).
- [ ] Optional: the HR Help Center demo (41 articles) - skills articles cluster apart from leave/vacation articles.

Scripts: `02_chat_parameter_experiments.py`, `03_use_cases_extraction_and_classification.py`, `04_embeddings_and_similarity.py`.
