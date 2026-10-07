# Runbook 02 - Dedicated AI clusters, custom model and endpoint

> Costs money: a hosting cluster commits you to 744 unit-hours/month. Estimate first:
> `python part2-genai-professional/m2_oci_genai_service/06_dedicated_cluster_cost_calculator.py`. Delete clusters when done.

## 0. Pre-check service limits
- [ ] Console -> *Governance & Administration -> Tenancy Management -> Limits, Quotas and Usage* -> filter "Generative AI". Limits are **0 by default**; request an increase for the unit type you need (Small/Large Cohere, Large Meta).
- [ ] The base model must match the unit type (small Cohere units -> Command Light in the demo).

## 1. Fine-tuning cluster
- [ ] Generative AI -> **Dedicated AI clusters -> Create**; compartment, name (e.g. `custom-fine-tuning`), purpose **Fine-tuning**, base model; confirm the 1 unit-hour minimum commitment; Create (a few minutes).

## 2. Hosting cluster
- [ ] Create another cluster, purpose **Hosting**; note the 744 unit-hour monthly commitment. A T-Few hosting cluster shows capacity **50** endpoints.

## 3. Training data
- [ ] Prepare CSV (human request, assistant utterance) -> `python .../05_prepare_finetune_dataset.py build`.
- [ ] Validate: `python .../05_prepare_finetune_dataset.py validate <file>.jsonl` (JSONL, `prompt` + `completion`, UTF-8).
- [ ] Upload the JSONL to an **Object Storage bucket** and add IAM policies so Generative AI can read it.

## 4. Custom model (fine-tuning)
- [ ] **Custom models -> Create model** -> new model -> base model -> method **T-Few** (or Vanilla) -> pick the fine-tuning cluster -> leave hyper-parameters at defaults -> choose the training file (console previews 5 lines) -> **Submit**.
- [ ] Review **accuracy** (0.98 in the demo) and **loss** (trending to 0).

## 5. Endpoint
- [ ] **Endpoints -> Create endpoint** -> choose the custom model -> choose the hosting cluster -> optional content moderation -> Create (status Active).
- [ ] Test in Playground: compare **base vs custom** on prompts *not* in the training data, at temperature 0, 1 and 5 (`08_custom_model_inference.py` does this from code).

## 6. Clean up
- [ ] Delete endpoint -> delete hosting cluster -> delete fine-tuning cluster -> delete custom model -> remove the bucket if unused.
