# Part 1 - AI Foundations labs

| Lab | Course module | Script | Notebook |
|---|---|---|---|
| 01 | ML Demo 1 - logistic regression on Iris | `lab01_logistic_regression_iris.py` | `notebooks/lab01_...ipynb` |
| 02 | ML Demo 2 - split, standardize, evaluate, predict | `lab02_iris_pipeline_scaling_eval.py` | `notebooks/lab02_...ipynb` |
| 03 | Deep learning - MLP on concentric circles | `lab03_mlp_circles_decision_boundary.py` | `notebooks/lab03_...ipynb` |
| 04 | OCI Data Science - ADS prepare/verify/save/deploy | `lab04_oci_data_science_ads_workflow.py` | `notebooks/lab04_...ipynb` |

```bash
pip install -r requirements.txt
python part1-ai-foundations/lab02_iris_pipeline_scaling_eval.py          # accuracy 1.0
python part1-ai-foundations/lab03_mlp_circles_decision_boundary.py --no-show   # saves outputs/lab03_decision_boundaries.png
```
Scripts use `# %%` cells (run as scripts, or cell-by-cell in VS Code). Regenerate notebooks with `python scripts/py_to_ipynb.py`. Lab 04 part B needs an OCI Data Science notebook session (`RUN_ADS = True`).
