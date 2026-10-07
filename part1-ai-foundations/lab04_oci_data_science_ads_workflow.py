# %% [markdown]
# # Lab 04 - OCI Data Science: train -> prepare -> verify -> save -> deploy -> predict (course Module 26)
#
# The course demo ran this inside an **OCI Data Science notebook session** using the Accelerated Data
# Science (ADS) SDK and a `generalml` Conda environment (Python 3.8 at the time).
#
# * Part A (runs anywhere): train a RandomForest on Iris exactly like the demo.
# * Part B (needs an OCI Data Science notebook session + `pip install oracle-ads`): the ADS model lifecycle.
#   Not executed in the offline CI - run it in your own tenancy.
#
# Lifecycle shown by `model.summary_status()`: initiate -> prepare -> verify -> save -> deploy -> predict.

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "iris.csv" if "__file__" in globals() else Path("data/iris.csv")

# %% [markdown]
# ## Part A - train the model (as in the demo)

# %%
iris = pd.read_csv(DATA)
X = iris.drop(columns=["Id", "Species"])
y = iris["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("test accuracy:", round(clf.score(X_test, y_test), 3))

# %% [markdown]
# ## Part B - ADS model lifecycle (run inside OCI Data Science)
# Set `RUN_ADS = True` in your notebook session.

# %%
RUN_ADS = False

if RUN_ADS:
    import tempfile

    import ads
    from ads.model.framework.sklearn_model import SklearnModel

    ads.set_auth(auth="resource_principal")  # notebook sessions authenticate with a resource principal

    model = SklearnModel(estimator=clf, artifact_dir=tempfile.mkdtemp())

    # 1) prepare - auto-generates score.py, runtime.yaml, model.pkl ... (no manual config/coding)
    model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",  # pick the env you installed via Environment Explorer
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=X_train,
        y_sample=y_train,
        as_onnx=False,
        force_overwrite=True,
    )

    # 2) status table: which lifecycle steps are done / available
    print(model.summary_status())

    # 3) verify - simulates deployment locally for debugging (the demo ran the first 10 test rows)
    print(model.verify(X_test.head(10)))

    # 4) save -> Model Catalog (stores metadata + provenance, shareable across the team)
    model_id = model.save(display_name="iris-random-forest")
    print("saved model:", model_id)

    # 5) deploy -> managed HTTP endpoint (the demo described but did not execute this step)
    # model.deploy(display_name="iris-rf-deployment")
    # 6) predict via the deployed endpoint
    # print(model.predict(X_test.head(5)))
