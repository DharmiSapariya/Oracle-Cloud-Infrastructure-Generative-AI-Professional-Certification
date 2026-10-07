# %% [markdown]
# # Lab 02 - Full ML pipeline: split, standardize, train, evaluate, predict (course Module 10, "ML Demo 2")
#
# New pieces compared with Lab 01:
# * `train_test_split(..., random_state=42)` - reproducible split
# * `StandardScaler` - mean 0 / std 1 so large-magnitude features can't dominate
# * `accuracy_score` - correct predictions / total predictions on **unseen** data (guards against overfitting)

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "data" / "iris.csv" if "__file__" in globals() else Path("data/iris.csv")

# %% [markdown]
# ## 1. Load + features/labels (same as Lab 01)

# %%
iris_data = pd.read_csv(DATA)
X = iris_data.drop(columns=["Id", "Species"])
y = iris_data["Species"]

# %% [markdown]
# ## 2. Train/test split
# `random_state` seeds the shuffle so the split (and your results) are reproducible.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), "training rows /", len(X_test), "test rows")

# %% [markdown]
# ## 3. Standardize features
# Fit the scaler on the **training** data only, then apply the same transform to the test data
# (fitting on test data would leak information).

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("train mean ~0:", X_train_scaled.mean(axis=0).round(3), "| std ~1:", X_train_scaled.std(axis=0).round(3))

# %% [markdown]
# ## 4. Train

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## 5. Evaluate on the held-out test set

# %%
y_pred = model.predict(X_test_scaled)
print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## 6. Predict on brand-new samples
# Crucially the new data goes through the **same** scaler before prediction.
# These three rows are illustrative (setosa-like, virginica-like, setosa-like).

# %%
new_samples = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3],
        [4.9, 3.1, 1.5, 0.1],
    ]
)
new_scaled = scaler.transform(pd.DataFrame(new_samples, columns=X.columns))
for row, pred in zip(new_samples.tolist(), model.predict(new_scaled)):
    print(f"{row} -> {pred}")

# %% [markdown]
# ## Why standardize? A tiny demonstration
# Without scaling, a feature measured in thousands (square footage) swamps one measured in single
# digits (bedrooms). Euclidean-style methods and gradient-based optimizers are both sensitive to this.

# %%
house = pd.DataFrame({"sqft": [1000, 2500, 5000], "bedrooms": [1, 3, 6]})
print(house.describe().loc[["mean", "std"]])
print(pd.DataFrame(StandardScaler().fit_transform(house), columns=house.columns).describe().loc[["mean", "std"]].round(3))
