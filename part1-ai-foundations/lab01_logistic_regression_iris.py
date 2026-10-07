# %% [markdown]
# # Lab 01 - Logistic Regression on the Iris dataset (course Module 9, "ML Demo 1")
#
# Workflow from the course: **load data -> preprocess -> train -> evaluate -> predict**.
# Iris has 150 rows, 3 species (multi-class) and 4 features, so this is a
# *multi-class classification* problem solved with logistic regression.

# %%
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).parent / "data" / "iris.csv" if "__file__" in globals() else Path("data/iris.csv")

# %% [markdown]
# ## 1. Load the dataset and inspect it

# %%
iris_data = pd.read_csv(DATA)
print(iris_data.head())  # first 5 rows -> understand the structure

# %% [markdown]
# ## 2. Split into features (X) and label (y)
# The `Id` column is not a useful feature and `Species` is the target, so both are dropped from X.

# %%
X = iris_data.drop(columns=["Id", "Species"])
y = iris_data["Species"]
print("X shape:", X.shape, "| classes:", sorted(y.unique()))

# %% [markdown]
# ## 3. Create and train the model
# `LogisticRegression()` with default settings. `max_iter` is raised only to avoid a convergence warning.

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X, y)  # learns the relationship between features and labels

# %% [markdown]
# ## 4. Predict on new, unseen flowers
# (sepal length, sepal width, petal length, petal width) in cm.

# %%
new_flowers = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2], [6.3, 2.9, 5.6, 1.8]], columns=X.columns
)
for row, pred in zip(new_flowers.values.tolist(), model.predict(new_flowers)):
    print(f"{row} -> {pred}")

# %% [markdown]
# ## 5. Peek at the probabilities (what the sigmoid/softmax actually outputs)

# %%
proba = pd.DataFrame(model.predict_proba(new_flowers), columns=model.classes_).round(3)
print(proba)

if __name__ == "__main__":
    pass
