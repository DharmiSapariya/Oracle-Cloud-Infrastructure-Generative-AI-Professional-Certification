# %% [markdown]
# # Lab 03 - Deep learning demo: classifying circular data with an MLP (course Module 15)
#
# `make_circles` produces two concentric rings that **cannot** be separated by a straight line, so
# a linear model fails. A Multilayer Perceptron with ReLU hidden units learns a non-linear boundary,
# and the boundary gets better as you add neurons.
#
# Parameters of `make_circles` (as in the course):
# * `n_samples=300` -> 150 points per class
# * `noise`   -> how scattered the points are
# * `factor`  -> ratio between inner and outer circle radius (0.5 vs 0.7 changes the gap)
# * `random_state` -> reproducible data

# %%
import argparse
from pathlib import Path

import matplotlib

if "__file__" not in globals():  # running as a notebook: use the inline backend, but never block on a window
    pass

import numpy as np
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

OUT_DIR = (Path(__file__).parent / "outputs") if "__file__" in globals() else Path("outputs")

# %%
def make_data(n_samples=300, noise=0.1, factor=0.5, random_state=42):
    return make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)


def train_mlp(X, y, hidden_layer_size, random_state=42, max_iter=2000):
    """One hidden layer with `hidden_layer_size` neurons, ReLU activation (non-linear boundary)."""
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation="relu",
        solver="lbfgs",  # converges reliably on small datasets like this one
        max_iter=max_iter,
        random_state=random_state,  # initial weights/biases
    )
    clf.fit(X, y)
    return clf


def decision_grid(X, clf, resolution=100, pad=0.5):
    """Predict labels on a 100x100 grid across the data range -> used to draw the decision boundary."""
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, zz


def update_plot(hidden_layer_size, X=None, y=None, ax=None):
    """Same name as the course's slider callback. Returns training accuracy."""
    import matplotlib.pyplot as plt

    if X is None:
        X, y = make_data()
    clf = train_mlp(X, y, hidden_layer_size)
    xx, yy, zz = decision_grid(X, clf)
    ax = ax or plt.gca()
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="RdYlGn")  # regions for label 0 vs 1
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", edgecolor="k", s=18, label="label 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="green", edgecolor="k", s=18, label="label 1")
    acc = clf.score(X, y)
    ax.set_title(f"{hidden_layer_size} hidden neuron(s) - acc {acc:.2f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return acc


def main(no_show=False):
    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = make_data()
    sizes = [1, 2, 3, 4, 5, 6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    print("hidden neurons -> training accuracy")
    for ax, n in zip(axes.ravel(), sizes):
        acc = update_plot(n, X, y, ax)
        print(f"  {n} -> {acc:.3f}")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("MLP decision boundary vs number of hidden neurons (make_circles)")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "lab03_decision_boundaries.png"
    fig.savefig(out, dpi=130)
    print("saved", out)
    if not no_show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    main(ap.parse_known_args()[0].no_show)  # parse_known_args: harmless inside Jupyter

# %% [markdown]
# ## Interactive slider (notebook only)
# In Jupyter you can reproduce the course's `hidden layer size` slider:
# ```python
# from ipywidgets import interact, IntSlider
# interact(lambda n: update_plot(n), n=IntSlider(min=1, max=8, value=1))
# ```
