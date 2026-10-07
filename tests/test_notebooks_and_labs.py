"""Execute every generated notebook's code cells headlessly - guarantees notebooks and scripts stay in sync."""
import os
from pathlib import Path

import matplotlib
import nbformat
import pytest

matplotlib.use("Agg")
NB_DIR = Path(__file__).resolve().parents[1] / "part1-ai-foundations" / "notebooks"


@pytest.mark.parametrize("nb_path", sorted(NB_DIR.glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook_runs(nb_path, monkeypatch):
    monkeypatch.chdir(NB_DIR)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    nb = nbformat.read(nb_path, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    exec(compile(code, str(nb_path), "exec"), {"__name__": "__notebook__"})


def test_mlp_more_neurons_beat_one_neuron():
    import importlib.util

    root = Path(__file__).resolve().parents[1] / "part1-ai-foundations"
    spec = importlib.util.spec_from_file_location("lab03", root / "lab03_mlp_circles_decision_boundary.py")
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    X, y = lab.make_data()
    acc = {n: lab.train_mlp(X, y, n).score(X, y) for n in (1, 4)}
    assert acc[1] < 0.6 < 0.9 < acc[4]


def test_iris_pipeline_reaches_course_accuracy():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(NB_DIR.parent / "data" / "iris.csv")
    assert df.shape == (150, 6) and df.Species.nunique() == 3
    X, y = df.drop(columns=["Id", "Species"]), df.Species
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    assert accuracy_score(yte, m.predict(sc.transform(Xte))) == 1.0
