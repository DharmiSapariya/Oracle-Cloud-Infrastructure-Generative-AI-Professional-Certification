"""Convert the `# %%` percent-format lab scripts into .ipynb notebooks (stored in part1-ai-foundations/notebooks)."""
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "part1-ai-foundations"


def convert(py: Path) -> nbformat.NotebookNode:
    cells, kind, buf = [], None, []

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in py.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^# %%(?: \[(markdown)\])?", line)
        if m:
            flush()
            kind, buf = ("markdown" if m.group(1) else "code"), []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    out = SRC / "notebooks"
    out.mkdir(exist_ok=True)
    for py in sorted(SRC.glob("lab0*.py")):
        # notebooks run from part1-ai-foundations/notebooks, so data lives one level up
        nb = convert(py)
        for c in nb.cells:
            if c.cell_type == "code":
                c.source = c.source.replace('Path(__file__).parent / "data" / "iris.csv" if "__file__" in globals() else Path("data/iris.csv")',
                                            'Path("../data/iris.csv")')
        nbformat.write(nb, out / (py.stem + ".ipynb"))
        print("wrote", out / (py.stem + ".ipynb"))
