"""Every relative link in every Markdown file must point at something that exists."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FILES = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts and ".venv" not in p.parts]


@pytest.mark.parametrize("md", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(md):
    broken = []
    for target in LINK.findall(md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#")[0]
        if path and not (md.parent / path).resolve().exists():
            broken.append(target)
    assert not broken, f"{md.relative_to(ROOT)} has broken links: {broken}"
