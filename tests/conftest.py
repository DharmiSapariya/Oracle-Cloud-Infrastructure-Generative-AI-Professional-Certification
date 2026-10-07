import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "part2-genai-professional", ROOT / "part2-genai-professional" / "m3_rag_langchain",
          ROOT / "part2-genai-professional" / "m1_llm_fundamentals", ROOT / "part2-genai-professional" / "m2_oci_genai_service"):
    sys.path.insert(0, str(p))


def load_module(name, path):
    """Import a file (even with a digit-leading name like 05_x.py) and register it so @dataclass works."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
