"""Course demo: "Creating an Endpoint for the Custom Model" - test the fine-tuned model.

Qualitative evaluation = compare base model vs custom model on the SAME prompts that were NOT in the
training data. Course result: the custom model returned correctly rephrased assistant utterances and
stayed consistent from temperature 0 to 5, while the base model (Cohere Command Light) produced
off-task text.

Setup: set OCI_CUSTOM_ENDPOINT_ID in .env to your hosting-cluster endpoint OCID and OCI_CHAT_API_FORMAT
to the base model's family (COHERE / COHEREV2 / GENERIC).  The base model is OCI_CHAT_MODEL_ID.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import build_chat_details, extract_text, get_inference_client, load_settings, require  # noqa: E402

TEST_FILE = Path(__file__).parent / "data" / "rephrasing_test.jsonl"
INSTRUCTION = "Turn this message to a virtual assistant into the correct action: "

if __name__ == "__main__":
    s = load_settings()
    require(s, "compartment_id", "custom_endpoint_id")
    c = get_inference_client(s)
    tests = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines()[:3]]

    for t in tests:
        prompt = INSTRUCTION + t["prompt"]
        print(f"\nREQUEST   : {t['prompt']}\nEXPECTED  : {t['completion']}")
        base = extract_text(c.chat(build_chat_details(s, prompt, max_tokens=60, temperature=0)))
        print(f"BASE      : {base.strip()}")
        for temp in (0, 1, 5):
            out = extract_text(c.chat(build_chat_details(s, prompt, max_tokens=60, temperature=temp, endpoint_id=s.custom_endpoint_id)))
            print(f"CUSTOM T={temp}: {out.strip()}")
