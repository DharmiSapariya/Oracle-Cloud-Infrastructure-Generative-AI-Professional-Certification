"""Course demo: "OCI Generative AI Inference API" - from console to code.

Playground example: generate a job description for a "senior data visualization expert", then
"View Code" -> Python. This script is that generated code, cleaned up.

Response fields to look for (Cohere v1 format): status 200, chat history (roles USER / CHATBOT),
finish reason "COMPLETE", and the text itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import build_chat_details, extract_text, get_inference_client, load_settings, require  # noqa: E402

PROMPT = "Generate a job description for a data visualization expert with three qualifications."

if __name__ == "__main__":
    settings = load_settings()
    require(settings, "compartment_id")
    client = get_inference_client(settings)

    details = build_chat_details(settings, PROMPT, max_tokens=600, temperature=0)
    response = client.chat(details)

    print("status      :", response.status)  # 200 -> success
    print("api format  :", response.data.chat_response.api_format)
    print("model       :", response.data.model_id, "| version:", response.data.model_version)
    print("finish      :", getattr(response.data.chat_response, "finish_reason", None))
    print("\n" + extract_text(response))
