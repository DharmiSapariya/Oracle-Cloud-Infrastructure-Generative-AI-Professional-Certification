"""Course demo: "Chat Models in Action" - parameter experimentation.

* temperature 0 (deterministic) vs 1 (more random) on the same poem prompt
* preamble override ("answer in a pirate tone") - changes style, is NOT fine-tuning
* top-p / top-k and the frequency / presence penalties
* context retention across turns: "expand on number two above"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import chat, get_inference_client, load_settings, require  # noqa: E402

POEM = "Write a four-line poem about Cohere Embed V3 in the style of Rudyard Kipling."

if __name__ == "__main__":
    s = load_settings()
    require(s, "compartment_id")
    c = get_inference_client(s)

    print("##### temperature 0 (run twice -> expect identical output)")
    for _ in range(2):
        print(chat(c, s, POEM, temperature=0, max_tokens=200), "\n")

    print("##### temperature 1 (run twice -> expect different output)")
    for _ in range(2):
        print(chat(c, s, POEM, temperature=1, max_tokens=200), "\n")

    print("##### preamble override: pirate tone (style change, not fine-tuning)")
    print(chat(c, s, "Plan a weekend in Lisbon.", preamble="You are a travel advisor. Answer in a pirate tone.", max_tokens=250))

    print("\n##### sampling knobs")
    q = "Name a country whose name starts with 'United'."
    print("top_k=3, top_p=0.75 :", chat(c, s, q, temperature=0.8, top_k=3, top_p=0.75, max_tokens=40))
    print("freq/presence pen.  :", chat(c, s, "Describe the ocean.", frequency_penalty=0.8, presence_penalty=0.6, max_tokens=120))

    print("\n##### context retention across turns")
    first = "Give me a short course outline with 5 numbered lessons on large language models."
    answer = chat(c, s, first, max_tokens=300)
    print(answer)
    history = [{"role": "USER", "message": first}, {"role": "CHATBOT", "message": answer}]
    print("\n--- follow-up ---")
    print(chat(c, s, "Expand on number two above.", chat_history=history, max_tokens=300))
