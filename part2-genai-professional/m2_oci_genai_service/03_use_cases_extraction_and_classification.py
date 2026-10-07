"""Course demo use cases 2 + 3: data extraction and few-shot text classification (sentiment)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import chat, get_inference_client, load_settings, require  # noqa: E402

# Short original paragraph (the course used a Wikipedia excerpt about NVIDIA).
TEXT = (
    "NVIDIA is a technology company headquartered in Santa Clara, California. It designs graphics "
    "processing units used for gaming, data centres and artificial intelligence, and its CUDA software "
    "platform lets developers run general-purpose computations on those GPUs."
)

FEW_SHOT_SENTIMENT = """Classify the sentiment of the message as positive, negative or neutral.

Message: I can't stand waiting in long lines at the grocery store.
Sentiment: negative

Message: The new update made the app so much faster!
Sentiment: positive

Message: The meeting is at 3 pm on Thursday.
Sentiment: neutral

Message: Learning a new language has been challenging but rewarding.
Sentiment:"""

if __name__ == "__main__":
    s = load_settings()
    require(s, "compartment_id")
    c = get_inference_client(s)
    print("### Extraction\n" + chat(c, s, f"Extract the entities mentioned in the text below.\n\n{TEXT}", max_tokens=300))
    print("\n### Few-shot classification (expected: positive)\n" + chat(c, s, FEW_SHOT_SENTIMENT, max_tokens=5))
