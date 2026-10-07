"""Course demo (Module 4, step "Chat with the agent") - but from code, via the Agent Runtime API.

Console flow: Generative AI Agents -> Chat -> pick agent + endpoint. Programmatic flow:
  1. create a session  (a session keeps context: the demo's 3rd question never named the course,
     yet the agent understood it - that is session memory; default idle timeout 3600 s)
  2. chat(agent_endpoint_id, ChatDetails(user_message, session_id))
  3. read the answer + citations (title, doc id, source location) - groundedness you can trace

Requires an agent endpoint (see docs/runbooks/04-genai-agents-object-storage.md) and OCI_AGENT_ENDPOINT_ID in .env.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import load_oci_config, load_settings, require  # noqa: E402

DEMO_QUESTIONS = [
    "Please tell me about Oracle free tier",
    "How many modules are there in Oracle AI Foundations course?",
    "Who are the instructors for this course?",  # relies on session memory
]


def format_citations(message) -> list[str]:
    cites = getattr(message.content, "citations", None) or []
    return [f"{c.title or '?'} (doc {c.doc_id or '?'})" for c in cites]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*", default=DEMO_QUESTIONS)
    args = ap.parse_args()

    import oci

    s = load_settings()
    require(s, "agent_endpoint_id")
    client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(
        config=load_oci_config(s), service_endpoint=s.agent_runtime_endpoint
    )
    models = oci.generative_ai_agent_runtime.models

    session = client.create_session(
        models.CreateSessionDetails(display_name="cert-lab-session", description="course demo session"),
        s.agent_endpoint_id,
    ).data
    for q in args.questions:
        reply = client.chat(
            s.agent_endpoint_id,
            models.ChatDetails(user_message=q, session_id=session.id, should_stream=False),
        ).data
        print(f"\nQ: {q}\nA: {reply.message.content.text}")
        for c in format_citations(reply.message):
            print("   cited:", c)
