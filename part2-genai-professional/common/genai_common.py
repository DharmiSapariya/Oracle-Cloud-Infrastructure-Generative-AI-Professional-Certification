"""Shared configuration + thin wrappers around the OCI Generative AI Inference API.

Mirrors the steps of the course's "Inference API" demo:

  1. import oci
  2. compartment id + config profile -> load config file (authentication)
  3. service endpoint (region specific)
  4. GenerativeAiInferenceClient(endpoint, retry strategy = none, timeout = (10s connect, 240s read))
  5. ChatDetails  (holds the whole request)
  6. a chat request object (Cohere / generic) with message, max_tokens, temperature, ...
  7. serving mode = on-demand + model id
  8. send -> response (status 200, chat history, finish reason, text ...)

Request *building* is separated from *sending* so the building logic can be unit-tested offline.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- settings
def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without requiring python-dotenv."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    config_file: str = "~/.oci/config"
    profile: str = "DEFAULT"
    compartment_id: str = ""
    endpoint: str = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
    chat_model_id: str = "cohere.command-a-03-2025"
    embed_model_id: str = "cohere.embed-v4.0"
    chat_api_format: str = ""  # "", COHERE, COHEREV2 or GENERIC (auto-detected from model id when empty)
    db_user: str = ""
    db_password: str = ""
    db_dsn: str = ""
    table_name: str = "RAG_DOCS"
    custom_endpoint_id: str = ""  # hosting-cluster endpoint of a fine-tuned (custom) model
    agent_endpoint_id: str = ""
    agent_runtime_endpoint: str = "https://agent-runtime.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
    extra: dict = field(default_factory=dict)


def load_settings() -> Settings:
    """Read .env (if present) + environment variables into a Settings object."""
    _load_dotenv(REPO_ROOT / ".env")
    g = os.environ.get
    return Settings(
        config_file=os.path.expanduser(g("OCI_CONFIG_FILE", "~/.oci/config")),
        profile=g("OCI_CONFIG_PROFILE", "DEFAULT"),
        compartment_id=g("OCI_COMPARTMENT_ID", ""),
        endpoint=g("OCI_GENAI_ENDPOINT", Settings.endpoint),
        chat_model_id=g("OCI_CHAT_MODEL_ID", Settings.chat_model_id),
        embed_model_id=g("OCI_EMBED_MODEL_ID", Settings.embed_model_id),
        chat_api_format=g("OCI_CHAT_API_FORMAT", "").upper(),
        db_user=g("DB_USER", ""),
        db_password=g("DB_PASSWORD", ""),
        db_dsn=g("DB_DSN", ""),
        table_name=g("VECTOR_TABLE_NAME", "RAG_DOCS"),
        custom_endpoint_id=g("OCI_CUSTOM_ENDPOINT_ID", ""),
        agent_endpoint_id=g("OCI_AGENT_ENDPOINT_ID", ""),
        agent_runtime_endpoint=g("OCI_AGENT_RUNTIME_ENDPOINT", Settings.agent_runtime_endpoint),
    )


def require(settings: Settings, *names: str) -> None:
    """Exit with a friendly message if required settings are missing/placeholder."""
    missing = [n for n in names if not getattr(settings, n) or "replace_me" in str(getattr(settings, n))]
    if missing:
        sys.exit(
            "Missing settings: " + ", ".join(missing) + "\nCopy .env.example to .env and fill them in "
            "(see docs/runbooks/00-oci-setup.md)."
        )


# --------------------------------------------------------------------------- auth + clients
def load_oci_config(settings: Settings) -> dict:
    import oci

    config = oci.config.from_file(settings.config_file, settings.profile)
    oci.config.validate_config(config)  # fails early on a bad key / fingerprint / tenancy
    return config


def get_inference_client(settings: Settings):
    import oci

    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=load_oci_config(settings),
        service_endpoint=settings.endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),  # as in the demo
        timeout=(10, 240),  # (connect, read) seconds
    )


# --------------------------------------------------------------------------- chat
def detect_api_format(model_id: str, override: str = "") -> str:
    """Pick the request format for a model.

    * cohere.command-r*  -> COHERE   (the format used in the course demo)
    * other cohere.*     -> COHEREV2 (Command A family)
    * everything else    -> GENERIC  (Meta Llama, etc.)
    """
    if override:
        return override.upper()
    mid = model_id.lower()
    if mid.startswith("cohere.command-r") or mid in ("cohere.command", "cohere.command-light"):
        return "COHERE"
    if mid.startswith("cohere."):
        return "COHEREV2"
    return "GENERIC"


def build_chat_details(
    settings: Settings,
    message: str,
    *,
    max_tokens: int = 600,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    preamble: Optional[str] = None,
    chat_history: Optional[Sequence[dict]] = None,
    model_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
):
    """Build the ChatDetails request object (no network call).

    Pass `endpoint_id` to call a fine-tuned model hosted on a dedicated AI cluster (DedicatedServingMode)
    instead of an on-demand pretrained model; set OCI_CHAT_API_FORMAT to match the custom model's base.

    `preamble` is the "preamble override" from the playground: it changes tone/behaviour (e.g.
    "answer in a pirate tone") without any fine-tuning. For COHEREV2/GENERIC models it is sent as a
    system message. `chat_history` is a list of {"role": "USER"|"CHATBOT", "message": str}.
    """
    from oci.generative_ai_inference import models

    model_id = model_id or settings.chat_model_id
    fmt = detect_api_format(model_id, settings.chat_api_format)
    history = list(chat_history or [])

    if fmt == "COHERE":
        req = models.CohereChatRequest()
        req.message = message
        req.max_tokens = max_tokens
        req.temperature = temperature
        req.frequency_penalty = frequency_penalty
        req.presence_penalty = presence_penalty
        if top_p is not None:
            req.top_p = top_p
        if top_k is not None:
            req.top_k = top_k
        if preamble:
            req.preamble_override = preamble
        if history:
            req.chat_history = [
                models.CohereUserMessage(message=h["message"])
                if h["role"].upper() == "USER"
                else models.CohereChatBotMessage(message=h["message"])
                for h in history
            ]
    elif fmt == "COHEREV2":
        msgs = []
        if preamble:
            msgs.append(models.CohereSystemMessageV2(content=[models.CohereTextContentV2(text=preamble)]))
        for h in history:
            cls = models.CohereUserMessageV2 if h["role"].upper() == "USER" else models.CohereAssistantMessageV2
            msgs.append(cls(content=[models.CohereTextContentV2(text=h["message"])]))
        msgs.append(models.CohereUserMessageV2(content=[models.CohereTextContentV2(text=message)]))
        req = models.CohereChatRequestV2(messages=msgs, max_tokens=max_tokens, temperature=temperature)
        req.frequency_penalty = frequency_penalty
        req.presence_penalty = presence_penalty
        if top_p is not None:
            req.top_p = top_p
        if top_k is not None:
            req.top_k = top_k
    else:  # GENERIC (Meta Llama and other non-Cohere models)
        msgs = []
        if preamble:
            msgs.append(models.SystemMessage(content=[models.TextContent(text=preamble)]))
        for h in history:
            cls = models.UserMessage if h["role"].upper() == "USER" else models.AssistantMessage
            msgs.append(cls(content=[models.TextContent(text=h["message"])]))
        msgs.append(models.UserMessage(content=[models.TextContent(text=message)]))
        req = models.GenericChatRequest(messages=msgs, max_tokens=max_tokens, temperature=temperature)
        req.frequency_penalty = frequency_penalty
        req.presence_penalty = presence_penalty
        if top_p is not None:
            req.top_p = top_p
        if top_k is not None:
            req.top_k = top_k

    details = models.ChatDetails()
    if endpoint_id:
        details.serving_mode = models.DedicatedServingMode(endpoint_id=endpoint_id)  # your hosting-cluster endpoint
    else:
        details.serving_mode = models.OnDemandServingMode(model_id=model_id)  # on-demand, not a dedicated cluster
    details.chat_request = req
    details.compartment_id = settings.compartment_id
    return details


def extract_text(chat_response) -> str:
    """Pull the answer text out of a chat response regardless of API format."""
    inner = chat_response.data.chat_response
    if hasattr(inner, "text") and inner.text is not None:  # COHERE v1
        return inner.text
    msg = getattr(inner, "message", None)  # COHEREV2
    if msg is not None and getattr(msg, "content", None):
        return "".join(getattr(c, "text", "") or "" for c in msg.content)
    choices = getattr(inner, "choices", None)  # GENERIC
    if choices:
        return "".join(getattr(c, "text", "") or "" for c in choices[0].message.content)
    return str(inner)


def chat(client, settings: Settings, message: str, **kwargs) -> str:
    """Send a chat request and return just the text."""
    response = client.chat(build_chat_details(settings, message, **kwargs))
    return extract_text(response)


# --------------------------------------------------------------------------- embeddings
def build_embed_details(settings: Settings, texts: Sequence[str], input_type: str = "SEARCH_DOCUMENT"):
    from oci.generative_ai_inference import models

    if len(texts) > 96:
        raise ValueError("The embedding API accepts at most 96 inputs per run (course: 'max 96 inputs per run').")
    details = models.EmbedTextDetails()
    details.inputs = list(texts)
    details.serving_mode = models.OnDemandServingMode(model_id=settings.embed_model_id)
    details.compartment_id = settings.compartment_id
    details.input_type = input_type  # SEARCH_DOCUMENT for corpus text, SEARCH_QUERY for questions
    details.truncate = "END"  # inputs longer than the model limit are cut instead of failing
    return details


def embed(client, settings: Settings, texts: Iterable[str], input_type: str = "SEARCH_DOCUMENT") -> list[list[float]]:
    """Embed any number of texts, batching to respect the 96-inputs-per-call limit."""
    texts = list(texts)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 96):
        resp = client.embed_text(build_embed_details(settings, texts[i : i + 96], input_type))
        vectors.extend(resp.data.embeddings)
    return vectors
