import pytest

pytest.importorskip("oci")
from common.genai_common import (Settings, build_chat_details, build_embed_details, detect_api_format,
                                 )

S = Settings(compartment_id="ocid1.compartment.oc1..test")


def test_api_format_detection():
    assert detect_api_format("cohere.command-r-plus-08-2024") == "COHERE"
    assert detect_api_format("cohere.command-a-03-2025") == "COHEREV2"
    assert detect_api_format("meta.llama-3.3-70b-instruct") == "GENERIC"
    assert detect_api_format("anything", "generic") == "GENERIC"


def test_cohere_v1_request_carries_parameters_and_history():
    d = build_chat_details(S, "hi", model_id="cohere.command-r-plus-08-2024", max_tokens=600, temperature=0,
                           top_p=0.75, top_k=5, preamble="pirate",
                           chat_history=[{"role": "USER", "message": "q"}, {"role": "CHATBOT", "message": "a"}])
    r = d.chat_request
    assert (r.message, r.max_tokens, r.temperature, r.top_p, r.top_k, r.preamble_override) == ("hi", 600, 0, 0.75, 5, "pirate")
    assert len(r.chat_history) == 2 and d.serving_mode.serving_type == "ON_DEMAND" and d.compartment_id == S.compartment_id


def test_v2_and_generic_put_preamble_first_and_question_last():
    for model in ("cohere.command-a-03-2025", "meta.llama-3.3-70b-instruct"):
        msgs = build_chat_details(S, "final?", model_id=model, preamble="sys").chat_request.messages
        assert msgs[0].role == "SYSTEM" and msgs[-1].role == "USER" and msgs[-1].content[0].text == "final?"


def test_dedicated_endpoint_switches_serving_mode():
    d = build_chat_details(S, "x", model_id="cohere.command-r-16k", endpoint_id="ocid1.generativeaiendpoint.oc1..e")
    assert d.serving_mode.serving_type == "DEDICATED" and d.serving_mode.endpoint_id.endswith("..e")


def test_embed_request_and_96_limit():
    d = build_embed_details(S, ["a", "b"], input_type="SEARCH_QUERY")
    assert d.input_type == "SEARCH_QUERY" and d.truncate == "END" and d.inputs == ["a", "b"]
    with pytest.raises(ValueError):
        build_embed_details(S, ["x"] * 97)
