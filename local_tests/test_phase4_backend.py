"""Phase 4 tests for the OpenAI-compatible backend (local Ollama path).

Covers: Gemini-style -> JSON Schema conversion (nested types), transient-error
classification and the bounded retry decorator, request kwargs
(max_output_tokens, no SDK retries), and the malformed tool-JSON path.

Runs under pytest, or directly: python local_tests/test_phase4_backend.py
"""

import os
import sys
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import httpx
import openai

from utils.agent_infrastructure import vlm_backends as vb
from agents.tools.registry import build_tools_for_scaffold

_REQ = httpx.Request("POST", "http://host.docker.internal:11434/v1/responses")


def _status_error(cls, status):
    return cls(f"status {status}", response=httpx.Response(status, request=_REQ), body=None)


def _make_backend(env=None, **kwargs):
    base_env = {"OPENAI_API_KEY": "ollama", "OPENAI_BASE_URL": "http://host.docker.internal:11434/v1"}
    base_env.update(env or {})
    cleared = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    cleared.update({k: v for k, v in base_env.items() if v is not None})
    with mock.patch.dict(os.environ, cleared, clear=True):
        return vb.OpenAIBackend(
            "qwen3-vl:8b-instruct",
            tools=kwargs.pop("tools", build_tools_for_scaffold("simple")),
            system_instruction="sys",
            **kwargs,
        )


# --------------------------------------------------------------------------
# Schema conversion
# --------------------------------------------------------------------------

def test_schema_scalar_types():
    assert vb.gemini_schema_to_json_schema({"type_": "STRING"}) == {"type": "string"}
    assert vb.gemini_schema_to_json_schema({"type_": "INTEGER"}) == {"type": "integer"}
    assert vb.gemini_schema_to_json_schema({"type_": "NUMBER"}) == {"type": "number"}
    assert vb.gemini_schema_to_json_schema({"type_": "BOOLEAN"}) == {"type": "boolean"}
    # JSON-Schema style input is accepted as-is.
    assert vb.gemini_schema_to_json_schema({"type": "number"}) == {"type": "number"}


def test_schema_array_of_objects_nested():
    schema = {
        "type_": "OBJECT",
        "properties": {
            "entries": {
                "type_": "ARRAY",
                "description": "list",
                "items": {
                    "type_": "OBJECT",
                    "properties": {
                        "id": {"type_": "STRING"},
                        "importance": {"type_": "INTEGER"},
                        "weight": {"type_": "NUMBER"},
                        "tags": {"type_": "ARRAY", "items": {"type_": "STRING"}},
                        "meta": {"type_": "OBJECT", "properties": {"ok": {"type_": "BOOLEAN"}}, "required": ["ok"]},
                    },
                    "required": ["id", "not_a_property"],
                },
            },
            "mode": {"type_": "STRING", "enum": ["a", "b"]},
        },
        "required": ["entries"],
    }
    out = vb.gemini_schema_to_json_schema(schema)
    entries = out["properties"]["entries"]
    assert entries["type"] == "array" and entries["description"] == "list"
    item = entries["items"]
    assert item["type"] == "object"
    assert item["properties"]["importance"] == {"type": "integer"}
    assert item["properties"]["weight"] == {"type": "number"}
    assert item["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert item["properties"]["meta"] == {
        "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]
    }
    # required entries that name no property are dropped (invalid JSON Schema otherwise)
    assert item["required"] == ["id"]
    assert out["properties"]["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert out["required"] == ["entries"]


def _walk_schema(node, path="$"):
    """Yield (path, node) for every schema node."""
    yield path, node
    if node.get("type") == "array":
        yield from _walk_schema(node["items"], path + "[]")
    if node.get("type") == "object":
        for name, child in node.get("properties", {}).items():
            yield from _walk_schema(child, f"{path}.{name}")


def test_real_simple_tools_convert_with_correct_types():
    backend = _make_backend()
    tools = {t["name"]: t for t in backend._tools_openai}
    # Every simple-scaffold tool converts (a fixed count went stale with each new tool).
    assert set(tools) == {t["name"] for t in build_tools_for_scaffold("simple")}
    assert tools["buy_item"]["parameters"]["properties"]["count"]["type"] == "integer"
    assert tools["go_to_exit"]["parameters"]["properties"]["exit"]["type"] == "string"
    for tool in tools.values():
        assert tool["type"] == "function" and tool["parameters"]["type"] == "object"
        for path, node in _walk_schema(tool["parameters"]):
            assert "type_" not in node, path
            assert node["type"] in {"string", "integer", "number", "boolean", "array", "object"}, path

    mem_item = tools["process_memory"]["parameters"]["properties"]["entries"]["items"]
    assert mem_item["type"] == "object"
    assert mem_item["properties"]["importance"]["type"] == "integer"
    assert mem_item["properties"]["content"]["type"] == "string"

    run_skill_args = tools["run_skill"]["parameters"]["properties"]["args"]
    assert run_skill_args["type"] == "object"
    assert run_skill_args["properties"]["x"]["type"] == "integer"

    window = tools["process_trajectory_history"]["parameters"]["properties"]["window_range"]
    assert window == {**window, "type": "array", "items": {"type": "integer"}}

    replan_edit = tools["replan_objectives"]["parameters"]["properties"]["edits"]["items"]
    assert replan_edit["properties"]["objective"]["properties"]["optional"]["type"] == "boolean"
    assert tools["navigate_to"]["parameters"]["properties"]["consider_npcs"]["type"] == "boolean"
    assert tools["get_walkthrough"]["parameters"]["properties"]["part"]["type"] == "integer"
    assert tools["get_walkthrough"]["parameters"]["required"] == ["part"]


def test_tool_with_no_parameters_still_valid():
    backend = _make_backend(tools=[{"name": "noop", "description": "d", "parameters": {"type_": "OBJECT", "properties": {}, "required": []}}])
    assert backend._tools_openai[0]["parameters"] == {"type": "object", "properties": {}, "required": []}


# --------------------------------------------------------------------------
# Retry classification and policy
# --------------------------------------------------------------------------

def test_transient_classification():
    transient = [
        openai.APITimeoutError(request=_REQ),
        openai.APIConnectionError(request=_REQ),
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.InternalServerError, 500),
        _status_error(openai.InternalServerError, 503),
        _status_error(openai.APIStatusError, 408),
    ]
    permanent = [
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.NotFoundError, 404),
        _status_error(openai.UnprocessableEntityError, 422),
        ValueError("bug"),
        TypeError("unexpected keyword argument"),
    ]
    for exc in transient:
        assert vb.is_transient_api_error(exc), type(exc).__name__
    for exc in permanent:
        assert not vb.is_transient_api_error(exc), type(exc).__name__


class _Flaky:
    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self, *_a, **_k):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_retry_recovers_from_transient_errors():
    flaky = _Flaky([openai.APITimeoutError(request=_REQ), _status_error(openai.InternalServerError, 500)])
    sleeps = []
    wrapped = vb.retry_transient_errors(max_attempts=3)(flaky)
    with mock.patch.object(vb.time, "sleep", side_effect=sleeps.append):
        assert wrapped(object()) == "ok"
    assert flaky.calls == 3
    assert len(sleeps) == 2 and sum(sleeps) <= 30.0


def test_retry_fails_fast_on_4xx():
    err = _status_error(openai.BadRequestError, 400)
    flaky = _Flaky([err])
    wrapped = vb.retry_transient_errors(max_attempts=3)(flaky)
    with mock.patch.object(vb.time, "sleep") as sleep:
        try:
            wrapped(object())
            raise AssertionError("expected BadRequestError")
        except openai.BadRequestError as e:
            assert e is err
    assert flaky.calls == 1
    sleep.assert_not_called()


def test_retry_bounded_attempts_and_wait_and_reraises_original():
    errors = [openai.APIConnectionError(request=_REQ) for _ in range(10)]
    flaky = _Flaky(errors)
    sleeps = []
    wrapped = vb.retry_transient_errors(max_attempts=3, initial_delay=20.0, max_total_wait=30.0)(flaky)
    with mock.patch.object(vb.time, "sleep", side_effect=sleeps.append):
        try:
            wrapped(object())
            raise AssertionError("expected APIConnectionError")
        except openai.APIConnectionError:
            pass
    assert flaky.calls == 3
    assert sum(sleeps) <= 30.0 + 1e-9


def test_backend_settings_for_compatible_server():
    backend = _make_backend()
    # The local default cap: 1024 until speed-plan Phase 3 measured replies at p99=152
    # tokens and lowered it (vb.DEFAULT_LOCAL_MAX_OUTPUT_TOKENS, 256).
    assert backend.max_output_tokens == vb.DEFAULT_LOCAL_MAX_OUTPUT_TOKENS == 256
    assert backend.request_timeout_s == 240.0
    assert backend.max_attempts == 3
    assert backend.temperature is None and backend.top_p is None
    assert backend.client.max_retries == 0
    assert backend.client.timeout.read == 240.0
    assert backend.call_deadline_s == 3 * 240.0 + 30.0


def test_backend_settings_hosted_openai_and_overrides():
    hosted = _make_backend(env={"OPENAI_BASE_URL": None})
    assert hosted.max_output_tokens is None  # reasoning tokens count against the cap on OpenAI
    assert hosted.request_timeout_s == 600.0
    tuned = _make_backend(env={
        "OPENAI_MAX_OUTPUT_TOKENS": "512", "OPENAI_REQUEST_TIMEOUT_S": "90",
        "OPENAI_MAX_ATTEMPTS": "2", "OPENAI_TEMPERATURE": "0.7", "OPENAI_TOP_P": "0.8",
    })
    assert (tuned.max_output_tokens, tuned.request_timeout_s, tuned.max_attempts) == (512, 90.0, 2)
    assert (tuned.temperature, tuned.top_p) == (0.7, 0.8)
    off = _make_backend(env={"OPENAI_MAX_OUTPUT_TOKENS": "none"})
    assert off.max_output_tokens is None


def test_call_responses_sends_cap_and_retries_transient_only():
    backend = _make_backend(env={"OPENAI_TEMPERATURE": "0.7"})
    seen = []
    flaky = _Flaky([openai.APITimeoutError(request=_REQ)], result="resp")

    def fake_create(**kwargs):
        seen.append(kwargs)
        return flaky()

    with mock.patch.object(backend.client.responses, "create", side_effect=fake_create), \
            mock.patch.object(vb.time, "sleep"):
        assert backend._call_responses("sys", "hello", tools=backend._tools_openai) == "resp"
    assert len(seen) == 2
    kw = seen[-1]
    assert kw["max_output_tokens"] == vb.DEFAULT_LOCAL_MAX_OUTPUT_TOKENS
    assert kw["temperature"] == 0.7
    assert "top_p" not in kw
    assert kw["extra_body"]["prompt_cache_key"].startswith("sys-")
    assert len(kw["tools"]) == len(build_tools_for_scaffold("simple"))

    calls = []
    with mock.patch.object(backend.client.responses, "create",
                           side_effect=lambda **k: calls.append(k) or (_ for _ in ()).throw(
                               _status_error(openai.BadRequestError, 400))), \
            mock.patch.object(vb.time, "sleep"):
        try:
            backend._call_responses("sys", "hello")
            raise AssertionError("expected BadRequestError")
        except openai.BadRequestError:
            pass
    assert len(calls) == 1


def test_frame_goes_last_so_a_new_frame_keeps_the_prompt_text_cached():
    # Ollama renders a message's images before its text, so a frame inside the text
    # message broke the prefix cache at every new screen (cached stuck at 8,914 live).
    from PIL import Image

    backend = _make_backend()
    sent = []

    def fake_create(**kwargs):
        sent.append(kwargs["input"])
        return type("R", (), {"output": [], "usage": None, "output_text": ""})()

    with mock.patch.object(backend.client.responses, "create", side_effect=fake_create):
        for colour in ("red", "blue"):
            backend.get_query(Image.new("RGB", (240, 160), colour), "the same prompt", "test")
    first, second = sent
    assert first[0] == second[0] == {"role": "user", "content": [{"type": "input_text", "text": "the same prompt"}]}
    assert [p["type"] for p in first[-1]["content"]] == ["input_image"]
    assert first[-1] != second[-1], "the two frames differ only in the last message"
    assert len(first) == 2
    assert backend._responses_input(None, "text only") == [first[0] | {"content": [{"type": "input_text", "text": "text only"}]}]


# --------------------------------------------------------------------------
# Malformed tool-call JSON
# --------------------------------------------------------------------------

def _fc_item(arguments):
    return {"type": "function_call", "name": "press_buttons", "arguments": arguments}


def _first_call(adapter):
    return adapter.candidates[0].content.parts[0].function_call


def test_adapter_valid_tool_json():
    fc = _first_call(vb._openai_responses_adapter(type("R", (), {"output": [_fc_item('{"buttons": ["A"], "reasoning": "r"}')]})()))
    assert fc.name == "press_buttons"
    assert fc.args == {"buttons": ["A"], "reasoning": "r"}
    assert fc.parse_error is None


def test_adapter_malformed_tool_json_reports_error():
    fc = _first_call(vb._openai_responses_adapter(type("R", (), {"output": [_fc_item('{"buttons": ["A"')]})()))
    assert fc.args == {}
    assert fc.parse_error and "not valid JSON" in fc.parse_error
    assert '{"buttons": ["A"' in fc.parse_error  # raw text kept for the model to see


def test_adapter_non_object_tool_json_reports_error():
    fc = _first_call(vb._openai_responses_adapter(type("R", (), {"output": [_fc_item("[1, 2]")]})()))
    assert fc.args == {}
    assert fc.parse_error and "JSON object" in fc.parse_error


def _run_all():
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report every failure, not just the first
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
