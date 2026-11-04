"""Phase 4 tests for PokeAgent context budget, tool-error feedback, text-only
re-prompt and the no-stacking wall-clock guard. Also checks that SIMPLE.md
describes exactly the tools the simple scaffold has.

Runs under pytest, or directly: python local_tests/test_phase4_agent.py
"""

import json
import os
import sys
import threading
import time
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib

# agents/__init__.py re-exports the PokeAgent class under the module's name, so
# `import agents.PokeAgent as pa` would bind the class; fetch the module itself.
pa = importlib.import_module("agents.PokeAgent")
from agents.subagents.utils.runtime import PokeAgentRuntime
from agents.tools.registry import build_tools_for_scaffold
from utils.agent_infrastructure import vlm_backends as vb


def _bare_agent():
    """A PokeAgent with only the state the tested methods use (no server/VLM)."""
    agent = pa.PokeAgent.__new__(pa.PokeAgent)
    agent.step_count = 0
    agent.conversation_history = []
    agent.recent_function_results = []
    agent.text_only_reply_count = 0
    agent.text_only_noop_count = 0
    agent._last_prompt_usage = None
    # Per-step timing (speed plan, Phase 0): __init__ sets these, and run_step writes them.
    agent._step_timing = {}
    agent._post_step_s = None
    agent.scaffold = "simple"
    agent.backend = "openai"
    agent.model = "qwen3-vl:8b-instruct"
    agent.optimization_enabled = False
    agent.tools = build_tools_for_scaffold("simple")
    agent.runtime = PokeAgentRuntime(
        initial_step=0,
        publish_history=agent._add_to_history,
        publish_function_result=agent._store_function_result_for_context,
        on_step_change=lambda step: setattr(agent, "step_count", step),
    )
    agent.vlm = mock.MagicMock()
    agent.vlm.backend = mock.MagicMock(spec=[])  # no call_deadline_s attribute
    agent.llm_logger = mock.MagicMock()
    agent.mcp_adapter = mock.MagicMock()
    agent.mcp_adapter.call_tool.return_value = {"success": True, "message": "ok"}
    agent._wait_for_actions_complete = lambda *a, **k: None
    return agent


# --------------------------------------------------------------------------
# Truncation / context budget
# --------------------------------------------------------------------------

def test_truncate_middle_short_and_non_string_unchanged():
    assert pa.truncate_middle("abc", 10) == "abc"
    assert pa.truncate_middle({"a": 1}, 1) == {"a": 1}
    assert pa.truncate_middle(None, 5) is None


def test_truncate_middle_keeps_head_and_tail_with_marker():
    text = "HEAD" + "x" * 5000 + "TAIL"
    out = pa.truncate_middle(text, 1500)
    assert len(out) <= 1500
    assert out.startswith("HEAD") and out.endswith("TAIL")
    marker_start = out.index("... [") + len("... [")
    omitted = int(out[marker_start:out.index(" chars truncated]")])
    kept = len(out) - len(f"\n... [{omitted} chars truncated] ...\n")
    assert omitted + kept == len(text)


def test_history_caps_tool_results_without_mutating_caller():
    agent = _bare_agent()
    big = json.dumps({"success": True, "walkthrough": "w" * 8000})
    calls = [{"name": "get_walkthrough", "args": {"part": 1}, "result": big}]
    agent._add_to_history("p", "thinking " * 500, tool_calls=calls, step_number=1)
    entry = agent.conversation_history[-1]
    assert len(entry["tool_calls"][0]["result"]) <= pa.HISTORY_TOOL_RESULT_MAX_CHARS
    assert len(entry["llm_response"]) <= pa.HISTORY_RESPONSE_MAX_CHARS
    assert calls[0]["result"] == big  # caller's list untouched (full result still logged)
    # Replayed as one compact entry: the walkthrough text is not repeated every step.
    # (This used to assert the verbose layout's "chars truncated" marker.)
    history = agent._format_action_history()
    assert history.startswith('[1] (?)->(?) | get_walkthrough {"part": 1} | ok'), history
    assert "w" * 50 not in history and len(history) < 600


def test_previous_results_block_is_capped():
    agent = _bare_agent()
    for name in ("a", "b", "c"):
        agent._store_function_result_for_context(name, name * 20000)
    block = agent._get_function_results_context()
    # Three results, each capped, and their sum bounded by the block budget
    # (plus headers and the 500-char floor for older results).
    assert len(block) < pa.PREVIOUS_RESULTS_TOTAL_MAX_CHARS + 2 * 500 + 600
    assert "🔧 Function: c" in block and "chars truncated" in block
    assert agent.recent_function_results == []


def test_walkthrough_sized_result_survives_previous_block():
    agent = _bare_agent()
    part = "W" * 7900
    agent._store_function_result_for_context("get_walkthrough", part)
    assert part in agent._get_function_results_context()


def test_fit_prompt_drops_oldest_history_until_under_budget():
    agent = _bare_agent()
    for step in range(1, 21):
        agent._add_to_history("p", f"step{step} " + "z" * 1400, step_number=step)
    agent._store_function_result_for_context("press_buttons", "{\"success\": true}")

    def build():
        return agent._format_action_history() + agent._get_function_results_context()

    full_estimate = agent._estimate_prompt_tokens(build(), has_image=True)
    agent.recent_function_results = [{"function_name": "press_buttons", "result": "{\"success\": true}", "timestamp": 0}]
    # Since speed-plan Phase 2 only the newest entry replays its reply ("why"); older
    # ones are one short line (~8 tokens here), so a 50-token cut forces several drops.
    # (It was 500 when every entry kept a ~70-token "why", and 3000 with the verbose
    # layout, which replayed every 1400-char reply in full.)
    budget = full_estimate - 50
    with mock.patch.object(pa, "PROMPT_TOKEN_BUDGET", budget):
        prompt = agent._fit_prompt_to_budget(build, has_image=True)
    assert agent._estimate_prompt_tokens(prompt, has_image=True) <= budget
    assert len(agent.conversation_history) < 20
    assert agent.conversation_history[-1]["llm_response"].startswith("step20")  # newest kept
    assert "RESULTS FROM PREVIOUS STEP" in prompt  # one-shot block restored on rebuild


def test_estimate_uses_actual_usage_when_known():
    agent = _bare_agent()
    usage = type("U", (), {"prompt_token_count": 9000})()
    agent._record_prompt_usage(type("R", (), {"usage_metadata": usage})(), "x" * 3200)
    assert agent._estimate_prompt_tokens("x" * 6400, has_image=True) == 9000 + 1000


# --------------------------------------------------------------------------
# Malformed tool JSON -> explicit error, tool not executed
# --------------------------------------------------------------------------

def test_malformed_tool_json_returns_error_to_model():
    agent = _bare_agent()
    response = vb._openai_responses_adapter(type("R", (), {"output": [
        {"type": "function_call", "name": "press_buttons", "arguments": '{"buttons": ["A"'}
    ]})())
    tool_calls = []
    with mock.patch.object(agent, "_execute_function_call", side_effect=AssertionError("must not run")):
        assert agent._handle_vlm_function_calls(response, tool_calls, 0, 5) is True
    assert len(tool_calls) == 1
    result = json.loads(tool_calls[0]["result"])
    assert result["success"] is False and "not valid JSON" in result["error"]


# --------------------------------------------------------------------------
# Text-only replies: one re-prompt, then a logged no-op
# --------------------------------------------------------------------------

def _text_adapter(text):
    return vb._openai_responses_adapter(type("R", (), {"output": [
        {"type": "message", "content": [{"type": "output_text", "text": text}]}
    ]})())


def _tool_adapter():
    return vb._openai_responses_adapter(type("R", (), {"output": [
        {"type": "function_call", "name": "press_buttons",
         "arguments": json.dumps({"buttons": ["A"], "reasoning": "advance"})}
    ]})())


def _run_text_step(agent, responses):
    agent.vlm.get_text_query.side_effect = list(responses)
    with mock.patch.object(pa, "get_run_data_manager", return_value=None), \
            mock.patch.dict(os.environ, {"RUN_DATA_ID": ""}):
        return agent.run_step("PROMPT", screenshot_b64=None)


def test_text_only_reply_is_reprompted_once_and_recovers():
    agent = _bare_agent()
    ok, _ = _run_text_step(agent, [_text_adapter("ANALYZE: I should press A."), _tool_adapter()])
    assert ok
    prompts = [c.args[0] for c in agent.vlm.get_text_query.call_args_list]
    assert prompts == ["PROMPT", "PROMPT" + pa.TEXT_ONLY_REPROMPT]
    assert agent.text_only_reply_count == 1 and agent.text_only_noop_count == 0
    pressed = [c for c in agent.mcp_adapter.call_tool.call_args_list if c.args[0] == "press_buttons"]
    assert len(pressed) == 1


def test_text_only_twice_records_noop_and_feedback():
    agent = _bare_agent()
    ok, _ = _run_text_step(agent, [_text_adapter("thinking"), _text_adapter("still thinking")])
    assert ok
    assert agent.vlm.get_text_query.call_count == 2  # exactly one re-prompt
    assert agent.text_only_reply_count == 1 and agent.text_only_noop_count == 1
    assert agent.conversation_history[-1]["action_details"].startswith("NO-OP")
    assert agent.recent_function_results[-1]["function_name"] == "no_tool_call"
    assert "ACTION: NO-OP" in agent._format_action_history()


# --------------------------------------------------------------------------
# Wall-clock guard never stacks a second request on a running one
# --------------------------------------------------------------------------

def test_guard_does_not_stack_requests_while_first_still_running():
    agent = _bare_agent()
    started = []
    release = threading.Event()

    def slow_call():
        started.append(time.time())
        release.wait(5)
        return "late"

    with mock.patch.object(agent, "_vlm_guard_timeout_s", return_value=0.2), \
            mock.patch.object(pa, "VLM_GUARD_GRACE_S", 0.2):
        try:
            agent._call_vlm_with_guard(slow_call)
            raise AssertionError("expected TimeoutError")
        except TimeoutError:
            pass
    release.set()
    assert len(started) == 1


def test_guard_uses_late_result_from_grace_period():
    agent = _bare_agent()
    calls = []

    def call():
        calls.append(1)
        time.sleep(0.3)
        return "done"

    with mock.patch.object(agent, "_vlm_guard_timeout_s", return_value=0.1), \
            mock.patch.object(pa, "VLM_GUARD_GRACE_S", 2.0):
        assert agent._call_vlm_with_guard(call) == "done"
    assert len(calls) == 1


def test_guard_sized_from_backend_deadline():
    agent = _bare_agent()
    agent.vlm.backend = type("B", (), {"call_deadline_s": 750.0})()
    assert agent._vlm_guard_timeout_s() == 780.0
    agent.vlm.backend = object()
    assert agent._vlm_guard_timeout_s() == pa.VLM_GUARD_DEFAULT_S


# --------------------------------------------------------------------------
# SIMPLE.md matches the simple scaffold's real tool list
# --------------------------------------------------------------------------

def test_simple_directive_lists_exactly_the_simple_tools():
    path = os.path.join(REPO_ROOT, "agents", "prompts", "pokeagent-directives", "SIMPLE.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    simple = {t["name"] for t in build_tools_for_scaffold("simple")}
    for name in simple:
        assert f"`{name}(" in text, f"SIMPLE.md does not describe {name}"
    everything = set()
    for scaffold in ("pokeagent", "autonomous_cli", "continualharness", "simplest", "simple"):
        everything |= {t["name"] for t in build_tools_for_scaffold(scaffold)}
    for name in everything - simple:
        assert f"`{name}" not in text, f"SIMPLE.md mentions {name}, which simple does not have"
    assert "NEVER save the game" in text
    assert "no walkthrough" not in text.lower()


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
