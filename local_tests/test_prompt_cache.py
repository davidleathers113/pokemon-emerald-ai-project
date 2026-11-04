"""Speed plan, Phase 1: the prompt is ordered so Ollama can reuse the last step's prompt.

David (2026-11-04) approved a plan to make each decision faster. Ollama reuses the cached
prompt up to the first byte that differs; the prompt used to open with "# Step: N" and
the previous result, so only the system prompt and tools (8.25K tokens) were ever reused
and ~5.6K tokens were read again every step. Two consecutive steps on one map (the
player one tile further on) must now share everything up to the per-step tail.

Same in-process server harness as test_movement_feedback.py; the agent is the bare one
from test_phase4_agent.py (no model).

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_prompt_cache.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import ROUTE101, server
from test_phase4_agent import _bare_agent

LASTING = ("<objectives>", "<long_term_memory>", "<map_info>")
CHANGING = ("<last_result>", "<recent_steps>", "<state>")


def _agent():
    agent = _bare_agent()
    agent.bootstrap_active = False  # no evolved-prompt bootstrap (PokeAgent.__init__ sets it)

    def tool(name, args):
        overviews = {
            "get_memory_overview": "=== LONG-TERM MEMORY OVERVIEW ===\n[1] routes/oldale: north of Route 101",
            "get_skill_overview": "No skills learned yet.",
            "get_subagent_overview": "No subagents registered yet.",
        }
        return {"success": True, "overview": overviews.get(name, "")}

    agent.mcp_adapter.call_tool.side_effect = tool
    return agent


def _state(s):
    return json.dumps(asyncio.run(s.srv.mcp_get_game_state()), default=str)


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_two_steps_on_one_map_share_everything_before_the_tail():
    from agents.objectives.direct_objectives import DirectObjectiveManager

    s = server()
    s.load(ROUTE101, wild_encounters=False)
    real = s.srv.direct_objectives_manager
    try:
        m = DirectObjectiveManager()
        m.load_categorized_full_game_sequence(start_story_index=5)
        s.srv.direct_objectives_manager = m
        agent = _agent()
        first = agent._build_structured_prompt(_state(s), 41)
        s.call("mcp_press_buttons", {"buttons": ["LEFT"], "reasoning": "test"})
        time.sleep(0.5)
        agent.step_count += 1
        agent._store_function_result_for_context("press_buttons", json.dumps({"success": True, "summary": "moved"}))
        second = agent._build_structured_prompt(_state(s), 42)
    finally:
        s.srv.direct_objectives_manager = real
    print("    sections:", [t for t in LASTING + CHANGING if t in second])
    for tag in LASTING:
        assert tag in first and tag in second, (tag, first[:600])
    shared = _common_prefix(first, second)
    print(f"    shared prefix {shared} of {len(second)} chars; differs at: {second[shared:shared + 60]!r}")
    assert shared >= second.index("</map_info>"), "a lasting section differs between the two steps"
    tail = min(second.index(t) for t in CHANGING if t in second)
    assert shared >= tail - 1, (shared, tail)
    assert second.rstrip().endswith("Step: 42"), second[-40:]
    # The legend is in the system instructions now, not repeated each step.
    assert "(Legend: " not in second
    # The overviews were fetched once for both steps (cached between store writes).
    fetched = [c.args[0] for c in agent.mcp_adapter.call_tool.call_args_list]
    assert fetched.count("get_memory_overview") == 1, fetched


def test_a_store_write_refreshes_the_overviews():
    agent = _agent()
    agent._gather_store_context()
    agent._gather_store_context()
    assert [c.args[0] for c in agent.mcp_adapter.call_tool.call_args_list].count("get_memory_overview") == 1
    agent._note_store_write("process_memory")
    agent._gather_store_context()
    assert [c.args[0] for c in agent.mcp_adapter.call_tool.call_args_list].count("get_memory_overview") == 2
    agent._note_store_write("press_buttons")  # not a store write
    agent._gather_store_context()
    assert [c.args[0] for c in agent.mcp_adapter.call_tool.call_args_list].count("get_memory_overview") == 2


def _run_all():
    import test_movement_feedback as harness

    failures = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        started = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - started:.1f}s)")
        except Exception as e:  # report every failure, not just the first
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    if harness._SERVER is not None:
        harness._SERVER.stop()
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
