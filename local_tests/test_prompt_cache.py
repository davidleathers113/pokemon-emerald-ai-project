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


# Speed plan, Phase 2: characters from the first per-step section to the end, by fixture.
# Measured when it was written: ROUTE 112 9,043 → 3,938 (its Map Data JSON was ~5K),
# MT CHIMNEY 4,203 → 3,395, Isaac's battle 922 (no map). The ceilings leave ~15% room.
TAIL_CEILINGS = {
    "test_route111_south_to_113.state": 4600,
    "test_mt_chimney_busy_grunt.state": 3900,
    "test_switch_question.state": 1100,
}


def test_the_prompt_leaves_out_what_it_repeats():
    """Speed plan, Phase 2 (golden test on 3 fixtures: overworld, overworld with people,
    battle): no Map Data JSON (its people and things become one line in <map_info>), no
    second action history, no PROGRESS counts, no empty skill/subagent stubs, and the
    per-step tail under its ceiling."""
    s = server()
    for fixture, ceiling in TAIL_CEILINGS.items():
        s.load(Path(__file__).resolve().parent / fixture, wild_encounters=False)
        prompt = _agent()._build_structured_prompt(_state(s), 5)
        tail = len(prompt) - min(prompt.index(t) for t in CHANGING if t in prompt)
        print(f"    {fixture}: prompt {len(prompt)}, per-step tail {tail} (ceiling {ceiling})")
        for gone in ("Map Data (JSON)", "RECENT ACTION HISTORY", "📊 PROGRESS", "No skills learned yet",
                     "No subagents registered yet", "(Legend: "):
            assert gone not in prompt, (fixture, gone)
        assert tail <= ceiling, (fixture, tail)
        if fixture == "test_mt_chimney_busy_grunt.state":
            info = prompt[prompt.index("<map_info>"):prompt.index("</map_info>")]
            # In this save the map JSON's objects are filtered out by story flags; its
            # two signs (the meteorite machine, the route sign) remain.
            assert "PEOPLE & THINGS" in info and "sign at (14, 6)" in info, info


def test_people_and_things_come_from_the_map_json():
    """The map JSON's objects (as the live MT CHIMNEY prompt had them) and its signs
    become one line; the rest of the JSON is in EXITS and the ASCII map already."""
    from test_phase4_agent import pa  # the agents.PokeAgent module

    line = json.dumps({
        "objects": [
            {"x": 19, "y": 39, "graphics_id": "OBJ_EVENT_GFX_OLD_WOMAN", "trainer_type": "TRAINER_TYPE_NONE"},
            {"x": 16, "y": 18, "graphics_id": "OBJ_EVENT_GFX_EXPERT_F", "trainer_type": "TRAINER_TYPE_NORMAL"},
            {"x": 2, "y": 4, "graphics_id": "OBJ_EVENT_GFX_ITEM_BALL", "trainer_type": "TRAINER_TYPE_NONE"},
        ],
        "bg_events": [{"type": "sign", "x": 14, "y": 6}, {"type": "hidden_item", "x": 3, "y": 9}],
        "warps": [{"x": 1, "y": 1}],
    })
    assert pa._map_things(line) == (
        "PEOPLE & THINGS (from the map file): OLD WOMAN at (19, 39); EXPERT F (trainer) at (16, 18); "
        "ITEM BALL at (2, 4); sign at (14, 6); hidden item at (3, 9)"
    ), pa._map_things(line)
    assert pa._map_things('{"objects": [], "bg_events": []}') == "" and pa._map_things("not json") == ""


def test_a_failed_result_is_sent_once():
    """Server refusals set "error" and "summary" to the same text: only one goes in."""
    agent = _agent()
    refusal = {"success": False, "error": "No exit matches 'west'.", "summary": "No exit matches 'west'."}
    agent._store_function_result_for_context("go_to_exit", json.dumps(refusal))
    block = agent._get_function_results_context()
    assert block.count("No exit matches 'west'.") == 1 and '"summary"' in block, block


def test_older_steps_are_one_line_each():
    """Only the newest step keeps its reasoning ("why"); older steps are one line."""
    agent = _agent()
    for step in range(1, 6):
        agent._add_to_history("p", f"I go north because step {step}", step_number=step)
    history = agent._format_action_history()
    print("    " + history.replace("\n", "\n    "))
    assert history.count("why:") == 1 and "step 5" in history.split("why:")[1], history


def test_brief_reasoning_is_opt_in_and_reaches_every_rationale_field():
    """Speed plan, Phase 3: BRIEF_REASONING=1 asks every tool's rationale for one line; off
    (the default) the ANALYZE/PLAN format stays, byte for byte. The registry itself is
    not changed (a copy per build)."""
    import os
    from unittest import mock

    from agents.tools import registry

    def rationale_fields(tools):
        return {
            (t["name"], name): spec["description"]
            for t in tools for name, spec in ((t.get("parameters") or {}).get("properties") or {}).items()
            if name in registry.REASONING_PARAMETER_NAMES
        }

    with mock.patch.dict(os.environ, {"BRIEF_REASONING": ""}):
        default = registry.build_tools_for_scaffold("simple")
    with mock.patch.dict(os.environ, {"BRIEF_REASONING": "1"}):
        brief = registry.build_tools_for_scaffold("simple")
    before, after = rationale_fields(default), rationale_fields(brief)
    print(f"    {len(after)} rationale fields; press_buttons default: {before[('press_buttons', 'reasoning')][:40]}…")
    assert before.keys() == after.keys() and len(after) > 10, len(after)
    assert "ANALYZE" in before[("press_buttons", "reasoning")]
    assert set(after.values()) == {registry.BRIEF_REASONING_DESCRIPTION}, set(after.values())
    # Everything else is the same.
    for d, b in zip(default, brief):
        assert d["name"] == b["name"] and d["description"] == b["description"]
        dp = {k: v for k, v in (d["parameters"].get("properties") or {}).items() if k not in registry.REASONING_PARAMETER_NAMES}
        bp = {k: v for k, v in (b["parameters"].get("properties") or {}).items() if k not in registry.REASONING_PARAMETER_NAMES}
        assert dp == bp, d["name"]
    with mock.patch.dict(os.environ, {"BRIEF_REASONING": ""}):
        assert registry.build_tools_for_scaffold("simple") == default  # the registry was not changed


def _game_state(screen="overworld", location="ROUTE 101", dialogue=False, battle=False):
    return {
        "location": location,
        "raw_state": {"game": {
            "game_state": screen, "is_in_battle": battle, "dialog_text": "Hi!" if dialogue else None,
            "dialogue_detected": {"has_dialogue": dialogue},
        }},
    }


def test_the_frame_goes_out_only_when_the_text_cannot_describe_the_screen():
    """Speed plan, Phase 4 (SCREENSHOT_POLICY=needed): dialogue, menus and other screens,
    a new map, the first step of a battle and every 5th step get the frame; a walk on a
    known map or a battle menu after its first step does not."""
    from test_phase4_agent import pa

    seen = {"location": "ROUTE 101", "in_battle": False, "frame_step": 10}
    cases = [
        (_game_state(), 11, False),
        (_game_state(), 15, True),                       # every 5 steps
        (_game_state(screen="menu"), 11, True),
        (_game_state(screen="title"), 11, True),
        (_game_state(dialogue=True), 11, True),
        (_game_state(location="OLDALE TOWN"), 11, True),  # first step on a new map
        (_game_state(screen="battle", battle=True), 11, True),  # a battle just started
    ]
    for state, step, want in cases:
        send, why = pa.frame_needed(state, seen, step)
        assert send is want, (state["raw_state"]["game"]["game_state"], state["location"], step, why)
    in_battle = {"location": "ROUTE 101", "in_battle": True, "frame_step": 10}
    assert pa.frame_needed(_game_state(screen="battle", battle=True), in_battle, 11)[0] is False


def test_the_screenshot_policy_is_always_unless_switched():
    import os
    from unittest import mock

    from test_phase4_agent import pa

    agent = _agent()
    with mock.patch.dict(os.environ, {"SCREENSHOT_POLICY": ""}):
        for step in range(1, 4):
            assert agent._apply_screenshot_policy(_game_state(), "FRAME", step) == "FRAME"
    agent = _agent()
    with mock.patch.dict(os.environ, {"SCREENSHOT_POLICY": "needed"}):
        sent = [agent._apply_screenshot_policy(_game_state(), "FRAME", step) for step in range(1, 8)]
    # Step 1: a new map; then every 5 steps (6); none on 2-5 and 7.
    assert sent == ["FRAME", None, None, None, None, "FRAME", None], sent
    assert agent._step_timing["frame"] == 0


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
