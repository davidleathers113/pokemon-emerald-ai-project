"""Agent side of the movement/dialogue changes (no server, no model):

- short-term memory replays each step as one compact entry: the reasoning once, the
  call without its reasoning argument, and a one-line outcome that keeps the server's
  BLOCKED / arrival summary and the auto-advanced dialogue;
- auto-advance is called before each model call, reaches the next prompt's RESULTS
  FROM PREVIOUS STEP, and is switched off by POKEAGENT_AUTO_DIALOGUE=0;
- go_to_exit is routed to the server.

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_dialogue_agent.py
"""
import json
import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_agent import _bare_agent, pa

REASONING = (
    "ANALYZE: Currently at Route 101 (13,2). Objective: Travel north to Oldale Town. "
    "Destination: North edge of map (13,0) leads to Oldale Town. PLAN: Move north 3 tiles "
    "to reach Oldale Town. Expected: arrive in Oldale Town after walking off the north edge."
)
BLOCKED = "BLOCKED moving UP at (13,2) on ROUTE 101: (13,1) is not walkable. 2 of 3 presses skipped."


def _blocked_step(agent, step=16):
    result = json.dumps({
        "success": True,
        "buttons_queued": ["UP", "UP", "UP"],
        "status": "BLOCKED",
        "summary": BLOCKED,
        "now": "(13,2) on ROUTE 101, facing UP",
    })
    calls = [{"name": "press_buttons", "args": {"buttons": ["UP", "UP", "UP"], "reasoning": REASONING}, "result": result}]
    agent._last_end_location = "ROUTE 101"
    agent._add_to_history(
        "p", f"{REASONING}\n\nAction executed: press_buttons", tool_calls=calls,
        start_coords=(12, 2), end_coords=(13, 2), step_number=step,
    )


def test_entry_is_one_header_line_plus_reasoning_once():
    agent = _bare_agent()
    _blocked_step(agent)
    history = agent._format_action_history()
    print(history)
    lines = history.splitlines()
    assert lines[0] == f"[16] (12,2)->(13,2) ROUTE 101 | press_buttons UP,UP,UP | {BLOCKED}", lines[0]
    assert lines[1].startswith("  why: ANALYZE: Currently at Route 101 (13,2).")
    assert len(lines[1]) <= len("  why: ") + pa.HISTORY_WHY_MAX_CHARS
    assert history.count("Travel north to Oldale") == 1  # was 3 (THINKING, args, result echo)
    assert '"reasoning"' not in history and "Action executed" not in history


def test_twenty_entries_stay_small():
    agent = _bare_agent()
    for step in range(1, 21):
        _blocked_step(agent, step)
    history = agent._format_action_history()
    # Same 20 entries in the old layout: ~980 chars each (measured on the live prompt).
    assert len(history) < 20 * 420, len(history)


def test_error_result_is_summarised():
    agent = _bare_agent()
    err = json.dumps({"success": False, "error": "No exit of ROUTE 101 matches 'west'."})
    agent._add_to_history("p", "", tool_calls=[{"name": "go_to_exit", "args": {"exit": "west", "reasoning": "x"}, "result": err}], step_number=3)
    history = agent._format_action_history()
    assert "[3] (?)->(?) | go_to_exit \"west\" | ERROR: No exit of ROUTE 101 matches 'west'." in history, history


def test_auto_dialogue_reaches_prompt_and_history():
    agent = _bare_agent()
    agent.server_url = "http://server"
    summary = "Dialogue read (auto-advanced with 3 A presses): MOM: See you, honey! The text box closed."
    reply = mock.MagicMock()
    reply.json.return_value = {"success": True, "presses": 3, "summary": summary}
    with mock.patch.dict(os.environ, {"POKEAGENT_AUTO_DIALOGUE": "1"}), \
            mock.patch.object(pa.requests, "post", return_value=reply) as post:
        agent._auto_advance_dialogue()
    assert post.call_args.args[0] == "http://server/mcp/auto_advance_dialogue"
    block = agent._get_function_results_context()
    assert "RESULTS FROM PREVIOUS STEP" in block and summary in block
    agent._add_to_history("p", "", tool_calls=[{"name": "press_buttons", "args": {"buttons": ["B"]}, "result": "{}"}], step_number=5)
    assert f"  dialogue: {summary}" in agent._format_action_history()


def test_auto_dialogue_marks_the_earlier_screen_note_as_out_of_date():
    """Seen live in Birch's lab: the press result said 'A text box is still open ...: press
    A to continue' above auto-advance's 'The text box closed.'; the model went by the
    older line and pressed A at May eleven times."""
    agent = _bare_agent()
    agent.server_url = "http://server"
    press = json.dumps({"status": "DONE", "summary": 'Pressed 1 button. A text box is still open "MAY: Hi": press A to continue.'})
    agent._store_function_result_for_context("press_buttons", press)
    summary = "Dialogue read (auto-advanced with 4 A presses): MAY: Hi The text box closed."
    reply = mock.MagicMock()
    reply.json.return_value = {"success": True, "presses": 4, "summary": summary}
    with mock.patch.dict(os.environ, {"POKEAGENT_AUTO_DIALOGUE": "1"}), \
            mock.patch.object(pa.requests, "post", return_value=reply):
        agent._auto_advance_dialogue()
    block = agent._get_function_results_context()
    assert block.index("press A to continue") < block.index("This is the screen now"), block
    assert agent._pending_dialogue_note == summary  # history stays compact


def test_auto_dialogue_does_nothing_when_no_box():
    agent = _bare_agent()
    agent.server_url = "http://server"
    reply = mock.MagicMock()
    reply.json.return_value = {"success": True, "presses": 0, "final_status": "none"}
    with mock.patch.object(pa.requests, "post", return_value=reply):
        agent._auto_advance_dialogue()
    assert agent.recent_function_results == []


def test_auto_dialogue_switch_off():
    agent = _bare_agent()
    agent.server_url = "http://server"
    with mock.patch.dict(os.environ, {"POKEAGENT_AUTO_DIALOGUE": "0"}), \
            mock.patch.object(pa.requests, "post") as post:
        agent._auto_advance_dialogue()
    post.assert_not_called()


def test_go_to_exit_is_routed_to_the_server():
    adapter = pa.MCPToolAdapter("http://server")
    reply = mock.MagicMock()
    reply.json.return_value = {"success": True, "summary": "Arrived"}
    with mock.patch.object(pa.requests, "post", return_value=reply) as post:
        adapter.call_tool("go_to_exit", {"exit": "north", "reasoning": "to Oldale"})
    assert post.call_args.args[0] == "http://server/mcp/go_to_exit"
    assert post.call_args.kwargs["timeout"] >= 120


def test_prompt_sections_measure_each_section():
    """Speed plan, Phase 0: each trajectory row says how big each "### " section of its
    prompt was, so the uncached part's make-up can be measured step by step."""
    prompt = "# Step: 3\nhead\n### OBJECTIVES\nstory\n### STATE\nmap\nmore\n"
    sizes = pa._prompt_sections(prompt)
    assert sizes == {
        "(head)": len("# Step: 3\nhead\n"),
        "OBJECTIVES": len("### OBJECTIVES\nstory\n"),
        "STATE": len("### STATE\nmap\nmore\n"),
    }, sizes
    assert sum(sizes.values()) == len(prompt)


def test_trajectory_rows_carry_the_step_timing():
    """The parts of a step (auto-advance, state fetch, prompt build, model call, tools…)
    reach the trajectory row, which local_tools/speed_report.py reads."""
    import tempfile

    from utils.data_persistence import run_data_manager as rdm

    agent = _bare_agent()
    agent._step_timing = {"state_s": 0.9, "vlm_s": 31.2, "tools_s": 4.0}
    manager = mock.MagicMock()
    agent._log_trajectory_for_step(manager, 7, {"location": "ROUTE 101"}, "### STATE\nx\n", "why", [], "")
    kwargs = manager.log_trajectory.call_args.kwargs
    assert kwargs["timing"] == {"state_s": 0.9, "vlm_s": 31.2, "tools_s": 4.0}, kwargs
    assert kwargs["prompt_sections"] == {"STATE": len("### STATE\nx\n")}, kwargs
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trajectory_history.jsonl"
        with mock.patch.object(rdm, "get_cache_path", lambda name: path):
            rdm.RunDataManager.log_trajectory(
                mock.MagicMock(), step=7, reasoning="why", action={}, pre_state={"location": "ROUTE 101"},
                timing=kwargs["timing"], prompt_sections=kwargs["prompt_sections"],
            )
        row = json.loads(path.read_text().splitlines()[-1])
    assert row["timing"]["vlm_s"] == 31.2 and row["prompt_sections"] == {"STATE": 12}, row


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
