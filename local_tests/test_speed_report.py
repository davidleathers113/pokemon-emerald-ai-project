"""local_tools/speed_report.py: the numbers each phase of the speed plan is judged by
(no game needed).

  python3 local_tests/test_speed_report.py
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_tools"))
from speed_report import report

START = datetime(2026, 11, 4, 15, 0, 0)
FAILED = (
    'RESULTS FROM PREVIOUS STEP:\n🔧 Function: go_to_exit\nResult:\n{\n  "success": false,\n'
    '  "error": "Cannot reach \'ROUTE 112\' from (14, 5) on JAGGED PASS: no path",\n  "summary": "x"\n}\n### OBJECTIVES\n'
)
OK = 'RESULTS FROM PREVIOUS STEP:\n🔧 Function: use_move\nResult:\n{\n  "success": true\n}\n### OBJECTIVES\n'


def _write(tmp, rows, calls):
    run = Path(tmp) / "20261104_150000_run"
    run.mkdir()
    (run / "trajectory_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = Path(tmp) / "llm_log.jsonl"
    log.write_text("".join(json.dumps(c) + "\n" for c in calls))
    return run, log


def _rows():
    rows = []
    for i in range(6):
        rows.append({
            "step": i + 1,
            "timestamp": (START + timedelta(seconds=40 * i)).isoformat(),
            "pre_state": {"is_in_battle": i >= 4},
            "location": "JAGGED PASS", "player_coords": [14, 5 + i],
            "action": {"tool_calls": [{"name": "complete_direct_objective" if i == 2 else "go_to_exit", "args": {}}]},
            "llm_prompt": FAILED if i in (1, 2) else OK,
            "timing": {"state_s": 0.9 + i / 10, "vlm_s": 30.0 + i},
            "prompt_sections": {"STATE": 4000 + i, "OBJECTIVES": 2000},
        })
    return rows


def _calls():
    calls = []
    for i in range(6):
        usage = {"prompt_tokens": 14000, "completion_tokens": 100 + i, "cached_tokens": 8252}
        calls.append({
            "timestamp": (START + timedelta(seconds=40 * i - 5)).isoformat(),
            "type": "interaction", "agent_step": i + 1, "duration": 30.0 + i,
            "metadata": repr({"model": "qwen3-vl", "token_usage": usage}),
        })
    # A call from an earlier run with the same step number, outside the window.
    calls.append({"timestamp": "2026-11-04T09:00:00", "type": "interaction", "agent_step": 1, "duration": 99.0,
                  "metadata": repr({"token_usage": {"prompt_tokens": 1, "cached_tokens": 0, "completion_tokens": 1}})})
    return calls


def test_speed_numbers_come_from_the_rows_and_the_llm_log():
    with tempfile.TemporaryDirectory() as tmp:
        run, log = _write(tmp, _rows(), _calls())
        r = report(run, 60, llm_log=log)
    assert r["steps"] == 6 and r["step_s"]["median"] == 40.0, r["step_s"]
    assert r["timing_s"]["vlm_s"]["n"] == 6 and r["timing_s"]["vlm_s"]["median"] == 32.5, r["timing_s"]
    assert r["prompt_section_chars"]["STATE"]["median"] == 4002.5, r["prompt_section_chars"]
    over, battle = r["model"]["overworld"], r["model"]["battle"]
    assert over["model_s"]["n"] == 4 and battle["model_s"]["n"] == 2, r["model"]  # the old call left out
    assert over["uncached_tokens"]["median"] == 5748, over


def test_quality_counts_failures_by_cause_and_objectives():
    with tempfile.TemporaryDirectory() as tmp:
        run, log = _write(tmp, _rows(), _calls())
        q = report(run, 60, llm_log=log)["quality"]
    assert q["failed_calls_per_100_steps"] == 40.0, q  # 2 of the 5 results after the first row
    causes = q["failed_call_causes"]
    assert causes == {"go_to_exit: Cannot reach 'ROUTE ###' from (##, #) on JAGGED PASS": 2}, causes
    # Step 3's objective call succeeded (its result opens step 4's prompt); step 2's
    # failed result belongs to step 1's go_to_exit.
    assert q["objectives_completed"] == 1, q


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    total = sum(1 for n in globals() if n.startswith("test_"))
    print(f"{total - failures}/{total} passed")
    sys.exit(1 if failures else 0)
