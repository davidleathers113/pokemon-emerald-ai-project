"""How fast the AI decides, and whether it still plays well: one report over a window of steps.

David (2026-11-04) asked why the AI is slow (median step ~40 s, ~31 s of it the model)
and approved a plan to speed it up without playing worse. Every phase of that plan is
judged against this report: the same numbers before and after.

Speed, from the newest run (or --run):
  - step time: from one decision's trajectory row to the next (median, p90);
  - the agent's own breakdown (trajectory "timing": auto-advance, state fetch, prompt
    build, model call, tools), when the agent wrote one;
  - model time and tokens, from the run's llm_logs file: prompt, cached, uncached
    (prompt - cached), output; overworld and battle apart.
Quality, over the same steps:
  - stalls per hour (local_tools/stall_watch.stall_reason replayed over the rows);
  - calls that came back with "success": false, per 100 steps;
  - objectives completed per hour (complete_direct_objective calls that succeeded).

  python3 local_tools/speed_report.py [--steps 60] [--run DIR] [--json]
"""
import argparse
import ast
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Rows a stall replay looks back over, and rows skipped after one (the watcher pauses
# the agent at a stall, so the same stuck stretch is one event, not several; 8 is
# stall_watch's longest run, NO_PROGRESS).
STALL_WINDOW = 30
STALL_SKIP = 8
# How long before the first row of the window its model call may have been logged.
MODEL_CALL_SLACK_S = 180


def _stall_reason(rows):
    """local_tools/stall_watch.stall_reason (a sibling script, not a package)."""
    if str(REPO / "local_tools") not in sys.path:
        sys.path.insert(0, str(REPO / "local_tools"))
    from stall_watch import stall_reason

    return stall_reason(rows)


def _time(row):
    try:
        return datetime.fromisoformat(row["timestamp"])
    except (KeyError, ValueError, TypeError):
        return None


def _pct(values, share):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(share * len(values)))]


def _summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": round(statistics.median(values), 1), "p90": round(_pct(values, 0.9), 1)}


def _newest_run():
    runs = sorted(REPO.glob(".pokeagent_cache/*_ai-local-*/trajectory_history.jsonl"), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None


def _rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _llm_log_for(run_dir):
    """llm_logs/llm_log_<run start>.jsonl: the run dir starts with the same stamp."""
    stamp = run_dir.name[:15]  # 20261104_151335
    path = REPO / "llm_logs" / f"llm_log_{stamp}.jsonl"
    return path if path.exists() else None


def _call(row):
    action = row.get("action") or {}
    calls = action.get("tool_calls") if isinstance(action, dict) else None
    return (calls or [{}])[0].get("name")


def _previous_result(prompt):
    """The previous step's result block, which opens the next prompt ("" if none)."""
    text = str(prompt or "")
    start = text.find("RESULTS FROM PREVIOUS STEP")
    if start == -1:
        return ""
    # The block ends at the next section: "</last_result>" (tagged prompts) or "### ".
    ends = [i for i in (text.find("</last_result>", start), text.find("### ", start)) if i != -1]
    return text[start:min(ends) if ends else len(text)]


def _result_failed(prompt):
    """Did the previous step's call fail?"""
    return '"success": false' in _previous_result(prompt)


def _failure_cause(prompt):
    """'navigate_to: Cannot reach (#, #) from (#, #) on ROUTE #': the failed call's tool
    and the first clause of its error, digits masked so one cause counts once."""
    block = _previous_result(prompt)
    tool = "?"
    at = block.find("Function: ")
    if at != -1:
        tool = block[at + len("Function: "):].split()[0] if block[at + len("Function: "):].split() else "?"
    error = ""
    # Prompts since Phase 2 leave out an "error" that only repeats "summary".
    for key in ('"error": "', '"summary": "'):
        at = block.find(key)
        if at != -1:
            error = block[at + len(key):].split('",')[0]
            break
    first = error.split(". ")[0].split(": ")[0][:70]
    return f"{tool}: " + "".join("#" if ch.isdigit() else ch for ch in first)


def report(run_dir, steps, llm_log=None):
    """The report over a run's last `steps` steps; llm_log defaults to the run's file."""
    rows = sorted((r for r in _rows(run_dir / "trajectory_history.jsonl") if _time(r)), key=_time)
    # A restarted run's file begins with the previous run's rows; keep this run's steps.
    rows = rows[-steps:]
    if len(rows) < 2:
        return {"error": f"only {len(rows)} steps in {run_dir}"}
    times = [_time(r) for r in rows]
    step_s = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    hours = (times[-1] - times[0]).total_seconds() / 3600 or None

    timing = {}
    for row in rows:
        for key, value in (row.get("timing") or {}).items():
            if isinstance(value, (int, float)):
                timing.setdefault(key, []).append(value)
    sections = {}
    for row in rows:
        for key, value in (row.get("prompt_sections") or {}).items():
            sections.setdefault(key, []).append(value)

    in_battle = {r.get("step"): bool((r.get("pre_state") or {}).get("is_in_battle")) for r in rows}
    model = {"overworld": {}, "battle": {}}
    log = llm_log or _llm_log_for(run_dir)
    if log is not None:
        for entry in _rows(log):
            if entry.get("agent_step") not in in_battle or entry.get("type") != "interaction":
                continue
            # The file can hold earlier runs' calls with the same step numbers. A step's
            # model call ends before its row is written, so the window opens earlier.
            when = _time(entry)
            if when is None or not times[0] - timedelta(seconds=MODEL_CALL_SLACK_S) <= when <= times[-1]:
                continue
            try:
                meta = ast.literal_eval(entry["metadata"]) if isinstance(entry.get("metadata"), str) else entry.get("metadata") or {}
            except (ValueError, SyntaxError):
                meta = {}
            usage = meta.get("token_usage") or {}
            mode = model["battle" if in_battle[entry["agent_step"]] else "overworld"]
            prompt, cached = usage.get("prompt_tokens"), usage.get("cached_tokens") or 0
            mode.setdefault("model_s", []).append(entry.get("duration"))
            mode.setdefault("prompt_tokens", []).append(prompt)
            mode.setdefault("cached_tokens", []).append(cached)
            mode.setdefault("uncached_tokens", []).append(prompt - cached if prompt is not None else None)
            mode.setdefault("output_tokens", []).append(usage.get("completion_tokens"))

    stalls, skip_until = [], -1
    for i in range(len(rows)):
        if i < skip_until:
            continue
        why = _stall_reason(rows[max(0, i - STALL_WINDOW + 1): i + 1])
        if why:
            stalls.append(f"{rows[i]['timestamp'][11:19]} {why.split(':')[0]}")
            skip_until = i + STALL_SKIP
    failed = sum(1 for r in rows[1:] if _result_failed(r.get("llm_prompt")))
    # Each failed call costs a whole step: the commonest causes are a speed item too.
    causes = {}
    for r in rows[1:]:
        if _result_failed(r.get("llm_prompt")):
            cause = _failure_cause(r.get("llm_prompt"))
            causes[cause] = causes.get(cause, 0) + 1
    completed = sum(
        1 for r, nxt in zip(rows, rows[1:])
        if _call(r) == "complete_direct_objective" and not _result_failed(nxt.get("llm_prompt"))
    )
    return {
        "run": run_dir.name,
        "steps": len(rows),
        "from": rows[0]["timestamp"],
        "to": rows[-1]["timestamp"],
        "step_s": _summary(step_s),
        "timing_s": {k: _summary(v) for k, v in timing.items()},
        "prompt_section_chars": {k: _summary(v) for k, v in sections.items()},
        "model": {mode: {k: _summary(v) for k, v in data.items()} for mode, data in model.items() if data},
        "quality": {
            "stalls": len(stalls),
            "stalls_per_hour": round(len(stalls) / hours, 2) if hours else None,
            "stall_events": stalls,
            "failed_calls_per_100_steps": round(100 * failed / (len(rows) - 1), 1),
            "failed_call_causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
            "objectives_completed": completed,
            "objectives_per_hour": round(completed / hours, 2) if hours else None,
        },
    }


def _print(result):
    if "error" in result:
        print(result["error"])
        return
    print(f"{result['run']}: {result['steps']} steps, {result['from'][:19]} → {result['to'][:19]}")
    s = result["step_s"]
    print(f"  step time: median {s.get('median')} s, p90 {s.get('p90')} s")
    for key, value in result["timing_s"].items():
        print(f"  {key}: median {value.get('median')} s, p90 {value.get('p90')} s")
    for mode, data in result["model"].items():
        parts = [f"{k} {v.get('median')}/{v.get('p90')}" for k, v in data.items()]
        print(f"  {mode} (n={next(iter(data.values())).get('n')}, median/p90): " + ", ".join(parts))
    if result["prompt_section_chars"]:
        widest = sorted(result["prompt_section_chars"].items(), key=lambda kv: -(kv[1].get("median") or 0))
        print("  prompt sections (chars, median): " + ", ".join(f"{k} {v.get('median')}" for k, v in widest))
    q = result["quality"]
    print(
        f"  quality: {q['stalls']} stalls ({q['stalls_per_hour']}/h), {q['failed_calls_per_100_steps']} failed calls"
        f" per 100 steps, {q['objectives_completed']} objectives ({q['objectives_per_hour']}/h)"
    )
    for event in q["stall_events"]:
        print(f"    stall {event}")
    for cause, count in q["failed_call_causes"].items():
        print(f"    failed x{count}: {cause}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--run", type=Path, help="a .pokeagent_cache run directory (default: the newest)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run_dir = args.run or _newest_run()
    if run_dir is None:
        print("no run found under .pokeagent_cache")
        return 1
    result = report(run_dir, args.steps)
    print(json.dumps(result, indent=2)) if args.json else _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
