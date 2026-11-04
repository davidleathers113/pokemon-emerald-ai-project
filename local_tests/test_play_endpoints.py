"""The controller page's read-only endpoints: /objectives, /dashboard, /ai_feed.

David (2026-11-04): "it would be cool to be able to see what the LLM is doing and maybe
a module with the pokemon party and a checklist of where we are at in the game and what
objectives we've accomplished and what has yet to be done." server/play.html renders
these three (server/play_panels.js; local_tests/test_play_panels.js checks the
rendering with node).

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_play_endpoints.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent


def test_objectives_report_done_current_and_upcoming_from_memory():
    """The live sequence, not objectives.json: story index 33 means 33 done before it
    (as loaded), the 34th current, the rest upcoming; battling the same way."""
    from agents.objectives.direct_objectives import DirectObjectiveManager

    s = server()
    real = s.srv.direct_objectives_manager
    try:
        m = DirectObjectiveManager()
        m.load_categorized_full_game_sequence(start_story_index=33, start_battling_index=2)
        s.srv.direct_objectives_manager = m
        r = asyncio.run(s.srv.get_objectives())
        story = r["story"]
        print("   ", story["done"], "of", story["total"], "current", story["current_id"])
        assert r["mode"] == "categorized", r["mode"]
        assert story["total"] == len(m.story_sequence) and story["done"] == 33, story["done"]
        assert story["current_id"] == m.story_sequence[33].id
        statuses = [o["status"] for o in story["objectives"]]
        assert statuses[:33] == ["done"] * 33 and statuses[33] == "current" and set(statuses[34:]) == {"upcoming"}
        assert story["objectives"][0]["completed_at"], story["objectives"][0]
        assert r["battling"]["current_id"] == m.battling_sequence[2].id, r["battling"]["current_id"]
        json.dumps(r)  # served as JSON
        s.srv.direct_objectives_manager = None
        empty = asyncio.run(s.srv.get_objectives())
        assert empty["story"]["total"] == 0 and empty["story"]["current_id"] is None, empty
    finally:
        s.srv.direct_objectives_manager = real


def test_dashboard_shows_the_party_place_badges_and_balls():
    """test_mt_chimney_busy_grunt.state (live): BRELOOM Lv37 leads a party of six on MT
    CHIMNEY; the EXP line from the prompt is there too."""
    s = server()
    s.load(HERE / "test_mt_chimney_busy_grunt.state", wild_encounters=False)
    started = time.time()
    r = asyncio.run(s.srv.get_dashboard())
    took = time.time() - started
    print("   ", r["location"], r["position"], r["money"], sum(r["badges"]), r["balls"], f"{took:.2f}s")
    print("   ", [(c["species"], c["level"], c["hp"], [(m["name"], m["type"], m["pp"], m["max_pp"]) for m in c["moves"]]) for c in r["party"][:2]])
    assert r["location"] == "MT CHIMNEY" and r["position"] == [23, 16], (r["location"], r["position"])
    assert len(r["party"]) == 6 and r["party"][0]["species"] == "BRELOOM" and r["party"][0]["lead"], r["party"][0]
    assert not any(c["lead"] for c in r["party"][1:])
    moves = r["party"][0]["moves"]
    assert len(moves) == 4 and all(m["type"] and 0 <= m["pp"] <= m["max_pp"] for m in moves), moves
    assert {"name": "SKY UPPERCUT", "type": "FIGHTING"}.items() <= next(m for m in moves if m["name"] == "SKY UPPERCUT").items()
    assert len(r["badges"]) == 8 and sum(r["badges"]) >= 3, r["badges"]
    assert r["balls"] and all(b["count"] > 0 for b in r["balls"]), r["balls"]
    assert r["exp_note"].startswith("EXP: BRELOOM is Lv"), r["exp_note"]
    assert took < 1.0, took  # one read under the memory lock: no visible cost to the game
    json.dumps(r)


def test_ai_feed_reads_the_trajectory_and_pause_events():
    """Steps from the trajectory file, newest first: the call, its result (from the next
    step's prompt), how long it took; the newest has no result yet and is "in flight".
    Pause and resume, with who did it, appear as events."""
    from utils.data_persistence.run_data_manager import get_cache_path

    s = server()
    path = get_cache_path("trajectory_history.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = path.read_bytes() if path.exists() else None
    result_text = 'RESULTS FROM PREVIOUS STEP:\n{\n  "success": false,\n  "error": "No exit of X \\"Y\\"",\n  "summary": "No exit of X \\"Y\\""\n}\n### OBJECTIVES'
    rows = [
        {"step": 7, "timestamp": "2026-11-04T16:00:00", "reasoning": "go west",
         "action": {"tool_calls": [{"name": "go_to_exit", "args": {"exit": "west", "reasoning": "go west"}}]}, "llm_prompt": "x"},
        {"step": 8, "timestamp": "2026-11-04T16:00:33.5", "reasoning": "talk",
         "action": {"tool_calls": [{"name": "talk_to", "args": {"x": 3, "y": 4}}]}, "llm_prompt": result_text},
    ]
    try:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        asyncio.run(s.srv.set_agent_pause({"paused": False}))
        asyncio.run(s.srv.set_agent_pause({"paused": True, "source": "stall_watch", "reason": "SAME_TILE: test"}))
        asyncio.run(s.srv.set_agent_pause({"paused": False, "source": "human"}))
        feed = asyncio.run(s.srv.get_ai_feed())
        print("   ", json.dumps(feed)[:400])
        newest, older = feed["steps"]
        assert newest["step"] == 8 and newest["result"] is None and newest["duration_s"] is None, newest
        assert older["tool"] == "go_to_exit" and older["args"] == {"exit": "west"}, older
        assert older["result"] == 'No exit of X "Y"' and older["success"] is False and older["duration_s"] == 33.5, older
        assert feed["in_flight"] == {"since": "2026-11-04T16:00:33.5", "step": 9}, feed["in_flight"]
        kinds = [(e["kind"], e["source"]) for e in feed["events"][:2]]
        assert kinds == [("resume", "human"), ("pause", "stall_watch")], kinds
        assert feed["events"][1]["text"] == "SAME_TILE: test"
    finally:
        if saved is None:
            path.unlink()
        else:
            path.write_bytes(saved)


def test_player_position_is_light():
    """Speed plan, Phase 5: the agent reads where a step ended from /player_position (it
    fetched the whole game state for that before)."""
    s = server()
    s.load(HERE / "test_mt_chimney_busy_grunt.state", wild_encounters=False)
    started = time.time()
    r = asyncio.run(s.srv.get_player_position())
    took = time.time() - started
    print("   ", r, f"{took:.3f}s")
    assert r == {"location": "MT CHIMNEY", "x": 23, "y": 16}, r
    assert took < 0.2, took


def test_the_page_and_its_renderers_are_served():
    s = server()
    page = asyncio.run(s.srv.get_play()).body.decode()
    assert 'id="feed"' in page and 'id="party"' in page and 'id="progress"' in page, "panels missing"
    assert '<script src="/play_panels.js"></script>' in page
    js = asyncio.run(s.srv.get_play_panels())
    assert js.media_type == "application/javascript" and b"renderFeed" in js.body


def _run_all():
    import test_movement_feedback as harness

    failures = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    if only:
        tests = [(n, f) for n, f in tests if any(o in n for o in only)]
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
