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


# The macro hold is at the HTTP edge (server/app.py MacroHoldMiddleware), so these tests
# talk to the app over ASGI as /play and curl do, on one event loop as under uvicorn.
ROUTE101 = HERE / "test_movement_route101.state"
WALK = {"place": "Oldale Town Pokemon Center", "reasoning": "test"}


def _client(s):
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=s.srv.app), base_url="http://play", timeout=300)


async def _until_macro_presses(s, timeout=20.0):
    """Wait for the macro's first press of its own, read from the sequence reports (not
    /macro_status), so a server without the hold still gets the outside press mid-macro."""
    first = max(s.srv.sequence_reports, default=0)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if max(s.srv.sequence_reports, default=0) > first:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the macro never pressed a button")


def _manual_sequences(s):
    return [r for r in s.srv.sequence_reports.values() if r.get("source") == "manual"]


def test_a_press_from_play_is_refused_while_a_macro_runs():
    """Seen live (2026-11-05, Elite Four): use_item("REVIVE", "MANECTRIC") failed with
    "Could not put the START menu cursor on BAG": a fast A from the /play page landed
    mid-macro. go_to from ROUTE 101 to OLDALE TOWN's Pokémon Center is a long macro
    with helper macros inside it (navigate_to, go_to_exit): while it runs, /play's press
    (/action, source "manual") and an operator's press_buttons are refused with a message
    naming go_to, nothing of theirs is queued, pause and status still answer, and the
    walk arrives as it does alone (test_macros test_go_to_walks_map_after_map). After it,
    /action presses again, and the feed has the macro's start and end."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)

    async def scenario():
        async with _client(s) as client:
            walk = asyncio.create_task(client.post("/mcp/go_to", json=WALK))
            try:
                await _until_macro_presses(s)
                before = len(_manual_sequences(s))
                press = await client.post("/action", json={"buttons": ["A"], "speed": "fast", "source": "manual"})
                operator = await client.post("/mcp/press_buttons", json={"buttons": ["B"], "source": "implementor"})
                queued = len(_manual_sequences(s)) - before
                status = await client.get("/macro_status")
                health = (await client.get("/health")).json()
                pause = await client.post("/agent_pause", json={"paused": False, "source": "test"})
                feed = await client.get("/ai_feed")
            finally:
                walked = (await walk).json()
            after = await client.post("/action", json={"buttons": ["B"], "speed": "fast", "source": "manual"})
            idle = (await client.get("/macro_status")).json()
            events = (await client.get("/ai_feed")).json()["events"]
            return press, operator, queued, status, health, pause, feed, walked, after, idle, events

    press, operator, queued, status, health, pause, feed, walked, after, idle, events = asyncio.run(scenario())
    print("   ", press.status_code, press.text[:200])
    print("   ", walked.get("summary") or walked)
    assert press.status_code == 409, (press.status_code, press.text)
    assert queued == 0, f"{queued} refused press(es) reached the button queue"
    assert press.json()["error"].startswith("a macro (go_to) is running; try again when it ends."), press.json()
    assert press.json()["macro"]["tool"] == "go_to", press.json()
    assert operator.status_code == 409 and "go_to" in operator.json()["error"], (operator.status_code, operator.text)
    assert status.status_code == 200, (status.status_code, status.text)
    status = status.json()
    assert status["running"] and status["tool"] == "go_to" and status["since"] and status["refused"] == 2, status
    assert health["macro"]["running"] and health["macro"]["tool"] == "go_to", health
    assert pause.status_code == 200 and feed.status_code == 200, (pause.status_code, feed.status_code)
    assert walked["success"] is True and walked["status"] == "ARRIVED", walked
    assert s.srv._location_name() == "OLDALE TOWN POKEMON CENTER 1F", s.srv._location_name()
    assert after.status_code == 200 and after.json()["status"] == "success", (after.status_code, after.text)
    assert idle == {"running": False, "tool": None, "since": None, "elapsed_s": None, "refused": 0, "forced": 0}, idle
    start = next(e for e in events if e["kind"] == "macro_start")
    end = next(e for e in events if e["kind"] == "macro_end")
    assert events.index(end) < events.index(start), events  # newest first
    assert start["source"] == end["source"] == "go_to", (start, end)
    assert "2 press request(s) from other clients refused" in end["text"], end


def test_force_presses_through_a_macro_hold():
    """An operator's emergency: "force": true on /action goes through while go_to holds
    the buttons, and is counted in /macro_status."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)

    async def scenario():
        async with _client(s) as client:
            walk = asyncio.create_task(client.post("/mcp/go_to", json=WALK))
            try:
                await _until_macro_presses(s)
                before = len(_manual_sequences(s))
                # B does nothing on its own in the overworld, so the walk can go on.
                forced = await client.post("/action", json={"buttons": ["B"], "speed": "fast", "source": "manual", "force": True})
                queued = len(_manual_sequences(s)) - before
                status = await client.get("/macro_status")
            finally:
                await walk
            return forced, queued, status

    forced, queued, status = asyncio.run(scenario())
    print("   ", forced.status_code, status.text)
    assert forced.status_code == 200 and forced.json()["status"] == "success", (forced.status_code, forced.text)
    assert queued == 1, queued
    assert status.status_code == 200, (status.status_code, status.text)
    status = status.json()
    assert status["running"] and status["tool"] == "go_to" and status["forced"] == 1, status


def test_the_agent_reads_a_macro_hold_refusal_as_its_tool_result():
    """The agent's own call (agents/PokeAgent.py MCPToolAdapter.call_tool) refused by the
    hold while someone else's go_to runs: the model gets the server's message and the
    macro status, not raise_for_status's "409 Client Error: Conflict for url". Any other
    non-2xx answer keeps today's result."""
    import importlib
    from unittest import mock

    import httpx

    pa = importlib.import_module("agents.PokeAgent")  # the module, not the re-exported class
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    adapter = pa.MCPToolAdapter("http://play")

    async def scenario():
        loop = asyncio.get_running_loop()
        async with _client(s) as client:
            def post(url, json=None, timeout=None):  # requests.post, through this loop's client
                return asyncio.run_coroutine_threadsafe(client.post(url, json=json), loop).result()

            walk = asyncio.create_task(client.post("/mcp/go_to", json=WALK))
            try:
                await _until_macro_presses(s)
                with mock.patch.object(pa.requests, "post", post):
                    return await asyncio.to_thread(adapter.call_tool, "press_buttons", {"buttons": ["A"], "reasoning": "test"})
            finally:
                await walk

    result = asyncio.run(scenario())
    print("   ", result)
    assert result["success"] is False, result
    assert result["error"].startswith("a macro (go_to) is running; try again when it ends."), result
    assert result["macro"]["running"] and result["macro"]["tool"] == "go_to", result

    other = httpx.Response(500, json={"detail": "boom"}, request=httpx.Request("POST", "http://play/mcp/press_buttons"))
    with mock.patch.object(pa.requests, "post", lambda url, json=None, timeout=None: other):
        result = adapter.call_tool("press_buttons", {"buttons": ["A"]})
    assert result["success"] is False and "500 Internal Server Error" in result["error"], result


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
