"""Movement feedback: a blocked direction press is detected, reported and not repeated.

Runs the real server game loop in-process (thread) on real savestates and calls the
/mcp endpoints as coroutines, so the button timing, the per-press judgement and the
sequence reports are the production code paths.

Fixtures:
  test_movement_route101.state  ROUTE 101 (13,2) facing north, trees above (the
                                position the live agent pressed UP from 9 times)
  test_phase23_overworld.state  Brendan's house 1F (4,5); Mom stands at (2,6)

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_movement_feedback.py
"""
import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ROUTE101 = REPO / "local_tests" / "test_movement_route101.state"
HOUSE = REPO / "local_tests" / "test_phase23_overworld.state"


class InProcessServer:
    """server.app with its game loop on a thread and a real emulator."""

    def __init__(self):
        os.chdir(tempfile.mkdtemp())  # importing the server creates cache dirs relative to cwd
        from pokemon_env.emulator import EmeraldEmulator
        from server import app as srv

        self.srv = srv
        env = EmeraldEmulator(rom_path=str(REPO / "Emerald-GBAdvance" / "rom.gba"))
        env.initialize()
        srv.env = env
        srv.running = True
        self.thread = threading.Thread(target=srv.game_loop, daemon=True)
        self.thread.start()

    def load(self, state_path):
        srv = self.srv
        # getattr: the same harness must also run against the pre-change server.
        with srv.memory_lock:
            srv.action_queue.clear()
            srv.current_action = None
            srv.action_frames_remaining = 0
            srv.release_frames_remaining = 0
            for name in ("_move_check", "_active_action_data", "last_blocked_press"):
                if hasattr(srv, name):
                    setattr(srv, name, None)
            srv.env.load_state(str(state_path))
            # Right after a load the game ignores input for a while (seen on both
            # fixtures); run past that so every test starts from a playable state.
            srv.env.tick(300)

    def call(self, endpoint, payload):
        if not hasattr(self.srv, endpoint):
            raise AssertionError(f"server has no endpoint {endpoint}")
        result = asyncio.run(getattr(self.srv, endpoint)(payload))
        if "summary" not in result:
            self.wait_idle()  # pre-change server: let the queued presses run before asserting
        return result

    def snapshot(self):
        """Player tile, map and facing read straight from RAM (independent of the server)."""
        with self.srv.memory_lock:
            return read_player(self.srv.env.memory_reader)

    def wait_idle(self, timeout=30.0):
        """For the pre-change server, whose endpoints return before the presses run."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = asyncio.run(self.srv.get_queue_status())
            if status["queue_empty"]:
                time.sleep(0.5)
                return
            time.sleep(0.05)

    def stop(self):
        self.srv.running = False
        self.thread.join(timeout=5)


class Player:
    def __init__(self, x, y, map_id, facing):
        self.x, self.y, self.map_id, self.facing = x, y, map_id, facing

    def __repr__(self):
        return f"Player({self.x},{self.y} map={self.map_id} facing={self.facing})"


def read_player(reader):
    # pokeemerald: gPlayerAvatar.objectEventId (+5); ObjectEvent.facingDirection (+0x18).
    obj = reader._read_u8(0x02037590 + 5)
    facing = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}.get(reader._read_u16(0x02037350 + obj * 0x24 + 0x18) & 0xF)
    sb1 = reader._read_u32(0x03005D8C)
    x, y = reader._read_s16(sb1), reader._read_s16(sb1 + 2)
    return Player(x, y, (reader._read_u8(sb1 + 4), reader._read_u8(sb1 + 5)), facing)


_SERVER = None


def server():
    global _SERVER
    if _SERVER is None:
        _SERVER = InProcessServer()
    return _SERVER


def test_blocked_press_into_trees_is_reported_and_rest_skipped():
    s = server()
    s.load(ROUTE101)
    snap = s.snapshot()
    assert (snap.x, snap.y, snap.facing) == (13, 2, "UP"), snap
    r = s.call("mcp_press_buttons", {"buttons": ["UP", "UP", "UP"], "reasoning": "go north"})
    print("   ", r)
    assert r.get("status") == "BLOCKED", r
    assert r["summary"] == (
        "BLOCKED moving UP at (13,2) on ROUTE 101: (13,1) is not walkable. 2 of 3 presses skipped. "
        'To go north to OLDALE TOWN use go_to_exit("north").'
    ), r["summary"]
    snap = s.snapshot()
    assert (snap.x, snap.y) == (13, 2)


def test_identical_repeat_of_blocked_press_is_refused():
    s = server()
    s.load(ROUTE101)
    s.call("mcp_press_buttons", {"buttons": ["UP"], "reasoning": "north"})
    executed_before = s.srv.latest_metrics.get("total_actions", 0)
    r = s.call("mcp_press_buttons", {"buttons": ["UP", "UP"], "reasoning": "north again"})
    print("   ", r)
    assert r["success"] is False and r.get("status") == "REFUSED", r
    assert "navigate_to" in r["error"] and 'go_to_exit("north")' in r["error"], r
    assert s.srv.latest_metrics.get("total_actions", 0) == executed_before  # nothing queued
    # Another direction from the same tile is still allowed, and moving clears the block.
    r = s.call("mcp_press_buttons", {"buttons": ["LEFT"], "reasoning": "west"})
    assert r["success"] and r.get("status") == "DONE", r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (12, 2), snap


def test_turning_in_place_is_not_blocked_but_the_next_press_is():
    s = server()
    s.load(HOUSE)
    r = s.call("mcp_press_buttons", {"buttons": ["LEFT", "LEFT"], "reasoning": "to Mom"})
    assert r.get("status") == "DONE", r
    snap = s.snapshot()
    assert (snap.x, snap.y, snap.facing) == (2, 5, "LEFT"), snap
    # Mom stands at (2,6). Facing west, DOWN only turns the player: not BLOCKED.
    r = s.call("mcp_press_buttons", {"buttons": ["DOWN"], "reasoning": "face Mom"})
    print("   ", r)
    assert r.get("status") == "DONE", r
    snap = s.snapshot()
    assert (snap.x, snap.y, snap.facing) == (2, 5, "DOWN"), snap
    # Now facing her, DOWN again cannot move: BLOCKED, and a person is named as the cause.
    r = s.call("mcp_press_buttons", {"buttons": ["DOWN", "DOWN"], "reasoning": "walk into Mom"})
    print("   ", r)
    assert r.get("status") == "BLOCKED", r
    assert "(2,6) is occupied by someone" in r["summary"] and "1 of 2 presses skipped" in r["summary"], r
    # A person may walk away, so the same press is not refused afterwards.
    r = s.call("mcp_press_buttons", {"buttons": ["DOWN"], "reasoning": "again"})
    assert r.get("status") != "REFUSED", r


def test_two_presses_same_direction_blocked_after_turn():
    """Facing west, UP into a tree: the 1st press turns, the 2nd is the BLOCKED one."""
    s = server()
    s.load(ROUTE101)
    s.call("mcp_press_buttons", {"buttons": ["LEFT"], "reasoning": "west"})  # (12,2) facing west
    r = s.call("mcp_press_buttons", {"buttons": ["UP", "UP", "UP"], "reasoning": "north"})
    print("   ", r)
    # (12,1) is a tree. Whether the first UP only turns or bumps, the run stops by the 2nd.
    assert r.get("status") == "BLOCKED", r
    assert "(12,1) is not walkable" in r["summary"], r
    assert ("1 of 3 presses skipped" in r["summary"]) or ("2 of 3 presses skipped" in r["summary"]), r


def test_a_press_that_does_nothing_says_so():
    """Seen live: 10 A presses at a Pokemon Center door, 6 tiles from the nurse."""
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "talk"})
    print("   ", r)
    assert "Nothing happened" in r["summary"] and "(13,1)" in r["summary"], r
    # Seen live: after a battle the model kept "fighting" for 13 minutes on empty grass.
    assert "no battle is going on" in r["summary"], r
    assert "in a row" not in r["summary"], r
    r = s.call("mcp_press_buttons", {"buttons": ["A", "A"], "reasoning": "attack"})
    assert "A has now done nothing here 2 times in a row" in r["summary"], r
    # Facing Mom, A opens her dialogue: no such note.
    s.load(HOUSE)
    s.call("mcp_press_buttons", {"buttons": ["LEFT", "LEFT", "DOWN"], "reasoning": "to Mom"})
    r = s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "talk to Mom"})
    print("   ", r)
    assert "Nothing happened" not in r["summary"], r


def test_ledge_jump_is_not_blocked_and_no_press_is_lost():
    s = server()
    s.load(ROUTE101)
    s.call("mcp_press_buttons", {"buttons": ["LEFT"] * 5, "reasoning": "to x=8"})
    snap = s.snapshot()
    assert (snap.x, snap.y) == (8, 2), snap
    # (8,6) is a ledge: the 4th DOWN jumps from (8,5) to (8,7). The 5th DOWN used to be
    # swallowed by the jump; now it waits for the landing.
    r = s.call("mcp_press_buttons", {"buttons": ["DOWN"] * 6, "reasoning": "down over the ledge"})
    print("   ", r)
    snap = s.snapshot()
    assert (snap.x, snap.y) == (8, 9), snap
    assert r.get("status") == "DONE", r


def _run_all():
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        started = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - started:.1f}s)")
        except Exception as e:  # report every failure, not just the first
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    if _SERVER is not None:
        _SERVER.stop()
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
