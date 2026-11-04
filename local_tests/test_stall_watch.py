"""local_tools/stall_watch.py: which runs of decisions count as stuck (no game needed).

  python3 local_tests/test_stall_watch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_tools"))
from stall_watch import rows_after, stall_reason


def _row(tool, args=None, loc="ROUTE 117", xy=(47, 5), ts="t"):
    return {"timestamp": ts, "location": loc, "player_coords": list(xy),
            "action": {"tool_calls": [{"name": tool, "args": {**(args or {}), "reasoning": "x"}}]}}


def test_three_walks_on_one_tile_are_a_stall():
    """Seen live: train / navigate_to / train, all from (47,5) on ROUTE 117."""
    # A row is where its decision was made; the next row shows where the call left the
    # player. (Before 2026-11-04 the third call's own row was taken as its end.)
    rows = [_row("train", {"trainee": "BRELOOM"}), _row("navigate_to", {"x": 47, "y": 4}), _row("train", {"trainee": "BRELOOM"})]
    assert stall_reason(rows) == ""  # the third call hasn't finished yet
    rows.append(_row("get_walkthrough", {"part": 1}))
    assert stall_reason(rows).startswith("SAME_TILE: 3 walk/train calls left the player at (47, 5) on ROUTE 117")


def test_a_call_that_moved_the_player_is_not_stuck():
    """Seen live: navigate_to, navigate_to (an unreachable door), then go_to_exit("east")
    from (12,4) on VERDANTURF TOWN reached ROUTE 117 (0,6), and was reported as a stall."""
    rows = [
        _row("navigate_to", {"x": 12, "y": 2}, loc="VERDANTURF TOWN", xy=(12, 4)),
        _row("navigate_to", {"x": 12, "y": 2}, loc="VERDANTURF TOWN", xy=(12, 4)),
        _row("go_to_exit", {"exit": "east"}, loc="VERDANTURF TOWN", xy=(12, 4)),
        _row("navigate_to", {"x": 11, "y": 1}, loc="ROUTE 117", xy=(0, 6)),
    ]
    assert stall_reason(rows) == ""
    # The same three, all failing, are still a stall.
    rows[-1] = _row("navigate_to", {"x": 11, "y": 1}, loc="VERDANTURF TOWN", xy=(12, 4))
    assert stall_reason(rows).startswith("SAME_TILE: 3 walk/train calls left the player at (12, 4) on VERDANTURF TOWN"), stall_reason(rows)


def test_rows_from_before_the_watch_started_are_not_judged():
    """Seen live: a restart's new trajectory file began with the previous run's rows, and
    the stall already reported and handled paused the agent again after the unpause."""
    old = [_row("navigate_to", {"x": 11, "y": 1}, xy=(0, 6), ts=f"2026-11-04T06:1{i}:00") for i in range(5)]
    assert stall_reason(old).startswith("SAME_TILE")
    assert rows_after(old, "2026-11-04T06:17:36") == []
    new = _row("go_to", {"place": "Route 117"}, xy=(0, 6), ts="2026-11-04T06:18:10")
    assert rows_after(old + [new], "2026-11-04T06:17:36") == [new]
    assert rows_after([{"timestamp": "garbled"}, new], "2026-11-04T06:17:36") == [new]


def test_battles_and_progress_are_not_stalls():
    battle = [_row("use_move", {"move": "ASTONISH"}) for _ in range(6)]
    assert stall_reason(battle[:3]) == ""  # still on one tile, but battling
    walking = [_row("navigate_to", {"x": i, "y": 5}, xy=(40 + i, 5)) for i in range(5)]
    assert stall_reason(walking) == ""
    two = [_row("go_to_exit", {"exit": "west"}), _row("go_to_exit", {"exit": "west"})]
    assert stall_reason(two) == ""  # not enough to say


def test_a_long_train_call_is_progress():
    """train can step in the grass for minutes and end on its own tile."""
    rows = [
        _row("train", {"trainee": "BRELOOM"}, ts="2026-11-04T05:00:00"),
        _row("train", {"trainee": "BRELOOM"}, ts="2026-11-04T05:04:00"),
        _row("train", {"trainee": "BRELOOM"}, ts="2026-11-04T05:08:00"),
        _row("use_move", {"move": "ABSORB"}, ts="2026-11-04T05:12:00"),
    ]
    assert stall_reason(rows) == ""
    quick = [_row("train", {"trainee": "BRELOOM"}, ts=f"2026-11-04T05:0{i}:00") for i in range(4)]
    assert stall_reason(quick).startswith("SAME_TILE"), stall_reason(quick)


def test_the_same_call_four_times_is_a_stall():
    rows = [_row("press_buttons", {"buttons": ["A"]}, xy=(9, 7)) for _ in range(4)]
    assert stall_reason(rows).startswith('REPEAT: press_buttons({"buttons": ["A"]}) 4 times in a row')
    rows[-1] = _row("press_buttons", {"buttons": ["B"]}, xy=(9, 7))
    assert stall_reason(rows) == ""


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
