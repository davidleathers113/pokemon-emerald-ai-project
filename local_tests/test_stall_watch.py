"""local_tools/stall_watch.py: which runs of decisions count as stuck (no game needed).

  python3 local_tests/test_stall_watch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_tools"))
from stall_watch import stall_reason


def _row(tool, args=None, loc="ROUTE 117", xy=(47, 5), ts="t"):
    return {"timestamp": ts, "location": loc, "player_coords": list(xy),
            "action": {"tool_calls": [{"name": tool, "args": {**(args or {}), "reasoning": "x"}}]}}


def test_three_walks_on_one_tile_are_a_stall():
    """Seen live: train / navigate_to / train, all from (47,5) on ROUTE 117."""
    rows = [_row("train", {"trainee": "BRELOOM"}), _row("navigate_to", {"x": 47, "y": 4}), _row("train", {"trainee": "BRELOOM"})]
    assert stall_reason(rows).startswith("SAME_TILE: 3 walk/train calls left the player at (47, 5) on ROUTE 117")


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
