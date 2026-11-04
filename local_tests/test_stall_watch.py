"""local_tools/stall_watch.py: which runs of decisions count as stuck (no game needed).

  python3 local_tests/test_stall_watch.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_tools"))
from stall_watch import newest_time, rows_after, stall_reason


def _row(tool, args=None, loc="ROUTE 117", xy=(47, 5), ts="t"):
    return {"timestamp": ts, "location": loc, "player_coords": list(xy),
            "action": {"tool_calls": [{"name": tool, "args": {**(args or {}), "reasoning": "x"}}]}}


def _prompt(foe_hp=30, lead="1. Combusken (Lv.24) HP: 68/69 Status: OK"):
    return (
        "=== PARTY STATUS ===\nPokemon Party (2 pokemon):\n  1. Marill (Lv.13) HP: 0/44 Status: OK\n"
        f"  2. {lead[3:]}\n"
        f"Wild battle vs ZIGZAGOON Lv13 (NORMAL), HP {foe_hp}/40\nYou: MARILL Lv13 (WATER), HP 0/44\n"
    )


def test_two_calls_in_turn_with_nothing_changing_are_a_stall():
    """Seen live 07:06-07:36: MARILL fainted and the game's replacement menu was up; the
    model alternated press_buttons(A) and switch_pokemon for 30 minutes at (17,14). Neither
    SAME_TILE nor REPEAT fired."""
    rows = []
    for i in range(8):
        call = ("press_buttons", {"buttons": ["A"]}) if i % 2 == 0 else ("switch_pokemon", {"pokemon": "COMBUSKEN"})
        row = _row(call[0], call[1], xy=(17, 14), ts=f"2026-11-04T07:0{i}:00")
        row["llm_prompt"] = _prompt()
        rows.append(row)
    assert stall_reason(rows[:5]) == ""  # not yet
    assert stall_reason(rows[:6]).startswith('CYCLE: press_buttons({"buttons": ["A"]}) / switch_pokemon('), stall_reason(rows[:6])
    # The same alternation while the battle moves on (foe HP falls) is a fight, not a stall.
    for i, row in enumerate(rows):
        row["llm_prompt"] = _prompt(foe_hp=40 - 4 * i)
    assert stall_reason(rows) == ""


def test_throwing_balls_that_get_used_up_is_not_a_repeat():
    """Seen live 08:42: four throw_ball("Great Ball") at a full-HP ELECTRIKE (Great Balls 8
    -> 4) paused the agent as a REPEAT. The ball count changed each time: a catch attempt."""
    rows = []
    for i in range(4):
        row = _row("throw_ball", {"ball": "Great Ball"}, xy=(20, 9), ts=f"2026-11-04T08:4{i}:00")
        row["llm_prompt"] = _prompt(foe_hp=40) + f"=== BAG STATUS ===\n  Poke Balls:\n    - Great Ball x{8 - i}\n"
        rows.append(row)
    assert stall_reason(rows) == "", stall_reason(rows)
    for row in rows:  # the same throws with nothing used up would be stuck
        row["llm_prompt"] = _prompt(foe_hp=40) + "=== BAG STATUS ===\n  Poke Balls:\n    - Great Ball x8\n"
    assert stall_reason(rows).startswith('REPEAT: throw_ball({"ball": "Great Ball"})'), stall_reason(rows)


def test_eight_decisions_with_nothing_changing_are_a_stall():
    calls = [("press_buttons", {"buttons": ["A"]}), ("press_buttons", {"buttons": ["B"]}), ("use_move", {"move": "1"}),
             ("switch_pokemon", {"pokemon": "2"}), ("press_buttons", {"buttons": ["RIGHT"]}), ("use_move", {"move": "2"}),
             ("press_buttons", {"buttons": ["START"]}), ("auto_advance_dialogue", {})]
    rows = []
    for i, (name, args) in enumerate(calls):
        row = _row(name, args, xy=(17, 14), ts=f"2026-11-04T07:1{i}:00")
        row["llm_prompt"] = _prompt()
        rows.append(row)
    assert stall_reason(rows[:7]) == ""
    assert stall_reason(rows).startswith("NO_PROGRESS: 8 decisions at (17, 14) on ROUTE 117"), stall_reason(rows)
    rows[3]["llm_prompt"] = _prompt(foe_hp=12)
    assert stall_reason(rows) == ""


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


def test_the_cutoff_is_taken_from_the_rows_own_clock():
    """Seen live: rows are stamped in UTC (06:35) and the Mac's clock read 01:35; a cutoff
    from the Mac's clock let every row through, and 11 s after an unpause the REPEAT
    already handled (use_move LEECH SEED x4) paused the agent again."""
    seeded = [_row("use_move", {"move": "LEECH SEED"}, ts=f"2026-11-04T06:34:4{i}.5") for i in range(4)]
    assert stall_reason(seeded).startswith("REPEAT")
    cutoff = newest_time(seeded)  # unpaused, no decision since
    assert cutoff == "2026-11-04T06:34:43.500000"
    assert stall_reason(rows_after(seeded, cutoff)) == ""
    after = seeded + [_row("use_move", {"move": "LEECH SEED"}, ts="2026-11-04T06:35:30")]
    assert rows_after(after, cutoff) == after[-1:]  # only the new decision is judged
    assert newest_time([]) == "0001-01-01T00:00:00" and rows_after(seeded, newest_time([])) == seeded


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


def _bounce_rows(minutes):
    """The live rows of 12:40-12:54 UTC: go_to_exit("west") from ROUTE 111 (0,66),
    go_to_exit("north edge of the map") and go_to("Route 113") from ROUTE 112 (39,46),
    in turn, about 40 s apart."""
    from datetime import datetime, timedelta

    calls = [
        ("go_to_exit", {"exit": "west"}, "ROUTE 111", (0, 66)),
        ("go_to_exit", {"exit": "north edge of the map"}, "ROUTE 112", (39, 46)),
        ("go_to", {"place": "Route 113"}, "ROUTE 112", (39, 46)),
    ]
    start = datetime(2026, 11, 4, 12, 40, 12)
    rows = []
    for i in range(int(minutes * 60 / 40) + 1):
        tool, args, loc, xy = calls[i % 3]
        row = _row(tool, args, loc=loc, xy=xy, ts=(start + timedelta(seconds=40 * i)).isoformat())
        row["llm_prompt"] = "=== PARTY STATUS ===\n  1. Breloom (Lv.27) HP: 72/75 Status: OK\n"
        row["outcome"] = {"success": True, "objectives_completed": []}
        rows.append(row)
    return rows


def test_going_back_and_forth_between_two_maps_is_a_stall():
    """Seen live 12:40-12:54 UTC: three calls in turn bounced the player between ROUTE 111
    and ROUTE 112 with the story objective unchanged. Each call changed the tile, so no
    other check fired."""
    rows = _bounce_rows(6)
    why = stall_reason(rows)
    assert why.startswith("WANDER: ") and "ROUTE 111 → ROUTE 112 → ROUTE 111" in why, why
    assert stall_reason(_bounce_rows(4)) == ""  # not five minutes yet
    # An objective completed, or a party that changed (a fight, a heal): something happened.
    done = _bounce_rows(6)
    done[-3]["outcome"]["objectives_completed"] = ["story_route_112"]
    assert stall_reason(done) == ""
    fought = _bounce_rows(6)
    fought[-2]["llm_prompt"] = "=== PARTY STATUS ===\n  1. Breloom (Lv.27) HP: 40/75 Status: OK\n"
    assert stall_reason(fought) == ""
    # Travel through maps without coming back is not wandering.
    trip = _bounce_rows(6)
    for i, row in enumerate(trip):
        row["location"] = ("ROUTE 111", "ROUTE 112", "FIERY PATH")[min(i // 4, 2)]
    assert stall_reason(trip) == "", stall_reason(trip)


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
