"""Boulders, rocks and waterfalls on the way: the walk plans Rock Smash among Strength
pushes, never jams the way to an exit, says when only leaving the map helps, answers the
field moves' YES/NO by its cursor, and goes down and up waterfalls.

Seen live (2026-11-05):
- SEAFLOOR CAVERN ROOM1: an earlier walk pushed the boulder from (12,11) up column 12 to
  (12,7), where it fills the only corridor ((12,6) is a wall). Then go_to_exit("SEAFLOOR
  CAVERN ROOM2") failed with "the boulder at (5,11) was not pushed: the walk ended at
  (4,10)": the rock at (5,10) was walked through in the plan. The operator's way: smash
  (5,10), push (5,11) DOWN, walk row 11, push (12,11) RIGHT, up column 12 to (6,2).
- At the STRENGTH YES/NO a stray DOWN after A put the cursor on NO.
- VICTORY ROAD B2F: navigate_to said "No complete path" to the stairs at (19,12) and
  (5,26): the ways there go down a waterfall (and up one from (30,25)), and waterfalls
  were walls to the planner.

Fixtures (the live run's 2026-11-05 checkpoints, then the harness's own walks):
- test_victory_road_b1f.state: VICTORY ROAD B1F (17,26), the live checkpoint at 19:11;
  boulders (20,26), (21,25) and rocks (20,25), (21,26) on their map-file tiles. Party:
  AZUMARILL (WATERFALL, DIVE, SURF, STRENGTH), BRELOOM (ROCK SMASH), TROPIUS (FLY), ...;
  all eight badges.
- test_victory_road_b2f.state: VICTORY ROAD B2F (19,12), the island stairs, from the B1F
  fixture by go_to_exit("VICTORY ROAD B2F").
- test_victory_road_b2f_south.state: VICTORY ROAD B2F (31,25), beside the stairs to B1F
  (30,25), from the B2F fixture by navigate_to(31, 25) (Surf, up the west waterfall, down
  the east one).
- test_victory_road_b1f_on_stairs.state: VICTORY ROAD B1F (5,26) on the ladder just come
  up from B2F, from the B2F fixture by go_to_exit("VICTORY ROAD B1F (5, 26)").

SEAFLOOR CAVERN ROOM1 has no fixture (Dive is not on main yet): its tests plan on the
map file's tiles (map_file_state), as the room is after a map load.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_boulder_rocks.py [name-substring]
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

STATES = Path(__file__).resolve().parent
VR_B1F = STATES / "test_victory_road_b1f.state"
VR_B2F = STATES / "test_victory_road_b2f.state"
VR_B2F_SOUTH = STATES / "test_victory_road_b2f_south.state"
VR_B1F_ON_STAIRS = STATES / "test_victory_road_b1f_on_stairs.state"
BOULDER = "OBJ_EVENT_GFX_PUSHABLE_BOULDER"


def map_file_state(map_dir, boulders=None, moves=("STRENGTH", "ROCK_SMASH", "SURF", "WATERFALL")):
    """A pathfinding state for map_dir as a map load leaves it (boulders and rocks on their
    map-file tiles, unless `boulders` says where the boulders stand), with the party able
    to use `moves`: the planner's input without an emulator."""
    from utils.mapping.porymap_state import _readable_map_name, _static_grid, _static_tiles

    raw = json.loads((REPO / "pokemon_env" / "porymap" / "data" / "maps" / map_dir / "map.json").read_text())
    objects = raw.get("object_events") or []
    grid = [list(row) for row in _static_grid(map_dir)]
    porymap = {
        "grid": grid,
        "raw_tiles": [list(row) for row in _static_tiles(map_dir)],
        "dimensions": {"width": len(grid[0]), "height": len(grid)},
        "objects": [
            {"x": o["x"], "y": o["y"], "graphics_id": o["graphics_id"], "movement_type": o.get("movement_type", "")}
            for o in objects if o["graphics_id"] != BOULDER
        ],
        "boulders": boulders if boulders is not None else [(o["x"], o["y"]) for o in objects if o["graphics_id"] == BOULDER],
        "strength": "STRENGTH" in moves,
        "clearable": [(o["x"], o["y"]) for o in objects if o["graphics_id"] == "OBJ_EVENT_GFX_BREAKABLE_ROCK"]
        if "ROCK_SMASH" in moves else [],
        "surf": "SURF" in moves,
        "waterfall": "WATERFALL" in moves,
        "raw_warps": [(w["x"], w["y"]) for w in raw.get("warp_events") or []],
        "warps": [],
    }
    return {"map": {"porymap": porymap}, "player": {"location": _readable_map_name(raw["id"])}}


def _plan(state, start, goal):
    from utils.mapping.pathfinding import Pathfinder

    finder = Pathfinder()
    buttons = finder.find_path(start, goal, state, consider_npcs=True, allow_partial=False)
    return buttons, [(p["kind"], tuple(p["boulder"]), p["to"] and tuple(p["to"])) for p in finder.last_pushes]


ROOM1 = "SeafloorCavern_Room1"
ROOM1_ENTRANCE = (5, 17)  # the tile above the warp from SEAFLOOR CAVERN ENTRANCE (5,18)
ROOM1_TO_ROOM2 = (6, 2)


def test_seafloor_room1_smashes_the_rock_before_pushing():
    """The operator's way through ROOM1 from the entrance: smash (5,10), push (5,11) DOWN,
    push (12,11) RIGHT, then up column 12 to the stairs at (6,2)."""
    buttons, steps = _plan(map_file_state(ROOM1), ROOM1_ENTRANCE, ROOM1_TO_ROOM2)
    assert buttons, "no way through SEAFLOOR CAVERN ROOM1"
    assert steps == [("clear", (5, 10), None), ("push", (5, 11), (5, 12)), ("push", (12, 11), (13, 11))], steps


def test_a_rock_stops_a_push_and_is_no_floor():
    """Without ROCK SMASH the rock at (5,10) is a wall: no push is planned from its tile and
    the room has no way through (the live failure planned one from (5,10))."""
    buttons, steps = _plan(map_file_state(ROOM1, moves=("STRENGTH", "SURF")), ROOM1_ENTRANCE, ROOM1_TO_ROOM2)
    assert buttons is None, steps


def test_a_walk_to_a_tile_never_jams_the_corridor():
    """navigate_to(12, 8) in ROOM1: the shortest plan pushes the boulder (12,11) up column
    12 to (12,7), which seals the stairs at (6,2) (the live jam). The plan taken pushes it
    RIGHT instead and keeps every exit reachable."""
    from utils.mapping.pathfinding import Pathfinder

    class NoExitCheck(Pathfinder):
        def _reachable_exits(self, *args):
            return frozenset()

    state = map_file_state(ROOM1)
    jamming = NoExitCheck()
    assert jamming.find_path(ROOM1_ENTRANCE, (12, 8), state, consider_npcs=True, allow_partial=False)
    assert (12, 7) in {tuple(p["to"]) for p in jamming.last_pushes if p["to"]}, jamming.last_pushes
    buttons, steps = _plan(map_file_state(ROOM1), ROOM1_ENTRANCE, (12, 8))
    assert buttons and ("push", (12, 11), (12, 10)) not in steps, steps
    assert ("push", (12, 11), (13, 11)) in steps, steps


def test_a_jammed_room_says_to_leave_and_come_back():
    """With the boulder already at (12,7) (the live jam) there is no way to (6,2), and the
    answer says a map load (leave, come back) puts the boulders back."""
    from server import game_tools

    moves = {"STRENGTH", "ROCK_SMASH", "SURF"}
    game_tools.field_moves_reader = lambda: moves
    game_tools._field_moves_cache[:] = [0.0, set()]
    try:
        jammed = map_file_state(ROOM1, boulders=[(5, 11), (12, 7)])
        assert _plan(jammed, ROOM1_ENTRANCE, ROOM1_TO_ROOM2)[0] is None
        note = game_tools.boulder_reset_note(jammed, ROOM1_ENTRANCE, [ROOM1_TO_ROOM2])
        assert "leave this map and come back" in note, note
        assert game_tools.boulder_reset_note(map_file_state(ROOM1), ROOM1_ENTRANCE, [ROOM1_TO_ROOM2]) == ""
    finally:
        game_tools.field_moves_reader = None
        game_tools._field_moves_cache[:] = [0.0, set()]


def test_push_bounds_cover_the_dungeons():
    """The bounds hold the longest solutions measured on the map files: SEAFLOOR CAVERN
    ROOM8 (5,12) to (5,2) takes 9 pushes."""
    buttons, steps = _plan(map_file_state("SeafloorCavern_Room8"), (5, 12), (5, 2))
    assert buttons and sum(1 for kind, _, _ in steps if kind == "push") == 9, steps


VR_B2F_MAP = "VictoryRoad_B2F"


def test_b2f_plans_go_down_waterfalls_and_up_with_the_move():
    """VICTORY ROAD B2F on its map file: from the east stairs (43,2) and from (31,25) the
    ways to the island stairs (19,12) and to (5,26) go over the bridge and down the west
    waterfall, no WATERFALL needed; from the island to (31,25) the way goes up the west
    waterfall, so only with WATERFALL."""
    no_climb = ("STRENGTH", "ROCK_SMASH", "SURF")
    for start in ((43, 3), (31, 25)):
        for goal in ((19, 12), (5, 26)):
            buttons, steps = _plan(map_file_state(VR_B2F_MAP, moves=no_climb), start, goal)
            assert buttons, f"no way from {start} to {goal} without WATERFALL"
            assert not any(kind == "waterfall" for kind, _, _ in steps), steps
    assert _plan(map_file_state(VR_B2F_MAP, moves=no_climb), (19, 13), (31, 25))[0] is None
    buttons, steps = _plan(map_file_state(VR_B2F_MAP), (19, 13), (31, 25))
    assert buttons and "waterfall" in [kind for kind, _, _ in steps], steps


def test_presses_after_a_waterfall_ride_stay_on_the_path():
    """game_tools.jump_presses (it drops the press planned for a ledge's landing: the game
    jumps both tiles on one press) after a ride down a waterfall. A column from the top:
    water, two waterfall tiles, water, shore, a south ledge, floor, floor. The plan presses
    DOWN for the ride (three tiles), the shore, the ledge, its landing and the last tile.
    Counted one tile per press, the walk was two tiles behind at the ledge, missed it and
    kept the landing's press: one DOWN too many, a tile past the path. The column is made
    up (real behaviors and elevations) to keep the case to its two parts."""
    from pokemon_env.enums import MetatileBehavior as B
    from server import game_tools

    column = [
        (B.OCEAN_WATER, 1), (B.WATERFALL, 1), (B.WATERFALL, 1), (B.OCEAN_WATER, 1),
        (B.NORMAL, 3), (B.JUMP_SOUTH, 3), (B.NORMAL, 3), (B.NORMAL, 3),
    ]
    tiles = [[(0, behavior, 0, elevation)] for behavior, elevation in column]
    planned = ["DOWN"] * 5  # the ride, the shore, the ledge, its landing, the last tile
    assert game_tools.jump_presses((0, 0), planned, tiles) == ["DOWN"] * 4


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _server(state):
    from test_movement_feedback import server

    s = server()
    s.load(state, wild_encounters=False)
    return s


def test_navigate_to_climbs_and_rides_down_waterfalls():
    """From the island (19,12) to (31,25) and back: Surf, up the west waterfall with
    WATERFALL, across the lake, down the east one, land; back up the east one."""
    s = _server(VR_B2F)
    r = s.call("mcp_navigate_to", {"x": 31, "y": 25, "variance": "none", "reason": "across B2F"})
    print("   ", r.get("summary") or r.get("error"))
    assert (s.snapshot().x, s.snapshot().y) == (31, 25) and _location(s) == "VICTORY ROAD B2F", r
    assert "Climbed the waterfall" in r["summary"], r
    s.load(VR_B2F_SOUTH, wild_encounters=False)
    r = s.call("mcp_navigate_to", {"x": 19, "y": 13, "variance": "none", "reason": "back to the island"})
    print("   ", r.get("summary") or r.get("error"))
    assert (s.snapshot().x, s.snapshot().y) == (19, 13) and _location(s) == "VICTORY ROAD B2F", r


def test_navigate_to_a_warp_beyond_water_reports_the_warp():
    """navigate_to(19, 12) from (31,25): the walk lands on the island and takes the stairs
    at (19,12). It used to plan again on B1F and say "No path found from (17, 16)"."""
    s = _server(VR_B2F_SOUTH)
    r = s.call("mcp_navigate_to", {"x": 19, "y": 12, "variance": "none", "reason": "the island stairs"})
    print("   ", r.get("summary") or r.get("error"))
    assert _location(s) == "VICTORY ROAD B1F", r
    assert "No path found" not in str(r.get("summary")) + str(r.get("error")), r


def test_go_to_exit_from_the_stairs_just_come_up():
    """On VICTORY ROAD B1F's ladder (5,26), just come up from B2F: go_to_exit("VICTORY
    ROAD B2F") steps off and back on, and goes down."""
    s = _server(VR_B1F_ON_STAIRS)
    assert (s.snapshot().x, s.snapshot().y) == (5, 26)
    r = s.call("mcp_go_to_exit", {"exit": "VICTORY ROAD B2F", "reasoning": "back down"})
    print("   ", r.get("summary") or r.get("error"))
    assert r.get("success") is True and _location(s) == "VICTORY ROAD B2F", r


def test_victory_road_b1f_walk_smashes_and_pushes():
    """VICTORY ROAD B1F (17,26) to the ladder at (20,21): smash (20,25), push (20,26) DOWN,
    smash (21,26), the ROCK SMASH and STRENGTH questions answered by the walk."""
    s = _server(VR_B1F)
    r = s.call("mcp_go_to_exit", {"exit": "VICTORY ROAD 1F (20, 21)", "reasoning": "past the rocks"})
    print("   ", r.get("summary") or r.get("error"))
    assert r.get("success") is True and _location(s) == "VICTORY ROAD 1F", r
    assert r["summary"].count("used ROCK SMASH") == 2 and "pushed the boulder" in r["summary"], r


def _ask_with_cursor_on_no(s, stand, facing):
    """Walk to `stand`, face the object `facing` way, A, and once the field YES/NO takes
    input, a stray DOWN puts its cursor on NO (as seen live at the STRENGTH question)."""
    from server import menu_state as ms

    srv = s.srv
    s.call("mcp_navigate_to", {"x": stand[0], "y": stand[1], "variance": "none", "reason": "beside it"})
    assert (s.snapshot().x, s.snapshot().y) == tuple(stand)

    async def ask():
        await srv._macro_press([facing], "test")  # the object is in the way: this only turns
        await srv._macro_press(["A"], "test")
        asked, texts = await srv._talk_until(lambda: srv._mem(ms.task_running, ms.TASK_FIELD_YES_NO))
        assert asked, f"no YES/NO came up ({texts})"
        await srv._macro_press(["DOWN"], "test")
        assert srv._mem(ms.read_menu_cursor).cursor == 1, "the stray DOWN did not reach NO"
        assert await srv._answer_yes("test") is None
        for _ in range(6):  # "<MON> used ...!" and the rest of the text
            await srv._await_game(lambda: srv._box_waits_for_a() or srv._player_can_walk(), 900)
            if srv._player_can_walk():
                break
            if srv._box_waits_for_a():
                await srv._macro_press(["A"], "test")

    import asyncio

    asyncio.run(ask())


def test_rock_smash_question_is_answered_by_its_cursor():
    """The ROCK SMASH YES/NO with its cursor on NO: _answer_yes (what the walk's smash
    answers with) moves it to YES, and the rock at (20,25) is smashed."""
    from server.field_control import npc_at

    s = _server(VR_B1F)
    _ask_with_cursor_on_no(s, (19, 25), "RIGHT")
    with s.srv.memory_lock:
        assert not npc_at(s.srv.env.memory_reader, 20, 25), "the rock at (20,25) is still there"


def test_strength_question_is_answered_by_its_cursor():
    """The STRENGTH YES/NO at the boulder (20,26) with its cursor on NO: YES all the same,
    and STRENGTH is in use (FLAG_SYS_USE_STRENGTH)."""
    s = _server(VR_B1F)
    _ask_with_cursor_on_no(s, (19, 26), "RIGHT")
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader._is_event_flag_set(0x889), "STRENGTH is not in use"


def _run_all(only=""):
    import test_movement_feedback as harness

    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f) and only in n]
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
    sys.exit(1 if _run_all(sys.argv[1] if len(sys.argv) > 1 else "") else 0)
