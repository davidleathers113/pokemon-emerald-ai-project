"""go_to_exit walks through map edges, doors, stairs and exit mats; navigate_to reports
unwalkable or unreachable targets instead of silently stopping short.

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_movement_exits.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import HOUSE, ROUTE101, server


HERE = Path(__file__).resolve().parent
BATTLE = HERE / "test_battle_wurmple.state"  # Petalburg Woods, move menu vs Lyle's Wurmple


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def test_walking_tools_refuse_during_a_battle():
    """Seen live: between two of Bug Catcher Lyle's Wurmple the model called navigate_to
    and go_to_exit; the presses moved the battle cursor and the results said only "did
    not leave the map"."""
    s = server()
    s.load(BATTLE)
    for endpoint, payload in (
        ("mcp_go_to_exit", {"exit": "ROUTE 104", "reasoning": ""}),
        ("mcp_navigate_to", {"x": 16, "y": 39, "variance": "none", "reason": ""}),
    ):
        r = s.call(endpoint, payload)
        print("   ", r)
        assert r["success"] is False and "battle is still going on" in r["error"], r
        assert "buttons_queued" not in r, r
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.is_in_battle()
    # The battle itself still plays: A on the highlighted EMBER finishes this Wurmple.
    for _ in range(4):
        assert s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": ""})["success"]


def test_go_to_exit_north_edge_from_route101_reaches_oldale():
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    r = s.call("mcp_go_to_exit", {"exit": "north", "reasoning": "to Oldale"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "OLDALE TOWN", _location(s)
    assert "OLDALE TOWN" in r["summary"], r


def test_go_to_exit_that_arrives_late_is_not_a_failure():
    """Seen live: go_to_exit("METEOR FALLS 1F 1R") from ROUTE 114 went through the door,
    then answered "No exit of METEOR FALLS 1F 1R matches 'METEOR FALLS 1F 1R'": the map
    change had not shown by the time the walk was judged, so it planned again from the
    new map. Here the first wait for the map change is made to report the old map (a slow
    fade); the walk still counts as arrived."""
    import asyncio
    import dataclasses

    s = server()
    s.load(ROUTE101, wild_encounters=False)
    real = s.srv._await_map_change
    calls = []

    async def slow_fade(old_map_id, timeout_s):
        calls.append(timeout_s)
        snap = await real(old_map_id, timeout_s)
        if len(calls) == 1 and snap is not None:
            return dataclasses.replace(snap, map_id=old_map_id)
        return snap

    s.srv._await_map_change = slow_fade
    try:
        r = asyncio.run(s.srv.mcp_go_to_exit({"exit": "north", "reasoning": "test"}))
    finally:
        s.srv._await_map_change = real
    print("   ", r.get("summary"))
    assert _location(s) == "OLDALE TOWN", _location(s)
    assert r["success"] is True and "Arrived" in r["summary"], r


def test_go_to_exit_by_destination_name():
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    r = s.call("mcp_go_to_exit", {"exit": "Oldale Town", "reasoning": "to Oldale"})
    assert r["success"] is True and _location(s) == "OLDALE TOWN", r


def test_go_to_exit_south_from_a_corner_of_oldale_uses_the_real_opening():
    """Seen live: from Oldale (15,19) go_to_exit("south") pressed DOWN into Route 101's
    trees 20+ times. Oldale's whole south row is open ground, but Route 101 continues
    only under x=8-11, so the exit list and the route must both use that opening."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_navigate_to", {"x": 15, "y": 19, "variance": "none", "reason": "corner"})
    assert r["success"] is True, r
    r = s.call("mcp_go_to_exit", {"exit": "south", "reasoning": "back to Littleroot"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "ROUTE 101", _location(s)


def test_go_to_exit_through_a_building_door():
    """Route 101 -> Oldale, then into the POKEMON CENTER through its front door."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_go_to_exit", {"exit": "POKEMON CENTER", "reasoning": "heal"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "OLDALE TOWN POKEMON CENTER 1F", _location(s)


def test_go_to_exit_through_house_exit_mat():
    """House exits are arrow warps: they fire when you walk off the mat, not onto it."""
    s = server()
    s.load(HOUSE)
    r = s.call("mcp_go_to_exit", {"exit": "LITTLEROOT TOWN", "reasoning": "go outside"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "LITTLEROOT TOWN", _location(s)


def test_go_to_exit_up_the_stairs():
    s = server()
    s.load(HOUSE)
    r = s.call("mcp_go_to_exit", {"exit": "Brendans House 2F", "reasoning": "upstairs"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "LITTLEROOT TOWN BRENDANS HOUSE 2F", _location(s)


def test_every_exits_line_names_its_own_exit():
    """Seen live in ROUTE 112 CABLE CAR STATION: go_to_exit("door/warp at (6, 12) -> ROUTE
    112"), the EXITS line copied whole, answered "No exit … matches" (two doors lead to
    ROUTE 112). Every line the list prints names exactly its own exit: METEOR FALLS 1F 1R
    has two warps to B1F 1R; ROUTE 112 (test_route111_south_to_113.state is saved at
    (39,46) there) two doors into the CABLE CAR STATION."""
    from server import game_tools
    from server.field_control import match_exits

    s = server()
    for state in ("test_meteor_falls_before_scene.state", "test_route111_south_to_113.state"):
        s.load(HERE / state, wild_encounters=False)
        with s.srv.memory_lock:
            _, offset, source = game_tools.load_pathfinding_state(s.srv.env)
        exits = game_tools.list_map_exits(source["json_map"], source.get("porymap_map_name"), offset)
        print("   ", _location(s), [e["label"] for e in exits])
        assert len(exits) > 2, exits
        for e in exits:
            assert match_exits(exits, e["label"]) == [e], e["label"]
    # Two warps lead to FIERY PATH from here; the copied line walks to the one it names.
    r = s.call("mcp_go_to_exit", {"exit": "door/warp at (11, 36) -> FIERY PATH", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert _location(s) == "FIERY PATH" and "(11, 36)" in r["summary"], (_location(s), r)


def test_go_to_exit_unknown_name_lists_the_exits():
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    r = s.call("mcp_go_to_exit", {"exit": "west", "reasoning": ""})
    assert r["success"] is False, r
    assert "north edge -> OLDALE TOWN" in r["error"] and "south edge -> LITTLEROOT TOWN" in r["error"], r


def test_navigate_to_unwalkable_target_says_so():
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    # (13,0) is a tree and every tile around it is too: refuse with a reason.
    r = s.call("mcp_navigate_to", {"x": 13, "y": 0, "variance": "none", "reason": ""})
    print("   ", r)
    assert r["success"] is False and "not walkable" in r["error"], r
    # (12,1) is a tree; the nearest reachable tile is (12,2): walk there and say so.
    r = s.call("mcp_navigate_to", {"x": 12, "y": 1, "variance": "none", "reason": ""})
    print("   ", r)
    assert "not walkable" in r.get("note", ""), r
    assert r["summary"].startswith("Did not reach (12,1)."), r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (12, 2), snap
    # Off the map: point at go_to_exit instead of clamping to the edge.
    r = s.call("mcp_navigate_to", {"x": 9, "y": -1, "variance": "none", "reason": ""})
    assert r["success"] is False and "go_to_exit" in r["error"], r


def test_navigate_to_a_door_reports_the_new_map():
    """Seen live: navigate_to onto a shop door said "Did not reach", and the model walked
    back out and in again."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_navigate_to", {"x": 6, "y": 16, "variance": "none", "reason": "Pokemon Center door"})
    print("   ", r)
    assert _location(s) == "OLDALE TOWN POKEMON CENTER 1F", _location(s)
    assert r["summary"].startswith("Went through a door/warp: now at") and "POKEMON CENTER 1F" in r["summary"], r


def test_an_edge_exit_never_lands_in_a_dead_end_pocket():
    """Seen live: go_to_exit("east") from VERDANTURF TOWN (19,6) took the nearest edge tile
    and landed on ROUTE 117 (0,6), a one-tile notch walled in on three sides; every walk
    and train call from there failed until the player was walked out by hand. Route 117
    goes on from y=8. test_verdanturf_east.state is the player at (19,6)."""
    from server.game_tools import _edge_dead_ends

    assert 6 in _edge_dead_ends("VerdanturfTown", "MAP_ROUTE117", "right", 0)
    assert 8 not in _edge_dead_ends("VerdanturfTown", "MAP_ROUTE117", "right", 0)
    s = server()
    s.load(Path(__file__).resolve().parent / "test_verdanturf_east.state", wild_encounters=False)
    r = s.call("mcp_go_to_exit", {"exit": "east", "reasoning": "to Route 117"})
    print("   ", r.get("summary") or r)
    snap = s.snapshot()
    assert r["success"] is True and _location(s) == "ROUTE 117", r
    assert (snap.x, snap.y) != (0, 6), snap
    # From where it landed, Route 117 goes on east.
    r = s.call("mcp_press_buttons", {"buttons": ["RIGHT"], "reasoning": "on east"})
    assert r.get("status") == "DONE" and s.snapshot().x == snap.x + 1, r


def test_walks_smash_a_rock_the_party_can_clear():
    """Seen live on ROUTE 111: the breakable rocks at (19,100) and (18,101) block the only
    way north; with HM06 taught and the Dynamo Badge, every walk still said "no walkable
    path". test_route111_rocks.state: (19,103) below them, BRELOOM knows ROCK SMASH, the
    Dynamo Badge set (made from test_mauville_gym_live.state with RAM and teach_move).
    The rocks' hide flags are FLAG_TEMP_*: they come back each time the map loads."""
    from server import game_tools

    here = Path(__file__).resolve().parent
    s = server()
    s.load(here / "test_route111_rocks.state", wild_encounters=False)
    with s.srv.memory_lock:
        assert game_tools.clearable_obstacles("Route111") == {(18, 101): "ROCK_SMASH", (19, 100): "ROCK_SMASH"}
        areas = game_tools._route_areas("Route111")  # what go_to_exit and go_to route with
        assert areas[(19, 103)] == areas[(19, 99)], "the rocks should not split the route"
    # (19,99): where the player stood live after smashing (19,100) by hand.
    r = s.call("mcp_navigate_to", {"x": 19, "y": 99, "variance": "none", "reason": "north past the rocks"})
    print("   ", r.get("summary"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == (19, 99), (snap, r)
    assert "Used ROCK SMASH at" in r["summary"], r
    # Without the Dynamo Badge (RAM: FLAG_BADGE03_GET 0x869 cleared) the rocks are walls.
    s.load(here / "test_route111_rocks.state", wild_encounters=False)
    reader = s.srv.env.memory_reader
    with s.srv.memory_lock:
        addr = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET + 0x869 // 8
        s.srv.env.core.memory.u8[addr] = reader._read_u8(addr) & ~(1 << (0x869 % 8))
        reader._invalidate_mem_cache()
    game_tools._field_moves_cache[0] = 0.0  # forget the cached answer
    with s.srv.memory_lock:
        assert game_tools.clearable_obstacles("Route111") == {}
        areas = game_tools._route_areas("Route111")
        assert areas[(19, 103)] != areas[(19, 99)], "without the badge the rocks are walls"
    r = s.call("mcp_navigate_to", {"x": 19, "y": 99, "variance": "none", "reason": "north"})
    snap = s.snapshot()
    assert (snap.x, snap.y) != (19, 99), (snap, r)


# A loaded Mac, on purpose: the game loop paced at this many frames per second whether
# busy or idle. Four suites at once ran the emulator down to ~22 fps (its turbo target is
# 480), and the walks' waits, written in seconds, gave up while the game still played.
SLOW_FPS = 15.0


class _slow_game:
    def __init__(self, s, fps=SLOW_FPS):
        self.srv, self.fps = s.srv, fps

    def __enter__(self):
        self.saved = (self.srv.GAME_FPS, self.srv.TURBO_FPS)
        self.srv.GAME_FPS = self.srv.TURBO_FPS = self.fps

    def __exit__(self, *exc):
        self.srv.GAME_FPS, self.srv.TURBO_FPS = self.saved


def test_a_rock_is_smashed_when_the_emulator_runs_slow():
    """test_walks_smash_a_rock_the_party_can_clear failed under load: "BLOCKED moving UP at
    (18,102) on ROUTE 111: (18,101) is occupied by someone", no ROCK SMASH. At 15 fps
    the same: A was pressed whenever a box showed, so the presses landed while the lines
    still printed (only hurrying them), the ten allowed ran out with "BRELOOM used ROCK
    SMASH!" still up, and the wait after them (240 frames, but at most 10 s: 150 frames
    at 15 fps) ended before the rock was gone. Now A waits for each box's arrow or the
    YES/NO, and the waits count frames."""
    from server import game_tools

    s = server()
    s.load(HERE / "test_route111_rocks.state", wild_encounters=False)
    game_tools._field_moves_cache[0] = 0.0  # a test before may have cleared the badge
    with _slow_game(s):
        r = s.call("mcp_navigate_to", {"x": 19, "y": 99, "variance": "none", "reason": "north past the rocks"})
    print("   ", r.get("summary"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == (19, 99), (snap, r)
    assert "Used ROCK SMASH at" in r["summary"], r


def test_go_to_exit_through_a_door_when_the_emulator_runs_slow():
    """test_warp_routes' go_to_exit into LAVARIDGE TOWN GYM 1F failed under load: "Did
    not leave the map … the game took control of the player". The walk there is ~330
    game frames and the door (open, step in, fade, load, fade in) ~90-100 more, measured
    at 15 and 480 fps. The walk's wait was ~1.5x the walk at 80 fps in seconds and the
    door's 4 s, then 1 s more: at 8 fps the walk's wait ran out first and the door never
    got its frames (old code: "the walk ended without leaving the map")."""
    s = server()
    s.load(HERE / "test_lavaridge_after_chimney.state", wild_encounters=False)
    with _slow_game(s, fps=8.0):
        r = s.call("mcp_go_to_exit", {"exit": "LAVARIDGE TOWN GYM 1F", "reasoning": "test"})
    print("   ", r.get("summary"))
    assert r["success"] is True and "Arrived" in r["summary"], r
    assert _location(s) == "LAVARIDGE TOWN GYM 1F", _location(s)


def test_walks_go_down_jagged_pass_ledges_and_not_bumpy_slopes():
    """Seen live (15:47-15:49 UTC, REPEAT stall): from JAGGED PASS (14,5) every
    heal_at_pokecenter said "Cannot reach 'ROUTE 112' … no walkable path"; by hand the
    pass goes down ledge after ledge. Two causes: the south ledges are elevation 0 between
    elevation-3 floor, and the step off a ledge counted as an elevation change (pret
    IsElevationMismatchAt: elevation 0 never mismatches); and the bumpy slopes ((18,8)...)
    were drawn as one-way slides, though on foot they collide (pret CheckAcroBikeCollision).
    test_jagged_pass_top.state is the live save."""
    from server import game_tools

    s = server()
    s.load(HERE / "test_jagged_pass_top.state", wild_encounters=False)
    assert _location(s) == "JAGGED PASS", _location(s)
    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    grid = state["map"]["porymap"]["grid"]
    assert grid[8][18] == "#" and grid[9][14] == "↓", (grid[8][18], grid[9][14])
    path = game_tools.pathfinder.find_path((14, 5), (14, 40), state, consider_npcs=False, allow_partial=False)
    assert path, "no route down the pass"
    r = s.call("mcp_go_to", {"place": "Route 112", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    here = s.snapshot()
    # A hiker at (16,25) spots the player on the way down; his battle is a fair stop, once
    # past the ledges (the walk reads his speech through, as the heal walk does).
    assert _location(s) == "ROUTE 112" or ("a trainer battle started" in r.get("error", "") and here.y > 12), (here, r)
    # Seen live 16:05-16:07 UTC, the same cause: from ROUTE 112 (7,47) the CABLE CAR
    # STATION had "no walkable path"; by hand RIGHT jumped the east ledges on row 52
    # (elevation 0 between elevation-3 floor). test_route111_south_to_113.state is on
    # ROUTE 112.
    s.load(HERE / "test_route111_south_to_113.state", wild_encounters=False)
    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    for start in ((7, 47), (7, 52)):
        assert game_tools.pathfinder.find_path(start, (28, 27), state, consider_npcs=False, allow_partial=False), start


def test_walks_go_round_floor_switches():
    """Seen live in MAUVILLE CITY GYM: on the south-west switch (0,15) the barriers were
    open, then talk_to's walk to Wattson crossed the switches at (4,12) and (3,9) and
    flipped them shut. test_mauville_gym_switch1.state is the player on (0,15) just after
    pressing it; Wattson stands at (5,2)."""
    from server import game_tools
    from server.field_control import DIRECTION_DELTAS

    switches = {(4, 12), (3, 9), (8, 9)}
    s = server()
    s.load(Path(__file__).resolve().parent / "test_mauville_gym_switch1.state")
    with s.srv.memory_lock:
        state, offset, source = game_tools.load_pathfinding_state(s.srv.env)
        assert source["porymap_map_name"] == "MauvilleCity_Gym"
        assert switches <= set(state["map"]["porymap"]["triggers"]), state["map"]["porymap"]["triggers"]
        buttons = game_tools.pathfinder.find_path((0, 15), (5, 3), state, consider_npcs=True, allow_partial=False)
        # The same walk with no trigger list, as before: it crosses a switch.
        state["map"]["porymap"]["triggers"] = []
        old = game_tools.pathfinder.find_path((0, 15), (5, 3), state, consider_npcs=True, allow_partial=False)

    def tiles(presses):
        x, y, out = 0, 15, []
        for b in presses or []:
            dx, dy = DIRECTION_DELTAS[b]
            x, y = x + dx, y + dy
            out.append((x, y))
        return out

    print("    new:", tiles(buttons), "\n    old:", tiles(old))
    assert buttons and tiles(buttons)[-1] == (5, 3) and not switches & set(tiles(buttons)), tiles(buttons)
    assert switches & set(tiles(old)), tiles(old)
    r = s.call("mcp_navigate_to", {"x": 5, "y": 3, "variance": "none", "reason": "to Wattson"})
    print("   ", r.get("summary"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == (5, 3), (snap, r)
    # navigate_to's operator-only avoid list reaches the planner. (Not (4,10): that is the
    # only switch-free corridor, so avoiding it rightly falls back to crossing a switch.)
    s.load(Path(__file__).resolve().parent / "test_mauville_gym_switch1.state")
    r = s.call("mcp_navigate_to", {"x": 5, "y": 3, "variance": "none", "reason": "test", "avoid": [[7, 14]]})
    snap = s.snapshot()
    assert (snap.x, snap.y) == (5, 3), (snap, r)
    bad = s.call("mcp_navigate_to", {"x": 5, "y": 3, "reason": "test", "avoid": [["a"]]})
    assert bad["success"] is False and "avoid must be a list" in bad["error"], bad


def test_walks_never_step_on_a_door_they_are_not_going_to():
    """Seen live: from ROUTE 110 (16,89), right below the Cycling Road gate's door at
    (16,88), every navigate_to north ended inside ROUTE 110 SEASIDE CYCLING ROAD SOUTH
    ENTRANCE. test_route110_gate.state is that tile. The door you ask for still works."""
    s = server()
    here = Path(__file__).resolve().parent
    s.load(here / "test_route110_gate.state")
    r = s.call("mcp_navigate_to", {"x": 34, "y": 54, "variance": "none", "reason": "north to the rival"})
    print("   ", r.get("summary"))
    assert _location(s) == "ROUTE 110", (_location(s), r)
    s.load(here / "test_route110_gate.state")
    r = s.call("mcp_navigate_to", {"x": 16, "y": 88, "variance": "none", "reason": "the gate itself"})
    assert "CYCLING ROAD SOUTH ENTRANCE" in _location(s), (_location(s), r)
    # The grid marks MAUVILLE CITY BIKE SHOP's whole floor 'D'; blocking by that letter
    # walled the player in at (4,5). Only map.json's warp tiles count.
    s.load(here / "test_bike_shop.state")
    r = s.call("mcp_go_to_exit", {"exit": "MAUVILLE CITY", "reasoning": "test"})
    print("   ", r.get("summary"))
    assert _location(s) == "MAUVILLE CITY", (_location(s), r)
    # Seen in a test: walking toward grass beside ROUTE 117's Day Care door, the path
    # ran over the door (the warp tiles were dropped on the way to the planner).
    s.load(here / "test_route117_blocked.state")
    s.call("mcp_navigate_to", {"x": 50, "y": 1, "variance": "none", "reason": "test"})
    assert _location(s) == "ROUTE 117", _location(s)
    s.load(here / "test_route117_blocked.state")
    s.call("mcp_go_to_exit", {"exit": "ROUTE 117 POKEMON DAY CARE", "reasoning": "test"})
    assert _location(s) == "ROUTE 117 POKEMON DAY CARE", _location(s)  # asked for: still works


def test_a_walk_blocked_only_by_people_names_them():
    """Seen live: a Magma grunt on (13,2) in front of SPACE CENTER 1F's stairs gave a bare
    "No path found" (no fixture there: the grunts are gone once the scene is over). When a
    way exists with people left out, the answer names each person on it and, for a
    trainer not beaten yet, the talk_to that battles them.

    test_aqua_hideout_b2f.state (31,17): Matt, not beaten, stands at (23,19) in the only
    way west to (14,16). test_route117_blocked.state (47,5): the Day Care man at (47,4)
    stands in the gap north."""
    from server import game_tools

    s = server()
    s.load(HERE / "test_aqua_hideout_b2f.state", wild_encounters=False)
    # The party's field moves are kept for a second: the way west crosses water, and the
    # test before (another fixture's party) would otherwise still answer. Seen: run after
    # test_a_rock_is_smashed_when_the_emulator_runs_slow, no way west was found at all.
    game_tools._field_moves_cache[0] = 0.0
    with s.srv.memory_lock:
        r = game_tools.navigate_to_direct(s.srv.env, 14, 16, variance="none")
    said = r.get("note") or r.get("error") or ""
    print("   ", said)
    assert "Only people block the way: MATT at (23,19), a trainer not beaten yet: talk_to(23, 19) battles them" in said, r
    assert [(p["x"], p["y"], p["trainer"]) for p in r["people_in_the_way"]] == [(23, 19, True)], r

    s.load(HERE / "test_route117_blocked.state", wild_encounters=False)
    r = s.call("mcp_navigate_to", {"x": 45, "y": 0, "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    said = r.get("summary") or r.get("error") or ""
    assert "Only people block the way: OLD MAN at (47,4)." in said and "talk_to" not in said, r
    # A way that needs nobody to move says nothing about people.
    with s.srv.memory_lock:
        r = game_tools.navigate_to_direct(s.srv.env, 47, 5, variance="none")  # where the walk began
    assert r["success"] and r["path_end"] == [47, 5] and "people_in_the_way" not in r, r


def test_people_hidden_by_the_story_do_not_block_walks():
    """Seen live: five TEAM AQUA grunts and PROF BIRCH from a finished Route 110 scene
    (their hide flags set) stood in the pathfinder's object list across the only way
    north, so every walk toward the rival found "no complete path"."""
    from server import game_tools

    s = server()
    s.load(Path(__file__).resolve().parent / "test_route110_gate.state")
    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
        objects = state["map"]["porymap"]["objects"]
        path = game_tools.pathfinder.find_path((16, 89), (34, 56), state, consider_npcs=True, allow_partial=False)
    flags = {o.get("flag") for o in objects}
    assert "FLAG_HIDE_ROUTE_110_TEAM_AQUA" not in flags and "FLAG_HIDE_ROUTE_110_BIRCH" not in flags, flags
    assert any(o.get("graphics_id") == "OBJ_EVENT_GFX_CAMPER" for o in objects)  # trainers still count
    assert path, "the way north to the rival must be found"


def test_navigate_to_below_an_exit_mat_names_go_to_exit():
    """Seen live: in the Oldale Mart, navigate_to(4,8) from (3,7) stepped on the exit mat
    and the warp cut the walk short. The result said "STOPPED ... on OLDALE TOWN MART: an
    event is running" although the player was already outside.

    Superseded: (4,8) was a row the map builder added below the mats, where it listed the
    mats' warps; this test walked there and expected to arrive outside. pret's OLDALE TOWN
    MART is 11x8 and its warps are the mats (3,7) and (4,7), which EXITS now lists with
    "stand on it and press DOWN" (test_warp_routes). A tile below the map is answered as
    outside it, naming go_to_exit, which does take the mat."""
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    assert s.call("mcp_go_to_exit", {"exit": "MART", "reasoning": ""})["success"]
    assert _location(s) == "OLDALE TOWN MART", _location(s)
    start = s.snapshot()
    r = s.call("mcp_navigate_to", {"x": start.x + 1, "y": start.y + 1, "variance": "none", "reason": "leave"})
    print("   ", r)
    assert r["success"] is False and r.get("reason") == "out_of_map" and "go_to_exit" in r["error"], r
    r = s.call("mcp_go_to_exit", {"exit": "door/warp at (4, 7) -> OLDALE TOWN", "reasoning": "leave"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] and _location(s) == "OLDALE TOWN", (r, _location(s))


def test_navigate_to_toward_a_neighbouring_map_names_the_exit_call():
    """Seen live: "Destination: Route 101 (south edge)" in the reason, then one-tile
    navigate_to calls wobbling against Oldale's walls for minutes."""
    from server.game_tools import exit_hint_for_text

    assert 'go_to_exit("south")' in exit_hint_for_text("OLDALE TOWN", "Destination: Route101 (south edge)")
    assert exit_hint_for_text("OLDALE TOWN", "heal at the Pokemon Center") == ""
    assert exit_hint_for_text("OLDALE TOWN", "toward ROUTE 1010") == ""  # no prefix matches
    # Seen live on Route 101: "heading east toward the exit to Littleroot" (no "Town").
    assert 'go_to_exit("south")' in exit_hint_for_text("ROUTE 101", "toward the exit to Littleroot")
    # A copied route names both neighbours: list both, don't guess.
    both = exit_hint_for_text("ROUTE 101", "Route 103 → Oldale → Route 101 → Littleroot → Birch's Lab")
    assert 'OLDALE TOWN use go_to_exit("north")' in both and 'LITTLEROOT TOWN use go_to_exit("south")' in both, both

    s = server()
    s.load(ROUTE101, wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    start = s.snapshot()
    r = s.call("mcp_navigate_to", {
        "x": start.x, "y": start.y - 1, "variance": "none",
        "reason": "Destination: Route 101 (south edge). Walk south.",
    })
    print("   ", r)
    assert 'go_to_exit("south")' in r["summary"], r
    r = s.call("mcp_navigate_to", {"x": start.x, "y": start.y, "variance": "none", "reason": "step back"})
    assert "go_to_exit" not in r["summary"], r


def test_collision_bit_blocks_floor_tiles():
    """Seen live in Rusturf Tunnel: (6,11) is CAVE floor with collision 1. The game refused
    the step, the map drew '.', and navigate_to failed 5 times in a row."""
    from pokemon_env.enums import MetatileBehavior
    from utils.mapping.map_formatter import format_tile_to_symbol

    assert format_tile_to_symbol((521, MetatileBehavior.CAVE, 1, 0)) == "#"
    assert format_tile_to_symbol((513, MetatileBehavior.CAVE, 0, 3)) == "."
    assert format_tile_to_symbol((1, MetatileBehavior.TALL_GRASS, 1, 3)) == "#"
    assert format_tile_to_symbol((1, MetatileBehavior.TALL_GRASS, 0, 3)) == "~"
    assert format_tile_to_symbol((1, MetatileBehavior.NORMAL, 1, 0)) == "#"

    s = server()
    s.load(Path(__file__).resolve().parent / "test_rusturf_tunnel.state")  # (6,10), live
    r = s.call("mcp_navigate_to", {"x": 7, "y": 11, "variance": "none", "reason": ""})
    print("   ", r)
    assert r["success"] is False, r  # refused up front instead of pressing into the wall


def test_routes_treat_hm_obstacles_as_walls():
    """Seen live: the heal-first route from inside Rusturf Tunnel (23,5) led east to
    Verdanturf, through two BREAKABLE_ROCKs at (24,4)/(24,5) that need Rock Smash."""
    from server import game_tools

    s = server()
    s.load(Path(__file__).resolve().parent / "test_rusturf_tunnel.state")
    assert {(24, 4), (24, 5)} <= game_tools._obstacle_tiles("RusturfTunnel")
    # read_location used to call this map RUSTURF TUNNEL ALT (a MapLocation alias); it
    # now gives pret's name, the one ROM_TO_PORYMAP_MAP and exits use (test_map_enum.py).
    way = game_tools.way_to_pokemon_center("RUSTURF TUNNEL", (23, 5))
    print("   ", way)
    assert "VERDANTURF" not in way and "RUSTBORO CITY POKEMON CENTER 1F" in way, way
    saved = game_tools.flag_reader
    try:
        game_tools.flag_reader = lambda flag: True  # every hide flag set: rocks gone for good
        assert not game_tools._obstacle_tiles("RusturfTunnel")
    finally:
        game_tools.flag_reader = saved


def test_long_route_across_grass_is_still_found():
    """Seen live: with grass steps costing 8x, go_to_exit("west") from Route 116 (44,9)
    said "no walkable path" because the search cap counted cost, not steps."""
    from server import game_tools

    s = server()
    s.load(Path(__file__).resolve().parent / "test_route116_east.state")  # (39,10)
    s.call("mcp_press_buttons", {"buttons": ["RIGHT"] * 5, "reasoning": "east"})
    snap = s.snapshot()
    assert snap.x >= 43, snap
    with s.srv.memory_lock:
        plan = game_tools.plan_exit_route(s.srv.env, "west")
    assert plan.get("success"), plan.get("error")
    assert len(plan["buttons"]) >= snap.x, len(plan["buttons"])


def test_go_to_exit_replans_around_a_person_in_the_way():
    """Seen live: a person at (38,10) on Route 116 blocked every go_to_exit("west") from
    (39,10); each call planned the same path through them."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_route116_east.state")  # (39,10)
    r = s.call("mcp_go_to_exit", {"exit": "west", "reasoning": "to Rustboro"})
    print("   ", r.get("summary"))
    snap = s.snapshot()
    # Either it left for Rustboro, or a wild battle stopped it somewhere past the person.
    assert r.get("success") or (snap.x, snap.y) not in ((39, 10), (37, 10)), r
    assert "occupied by someone" not in str(r.get("summary")), r


def test_route_names_the_ladder_when_two_lead_to_the_same_map():
    """Seen live: the AI climbed GRANITE CAVE B1F's ladder at (4,21) to 1F (17,11), east of
    a row of east-facing ledges; Steven's Room (5,10) is west of them, reached only from
    the ladder at 1F (35,2). Counted two-way, the ledges made the door look a walk away.
    Ledges are one-way in routes now, and where two ladders lead to one map the hint
    names the ladder, in a form go_to_exit takes."""
    from server.field_control import match_exits
    from server.game_tools import way_to_place

    # Location names were GRANITE CAVE 1F ALT / B1F ALT (MapLocation aliases) and exits
    # were spelled "GRANITE CAVE B 1F"; both are now pret's names, GRANITE CAVE 1F / B1F,
    # so an exit's name is the location read after taking it (test_map_enum.py).
    hint = way_to_place("GRANITE CAVE 1F", (16, 11), "Granite Cave Steven's Room")
    print("   ", hint)
    assert "is through an exit of this map" not in hint, hint  # the door is beyond the ledges
    assert 'go_to_exit("GRANITE CAVE B1F (17, 11)")' in hint, hint
    # West of the ledges the door is just a walk.
    near = way_to_place("GRANITE CAVE 1F", (8, 6), "Granite Cave Steven's Room")
    assert near.startswith("GRANITE CAVE STEVENS ROOM is through an exit of this map"), near
    hint = way_to_place("GRANITE CAVE B1F", (6, 13), "Granite Cave Steven's Room")
    print("   ", hint)
    assert "GRANITE CAVE 1F (4, 21)" not in hint, hint
    exits = [
        {"kind": "warp", "x": 25, "y": 13, "listed_xy": (25, 13), "name": "GRANITE CAVE 1F"},
        {"kind": "warp", "x": 4, "y": 21, "listed_xy": (4, 21), "name": "GRANITE CAVE 1F"},
        {"kind": "warp", "x": 29, "y": 13, "listed_xy": (29, 13), "name": "GRANITE CAVE B2F"},
    ]
    assert match_exits(exits, "GRANITE CAVE 1F (4, 21)") == [exits[1]]
    assert match_exits(exits, "GRANITE CAVE 1F") == exits[:2]  # the name alone still matches both
    assert match_exits(exits, "(29, 13)") == [exits[2]]


def test_equally_far_places_pick_the_nearer_exit():
    """Seen live: at ROUTE 110 (16,92), seven tiles from Slateport, the heal hint sent the
    AI north to MAUVILLE (about 90 tiles); both are one map away, and the search just met
    Mauville first. The model wouldn't make the trek and dithered in place."""
    from server.game_tools import pokemon_center_route

    south = pokemon_center_route("ROUTE 110", (16, 92))
    print("   ", south)
    assert south[0][0] == "south" and south[-1][1] == "SLATEPORT CITY POKEMON CENTER 1F", south
    north = pokemon_center_route("ROUTE 110", (16, 8))
    assert north[0][0] == "north" and north[-1][1] == "MAUVILLE CITY POKEMON CENTER 1F", north


def test_maps_are_named_as_the_rom_numbers_them():
    """Seen live: inside ROUTE 117's Day Care (map 22,0) the prompt said ROUTE 116 TUNNELERS
    REST HOUSE and showed that house's exits; the AI walked at a door that wasn't there.
    MapLocation is now generated from the decomp's map_groups.json (test_map_enum.py)."""
    from utils.mapping.porymap_state import decomp_map_at

    assert decomp_map_at(22, 0) == ("Route117_PokemonDayCare", "ROUTE 117 POKEMON DAY CARE")
    assert decomp_map_at(0, 9) is not None and decomp_map_at(99, 0) is None
    s = server()
    s.load(Path(__file__).resolve().parent / "test_route117_daycare.state")
    assert _location(s) == "ROUTE 117 POKEMON DAY CARE", _location(s)
    r = s.call("mcp_go_to_exit", {"exit": "ROUTE 117", "reasoning": "test"})
    assert _location(s) == "ROUTE 117", (_location(s), r)
    s.load(ROUTE101, wild_encounters=False)
    assert _location(s) == "ROUTE 101"  # a name the table had right is unchanged


def test_loose_target_names_match_their_map():
    from server.game_tools import arrived_at

    # Was "RUSTURF TUNNEL ALT", a MapLocation alias read_location no longer gives.
    assert arrived_at("RUSTURF TUNNEL", None, "Rusturf Tunnel")
    assert arrived_at("DEWFORD TOWN GYM", None, "Dewford Gym")
    assert arrived_at("GRANITE CAVE 1F", None, "Granite Cave")
    # Seen live: on METEOR FALLS 1F 1R, meteor_falls_reached never showed as done.
    assert arrived_at("METEOR FALLS 1F 1R", None, "Meteor Falls")
    assert arrived_at("METEOR FALLS B1F 2R", None, "Meteor Falls")
    assert arrived_at("MAGMA HIDEOUT 3F 2R", None, "Magma Hideout")
    assert not arrived_at("ROUTE 114", None, "Meteor Falls")
    # Seen live: "Return to Route 112 Cable Car Station" (target "Route 112 Cable Car").
    assert arrived_at("ROUTE 112 CABLE CAR STATION", None, "Route 112 Cable Car")
    assert arrived_at("ROUTE 112 CABLE CAR STATION", None, "Route 112 Cable Car Station")
    assert not arrived_at("ROUTE 112", None, "Route 112 Cable Car")
    assert not arrived_at("MT CHIMNEY CABLE CAR STATION", None, "Route 112 Cable Car")
    assert not arrived_at("RUSTBORO CITY GYM", None, "Rustboro City")
    assert not arrived_at("ROUTE 104 MR BRINEYS HOUSE", None, "Route 104")
    # Seen live: "Visit Devon Corporation" never counted inside DEVON CORP 3F.
    assert arrived_at("RUSTBORO CITY DEVON CORP 3F", None, "Devon Corporation")
    assert arrived_at("RUSTBORO CITY DEVON CORP 1F", None, "Devon Corporation")
    # Seen live: in MAUVILLE CITY BIKE SHOP the model walked out to find "Rydel's Cycles".
    assert arrived_at("MAUVILLE CITY BIKE SHOP", None, "Rydel's Cycles")
    assert not arrived_at("MAUVILLE CITY", None, "Rydel's Cycles")
    assert not arrived_at("MT CHIMNEY", None, "Mt. Pyre")  # short words never stretch
    # Seen live: "Mr. Briney's Cottage" is ROUTE 104 MR BRINEYS HOUSE.
    assert arrived_at("ROUTE 104 MR BRINEYS HOUSE", None, "Mr. Briney's Cottage")
    # Seen live: go_to("Mauville City Poke Mart") "can't be reached on foot"; the map is
    # MAUVILLE CITY MART. Every spelling of the sign names it, and only it.
    for mart in ("Mauville City Poke Mart", "Mauville City PokéMart", "Mauville City Pokemart",
                 "Mauville City Poké Mart", "Mauville City Pokemon Mart", "Mauville Mart"):
        assert arrived_at("MAUVILLE CITY MART", None, mart), mart
    assert not arrived_at("MAUVILLE CITY", None, "Mauville City Poke Mart")
    for center in ("Oldale Poke Center", "Oldale PokéCenter", "Oldale Pokémon Center", "Oldale Town Pokemon Center"):
        assert arrived_at("OLDALE TOWN POKEMON CENTER 1F", None, center), center
    from server.game_tools import place_route

    route = place_route("ROUTE 117", (17, 9), "Mauville City Poke Mart")
    assert route and route[-1][1] == "MAUVILLE CITY MART", route
    from server.game_tools import way_to_place

    briney = way_to_place("RUSTBORO CITY", (12, 17), "Mr. Briney's Cottage")
    assert briney.endswith("takes the first step.") and "ROUTE 104 MR BRINEYS HOUSE" in briney, briney


MOSSDEEP_SHORE = HERE / "test_mossdeep_west_shore.state"  # MOSSDEEP CITY (16,6)


def test_place_names_match_spaces_case_and_punctuation_aside():
    """Seen live: go_to("MOSSDEEP CITY HOUSE 3") answered "can't be reached on foot or by
    Surf", standing in Mossdeep; the map is MOSSDEEP CITY HOUSE3. Every resolver of place
    names (go_to's place_route, objectives' arrived_at, way_to_place, fly_to's town,
    go_to_exit's exit names) reads them spaces, case and punctuation aside."""
    from server.field_control import match_exits
    from server.game_tools import (
        FlyDestination,
        arrived_at,
        fly_destination_named,
        place_route,
        unknown_place,
        way_to_place,
    )

    for spelling in ("MOSSDEEP CITY HOUSE 3", "mossdeep city house3", "Mossdeep-City House #3", "MOSSDEEPCITYHOUSE3"):
        assert unknown_place(spelling) is None, spelling
        assert arrived_at("MOSSDEEP CITY HOUSE3", None, spelling), spelling
        route = place_route("MOSSDEEP CITY", (16, 6), spelling)
        assert route and route[-1][1] == "MOSSDEEP CITY HOUSE3", (spelling, route)
    assert not arrived_at("MOSSDEEP CITY HOUSE2", None, "MOSSDEEP CITY HOUSE 3")
    assert "MOSSDEEP CITY HOUSE3" in way_to_place("MOSSDEEP CITY", (16, 6), "Mossdeep City House 3")
    # A route number written together, and the other way round.
    assert unknown_place("Route104") is None and arrived_at("ROUTE 104", None, "Route104")
    # fly_to's towns and go_to_exit's exits, spaces and accents aside.
    lilycove = FlyDestination(mapsec=5, town="LILYCOVE CITY", cell=(0, 0), landing_map="LILYCOVE CITY",
                              landing=(24, 15), visited=True)
    assert fly_destination_named([lilycove], "lily cove city") is lilycove
    exits = [{"kind": "warp", "label": "door/warp at (24, 14) -> LILYCOVE CITY POKEMON CENTER 1F",
              "name": "LILYCOVE CITY POKEMON CENTER 1F", "x": 24, "y": 14}]
    assert match_exits(exits, "Lilycove City Pokémon Center") == exits


def test_go_to_a_name_written_with_a_space_walks_there():
    """The live case: go_to("MOSSDEEP CITY HOUSE 3") from MOSSDEEP CITY's west shore."""
    s = server()
    s.load(MOSSDEEP_SHORE, wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "MOSSDEEP CITY HOUSE 3", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY HOUSE3", (r, _location(s))


def test_go_to_an_unknown_name_says_unknown_and_names_the_closest():
    """A name no map has is answered "unknown" with the closest real names (difflib),
    never "can't be reached", and nothing is walked."""
    s = server()
    s.load(MOSSDEEP_SHORE, wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "Flarbington Zqx Tower", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is False and r["status"] == "UNKNOWN_PLACE", r
    assert "Unknown place" in r["error"] and "can't be reached" not in r["error"], r
    assert len(r["suggestions"]) == 3 and all(name in r["error"] for name in r["suggestions"]), r
    snap = s.snapshot()
    assert _location(s) == "MOSSDEEP CITY" and (snap.x, snap.y) == (16, 6), snap
    # Close to real names: those are the ones suggested.
    r = s.call("mcp_go_to", {"place": "MOSSDEEP CITY HOUSE 9", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["status"] == "UNKNOWN_PLACE", r
    assert all(name.startswith("MOSSDEEP CITY HOUSE") for name in r["suggestions"]), r["suggestions"]


def test_exit_list_says_when_an_edge_cannot_be_walked():
    """Seen live: Route 104's "south edge of the map -> ROUTE 105" (all sea) sent the
    model south from the very spot where its objective was already done."""
    from utils.mapping.porymap_state import _format_porymap_info

    text = str(_format_porymap_info("ROUTE 104", (39, 62)))
    assert "south edge of the map (no way across on foot: water or cliff all along it) -> ROUTE 105" in text
    assert "east edge of the map (walk off at x=39, y=62-63) -> PETALBURG CITY" in text


def test_way_to_an_objective_target():
    """Seen live: objective target "Rustboro Gym"; the model left the gym and searched the
    city tile by tile for its trainers."""
    from server.game_tools import way_to_place

    gym = way_to_place("RUSTBORO CITY", (13, 21), "Rustboro Gym")
    print("   ", gym)
    assert gym == 'RUSTBORO CITY GYM is through an exit of this map: go_to_exit("RUSTBORO CITY GYM").', gym
    lab = way_to_place("LITTLEROOT TOWN", (7, 17), "Birch's Lab")
    assert 'go_to_exit("LITTLEROOT TOWN PROFESSOR BIRCHS LAB")' in lab, lab
    far = way_to_place("ROUTE 104", (39, 62), "Rustboro City")
    assert "PETALBURG WOODS → ROUTE 104 → RUSTBORO CITY" in far, far
    assert way_to_place("RUSTBORO CITY GYM", (5, 19), "Rustboro Gym") == ""  # already there
    # Seen live: "Route 116" routed into ROUTE 116 TUNNELERS REST HOUSE; the exact name wins.
    route = way_to_place("RUSTBORO CITY GYM", (5, 17), "Route 116")
    assert route.endswith('go_to_exit("RUSTBORO CITY") takes the first step.'), route
    assert "RUSTBORO CITY → ROUTE 116." in route and "TUNNELERS" not in route, route
    # Seen live: the 2F matched too and the route led upstairs, away from the nurse.
    center = way_to_place("PETALBURG WOODS", (7, 14), "Rustboro City Pokemon Center")
    assert center.count("go_to_exit") == 1 and center.endswith('takes the first step.'), center
    assert "RUSTBORO CITY POKEMON CENTER 1F. go_to_exit" in center and "2F" not in center, center
    assert way_to_place("ROUTE 104", (39, 62), "Route 104 North") == ""  # on the route; halves are arrived_at's job
    # A name no map has gets no route. (This used "Mr. Briney's Cottage" until COTTAGE was
    # taught to match HOUSE; see test_loose_target_names_match_their_map.)
    assert way_to_place("RUSTBORO CITY", (13, 21), "Battle Frontier Lounge") == ""
    # Seen live: inside the gym, "In Rustboro City Gym" in the reason produced
    # go_to_exit("RUSTBORO CITY") and pulled the model out of the gym.
    from server.game_tools import exit_hint_for_text

    for text in ("In Rustboro City Gym, facing UP", "Rustboro Gym trainers", "back in Rustboro"):
        assert exit_hint_for_text("RUSTBORO CITY GYM", text, (5, 17)) == "", text
    # Naming another map still works; from the gym the city door is the right first step.
    out = exit_hint_for_text("RUSTBORO CITY GYM", "then Route 104", (5, 17))
    assert "RUSTBORO CITY → ROUTE 104" in out and 'go_to_exit("RUSTBORO CITY")' in out, out


def test_navigate_to_toward_a_map_further_away_names_the_first_exit():
    """Seen live in Littleroot: "travel west through Route 102 to Petalburg City", then
    twenty minutes of one-tile walks. Route 102 is reached north via Route 101 and Oldale."""
    from server.game_tools import exit_hint_for_text

    # The hints read whether the party can Surf from the save loaded in the server, so this
    # test loads one whose party can't: it ran after whichever test came before it, and
    # once that was a Lilycove one (AZUMARILL with SURF) the Oldale → Lilycove hint below
    # offered a way over the sea.
    server().load(ROUTE101, wild_encounters=False)

    hint = exit_hint_for_text("LITTLEROOT TOWN", "travel west through Route 102 to Petalburg City")
    print("   ", hint)
    assert 'go_to_exit("north")' in hint, hint
    assert "ROUTE 101 → OLDALE TOWN → ROUTE 102 → PETALBURG CITY" in hint, hint
    assert hint.count("go_to_exit") == 1, hint  # both destinations start through the same exit
    # From Oldale, Petalburg lies west and Littleroot south: two exits, two lines.
    split = exit_hint_for_text("OLDALE TOWN", "Petalburg City or back to Littleroot")
    assert 'PETALBURG CITY is not next' in split and 'go_to_exit("west")' in split, split
    assert 'LITTLEROOT TOWN is not next' in split and 'go_to_exit("south")' in split, split
    # This map's own name is not a destination.
    assert exit_hint_for_text("LITTLEROOT TOWN", "explore Littleroot Town") == ""
    # Indoors the first step is the door.
    lab = exit_hint_for_text("LITTLEROOT TOWN PROFESSOR BIRCHS LAB", "go to Route 102")
    assert lab.startswith("ROUTE 102 is not next") and 'go_to_exit("LITTLEROOT TOWN")' in lab, lab
    # The north half of Route 104 is reached through Petalburg Woods, not across the trees.
    woods = exit_hint_for_text("PETALBURG CITY", "on to Rustboro City")
    assert "PETALBURG WOODS → ROUTE 104 → RUSTBORO CITY" in woods and 'go_to_exit("west")' in woods, woods
    # On foot there is no way from Oldale to Lilycove (it needs Surf), so no route is offered
    # rather than one that crosses the sea.
    assert exit_hint_for_text("OLDALE TOWN", "Lilycove City", (8, 10)) == ""


def test_navigate_to_walkable_target_reports_arrival():
    s = server()
    s.load(ROUTE101, wild_encounters=False)
    r = s.call("mcp_navigate_to", {"x": 9, "y": 0, "variance": "none", "reason": ""})
    print("   ", r)
    assert r["success"] is True and r["summary"].startswith("Reached (9,0)."), r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (9, 0), snap


# LILYCOVE CITY (24,15), outside the POKéMON CENTER, just after a Fly from Mossdeep
# (made from test_mossdeep_west_shore.state with fly_to("LILYCOVE CITY")).
LILYCOVE = HERE / "test_lilycove_pokecenter.state"
# pret LilycoveCity layout: rows 9-20, x 45-52 is the raised plateau east of the Center,
# ringed by one-way ledges (MB_JUMP_*); (49,14) is a cliff tile (metatile 0x6F, MB_NORMAL
# with collision), no ledge.
LILYCOVE_PLATEAU = (range(45, 53), range(9, 21))
JUMP_SYMBOLS = {"JUMP_SOUTH": "↓", "JUMP_NORTH": "↑", "JUMP_WEST": "←", "JUMP_EAST": "→"}


def test_go_to_exit_east_from_lilycove_jumps_each_ledge_with_one_press():
    """Seen live in LILYCOVE CITY: go_to_exit("east") from (24,15) climbed the plateau and
    stopped at "BLOCKED moving DOWN at (49,13): (49,14) is not walkable". The plan had one
    press per tile, two for each ledge (onto it, onto the landing), but the game jumps
    both tiles on one press (pret field_player_avatar.c PlayerJumpLedge): after the east
    ledge (47,10) every press was a tile east of the path."""
    from server import game_tools

    s = server()
    s.load(LILYCOVE, wild_encounters=False)
    snap = s.snapshot()
    assert _location(s) == "LILYCOVE CITY" and (snap.x, snap.y) == (24, 15), snap
    # The presses over the plateau: one RIGHT for the east ledge (47,10) and one DOWN for
    # the south ledge (48,11), as walked by hand from (45,10).
    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    tiles = state["map"]["porymap"]["raw_tiles"]
    planned = ["RIGHT", "RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN"]  # (45,10) to (48,13), a tile a press
    assert game_tools.jump_presses((45, 10), planned, tiles) == ["RIGHT", "RIGHT", "DOWN", "DOWN"]
    r = s.call("mcp_go_to_exit", {"exit": "east", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "ROUTE 124", (r, _location(s))
    assert "BLOCKED" not in r["summary"], r


def test_navigate_to_over_lilycove_ledges_ends_on_the_target():
    """navigate_to (48,13) from (45,10): over the east ledge (47,10) and the south ledge
    (48,11). With a press per tile it ended a tile off."""
    s = server()
    s.load(LILYCOVE, wild_encounters=False)
    r = s.call("mcp_navigate_to", {"x": 45, "y": 10, "variance": "none", "reason": "test"})
    assert r["success"] is True, r
    r = s.call("mcp_navigate_to", {"x": 48, "y": 13, "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert r["success"] is True and (snap.x, snap.y) == (48, 13), (r, snap)


def test_lilycove_ascii_draws_ledges_only_where_pret_has_them():
    """The ASCII the model reads draws a ledge arrow on a plateau tile exactly where pret's
    layout has a ledge the game jumps (MB_JUMP_SOUTH/NORTH/WEST/EAST; event_object_movement.c
    GetLedgeJumpDirection), pointing its way. The ledge corners (49,17) and (46,20) are a
    diagonal MB_JUMP_* with collision, never jumped, and were drawn '↓' ("SOUTH" is in
    "JUMP_SOUTHWEST"), so walks were planned over them. (49,14), reported live as a '↓', is
    a cliff tile and was drawn '#' already. LilycoveCity has no ASCII override
    (ascii_map_loader.MAP_OVERRIDES): the drawing comes from the layout."""
    from utils.mapping.ascii_map_loader import MAP_OVERRIDES
    from utils.mapping.pathfinding import tile_behavior_name
    from utils.mapping.porymap_state import _format_porymap_info, layout_tiles

    assert "LilycoveCity" not in MAP_OVERRIDES
    s = server()
    s.load(LILYCOVE, wild_encounters=False)
    with s.srv.memory_lock:
        parts = _format_porymap_info("LILYCOVE CITY", (24, 15), 6, s.srv.env.memory_reader).context_parts
    rows = parts[parts.index("\nASCII Map:") + 1].split("\n")
    tiles = layout_tiles("LilycoveCity")
    xs, ys = LILYCOVE_PLATEAU
    for y in ys:
        for x in xs:
            want = JUMP_SYMBOLS.get(tile_behavior_name(tiles[y][x]))
            drawn = rows[y][x]
            assert (drawn == want) if want else drawn not in JUMP_SYMBOLS.values(), ((x, y), drawn, want)
    assert rows[14][49] == "#" and rows[14][48] == "↓", rows[14][44:52]
    assert rows[17][49] == "#" and rows[20][46] == "#", (rows[17][44:53], rows[20][44:53])
    # The planner reads the same grid: no walk steps onto a corner.
    from server import game_tools

    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    grid = state["map"]["porymap"]["grid"]
    assert grid[17][49] == "#" and grid[20][46] == "#", (grid[17][44:53], grid[20][44:53])


def _run_all():
    import test_movement_feedback as harness

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
    if harness._SERVER is not None:
        harness._SERVER.stop()
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
