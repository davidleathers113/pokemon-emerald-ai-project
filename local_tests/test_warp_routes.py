"""Ways through warps within a place: Lavaridge Gym's holes and geysers (1F ↔ B1F),
Petalburg Gym's sign doors (warps within one map), and talk_to / navigate_to taking them
when the person or tile is on a part of the map the player can't walk to.

Seen live: in LAVARIDGE TOWN GYM 1F, talk_to(13,9) (Flannery) answered "Could not reach a
tile next to (13,9): No path found from (9, 9) to (12, 9)". Then in PETALBURG CITY GYM the
model cycled UP / [UP, A] at a room door for six calls: the doors are read, not walked
into, and [UP, A] after a BLOCKED UP was refused whole.

Static tests read only the decomp data (pret map.json, layouts); the live ones use the
same in-process server harness as test_movement_feedback.py.

Fixtures:
  test_lavaridge_after_chimney.state  LAVARIDGE TOWN (6,7) after Mt. Chimney, Flannery
                                      not beaten
  test_movement_route101.state        ROUTE 101 (13,2); the route sign stands at (5,9)
  test_aqua_hideout_b2f.state         AQUA HIDEOUT B2F (31,17), off the pad from (8,8),
                                      MATT's side (live save of 2026-11-05)
  test_mossdeep_gym_statue.state      MOSSDEEP CITY GYM (3,20) (live checkpoint of
                                      2026-11-05)

No savestate inside PETALBURG CITY GYM exists, so its sign doors and same-map arrival are
tested on the data and the arrival rule only; the Lavaridge crossing proves the live
stepping (between two maps).

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_warp_routes.py [test-name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import ROUTE101, server

HERE = Path(__file__).resolve().parent
LAVARIDGE = HERE / "test_lavaridge_after_chimney.state"
GYM_1F, GYM_B1F = "LavaridgeTown_Gym_1F", "LavaridgeTown_Gym_B1F"
FLANNERY = (13, 9)
# pret LavaridgeTown warp 1 → LavaridgeTown_Gym_1F warp 0: where the player walks in.
GYM_DOOR = (13, 18)
AQUA_B2F_STATE = HERE / "test_aqua_hideout_b2f.state"
MOSSDEEP_GYM_STATE = HERE / "test_mossdeep_gym_statue.state"
AQUA_B2F, AQUA_B1F, MOSSDEEP_GYM = "AquaHideout_B2F", "AquaHideout_B1F", "MossdeepCity_Gym"


def _beside(x, y):
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _read(s, fn):
    with s.srv.memory_lock:
        return fn(s.srv.env.memory_reader)


# --- the decomp data -----------------------------------------------------------------


def test_lavaridge_gym_maps_are_pret_and_their_grids_match_the_layout():
    """The two map.json files had been hand-edited (warps renumbered and dropped, the door
    put at y=19, below the 17x19 layout), so warp ids pointed at the wrong warps. Their
    ASCII overrides are now tile for tile the 17x19 layouts."""
    from server import game_tools
    from utils.mapping.porymap_state import _static_grid

    one, b1 = game_tools._raw_map_json(GYM_1F), game_tools._raw_map_json(GYM_B1F)
    assert len(one["warp_events"]) == 26 and len(b1["warp_events"]) == 24
    assert [(w["x"], w["y"]) for w in one["warp_events"][:2]] == [(13, 18), (14, 18)]
    for name in (GYM_1F, GYM_B1F):
        grid = _static_grid(name)
        assert (len(grid[0]), len(grid)) == (17, 19), (name, len(grid[0]), len(grid))
    # Each warp's destination warp sends the player back: holes and geysers come in pairs.
    for i, warp in enumerate(one["warp_events"][2:], start=2):
        back = b1["warp_events"][int(warp["dest_warp_id"])]
        assert int(back["dest_warp_id"]) == i, (i, warp, back)


def test_warps_that_only_mark_a_landing_spot_are_not_exits():
    """pret field_control_avatar.c: a warp event on plain floor never fires (it is where a
    geyser throws the player up to). Lavaridge 1F (3,6) is one; (0,10) is a hole."""
    from server import game_tools
    from utils.mapping.porymap_state import arrival_only_warps

    one = game_tools._raw_map_json(GYM_1F)["warp_events"]
    landing = {(one[i]["x"], one[i]["y"]) for i in arrival_only_warps(GYM_1F)}
    assert landing == {(3, 6), (4, 18), (13, 17)}, landing
    assert len(arrival_only_warps(GYM_B1F)) == 24 - 15  # B1F has 15 geysers
    exits = game_tools.list_map_exits({"warps": []}, GYM_1F)
    tiles = {(e["x"], e["y"]) for e in exits}
    assert (3, 6) not in tiles and (13, 17) not in tiles and (0, 10) in tiles, sorted(tiles)
    hole = next(e for e in exits if (e["x"], e["y"]) == (0, 10))
    assert hole["dest_map"] == "MAP_LAVARIDGE_TOWN_GYM_B1F" and hole["arrival"] == (0, 10), hole
    assert not any(f"({x}, {y})" in arg for _, arg, *_ in game_tools._walk_exits(GYM_1F) for x, y in landing)


def test_tile_route_to_flannery_is_the_holes_and_geysers_chain():
    """From the gym door, Flannery's side of 1F is reached only by dropping to B1F and
    riding geysers up; the last steps are the live-verified hole (2,3) → geyser (7,2) →
    hole (10,6) → geyser (12,12), which lands one tile east, at (13,12)."""
    from server import game_tools

    route = game_tools.tile_route(GYM_1F, GYM_DOOR, _beside(*FLANNERY))
    print("    from the door:", route)
    assert route, route
    # B1F was spelled "B 1F" in exit names, unlike the LAVARIDGE TOWN GYM B1F location
    # read_location gives there; exits now use that name (test_map_enum.py).
    assert route[-4:] == [
        ("LAVARIDGE TOWN GYM B1F (2, 3)", "LAVARIDGE TOWN GYM B1F"),
        ("LAVARIDGE TOWN GYM 1F (7, 2)", "LAVARIDGE TOWN GYM 1F"),
        ("LAVARIDGE TOWN GYM B1F (10, 6)", "LAVARIDGE TOWN GYM B1F"),
        ("LAVARIDGE TOWN GYM 1F (12, 12)", "LAVARIDGE TOWN GYM 1F"),
    ], route
    # Floors alternate: every step is a hole down or a geyser up.
    names = [name for _, name in route]
    assert all(a != b for a, b in zip(names, names[1:], strict=False)), names
    # From B1F (after the first drop) the way on is planned for the same target.
    below = game_tools.tile_route(GYM_1F, (10, 18), _beside(*FLANNERY), here_dir=GYM_B1F)
    assert below and below[-1] == route[-1], below
    # Already on her side (where the last geyser lands): nothing to cross.
    assert game_tools.tile_route(GYM_1F, (13, 12), _beside(*FLANNERY)) == []
    # Walls only: no tile to reach.
    assert game_tools.tile_route(GYM_1F, GYM_DOOR, [(13, 8)]) is None


def test_tile_route_in_petalburg_gym_goes_through_its_doors():
    """pret PetalburgCity_Gym is one 9x112 map; its rooms are joined by doors that warp
    within it (warp 2 (1,105) → warp 3's (7,85)). The static grid used to be the 9x8 lobby
    variant (ascii_map_loader, before the 4th badge) numbered from row 0, so no door joined
    anything."""
    from server import game_tools

    lobby = (4, 110)
    assert game_tools.tile_route("PetalburgCity_Gym", lobby, [(4, 82)]) == [
        ("PETALBURG CITY GYM (1, 105)", "PETALBURG CITY GYM")
    ]
    norman = game_tools.tile_route("PetalburgCity_Gym", lobby, _beside(4, 2))
    print("    to Norman:", norman)
    assert norman and all(name == "PETALBURG CITY GYM" for _, name in norman), norman
    assert norman[-1][0] in ("PETALBURG CITY GYM (7, 14)", "PETALBURG CITY GYM (1, 27)"), norman
    assert len(norman) == 4, norman
    assert game_tools.tile_route("PetalburgCity_Gym", lobby, [(2, 108)]) == []


def test_petalburg_gym_doors_are_read_and_land_within_the_map():
    """pret PetalburgCity_Gym: each room door is a warp in a wall with a sign on it
    (EventScript_SpeedRoomDoor: MSGBOX_YESNO, YES → warpdoor 7,85). go_to_exit is told to
    read it from the tile in front and that it lands within the same map."""
    from server import game_tools
    from utils.mapping.porymap_state import sign_door_warps

    warps = game_tools._raw_map_json("PetalburgCity_Gym")["warp_events"]
    doors = {(warps[i]["x"], warps[i]["y"]) for i in sign_door_warps("PetalburgCity_Gym")}
    assert {(1, 105), (7, 105), (7, 79), (7, 53), (1, 27), (7, 14)} <= doors, sorted(doors)
    assert (4, 111) not in doors  # the way out to PETALBURG CITY is an ordinary door mat
    exits = game_tools.list_map_exits({"warps": []}, "PetalburgCity_Gym")
    speed = next(e for e in exits if (e["x"], e["y"]) == (1, 105))
    assert speed["sign_door"] and speed["dest_map"] == "MAP_PETALBURG_CITY_GYM", speed
    assert speed["arrival"] == (7, 85), speed
    assert not any((e["x"], e["y"]) == (7, 85) for e in exits)  # a landing spot only
    assert not sign_door_warps(GYM_1F)


def test_petalburg_gym_warps_on_plain_floor_are_not_exits():
    """Seen live: Alexia's room's bottom-right "door" (6,59)/(7,59) is in pret's warp table,
    but nothing is there in the game: plain floor, nothing fires (pret
    TryStartWarpEventScript needs IsWarpMetatileBehavior). Neither is an exit nor a walk
    edge; the real door (7,53) is both. When the game's map buffer shows a warp behavior
    on such a tile now (a script laid a door there), go_to_exit may take it."""
    from pokemon_env.enums import MetatileBehavior
    from server import game_tools

    tiles = {(e["x"], e["y"]) for e in game_tools.list_map_exits({"warps": []}, "PetalburgCity_Gym")}
    assert (6, 59) not in tiles and (7, 59) not in tiles and (7, 53) in tiles, sorted(tiles)
    edges = [arg for _, arg, *_ in game_tools._walk_exits("PetalburgCity_Gym")]
    assert "PETALBURG CITY GYM (7, 53)" in edges, edges
    assert not any("(6, 59)" in arg or "(7, 59)" in arg for arg in edges), edges

    def mat_at_7_59(x, y):
        behavior = MetatileBehavior.SOUTH_ARROW_WARP if (x, y) == (7, 59) else MetatileBehavior.NORMAL
        return (1, behavior, 0, 3)

    tiles = {(e["x"], e["y"]) for e in game_tools.list_map_exits({"warps": []}, "PetalburgCity_Gym", None, mat_at_7_59)}
    assert (7, 59) in tiles and (6, 59) not in tiles, sorted(tiles)


def test_same_map_warp_counts_as_taken_at_its_landing_tile():
    """The map id does not change through a warp within the map, so go_to_exit judges it
    by where the player stands: the landing tile or one off it (a geyser throws the player
    one tile east, pret field_effect.c LavaridgeGymB1FWarpExitEffect_PopOut); still on the
    warp tile means it did not fire."""
    from server.game_tools import came_through_warp

    assert came_through_warp((7, 85), (1, 105), (7, 85))
    assert came_through_warp((12, 12), (12, 12), (13, 12)) is True  # landed, then hopped east
    assert not came_through_warp((12, 12), (12, 12), (12, 12))  # never left the warp tile
    assert not came_through_warp((7, 85), (1, 105), (1, 106))  # in front of the door still
    assert not came_through_warp((7, 85), (1, 105), (7, 87))


def test_exits_prompt_leaves_out_landing_spots():
    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_json_builder import build_json_map_for_llm
    from utils.mapping.porymap_state import _format_exits

    json_map = build_json_map_for_llm(GYM_1F, get_porymap_root())
    lines = _format_exits(json_map, json_map["ascii"].split("\n"))
    assert len(lines) == 26 - 3, lines
    assert not any("(3, 6)" in line for line in lines), lines


def _pret_warps(map_dir):
    from server import game_tools

    return [(int(w["x"]), int(w["y"])) for w in game_tools._raw_map_json(map_dir).get("warp_events") or []]


def test_every_map_lists_its_warps_on_pret_tiles():
    """Seen live: AQUA HIDEOUT B2F's pad (5,8) was listed at (6,8), a wall, and MOSSDEEP
    CITY GYM's (7,18) at (6,17): the builder moved a warp one tile toward any row or
    column of wall beside it, and off a map's last row into rows it added. 901 of the
    1289 warps on 372 maps were moved, and 206 maps' grids came out a row or column
    bigger than their tiles. Every map's listed warps are now its warp_events (pret
    src/field_control_avatar.c GetWarpEventAtMapPosition: a warp fires on its own tile)."""
    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_json_builder import build_json_map

    root = get_porymap_root()
    moved, sized, built_maps = [], [], 0
    for map_file in sorted((Path(root) / "data" / "maps").glob("*/map.json")):
        map_dir = map_file.parent.name
        try:
            built = build_json_map(map_dir, root, include_grid=True, include_ascii=True, badge_count=8)
        except ValueError:
            continue  # a tileset the parser does not know (SecretBase_*): no grid either way
        if not built:
            continue
        built_maps += 1
        listed = [(w["x"], w["y"]) for w in built["warps"]]
        if listed != _pret_warps(map_dir):
            moved.append(map_dir)
        tiles, grid = built.get("raw_tiles") or [], built.get("grid") or []
        if tiles and (len(grid) != len(tiles) or len(grid[0]) != len(tiles[0])):
            sized.append(map_dir)
    print(f"    {built_maps} maps; moved warps on {len(moved)}, resized grids on {len(sized)}")
    assert built_maps > 450, built_maps
    assert not moved, moved[:20]
    assert not sized, sized[:20]


def test_exits_lines_say_where_the_warp_is_and_how_it_fires():
    """The EXITS line names the warp's own tile; an exit mat (arrow warp, pret TryArrowWarp)
    is stood on and walked off, a door (MB_ANIMATED_DOOR, TryDoorWarp) is walked into from
    the tile below; pads fire on the step and need no note."""
    from pokemon_env.porymap_paths import get_porymap_root
    from utils.mapping.porymap_json_builder import build_json_map_for_llm
    from utils.mapping.porymap_state import _format_exits

    def lines(map_dir):
        json_map = build_json_map_for_llm(map_dir, get_porymap_root(), badge_count=8)
        return _format_exits(json_map, json_map["ascii"].split("\n"))

    b2f = lines(AQUA_B2F)
    for line in ("- door/warp at (5, 8) -> AQUA HIDEOUT B2F", "- door/warp at (31, 17) -> AQUA HIDEOUT B2F",
                 "- door/warp at (32, 20) -> AQUA HIDEOUT B1F", "- door/warp at (3, 3) -> AQUA HIDEOUT B1F"):
        assert line in b2f, (line, b2f)
    assert not any("(6, 8)" in line or "(33, 19)" in line for line in b2f), b2f
    gym = lines(MOSSDEEP_GYM)
    for x, y in ((7, 18), (3, 28), (1, 23), (23, 20)):
        assert f"- door/warp at ({x}, {y}) -> MOSSDEEP CITY GYM" in gym, (x, y, gym)
    assert "- door/warp at (6, 35) -> MOSSDEEP CITY (stand on it and press DOWN)" in gym, gym
    # pret OldaleTown warp_events[2]: the POKEMON CENTER's door.
    (x, y) = _pret_warps("OldaleTown")[2]
    door = f"- door/warp at ({x}, {y}) -> OLDALE TOWN POKEMON CENTER 1F (from ({x}, {y + 1}) press UP into it)"
    assert door in lines("OldaleTown"), (door, lines("OldaleTown"))


def test_aqua_hideout_route_to_matt_goes_up_to_b1f_and_back_down():
    """Seen live: from B2F's left room the planner found no way to MATT. pret's way: B1F
    (3,3), B1F's (12,1) back down to B2F's middle room, its pad (8,8) to (31,17). The
    moved warps had drawn the pad (5,8) as a door at (6,8) in the wall, which joined the
    left room to the middle one, so the route was a single pad from a room it can't be
    walked to from."""
    from server import game_tools

    beside_matt = [(24, 19), (22, 19), (23, 20), (23, 18)]
    route = game_tools.tile_route(AQUA_B2F, (3, 5), beside_matt)
    assert route == [
        ("AQUA HIDEOUT B1F (3, 3)", "AQUA HIDEOUT B1F"),
        ("AQUA HIDEOUT B2F (12, 1)", "AQUA HIDEOUT B2F"),
        ("AQUA HIDEOUT B2F (8, 8)", "AQUA HIDEOUT B2F"),
    ], route
    areas = game_tools._route_areas(AQUA_B2F)
    assert areas[(5, 7)] != areas[(7, 8)] and (6, 8) not in areas, (areas.get((5, 7)), areas.get((7, 8)))


def test_mossdeep_gym_route_to_tate_and_liza_is_pret_s_pads():
    """pret's way across MOSSDEEP CITY GYM: (3,28)→(1,23), (7,18)→(8,12), (11,3)→(11,35),
    (13,32)→(21,10), then the walk to (23,8) below TATE AND LIZA (23,7)."""
    from server import game_tools

    route = game_tools.tile_route(MOSSDEEP_GYM, (7, 34), [(23, 8)])
    assert [step for step, _ in route] == [
        "MOSSDEEP CITY GYM (3, 28)", "MOSSDEEP CITY GYM (7, 18)", "MOSSDEEP CITY GYM (11, 3)", "MOSSDEEP CITY GYM (13, 32)",
    ], route


def _live_exits(s):
    from server import game_tools
    from utils.mapping.porymap_state import _format_exits

    with s.srv.memory_lock:
        _, offset, source = game_tools.load_pathfinding_state(s.srv.env)
    json_map = source["json_map"]
    exits = game_tools.list_map_exits(json_map, source.get("porymap_map_name"), offset)
    return exits, _format_exits(json_map, json_map["ascii"].split("\n"))


def test_live_aqua_hideout_exits_and_the_pad_up_to_b1f():
    """On the live save at AQUA HIDEOUT B2F (31,17): the EXITS lines and go_to_exit's
    labels name pret's tiles, and go_to_exit takes the pad (32,20) up to B1F.

    Every way off this side passes (28,16) or (28,17), pret's AquaHideout_B2F coord
    events that start MATT's scene while VAR_TEMP_1 is 0. The scene is not what this tests,
    so VAR_TEMP_1 (0x4001, a SaveBlock1 var) is set to 1 in RAM, as the scene itself does."""
    s = server()
    s.load(AQUA_B2F_STATE, wild_encounters=False)
    assert s.srv._location_name() == "AQUA HIDEOUT B2F", s.srv._location_name()
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        sb1 = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
        s.srv.env.core.memory.u16[sb1 + reader.addresses.SAVE_BLOCK1_VARS_OFFSET + (0x4001 - 0x4000) * 2] = 1
        reader._invalidate_mem_cache()
    exits, lines = _live_exits(s)
    labels = [e["label"] for e in exits]
    print("   ", labels)
    assert "door/warp at (32, 20) -> AQUA HIDEOUT B1F" in labels, labels
    assert "door/warp at (5, 8) -> AQUA HIDEOUT B2F" in labels, labels
    assert "- door/warp at (32, 20) -> AQUA HIDEOUT B1F" in lines, lines
    r = s.call("mcp_go_to_exit", {"exit": "door/warp at (32, 20) -> AQUA HIDEOUT B1F", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] and s.srv._location_name() == "AQUA HIDEOUT B1F", r
    assert "(32, 20)" in r["summary"], r


def test_live_mossdeep_gym_exits_are_pret_s_pads():
    """Seen live at MOSSDEEP CITY GYM (3,20): EXITS gave (6,17), (3,27), (0,24), (24,19)
    for pret's (7,18), (3,28), (1,23), (23,20), and the walk to (6,17) failed."""
    s = server()
    s.load(MOSSDEEP_GYM_STATE, wild_encounters=False)
    assert s.srv._location_name() == "MOSSDEEP CITY GYM", s.srv._location_name()
    exits, lines = _live_exits(s)
    tiles = {(e["x"], e["y"]) for e in exits if e["kind"] == "warp"}
    print("   ", lines)
    assert {(7, 18), (3, 28), (1, 23), (23, 20), (8, 12), (11, 3), (13, 32), (21, 10)} <= tiles, tiles
    assert not tiles & {(6, 17), (3, 27), (0, 24), (24, 19)}, tiles
    assert "- door/warp at (7, 18) -> MOSSDEEP CITY GYM" in lines, lines


# --- live ------------------------------------------------------------------------------


def _enter_lavaridge_gym(s):
    s.load(LAVARIDGE, wild_encounters=False)
    r = s.call("mcp_go_to_exit", {"exit": "LAVARIDGE TOWN GYM 1F", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] and s.srv._location_name() == "LAVARIDGE TOWN GYM 1F", r
    snap = s.snapshot()
    assert (snap.x, snap.y) == GYM_DOOR, snap


def _talked_to_flannery(s, r):
    assert r["status"] in ("BATTLE", "TALKED"), r
    assert "Went through" in r["summary"] and "LAVARIDGE TOWN GYM B1F" in r["summary"], r
    snap = s.snapshot()
    assert s.srv._location_name() == "LAVARIDGE TOWN GYM 1F" or _read(s, lambda m: m.is_in_battle()), snap
    if not _read(s, lambda m: m.is_in_battle()):
        assert (snap.x, snap.y) in _beside(*FLANNERY), snap


def test_talk_to_flannery_crosses_the_gym():
    """The live call: talk_to(13,9) from the gym door. Its buried trainers may spot the
    player on the way; a trainer battle is a fair stop, reported as such."""
    s = server()
    _enter_lavaridge_gym(s)
    r = s.call("mcp_talk_to", {"x": FLANNERY[0], "y": FLANNERY[1], "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    if r["success"]:
        _talked_to_flannery(s, r)
    else:
        assert "a trainer battle started on the way" in r["error"], r
        assert "talk_to(13, 9) again" in r["error"], r
        assert _read(s, lambda m: m.is_in_battle()), r


def test_talk_to_flannery_reaches_her_when_her_trainers_are_beaten():
    """The whole crossing, end to end: the gym's trainers are marked beaten in RAM (their
    trainer flags, TRAINER_FLAGS_START + id in SaveBlock1, pret include/constants/flags.h),
    so none battles on the way and the walk must reach Flannery."""
    from utils.mapping.porymap_state import _TRAINER_FLAGS_START, _map_trainers

    s = server()
    _enter_lavaridge_gym(s)
    beaten = []
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        flags = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET
        for name in (GYM_1F, GYM_B1F):
            for _, _, trainer, trainer_id in _map_trainers(name):
                if "FLANNERY" in trainer:
                    continue
                flag = _TRAINER_FLAGS_START + trainer_id
                s.srv.env.core.memory.u8[flags + flag // 8] |= 1 << (flag % 8)
                beaten.append(trainer)
        reader._invalidate_mem_cache()
        # The gym's trainers use pret's generic `trainerbattle TRAINER_BATTLE_CONTINUE_SCRIPT,
        # TRAINER_COLE, ...`, which _map_trainers used to miss (and the TRAINERS prompt too).
        assert {"COLE", "GERALD", "AXLE", "DANIELLE", "JACE", "KEEGAN", "JEFF", "ELI"} <= set(beaten), beaten
        assert all(reader._is_event_flag_set(_TRAINER_FLAGS_START + t[3]) for t in _map_trainers(GYM_B1F))
    r = s.call("mcp_talk_to", {"x": FLANNERY[0], "y": FLANNERY[1], "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"], r
    _talked_to_flannery(s, r)


def _to_route101_sign(s):
    """ROUTE 101 (5,10), facing the route sign at (5,9) after a BLOCKED UP."""
    s.load(ROUTE101, wild_encounters=False)
    for _ in range(2):  # a ledge jump on the way can leave the first walk a tile short
        r = s.call("mcp_navigate_to", {"x": 5, "y": 10, "variance": "none", "reason": "to the sign"})
        snap = s.snapshot()
        if (snap.x, snap.y) == (5, 10):
            break
    assert (snap.x, snap.y) == (5, 10), (snap, r)
    for _ in range(2):  # the first UP may only turn the player
        blocked = s.srv.last_blocked_press
        if blocked and (blocked["button"], blocked["x"], blocked["y"]) == ("UP", 5, 10):
            break
        r = s.call("mcp_press_buttons", {"buttons": ["UP"], "reasoning": "walk into the sign"})
    blocked = s.srv.last_blocked_press
    assert blocked and (blocked["button"], blocked["x"], blocked["y"]) == ("UP", 5, 10), (blocked, r)


def _reading(s):
    from server.field_control import DIALOGUE_PLAIN

    return s.srv._field_status() == DIALOGUE_PLAIN


def test_up_then_a_after_a_blocked_up_still_presses_a():
    """Seen live at Petalburg Gym's door (7,79): [UP, A] after a BLOCKED UP was refused
    whole. The repeated UP is left out; the A reads what the player faces."""
    s = server()
    _to_route101_sign(s)
    r = s.call("mcp_press_buttons", {"buttons": ["UP", "A"], "reasoning": "read the sign"})
    print("   ", r)
    assert r["success"] and r.get("status") != "REFUSED", r
    assert r["buttons_queued"] == ["A"] and "Not pressed: UP" in r["summary"], r
    assert _reading(s), s.srv._field_status()
    # UP alone is still refused there.
    s.call("mcp_press_buttons", {"buttons": ["B"], "reasoning": "close"})
    for _ in range(4):
        if not _reading(s):
            break
        s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "close"})
    r = s.call("mcp_press_buttons", {"buttons": ["UP"], "reasoning": "again"})
    assert r.get("status") == "REFUSED", r


def test_a_after_a_blocked_step_in_one_call_is_pressed():
    """[UP, A] in one call, facing the sign: the UP is BLOCKED, and the A after it used to
    be skipped with the rest of the walk. Presses up to the next direction stay."""
    s = server()
    _to_route101_sign(s)
    s.srv.last_blocked_press = None  # this test is about the skip, not the repeat guard
    r = s.call("mcp_press_buttons", {"buttons": ["UP", "A", "LEFT"], "reasoning": "read the sign"})
    print("   ", r)
    assert r.get("status") == "BLOCKED" and "1 of 3 presses skipped" in r["summary"], r
    assert _reading(s), s.srv._field_status()
    snap = s.snapshot()
    assert (snap.x, snap.y) == (5, 10), snap


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
