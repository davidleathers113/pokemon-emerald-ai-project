"""Dive: go_to_exit and go_to dive down to the underwater maps and surface from them, when a
party Pokémon knows DIVE and the Mind Badge is set; underwater maps are walked on foot.

Seen live (2026-11-05, build ef1a194): after Steven gave HM08 (AZUMARILL knows DIVE, the
Mind Badge set), the AI stood on ROUTE 128 (25,0) pressing A on ordinary water four times.
The EXITS list said "dive edge of the map -> UNDERWATER ROUTE 128", and
go_to_exit("UNDERWATER ROUTE 128") answered "No exit of ROUTE 128 matches". By hand: Surf
to (38,27), A, "The sea is deep here… use DIVE?" YES; underwater, UP into the warp at
(38,26); in UNDERWATER SEAFLOOR CAVERN, B, "Light is filtering down… use DIVE?" YES; that
lands in SEAFLOOR CAVERN ENTRANCE (10,17).

Fixtures, made from the live run's checkpoint (SEAFLOOR CAVERN ROOM1 (13,13), 2026-11-05
21:54 UTC: party SABLEYE, AZUMARILL (ROLLOUT, DIVE, SURF, STRENGTH), TROPIUS, MANECTRIC,
BRELOOM, BLAZIKEN; seven badges) with the harness's own tools, a Repel in RAM and POKéNAV
calls off: go_to_exit("SEAFLOOR CAVERN ENTRANCE"), then
  test_seafloor_cavern_entrance.state  SEAFLOOR CAVERN ENTRANCE (10,2), on land
navigate_to(10,16) (it surfs), A, YES:
  test_underwater_seafloor_cavern.state  UNDERWATER SEAFLOOR CAVERN (6,5)
DOWN x3 (the south arrow warp at (6,7)):
  test_underwater_route128.state  UNDERWATER ROUTE 128 (38,27)
B, YES (ROUTE 128 (38,27), surfing), navigate_to(25,0):
  test_route128_dive.state  ROUTE 128 (25,0), surfing: the live stall's tile

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_dive.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server
from test_surf import FLAG_HAS_MATCH_CALL, _clear_flag

HERE = Path(__file__).resolve().parent
ROUTE128 = HERE / "test_route128_dive.state"
UNDERWATER_ROUTE128 = HERE / "test_underwater_route128.state"
UNDERWATER_CAVERN = HERE / "test_underwater_seafloor_cavern.state"
CAVERN_ENTRANCE = HERE / "test_seafloor_cavern_entrance.state"
FLAG_BADGE07_GET = 0x86D  # pret include/constants/flags.h (SYSTEM_FLAGS + 0xD), the Mind Badge


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _load(s, path, mind_badge=True):
    """A fixture, with POKéNAV calls off (as test_surf._load: a call rang on ROUTE 128 while
    the fixtures were made) and the Mind Badge cleared in RAM when mind_badge is False."""
    from server import game_tools

    s.load(path, wild_encounters=False)
    with s.srv.memory_lock:
        _clear_flag(s, FLAG_HAS_MATCH_CALL)
        if not mind_badge:
            _clear_flag(s, FLAG_BADGE07_GET)
        s.srv.env.memory_reader._invalidate_mem_cache()
    game_tools._field_moves_cache[0] = 0.0  # the party's field moves are kept for a second


def _exits_text(s, location, position):
    from utils.mapping.porymap_state import _format_porymap_info

    with s.srv.memory_lock:
        text = "\n".join(_format_porymap_info(location, position, 7, s.srv.env.memory_reader).context_parts)
    return [line for line in text.splitlines() if line.startswith("- ") and "->" in line]


def test_dive_links_come_from_the_map_data():
    """pret src/overworld.c SetDiveWarp: a map's "dive" / "emerge" connection lands on the
    same x, y; without one, the map script's `setdivewarp` (sFixedDiveWarp). UNDERWATER
    SEALED CHAMBER sets two, chosen by where the player is: no fixed link is assumed. The
    EXITS line is no "edge of the map" any more."""
    from utils.mapping.porymap_state import dive_links

    assert [(l.button, l.dest, l.tile) for l in dive_links("Route128")] == [("A", "Underwater_Route128", None)]
    assert [(l.button, l.dest, l.tile) for l in dive_links("Underwater_Route128")] == [("B", "Route128", None)]
    assert [(l.button, l.dest, l.tile) for l in dive_links("Underwater_SeafloorCavern")] == [
        ("B", "SeafloorCavern_Entrance", (10, 17))
    ]
    assert [(l.button, l.dest, l.tile) for l in dive_links("SeafloorCavern_Entrance")] == [
        ("A", "Underwater_SeafloorCavern", (6, 5))
    ]
    assert dive_links("Underwater_SealedChamber") == ()
    assert dive_links("Route101") == ()

    s = server()
    _load(s, ROUTE128)
    lines = _exits_text(s, "ROUTE 128", (25, 0))
    assert "- dive (A on deep water, needs DIVE) -> UNDERWATER ROUTE 128" in lines, lines
    assert not any("dive edge" in line for line in lines), lines


def test_go_to_exit_dives_from_the_deep_water():
    """The live failure: go_to_exit("UNDERWATER ROUTE 128") from (25,0). It surfs to a deep
    water tile (A dives from the player's own tile, pret TrySetDiveWarp), A, YES."""
    s = server()
    _load(s, ROUTE128)
    r = s.call("mcp_go_to_exit", {"exit": "UNDERWATER ROUTE 128", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "UNDERWATER ROUTE 128", (_location(s), r)
    assert "used DIVE at" in r["summary"], r
    from server import game_tools
    from utils.mapping.pathfinding import is_diveable
    from utils.mapping.porymap_state import layout_tiles

    # The same coordinates, below the deep water dived from.
    snap = s.snapshot()
    assert is_diveable(layout_tiles("Route128")[snap.y][snap.x]), snap
    # "dive" names it too, and so does its EXITS line copied whole.
    _load(s, ROUTE128)
    with s.srv.memory_lock:
        for query in ("dive", "dive (A on deep water, needs DIVE) -> UNDERWATER ROUTE 128"):
            plan = game_tools.plan_exit_route(s.srv.env, query)
            assert plan["success"] and plan["kind"] == "dive" and plan["press"] == "A", (query, plan)


def test_go_to_plans_down_through_the_seabed_to_the_cavern():
    """go_to("SEAFLOOR CAVERN ENTRANCE") from the live stall: down at the deep water whose
    underwater tile is nearest the warp to the cavern, through the warp, up from the
    cavern (its fixed dive warp lands on (10,17))."""
    from server import game_tools

    s = server()
    _load(s, ROUTE128)
    with s.srv.memory_lock:
        route = game_tools.place_route("ROUTE 128", (25, 0), "SEAFLOOR CAVERN ENTRANCE")
    assert [name for _, name in route] == [
        "UNDERWATER ROUTE 128", "UNDERWATER SEAFLOOR CAVERN", "SEAFLOOR CAVERN ENTRANCE"
    ], route
    r = s.call("mcp_go_to", {"place": "SEAFLOOR CAVERN ENTRANCE", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "SEAFLOOR CAVERN ENTRANCE", (_location(s), r)
    snap = s.snapshot()
    assert (snap.x, snap.y) == (10, 17), snap
    assert r["summary"].count("used DIVE at") == 2, r


def test_the_dive_spot_is_chosen_by_the_way_on():
    """With the next step known (go_to's), the spot is the one whose landing is nearest
    that exit, counting the way to the spot: landings in reach of the warp to the cavern
    at (38,26) first."""
    from server import game_tools
    from utils.mapping.porymap_state import dive_links

    s = server()
    _load(s, ROUTE128)
    with s.srv.memory_lock:
        link = dive_links("Route128")[0]
        toward = game_tools._dive_spot_order("Route128", link, (25, 0), "UNDERWATER SEAFLOOR CAVERN")
        nearest = game_tools._dive_spot_order("Route128", link, (25, 0))
        below = game_tools._step_distances("Underwater_Route128", [(38, 26)], True)
    print("    toward the warp:", toward[:3], "nearest:", nearest[:3])
    assert toward and toward[0][1] in below, toward[:3]
    # Every spot's landing that reaches the warp comes before any that doesn't.
    reach = [landing in below for _, landing in toward]
    assert reach == sorted(reach, reverse=True), reach
    # Without the next step, the nearest spot: no landing is weighed.
    here = game_tools._step_distances("Route128", [(25, 0)], True)
    assert here[nearest[0][0]] == min(here[tile] for tile, _ in nearest), nearest[:3]


def test_go_to_comes_back_up_to_the_surface():
    """From the cavern's entrance on land: Surf to its pool, dive (its fixed dive warp),
    the cavern's south arrow warp, then up from UNDERWATER ROUTE 128."""
    s = server()
    _load(s, CAVERN_ENTRANCE)
    r = s.call("mcp_go_to", {"place": "Route 128", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "ROUTE 128", (_location(s), r)
    assert "UNDERWATER SEAFLOOR CAVERN → UNDERWATER ROUTE 128 → ROUTE 128" in r["summary"], r


def test_an_underwater_map_is_not_the_map_above():
    """Every underwater map is named UNDERWATER …; "Route 128" means the sea above."""
    from server import game_tools

    assert not game_tools.arrived_at("UNDERWATER ROUTE 128", (38, 27), "Route 128")
    assert game_tools.arrived_at("UNDERWATER ROUTE 128", (38, 27), "Underwater Route 128")
    assert game_tools.arrived_at("ROUTE 128", (25, 0), "Route 128")
    s = server()
    _load(s, UNDERWATER_ROUTE128)
    r = s.call("mcp_go_to", {"place": "Route 128", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is True and _location(s) == "ROUTE 128", (_location(s), r)


def test_underwater_floor_is_walked_on_foot():
    """pret src/field_player_avatar.c: underwater the player walks (PLAYER_AVATAR_FLAG_
    UNDERWATER, not _SURFING), by collision and elevation. UNDERWATER SEAFLOOR CAVERN's
    floor is MB_DEEP_WATER at elevation 3: read as water to Surf onto, no walk crossed it
    from its warp at (6,7), with or without Surf."""
    from server import game_tools
    from utils.mapping.porymap_state import _static_grid

    areas = game_tools._walk_areas_without("Underwater_SeafloorCavern", frozenset(), True, False)
    assert (6, 5) in areas and (5, 7) in areas and areas[(6, 5)] == areas[(5, 7)], areas
    assert _static_grid("Underwater_SeafloorCavern")[5][6] == ".", _static_grid("Underwater_SeafloorCavern")
    s = server()
    _load(s, UNDERWATER_CAVERN)
    r = s.call("mcp_navigate_to", {"x": 4, "y": 5, "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == (4, 5) and _location(s) == "UNDERWATER SEAFLOOR CAVERN", (snap, r)


def test_a_warp_on_open_water_is_only_where_one_lands():
    """pret IsWarpMetatileBehavior: no warp fires on water. SEAFLOOR CAVERN ENTRANCE's warp
    at (10,18) (MB_DEEP_WATER, in the game's map buffer too) is where one lands; listed as
    an exit, it was the shortest way to UNDERWATER ROUTE 128 and stepping on it does nothing."""
    from server import game_tools
    from utils.mapping.porymap_state import arrival_only_warps

    assert 0 in arrival_only_warps("SeafloorCavern_Entrance")
    exits = game_tools.list_map_exits(game_tools._raw_map_json("SeafloorCavern_Entrance"), "SeafloorCavern_Entrance")
    assert not any(e["kind"] == "warp" and (e["x"], e["y"]) == (10, 18) for e in exits), exits


def _set_var(s, name, value):
    from server import game_tools

    reader = s.srv.env.memory_reader
    sb1 = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
    var_id = game_tools._var_ids()[name]
    s.srv.env.core.memory.u16[sb1 + reader.addresses.SAVE_BLOCK1_VARS_OFFSET + (var_id - 0x4000) * 2] = value
    reader._invalidate_mem_cache()
    game_tools._layout_cache.clear()  # scripted_layout keeps its answer for a second


def test_maps_read_from_the_decomp_use_the_layout_their_script_swaps_in():
    """Seen (the coordinator, reading pret): Route131_OnTransition always calls
    `setmaplayoutindex LAYOUT_ROUTE131_SKY_PILLAR`; there (36,6), the warp to SKY PILLAR
    ENTRANCE, is a door (MB_NON_ANIMATED_DOOR), in the map file's layout open sea. Read
    from ROUTE 128 (not the map in the game's buffer), ROUTE 131 still lists the door, and
    a route through it exists. Conditional swaps follow the live vars: RAM writes to
    VAR_SKY_PILLAR_STATE (pret SkyPillar_1F_OnTransition: _CLEAN below 2) and
    VAR_SOOTOPOLIS_CITY_STATE (SootopolisCity_EventScript_SetLayout: legends at 1-4)."""
    from server import game_tools
    from utils.mapping.porymap_json_builder import build_json_map_for_llm
    from utils.mapping.porymap_state import _format_exits, _get_pokeemerald_root, arrival_only_warps

    s = server()
    _load(s, ROUTE128)
    with s.srv.memory_lock:
        json_map = build_json_map_for_llm("Route131", _get_pokeemerald_root(), 7)
        lines = _format_exits(json_map, json_map["ascii"].split("\n"), True)
        assert "- door/warp at (36, 6) -> SKY PILLAR ENTRANCE" in lines, lines
        assert any(arg == "SKY PILLAR ENTRANCE" for _, arg, _, _, _ in game_tools._walk_exits("Route131"))
        assert 0 not in arrival_only_warps("Route131")
        assert game_tools.scripted_layout("Route131") == "LAYOUT_ROUTE131_SKY_PILLAR"

        _set_var(s, "VAR_SKY_PILLAR_STATE", 0)
        assert game_tools.scripted_layout("SkyPillar_1F") == "LAYOUT_SKY_PILLAR_1F_CLEAN"
        _set_var(s, "VAR_SKY_PILLAR_STATE", 2)
        assert game_tools.scripted_layout("SkyPillar_1F") == ""
        _set_var(s, "VAR_SOOTOPOLIS_CITY_STATE", 3)
        assert game_tools.scripted_layout("SootopolisCity") == "LAYOUT_SOOTOPOLIS_CITY_LEGENDS_BATTLE"
        _set_var(s, "VAR_SOOTOPOLIS_CITY_STATE", 0)
        assert game_tools.scripted_layout("SootopolisCity") == ""
        # Mirage Island is a coin flip per day (specialvar IsMirageIslandPresent): the map
        # file's layout stays.
        assert game_tools.scripted_layout("Route130") == ""

    # On the map itself, the layout the game drew wins (note_live_layout): the Shoal Cave
    # tide comes from the clock at entry (pret UpdateShoalTideFlag), which the script
    # reading can't tell. test_shoal_cave_high_tide.state is drawn at high tide.
    room, high = "ShoalCave_LowTideEntranceRoom", "LAYOUT_SHOAL_CAVE_HIGH_TIDE_ENTRANCE_ROOM"
    s.load(HERE / "test_shoal_cave_high_tide.state", wild_encounters=False)
    with s.srv.memory_lock:
        game_tools._layout_cache.clear()
        assert game_tools.scripted_layout(room) == ""
        game_tools._route_areas(room)  # planned before the game's layout is noted
        game_tools.note_live_layout(s.srv.env.memory_reader)
        assert game_tools.layout_now(room) == high
        assert game_tools._walk_areas_without.cache_info().currsize == 0, "planned from the old layout"
    _load(s, ROUTE128)
    with s.srv.memory_lock:
        game_tools.note_live_layout(s.srv.env.memory_reader)
        assert game_tools.layout_now(room) == ""  # left behind: its script again


def test_without_the_mind_badge_there_is_no_dive():
    """pret TrySetupDiveDownScript checks FLAG_BADGE07_GET first: without it A on deep
    water does nothing, so neither the walk nor the route dives."""
    s = server()
    _load(s, ROUTE128, mind_badge=False)
    r = s.call("mcp_go_to_exit", {"exit": "UNDERWATER ROUTE 128", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is False and "the Mind Badge" in (r.get("error") or ""), r
    assert _location(s) == "ROUTE 128"
    lines = _exits_text(s, "ROUTE 128", (25, 0))
    assert any("UNDERWATER ROUTE 128 (needs a party Pokémon that knows DIVE and the Mind Badge)" in l for l in lines), lines
    from server import game_tools

    with s.srv.memory_lock:
        assert not game_tools.can_dive()
        assert game_tools.place_route("ROUTE 128", (25, 0), "SEAFLOOR CAVERN ENTRANCE") is None


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
