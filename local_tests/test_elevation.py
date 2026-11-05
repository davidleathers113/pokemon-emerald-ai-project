"""Elevation: every walker (the tile planner, the route areas, the ASCII map) follows the
game's own elevation rule on every map, not only on bridges and at the shore.

pret src/event_object_movement.c: IsElevationMismatchAt refuses a step onto a tile whose
elevation is not the walker's (COLLISION_ELEVATION_MISMATCH in GetCollisionAtCoords),
except for a walker at 0 or a tile at 0 or 15; ObjectEventUpdateElevation then gives the
walker the tile's elevation, except on a tile at 15 (a bridge), where it keeps its own.
People collide only at compatible elevations (AreElevationsCompatible).

Seen live:
  MOSSDEEP CITY: go_to and heal_at_pokecenter from the west shore (16,6) kept walking DOWN
  into "(16,7) is not walkable"; the ASCII map drew '.' there. (16,6) is elevation 3 and
  (16,7) elevation 5. The map's ASCII override gave every tile elevation 3 and drew the
  sea as floor.
  FORTREE CITY: the heal walk was blocked at "(20,14) is not walkable": the treetop
  walkway (row 14, elevation 4) was joined to the ground at (20,13) (elevation 3). The
  ladders at x=12 and x=37 (elevation 0) are the ways up.

Fixtures:
  test_fortree_gym.state       FORTREE CITY GYM (2,19), facing UP (the gym's exit is the
                               way to the Pokémon Center).
  test_mossdeep_west_shore.state  MOSSDEEP CITY (16,6), the west shore, facing DOWN;
                               player elevation 3. Made without RAM edits from the live
                               run's checkpoint in MOSSDEEP CITY GYM (6,35) (2026-11-05
                               03:31): go_to_exit("MOSSDEEP CITY") to (38,10), then
                               navigate_to(16, 6), then env.save_state.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_elevation.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

FORTREE_GYM = Path(__file__).resolve().parent / "test_fortree_gym.state"
MOSSDEEP_SHORE = Path(__file__).resolve().parent / "test_mossdeep_west_shore.state"
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _game_state(map_dir, **porymap):
    """A pathfinding state for map_dir from the decomp alone (the tiles walks are planned
    on, its warp tiles), with porymap's extra fields."""
    from server.game_tools import _raw_map_json
    from utils.mapping.porymap_json_builder import build_json_map
    from utils.mapping.porymap_state import _get_pokeemerald_root

    built = build_json_map(map_dir, _get_pokeemerald_root(), include_grid=True, include_ascii=False, badge_count=8)
    warps = [(int(w["x"]), int(w["y"])) for w in _raw_map_json(map_dir).get("warp_events") or []]
    return {
        "player": {},
        "map": {"porymap": {
            "grid": built["grid"], "raw_tiles": built["raw_tiles"], "dimensions": built["dimensions"],
            "warps": [], "raw_warps": warps, **porymap,
        }},
    }


def _tiles_walked(start, buttons):
    x, y = start
    out = [(x, y)]
    for button in buttons:
        dx, dy = DELTAS[button]
        x, y = x + dx, y + dy
        out.append((x, y))
    return out


# --- the rule on real layout data ----------------------------------------------------


def test_a_step_between_floors_is_refused_and_a_tile_at_0_joins_them():
    """FORTREE CITY's tiles: (20,13) is 3, (20,14) is 4, the ladder (12,15)-(12,16) is 0,
    the bridge (30-33,14) is 15. Ground to walkway directly: refused. Through the ladder:
    allowed. Across the bridge a walker keeps the walkway's 4, so one on it at 3 can't step
    off onto the walkway."""
    from utils.mapping.pathfinding import Pathfinder

    finder = Pathfinder()
    md = finder._extract_map_data(_game_state("FortreeCity"))
    elevation = {p: md["raw_tiles"][p[1]][p[0]][3] for p in [(20, 13), (20, 14), (12, 14), (12, 15), (12, 16), (12, 17), (30, 14), (33, 14), (34, 14)]}
    assert elevation == {(20, 13): 3, (20, 14): 4, (12, 14): 4, (12, 15): 0, (12, 16): 0, (12, 17): 3,
                         (30, 14): 15, (33, 14): 15, (34, 14): 4}, elevation
    assert not finder._can_move_to((20, 13), (20, 14), md, 3)  # 3 -> 4
    assert not finder._can_move_to((20, 14), (20, 13), md, 4)  # 4 -> 3
    assert finder._can_move_to((12, 14), (12, 15), md, 4)  # 4 -> 0
    assert finder._can_move_to((12, 16), (12, 17), md, 0)  # 0 -> 3
    assert finder._can_move_to((33, 14), (34, 14), md, 4)  # off the bridge at the walkway's 4
    assert not finder._can_move_to((33, 14), (34, 14), md, 3)  # carried 3 onto 4: refused


def test_the_planner_climbs_by_the_ladder():
    """From the ground (20,13) to the walkway right below it (20,14): the plan goes round
    by a ladder (elevation 0), never straight down."""
    from utils.mapping.pathfinding import Pathfinder

    buttons = Pathfinder().find_path((20, 13), (20, 14), _game_state("FortreeCity"), allow_partial=False)
    walked = _tiles_walked((20, 13), buttons or [])
    print("   ", len(buttons or []), "presses")
    assert buttons and walked[-1] == (20, 14), buttons
    assert len(buttons) > 1 and ((12, 15) in walked or (37, 15) in walked), walked


def test_a_bridge_carries_each_floor_at_its_own_height():
    """ROUTE 120's south bridge (24-31, 67-68) is elevation 15 over water (1), with the
    walkway (4) at its ends. On the bridge a walker (4) goes on to the walkway and not into
    the water; a surfer under it (1) goes on to the water and not up onto the walkway. The
    player's live elevation decides which one starts on a bridge tile."""
    from utils.mapping.pathfinding import Pathfinder

    finder = Pathfinder()
    md = finder._extract_map_data(_game_state("Route120", surf=True))
    assert [md["raw_tiles"][y][x][3] for x, y in [(23, 67), (24, 67), (27, 67), (27, 66), (27, 69)]] == [4, 15, 15, 1, 1]
    assert finder._can_move_to((24, 67), (23, 67), md, 4) and not finder._can_move_to((24, 67), (23, 67), md, 1)
    assert finder._can_move_to((27, 68), (27, 69), md, 1) and not finder._can_move_to((27, 68), (27, 69), md, 4)

    surfer = Pathfinder().find_path(
        (27, 67), (27, 69), _game_state("Route120", surf=True, player_position=(27, 67), player_elevation=1),
        allow_partial=False,
    )
    assert surfer == ["DOWN", "DOWN"], surfer
    walker = Pathfinder().find_path(
        (27, 67), (22, 67), _game_state("Route120", surf=True, player_position=(27, 67), player_elevation=4),
        allow_partial=False,
    )
    assert walker == ["LEFT"] * 5, walker


def test_people_on_a_bridge_stop_only_walkers_at_their_elevation():
    """pret DoesObjectCollideWithObjectAt/AreElevationsCompatible: someone standing on the
    bridge (elevation 4) does not stop a surfer passing under it (1); someone at 1 there
    does."""
    from utils.mapping.pathfinding import Pathfinder

    def plan(elevation):
        people = [{"current_x": x, "current_y": y, "elevation": elevation} for x in range(24, 32) for y in (67, 68)]
        state = _game_state("Route120", surf=True, runtime_objects=people)
        return Pathfinder().find_path((27, 66), (27, 69), state, consider_npcs=True, allow_partial=False)

    on_top, in_the_water = plan(4), plan(1)
    print("   ", on_top, len(in_the_water or []))
    assert on_top == ["DOWN"] * 3, on_top
    assert in_the_water != ["DOWN"] * 3, in_the_water


def test_route_areas_follow_elevation():
    """The route areas use the planner's steps: MOSSDEEP CITY's west shore (3) and the town
    above it (5) are one area (stairs join them) though no step joins them at (16,6)-(16,7);
    the cliff edge (13,9), floor on the override's ASCII but collision in the layout, is in
    no area; under ROUTE 120's bridge the water joins the water beyond, and the bridge's
    tiles go with the walkway they carry."""
    from server.game_tools import _walk_areas_without

    mossdeep = _walk_areas_without("MossdeepCity", frozenset(), True, False)
    assert mossdeep[(16, 6)] == mossdeep[(16, 7)], (mossdeep.get((16, 6)), mossdeep.get((16, 7)))
    assert (13, 9) not in mossdeep and (8, 7) not in mossdeep  # cliff edge; the sea, without Surf

    route120 = _walk_areas_without("Route120", frozenset(), True, True)
    assert route120[(27, 66)] == route120[(27, 69)]
    assert route120[(27, 67)] == route120[(22, 67)]


def test_every_same_size_override_walks_on_the_layouts_own_tiles():
    """Every ASCII override the size of its pret layout (ascii_map_loader.MAP_OVERRIDES) is
    walked on map.bin's own words, cell for cell (pret: metatile id = word & 0x3FF,
    collision = (word >> 10) & 3, elevation = word >> 12), except its doors and warps, which
    keep the override's tile with the layout's elevation (override_walk_tiles).

    Checked after the live MOSSDEEP CITY "No path found" to the Space Center (2026-11-05):
    the planner's tiles there were right, and Team Magma walling the street was the cause
    (test_scene_walls)."""
    import json
    import struct

    from utils.mapping.ascii_map_loader import MAP_OVERRIDES
    from utils.mapping.porymap_json_builder import build_json_map
    from utils.mapping.porymap_state import _get_pokeemerald_root

    root = _get_pokeemerald_root()
    layouts = {
        entry["id"]: entry
        for entry in json.loads((root / "data" / "layouts" / "layouts.json").read_text())["layouts"]
        if "id" in entry
    }
    checked, differ = [], []
    for name in MAP_OVERRIDES:
        map_json = root / "data" / "maps" / name / "map.json"
        if not map_json.exists():
            continue  # an override of a map pret doesn't have (PetalburgCity_Gym_Lobby)
        data = json.loads(map_json.read_text())
        layout = layouts[data["layout"]]
        width, height = layout["width"], layout["height"]
        words = struct.unpack(f"<{width * height}H", (root / layout["blockdata_filepath"]).read_bytes()[: width * height * 2])
        built = build_json_map(name, root, include_grid=True, include_ascii=False, badge_count=8)
        tiles, grid = built["raw_tiles"], built["grid"]
        if len(tiles) != height or any(len(row) != width for row in tiles):
            continue  # not the layout's size: walked on the override's own tiles
        checked.append(name)
        warps = {(int(w["x"]), int(w["y"])) for w in data.get("warp_events") or []}
        for y in range(height):
            for x in range(width):
                word = words[y * width + x]
                tile = tiles[y][x]
                if (x, y) in warps or grid[y][x] in ("D", "S"):
                    if tile[3] != word >> 12:
                        differ.append((name, (x, y), tile, word))
                elif (tile[0], tile[2], tile[3]) != (word & 0x3FF, (word >> 10) & 3, word >> 12):
                    differ.append((name, (x, y), tile, word))
    print("   ", len(checked), "same-size overrides:", ", ".join(checked))
    assert "MossdeepCity" in checked and "FortreeCity" in checked, checked
    assert not differ, differ[:10]


# --- the walks on the live fixtures ---------------------------------------------------


def test_fortree_heal_walks_round_by_the_ladder():
    """The live failure: heal_at_pokecenter from the gym walked from the gym door down onto
    the walkway at (20,14) and stopped. Now it reaches the nurse."""
    s = server()
    s.load(FORTREE_GYM, wild_encounters=False)
    started = time.time()
    r = s.call("mcp_heal_at_pokecenter", {"reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"] is True and _location(s) == "FORTREE CITY POKEMON CENTER 1F", (r, _location(s))
    assert "not walkable" not in r.get("summary", ""), r


def test_mossdeep_heal_from_the_west_shore():
    s = server()
    s.load(MOSSDEEP_SHORE)
    snap = s.snapshot()
    assert (snap.x, snap.y) == (16, 6) and _location(s) == "MOSSDEEP CITY", snap
    started = time.time()
    r = s.call("mcp_heal_at_pokecenter", {"reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY POKEMON CENTER 1F", (r, _location(s))


def test_mossdeep_go_to_the_pokemon_center_from_the_west_shore():
    s = server()
    s.load(MOSSDEEP_SHORE)
    r = s.call("mcp_go_to", {"place": "Mossdeep City Pokemon Center", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY POKEMON CENTER 1F", (r, _location(s))


def test_mossdeep_map_shows_the_town_above_the_shore_as_another_floor():
    """The ASCII map on the west shore: (16,7) (elevation 5) is '^', not '.', while the
    shore beside the player (17,6) stays '.'; the legend says what '^' means. The player's
    elevation is read live (pret gObjectEvents[player].currentElevation) and the planner
    starts from it."""
    from server import game_tools
    from server.field_control import player_elevation
    from utils.mapping.porymap_state import _format_porymap_info

    s = server()
    s.load(MOSSDEEP_SHORE)
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        assert player_elevation(reader) == 3
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
        parts = _format_porymap_info("MOSSDEEP CITY", (16, 6), 7, reader).context_parts
    porymap = state["map"]["porymap"]
    assert porymap["player_position"] == (16, 6) and porymap["player_elevation"] == 3, porymap.get("player_elevation")
    rows = parts[parts.index("\nASCII Map:") + 1].split("\n")
    print("   ", rows[6][10:24], "/", rows[7][10:24])
    assert rows[6][16] == "P" and rows[6][17] == "." and rows[7][16] == "^", (rows[6], rows[7])
    legend = next(p for p in parts if p.startswith("(Legend:"))
    assert "'^' = floor at another elevation or cut off from you" in legend, legend


def _run_all(only=""):
    import test_movement_feedback as harness

    failures = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f) and only in n]
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
