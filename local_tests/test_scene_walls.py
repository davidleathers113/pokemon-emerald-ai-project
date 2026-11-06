"""Scene walls: people a story scene takes away stand in the way only until it plays, so
a walk whose only way is past them walks into the scene, lets it play, and goes on.

pret data/maps/MossdeepCity/scripts.inc: after the Mind Badge (MossdeepCity_Gym
TateAndLizaDefeated: clearflag FLAG_HIDE_MOSSDEEP_CITY_TEAM_MAGMA, setvar
VAR_MOSSDEEP_CITY_STATE 1) Maxie (45,25) and four grunts (44,23)-(44,26) stand across the
street to the Space Center. The coord events at (42,21), (41,22)-(41,24), (40,25), (40,26)
(VAR_MOSSDEEP_CITY_STATE 1) run MossdeepCity_EventScript_TeamMagmaEnterSpaceCenter, which
walks them into the Space Center and `removeobject`s them.

Seen live (2026-11-05, ef1a194): from MOSSDEEP CITY (39,15), navigate_to(64,15) answered
"No path found ... Target tile 'D' may be unreachable" and go_to("Mossdeep Space Center")
"no walkable path to the warp at (64, 15)". The planner counted Magma as a wall forever.
Walked by hand, the scene played at (40,25) and the street was open. (A scratch BFS over the
layout found the way because it ignored people; the planner's tiles and elevations match
pret's map.bin cell for cell, test_elevation's override check.)

Fixture:
  test_mossdeep_magma_street.state  MOSSDEEP CITY (39,15), facing RIGHT, Magma across the
      street. No live state between the Mind Badge and the scene was kept (the run's
      checkpoint is already inside the Space Center, VAR_MOSSDEEP_CITY_STATE 2), so it was
      made from the live checkpoint in MOSSDEEP CITY POKEMON CENTER 1F (7,4) (2026-11-05
      15:34, VAR_MOSSDEEP_CITY_STATE 0) with ONE RAM write, the two lines of pret's
      TateAndLizaDefeated that set up the scene: FLAG_HIDE_MOSSDEEP_CITY_TEAM_MAGMA (0x337)
      cleared and VAR_MOSSDEEP_CITY_STATE (0x405D) set to 1, in SaveBlock1. Then
      go_to_exit("MOSSDEEP CITY") (the map load spawns Magma from the flag), navigate_to(39,
      15), env.save_state.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_scene_walls.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_elevation import _game_state, _location, _tiles_walked
from test_movement_feedback import server

MAGMA_STREET = Path(__file__).resolve().parent / "test_mossdeep_magma_street.state"
SCENE_TILES = {(42, 21), (41, 22), (41, 23), (41, 24), (40, 25), (40, 26)}
MAGMA = {(44, 23), (44, 24), (44, 25), (44, 26), (45, 25)}
VAR_MOSSDEEP_CITY_STATE = 0x405D


class _Vars:
    """A memory reader that answers read_game_var from a dict (scene_clears reads only that)."""

    def __init__(self, values):
        self.values = values

    def read_game_var(self, var_id):
        return self.values.get(var_id)


class _Env:
    def __init__(self, values):
        self.memory_reader = _Vars(values)


def _mossdeep_with_magma(state_value):
    """MOSSDEEP CITY's decomp state with Magma standing (the hide flag clear) and the
    scenes scene_clears finds with VAR_MOSSDEEP_CITY_STATE at state_value."""
    from server import game_tools
    from utils.mapping.porymap_json_builder import build_json_map
    from utils.mapping.porymap_state import _get_pokeemerald_root

    objects = build_json_map("MossdeepCity", _get_pokeemerald_root(), include_grid=True, include_ascii=False,
                             badge_count=8)["objects"]
    scenes = game_tools.scene_clears(_Env({VAR_MOSSDEEP_CITY_STATE: state_value}), "MossdeepCity", objects)
    return _game_state("MossdeepCity", objects=objects, scenes=scenes, player_position=(39, 15), player_elevation=5), scenes


# --- the planner on the decomp's data -------------------------------------------------


def test_the_scene_frees_the_people_it_removes():
    """scene_clears reads the scene's `removeobject`s: its six tiles free Magma's five,
    and only while VAR_MOSSDEEP_CITY_STATE is 1 (0: before the badge; 2: it has played)."""
    _, scenes = _mossdeep_with_magma(1)
    print("   ", scenes)
    assert len(scenes) == 1, scenes
    assert set(scenes[0]["tiles"]) == SCENE_TILES and set(scenes[0]["clears"]) == MAGMA, scenes
    assert _mossdeep_with_magma(0)[1] == [] and _mossdeep_with_magma(2)[1] == []


def test_the_planner_walks_into_the_scene_when_magma_walls_the_street():
    """From (39,15) to the Space Center door (64,15): with Magma standing and no scene the
    planner finds no way (the live failure); with the scene it walks onto a scene tile
    before it reaches any tile Magma stands on, then on to the door."""
    from utils.mapping.pathfinding import Pathfinder

    state, _ = _mossdeep_with_magma(1)
    walled = dict(state["map"]["porymap"], scenes=[])
    assert Pathfinder().find_path((39, 15), (64, 15), {**state, "map": {"porymap": walled}},
                                  consider_npcs=True, allow_partial=False) is None

    finder = Pathfinder()
    buttons = finder.find_path((39, 15), (64, 15), state, consider_npcs=True, allow_partial=False)
    walked = _tiles_walked((39, 15), buttons or [])
    print("   ", len(buttons or []), "presses, scene at", finder.last_scene)
    assert buttons and walked[-1] == (64, 15), buttons
    assert finder.last_scene in SCENE_TILES, finder.last_scene
    first_scene = min(i for i, t in enumerate(walked) if t in SCENE_TILES)
    assert all(t not in MAGMA for t in walked[:first_scene]), walked


def test_a_walk_that_has_another_way_leaves_the_scene_alone():
    """The scene is walked into only when Magma walls the only way: (39,15) to the Pokémon
    Center door (28,16) never steps on a scene tile."""
    from utils.mapping.pathfinding import Pathfinder

    state, _ = _mossdeep_with_magma(1)
    finder = Pathfinder()
    buttons = finder.find_path((39, 15), (28, 16), state, consider_npcs=True, allow_partial=False)
    walked = _tiles_walked((39, 15), buttons or [])
    assert buttons and walked[-1] == (28, 16), buttons
    assert finder.last_scene is None and not SCENE_TILES & set(walked), walked


# --- the walks on the live fixture ----------------------------------------------------


def _at_the_street(s):
    s.load(MAGMA_STREET, wild_encounters=False)
    snap = s.snapshot()
    assert (snap.x, snap.y) == (39, 15) and _location(s) == "MOSSDEEP CITY", snap
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.read_game_var(VAR_MOSSDEEP_CITY_STATE) == 1


def test_navigate_to_the_space_center_door_through_the_scene():
    s = server()
    _at_the_street(s)
    started = time.time()
    r = s.call("mcp_navigate_to", {"x": 64, "y": 15, "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r.get("success") is True and _location(s) == "MOSSDEEP CITY SPACE CENTER 1F", (r, _location(s))
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.read_game_var(VAR_MOSSDEEP_CITY_STATE) == 2  # the scene played


def test_go_to_the_space_center_through_the_scene():
    s = server()
    _at_the_street(s)
    r = s.call("mcp_go_to", {"place": "Mossdeep Space Center", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY SPACE CENTER 1F", (r, _location(s))


def test_go_to_exit_house3_from_the_ninja_boy_through_the_scene():
    """MOSSDEEP CITY HOUSE3 (49,6) from (25,21), west of the street: the way east also
    runs past Magma."""
    s = server()
    _at_the_street(s)
    # The ninja boy beside (25,21) walks about and a POKéNAV call can ring: one more try.
    for _ in range(2):
        r = s.call("mcp_navigate_to", {"x": 25, "y": 21, "variance": "none", "reason": "test"})
        snap = s.snapshot()
        if (snap.x, snap.y) == (25, 21):
            break
    assert (snap.x, snap.y) == (25, 21) and _location(s) == "MOSSDEEP CITY", (r, snap)
    r = s.call("mcp_go_to_exit", {"exit": "MOSSDEEP CITY HOUSE3", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY HOUSE3", (r, _location(s))


def _step_in_front(s, buttons):
    """Move the person nearest the player (their current, previous and initial coords)
    onto the tile of the walk's first press in the direction the player already faces,
    so the walk about to start bumps into them as the ninja boy did: a press that turns
    the player first is judged a turn, not a block. pret struct ObjectEvent: +0x0C
    initialCoords, +0x10 currentCoords, +0x14 previousCoords (s16 x, y, map coords +
    MAP_OFFSET)."""
    from server.field_control import (
        DIRECTION_DELTAS,
        G_OBJECT_EVENTS,
        MAP_OFFSET,
        OBJECT_EVENT_COUNT,
        OBJECT_EVENT_SIZE,
        _player_object_base,
    )

    reader = s.srv.env.memory_reader
    player = _player_object_base(reader)
    px, py = reader.read_coordinates()
    x, y = px, py
    facing = s.snapshot().facing
    for button in buttons:
        dx, dy = DIRECTION_DELTAS.get(button, (0, 0))
        x, y = x + dx, y + dy
        if button == facing:
            break
        facing = button
    people = []
    for i in range(OBJECT_EVENT_COUNT):
        base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
        if base != player and reader._read_u8(base) & 1:
            ox, oy = reader._read_s16(base + 0x10) - MAP_OFFSET, reader._read_s16(base + 0x12) - MAP_OFFSET
            people.append((abs(ox - px) + abs(oy - py), base))
    base = min(people)[1]
    memory = s.srv.env.core.memory
    for offset in (0x0C, 0x10, 0x14):
        memory.u16[base + offset] = x + MAP_OFFSET
        memory.u16[base + offset + 2] = y + MAP_OFFSET
    # Frozen (ObjectEvent +0x01 bit 0) they stay on that tile: wandering on, they could
    # stand where a turning press leads, which is judged a turn, not a block, and the
    # walk loses a step for a reason this test is not about. The scene's releaseall
    # (pret UnfreezeObjectEvents) lets them go.
    memory.u8[base + 1] = memory.u8[base + 1] | 1
    reader._invalidate_mem_cache()
    return x, y


def test_a_person_in_the_way_before_the_scene_is_not_taken_for_it():
    """Seen in suite runs under load (2 of ~30): the ninja boy wandered into the first
    walks from (25,21); each was BLOCKED by him, and go_to_exit counted every stopped walk
    on a plan through the scene as the scene, so two blocks used up SCENE_REPLANS and the
    real scene at (40,25) then ended the call: "Did not leave the map ... STOPPED at
    (40,25) ... On the way: a scene played at (40, 25); a scene played at (40, 25)". Here a
    person is put in front of the first two walks, as they start."""
    s = server()
    _at_the_street(s)
    for _ in range(2):
        s.call("mcp_navigate_to", {"x": 25, "y": 21, "variance": "none", "reason": "test"})
        snap = s.snapshot()
        if (snap.x, snap.y) == (25, 21):
            break
    take_action = s.srv.take_action
    placed = []

    async def walk_into_someone(request, *args, **kwargs):
        if getattr(request, "source", None) == "go_to_exit" and len(placed) < 2:
            with s.srv.memory_lock:
                placed.append(_step_in_front(s, list(request.buttons)))
        return await take_action(request, *args, **kwargs)

    s.srv.take_action = walk_into_someone
    try:
        r = s.call("mcp_go_to_exit", {"exit": "MOSSDEEP CITY HOUSE3", "reasoning": "test"})
    finally:
        s.srv.take_action = take_action
    print("   ", placed, r.get("summary") or r)
    assert r["success"] is True and _location(s) == "MOSSDEEP CITY HOUSE3", (r, _location(s))
    # The walk round the person may reach the scene on another of its tiles (SCENE_TILES).
    scenes = [t for t in r.get("on_the_way", []) if t.startswith("a scene played at")]
    assert len(scenes) == 1, r


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
