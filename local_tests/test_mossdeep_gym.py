"""MOSSDEEP CITY GYM: statues where they stand, talk_to on a statue, floor switches that
move them (pret src/rotating_tile_puzzle.c), and the walk to TATE AND LIZA through the
switches and pads.

Seen live (2026-11-05, MOSSDEEP CITY GYM (3,20)): "someone" at (4,20) blocked the way,
and talk_to(4,20) said "nothing was said (nobody there, or they only turned)": a statue
(OBJ_EVENT_GFX_TRICK_HOUSE_STATUE, no script) that the yellow switch (2,21) had moved off
its map-file tile (5,21). The map drew it, and BLAKE (now (4,21)), on their map-file
tiles; walks could cross the switches; and no plan reached TATE AND LIZA (23,7).

pret, not the first guess: the gym's yellow arrows never move the player (their
metatiles are MB_NORMAL). A switch (a coord event running `moverotatingtileobjects N`)
moves every object standing on an arrow of color N one tile along it, statues and
trainers alike, with no collision check; the moves stay in SaveBlock1's object templates,
also across the gym's pads (field_screen_effect.c DoMossdeepGymWarp,
SKIP_OBJECT_EVENT_LOAD).

Fixtures:
  test_mossdeep_gym_statue.state    MOSSDEEP CITY GYM (3,20), the live checkpoint of
                                    2026-11-05: statue 14 at (4,20), BLAKE at (4,21)
  test_mossdeep_gym_entrance.state  MOSSDEEP CITY GYM (6,35), at the entrance
  test_mossdeep_west_shore.state    MOSSDEEP CITY (16,6), outside (the gym level gate)

RAM write, documented where used: the gym trainers' trainer flags set (beaten), for the
walk to TATE AND LIZA.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_mossdeep_gym.py [name-substring]
"""
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent
STATUE_STATE = HERE / "test_mossdeep_gym_statue.state"
GYM = "MossdeepCity_Gym"
# pret MossdeepCity_Gym map.json object_events: 14th a statue at (5,21), 3rd BLAKE at
# (5,20). In the fixture the yellow switch has moved them to (4,20) and (4,21).
STATUE_ID, BLAKE_ID = 14, 3
STATUE_NOW, BLAKE_NOW = (4, 20), (4, 21)
YELLOW_SWITCH = (2, 21)  # coord event MossdeepCity_Gym_EventScript_YellowFloorSwitch
BELOW_TATE = (23, 8)


def _places(s):
    """{local id: tile} read here straight from RAM, apart from the code under test: pret
    SaveBlock1 +0xC70 objectEventTemplates (24 bytes: +0 localId, +4 x, +6 y), then each
    loaded ObjectEvent (gObjectEvents 0x02037350, 0x24 bytes: +0 active bit, +8 localId,
    +0x10 currentCoords + MAP_OFFSET 7)."""
    with s.srv.memory_lock:
        r = s.srv.env.memory_reader
        sb1 = r._read_u32(r.addresses.SAVE_BLOCK1_PTR)
        places = {}
        for i in range(64):
            base = sb1 + 0xC70 + i * 24
            if r._read_u8(base):
                places[r._read_u8(base)] = (r._read_s16(base + 4), r._read_s16(base + 6))
        player = r._read_u8(0x02037590 + 5)
        for i in range(16):
            base = 0x02037350 + i * 0x24
            if i != player and r._read_u8(base) & 1 and r._read_u8(base + 8) in places:
                places[r._read_u8(base + 8)] = (r._read_s16(base + 0x10) - 7, r._read_s16(base + 0x12) - 7)
        return places


def _load(s):
    s.load(STATUE_STATE, wild_encounters=False)
    snap = s.snapshot()
    assert s.srv._location_name() == "MOSSDEEP CITY GYM" and (snap.x, snap.y) == (3, 20), snap


def _map_rows(s):
    from utils.mapping.porymap_state import _format_porymap_info

    with s.srv.memory_lock:
        parts = _format_porymap_info("MOSSDEEP CITY GYM", (3, 20), 6, s.srv.env.memory_reader).context_parts
    return parts[parts.index("\nASCII Map:") + 1].split("\n"), "\n".join(parts)


def _planned_objects(s):
    from server import game_tools

    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    return {(o["x"], o["y"]): o.get("graphics_id") for o in state["map"]["porymap"]["objects"]}


def test_statues_are_drawn_and_planned_where_they_stand():
    """'Z' on the statue's live tile, 'N' on BLAKE's; nothing on their map-file tiles; the
    planner's objects are on the same tiles."""
    s = server()
    _load(s)
    places = _places(s)
    assert places[STATUE_ID] == STATUE_NOW and places[BLAKE_ID] == BLAKE_NOW, places
    rows, _ = _map_rows(s)
    print("   ", rows[19:24])
    assert rows[20][4] == "Z" and rows[21][4] == "N", rows[20:22]
    assert rows[21][5] not in "ZN" and rows[20][5] not in "ZN", rows[20:22]
    planned = _planned_objects(s)
    assert planned.get(STATUE_NOW) == "OBJ_EVENT_GFX_TRICK_HOUSE_STATUE", planned
    assert BLAKE_NOW in planned and (5, 21) not in planned and (5, 20) not in planned, planned


def test_the_legend_names_statues():
    """The legend the model reads is SIMPLE.md's (PokeAgent strips the prompt's own)."""
    simple = (HERE.parent / "agents" / "prompts" / "pokeagent-directives" / "SIMPLE.md").read_text(encoding="utf-8")
    assert "'Z' = statue" in simple, simple[:1500]


def test_talk_to_a_statue_says_it_is_a_statue():
    s = server()
    _load(s)
    r = s.call("mcp_talk_to", {"x": STATUE_NOW[0], "y": STATUE_NOW[1], "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is False and "is a statue" in r["error"], r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (3, 20), snap  # no walk up to it


def test_a_floor_switch_moves_the_statues_and_the_plan_follows():
    """Stepping on the yellow switch (2,21): statue 14 on (4,20), a yellow Down arrow
    (metatile 0x251), goes to (4,21); BLAKE on (4,21), Right (0x250), to (5,21) (pret
    MoveRotatingTileObjects). The planner and the map read them there."""
    s = server()
    _load(s)
    r = s.call("mcp_navigate_to", {"x": YELLOW_SWITCH[0], "y": YELLOW_SWITCH[1], "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    deadline = time.time() + 20
    while time.time() < deadline and _places(s)[STATUE_ID] == STATUE_NOW:
        time.sleep(0.2)
    places = _places(s)
    assert places[STATUE_ID] == (4, 21) and places[BLAKE_ID] == (5, 21), places
    planned = _planned_objects(s)
    assert planned.get((4, 21)) == "OBJ_EVENT_GFX_TRICK_HOUSE_STATUE" and (5, 21) in planned, planned
    assert STATUE_NOW not in planned, planned


def test_walks_never_cross_a_floor_switch_they_do_not_mean_to_press():
    """From (3,20) the only way to the pad (1,23) is over the yellow switch (2,21). A walk
    over it moves the statues as a side effect, so go_to_exit won't take it (the old rule
    walked over a trigger when it was the only way); navigate_to plans the press itself."""
    s = server()
    _load(s)
    before = _places(s)
    r = s.call("mcp_go_to_exit", {"exit": "MOSSDEEP CITY GYM (1, 23)", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] is False and "floor switches" in r["error"], r
    assert _places(s) == before, (before, _places(s))
    snap = s.snapshot()
    assert (snap.x, snap.y) != YELLOW_SWITCH, snap


def test_navigate_to_tate_and_liza_goes_through_the_switches_and_pads():
    """The whole way from (3,20) to (23,8), below TATE AND LIZA (23,7): 18 switch presses
    and 5 pads (game_tools.puzzle_plan), planned again after each from where the objects
    stand. The gym's trainers are marked beaten in RAM (TRAINER_FLAGS_START + id in
    SaveBlock1's flags, pret include/constants/flags.h) so that none battles on the way:
    a battle would stop the walk, which is what it does elsewhere too."""
    from utils.mapping.porymap_state import _TRAINER_FLAGS_START, _map_trainers

    s = server()
    _load(s)
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        flags = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET
        for _, _, trainer, trainer_id in _map_trainers(GYM):
            if "TATE" in trainer:
                continue
            flag = _TRAINER_FLAGS_START + trainer_id
            s.srv.env.core.memory.u8[flags + flag // 8] |= 1 << (flag % 8)
        reader._invalidate_mem_cache()
    # The gym level gate (David, 2026-11-05: Lv48 here) refuses walks up to TATE AND LIZA
    # with this Lv30-42 party; this test is about the way through the puzzle, so it is
    # lowered for the walk (test_the_gym_level_gate_* cover the gate).
    with _gym_gate("MOSSDEEP CITY GYM=5"):
        r = s.call("mcp_navigate_to", {"x": BELOW_TATE[0], "y": BELOW_TATE[1], "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert r["success"] and (snap.x, snap.y) == BELOW_TATE, (r, snap)
    assert s.srv._location_name() == "MOSSDEEP CITY GYM", s.srv._location_name()


# The gym level gate (server/app.py GYM_MIN_LEVELS: MOSSDEEP CITY GYM 48). David
# (2026-11-05), after the loss to TATE & LIZA: "no gym rematch until all six party Pokémon
# are Lv 48+". The fixtures' party is AZUMARILL 40, TROPIUS 30-31, SABLEYE 37, BLAZIKEN 37,
# BRELOOM 42, MANECTRIC 37.
WEST_SHORE = HERE / "test_mossdeep_west_shore.state"  # MOSSDEEP CITY (16,6)
GYM_DOOR = (38, 9)  # pret MossdeepCity map.json warp_events: -> MAP_MOSSDEEP_CITY_GYM
LEADER = (23, 7)  # TATE (LIZA at (24,7)), MossdeepCity_Gym_EventScript_TateAndLiza


@contextmanager
def _gym_gate(value):
    """GYM_MIN_LEVELS, the environment override of the gate's table, for one block (None:
    unset, the table in server/app.py applies)."""
    old = os.environ.pop("GYM_MIN_LEVELS", None)
    if value is not None:
        os.environ["GYM_MIN_LEVELS"] = value
    try:
        yield
    finally:
        os.environ.pop("GYM_MIN_LEVELS", None)
        if old is not None:
            os.environ["GYM_MIN_LEVELS"] = old


def _assert_gated(r, *also):
    print("   ", r.get("error"))
    assert r["success"] is False, r
    for part in ("MOSSDEEP CITY GYM", "Lv48", "TROPIUS Lv", 'train("LOWEST", until_level=48)', *also):
        assert part in r["error"], (part, r["error"])
    assert "AZUMARILL" in r["error"] and "BRELOOM Lv42" in r["error"], r["error"]


def test_the_gym_level_gate_keeps_the_party_out_of_the_gym():
    """go_to, go_to_exit and navigate_to onto the door: each refused before a step."""
    s = server()
    with _gym_gate(None):
        for endpoint, payload in (
            ("mcp_go_to", {"place": "Mossdeep Gym"}),
            ("mcp_go_to_exit", {"exit": "MOSSDEEP CITY GYM"}),
            ("mcp_navigate_to", {"x": GYM_DOOR[0], "y": GYM_DOOR[1], "variance": "none"}),
        ):
            s.load(WEST_SHORE, wild_encounters=False)
            _assert_gated(s.call(endpoint, {**payload, "reasoning": "test", "reason": "test"}))
            snap = s.snapshot()
            assert (snap.x, snap.y) == (16, 6) and s.srv._location_name() == "MOSSDEEP CITY", (endpoint, snap)


def test_the_gym_level_gate_keeps_the_party_away_from_the_leader_inside():
    """Inside already: talk_to the leader, or a walk up to them, is refused; the refusal
    says walking out is allowed."""
    s = server()
    with _gym_gate(None):
        _load(s)
        _assert_gated(s.call("mcp_talk_to", {"x": LEADER[0], "y": LEADER[1], "reasoning": "test"}),
                      "TATE AND LIZA", "walk out of the gym")
        _assert_gated(s.call("mcp_navigate_to", {"x": BELOW_TATE[0], "y": BELOW_TATE[1], "variance": "none",
                                                 "reason": "test"}), "TATE AND LIZA")
        snap = s.snapshot()
        assert (snap.x, snap.y) == (3, 20), snap


def test_the_gym_level_gate_opens_when_lowered_or_off():
    s = server()
    with _gym_gate("off"):
        assert s.srv._gym_min_levels() == {}
    with _gym_gate("MOSSDEEP CITY GYM=5"):
        s.load(WEST_SHORE, wild_encounters=False)
        r = s.call("mcp_go_to", {"place": "Mossdeep Gym", "reasoning": "test"})
        print("   ", r.get("summary") or r.get("error"))
        assert r["success"] is True and s.srv._location_name() == "MOSSDEEP CITY GYM", (r, s.srv._location_name())


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
