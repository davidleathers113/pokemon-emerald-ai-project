"""Strength boulders: the map shows them where they stand, walks push them when the party
can use STRENGTH, and heal_at_pokecenter gets out of a dungeon (Escape Rope, or the walk
out through the boulders).

Seen live (2026-11-04, MAGMA HIDEOUT 1F): boulders at (5,22), (7,22) and (6,23) cut
(5,21) off from the exit to JAGGED PASS at (10,34), and no walk planned a way past them
(the ASCII map drew them as people, 'N'). By hand: face the boulder, A, "Would you like to
use STRENGTH?" YES, close the text, then each step into a boulder pushes it one tile (the
player stays put). heal_at_pokecenter there answered "no nurse is on this map": the routes
counted the boulders as walls at their map-file tiles, found no way out, and that empty
route was taken for "already in a Pokémon Center".

Fixture test_magma_hideout_1f.state: MAGMA HIDEOUT 1F (5,21) facing DOWN, boulders at
their spawn tiles (5,22), (7,22), (6,23), FLAG_SYS_USE_STRENGTH clear; party TROPIUS
(fainted), AZUMARILL (knows STRENGTH), SABLEYE, BLAZIKEN, BRELOOM, MANECTRIC (97/111);
Heat Badge set; one ESCAPE ROPE in the bag. Made from the live run's manual save in MAGMA
HIDEOUT 2F 2R (8,13): navigate_to(5, 21) walked 2F 2R → 2F 1R → 1F (the map load put the
boulders back), then env.save_state (the harness's RAM Repel undone first).

RAM writes, documented where used: the Heat Badge flag cleared, a boulder's coordinates,
the Escape Rope taken out of the bag.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_strength.py [name-substring]
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HIDEOUT = Path(__file__).resolve().parent / "test_magma_hideout_1f.state"
SPAWNS = {(5, 22), (7, 22), (6, 23)}
JAGGED_PASS_DOOR = (10, 34)
FLAG_BADGE04_GET = 0x86A  # pret include/constants/flags.h (SYSTEM_FLAGS + 0xA), the Heat Badge
FLAG_SYS_USE_STRENGTH = 0x889  # (SYSTEM_FLAGS + 0x29)
ESCAPE_ROPE = 85  # pret include/constants/items.h
OBJ_EVENT_GFX_PUSHABLE_BOULDER = 87  # pret include/constants/event_objects.h
G_OBJECT_EVENTS, OBJECT_EVENT_SIZE, MAP_OFFSET = 0x02037350, 0x24, 7  # pret struct ObjectEvent


def _boulders(s):
    from server.menu_state import find_objects_by_graphics

    with s.srv.memory_lock:
        return {(x, y) for x, y, _ in find_objects_by_graphics(s.srv.env.memory_reader, OBJ_EVENT_GFX_PUSHABLE_BOULDER)}


def _flag(s, flag):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader._is_event_flag_set(flag)


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _forget_field_moves():
    from server import game_tools

    game_tools._field_moves_cache[0] = 0.0  # the reader's answer is kept for a second
    game_tools._items_cache[0] = 0.0


def _load(s, heat_badge=True):
    """The fixture; without the Heat Badge (RAM: FLAG_BADGE04_GET cleared) the boulder
    script says only "It's a big boulder…" (pret EventScript_StrengthBoulder)."""
    s.load(HIDEOUT, wild_encounters=False)
    if not heat_badge:
        reader = s.srv.env.memory_reader
        with s.srv.memory_lock:
            addr = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET + FLAG_BADGE04_GET // 8
            s.srv.env.core.memory.u8[addr] = reader._read_u8(addr) & ~(1 << (FLAG_BADGE04_GET % 8))
            reader._invalidate_mem_cache()
    _forget_field_moves()
    assert _boulders(s) == SPAWNS and not _flag(s, FLAG_SYS_USE_STRENGTH)


def _move_boulder(s, at, to):
    """RAM: the boulder object on `at` moved to `to` (struct ObjectEvent currentCoords +0x10
    and previousCoords +0x14, both read by pret DoesObjectCollideWithObjectAt)."""
    with s.srv.memory_lock:
        reader, memory = s.srv.env.memory_reader, s.srv.env.core.memory
        for i in range(16):
            base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
            here = (reader._read_s16(base + 0x10) - MAP_OFFSET, reader._read_s16(base + 0x12) - MAP_OFFSET)
            if reader._read_u8(base + 5) == OBJ_EVENT_GFX_PUSHABLE_BOULDER and here == at:
                for offset in (0x10, 0x14):
                    memory.u16[base + offset] = to[0] + MAP_OFFSET
                    memory.u16[base + offset + 2] = to[1] + MAP_OFFSET
                reader._invalidate_mem_cache()
                return
    raise AssertionError(f"no boulder at {at}")


def _take_escape_ropes(s):
    """RAM: every ESCAPE ROPE slot of the ITEMS pocket emptied."""
    from server import menu_state as ms

    with s.srv.memory_lock:
        reader, memory = s.srv.env.memory_reader, s.srv.env.core.memory
        slots = reader._read_u32(ms.G_BAG_POCKETS)
        key = reader._get_security_key() & 0xFFFF
        for i in range(reader._read_u8(ms.G_BAG_POCKETS + 4)):
            if reader._read_u16(slots + 4 * i) == ESCAPE_ROPE:
                memory.u16[slots + 4 * i] = 0
                memory.u16[slots + 4 * i + 2] = key  # quantity 0, stored XOR the key
        reader._invalidate_mem_cache()
    _forget_field_moves()


# --- the map and the plan ------------------------------------------------------------


def test_the_map_shows_boulders_where_they_stand():
    """'R' at each boulder's live tile (never 'N' for a person), and the legend says so."""
    from utils.mapping.porymap_state import _format_porymap_info

    s = server()
    _load(s)
    _move_boulder(s, (5, 22), (4, 22))  # where the live run had pushed it

    with s.srv.memory_lock:
        parts = _format_porymap_info("MAGMA HIDEOUT 1F", (5, 21), 4, s.srv.env.memory_reader).context_parts
    text = "\n".join(parts)
    rows = parts[parts.index("\nASCII Map:") + 1].split("\n")
    for x, y in ((4, 22), (7, 22), (6, 23)):
        assert rows[y][x] == "R", (x, y, rows[y])
    assert rows[22][5] != "R" and rows[22][5] != "N", rows[22]
    assert "'R' = boulder" in text, text[-800:]


def test_the_plan_pushes_boulders_only_when_strength_can_be_used():
    """With STRENGTH and the Heat Badge the plan from (5,21) to the exit pushes boulders
    (each push a step that leaves the player where they are); without the badge there is
    no way (the boulders are walls)."""
    from server import game_tools

    s = server()
    _load(s)
    with s.srv.memory_lock:
        plan = game_tools.navigate_to_direct(s.srv.env, *JAGGED_PASS_DOOR, reason="test")
    print("   ", {k: plan.get(k) for k in ("success", "buttons", "push", "pushes", "note")})
    assert plan["success"] and plan.get("push"), plan
    assert plan["push"]["boulder"] == (5, 22) and plan["push"]["button"] == "DOWN", plan["push"]
    assert plan["buttons"] == [], plan  # the first push is the very next press
    assert not plan.get("note"), plan  # the whole way is planned

    _load(s, heat_badge=False)
    with s.srv.memory_lock:
        plan = game_tools.navigate_to_direct(s.srv.env, *JAGGED_PASS_DOOR, reason="test")
    print("   ", plan.get("error") or plan.get("note"))
    assert not plan.get("push") and (not plan["success"] or plan.get("note")), plan


def test_a_push_needs_a_free_floor_tile_behind_the_boulder():
    """pret TryPushBoulder: the tile behind must pass GetCollisionAtCoords (no wall, no
    elevation mismatch, no other object) and not be a door. The planner's rule, tile by tile."""
    from server import game_tools
    from utils.mapping.pathfinding import Pathfinder

    s = server()
    _load(s)
    with s.srv.memory_lock:
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    finder = Pathfinder()
    map_data = finder._extract_map_data(state)
    boulders = frozenset(SPAWNS)
    blocked = finder._get_blocked_positions(state, map_data, start_pos=(5, 21), consider_npcs=True)
    assert finder._push_target((5, 22), (0, 1), boulders, blocked, map_data) == (5, 23)
    assert finder._push_target((6, 23), (0, 1), boulders, blocked, map_data) is None  # (6,24) is rock
    assert finder._push_target((7, 22), (-1, 0), boulders, blocked, map_data) == (6, 22)
    assert finder._push_target((7, 22), (0, 1), frozenset({(7, 22), (7, 23)}), blocked, map_data) is None  # a boulder
    assert finder._push_target((5, 22), (0, -1), boulders, blocked, map_data) == (5, 21)


def test_a_boulder_that_does_not_move_is_reported():
    """A push into another boulder (RAM: (6,23) moved to (5,23), below (5,22)) does
    nothing; the push says so instead of claiming it."""
    s = server()
    _load(s)
    _move_boulder(s, (6, 23), (5, 23))
    moved, said = asyncio.run(s.srv._push_boulder((5, 22), "DOWN", "test"))
    print("   ", moved, said)
    assert moved is False and "did not move" in said, said
    assert _boulders(s) == {(5, 22), (7, 22), (5, 23)}
    assert _flag(s, FLAG_SYS_USE_STRENGTH), "STRENGTH should have been used on the boulder"
    assert s.srv._player_can_walk()


# --- the walks -----------------------------------------------------------------------


def test_navigate_to_pushes_the_boulders_out_of_the_way():
    """The live dead end: navigate_to the exit's corridor from (5,21). The walk uses
    STRENGTH on the first boulder, pushes as planned, checks each push in RAM."""
    s = server()
    _load(s)
    started = time.time()
    r = s.call("mcp_navigate_to", {"x": 10, "y": 30, "variance": "none", "reason": "to the exit"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    snap = s.snapshot()
    assert (snap.x, snap.y) == (10, 30), (snap, r)
    assert "STRENGTH" in r["summary"] and "pushed the boulder at (5, 22) DOWN to (5, 23)" in r["summary"], r
    # Boulders out of view are unloaded (field_control.loaded_boulders); the pushed ones
    # still loaded stand where the pushes left them, not on (5,22) or (7,22).
    assert not _boulders(s) & {(5, 22), (7, 22)}, _boulders(s)


def test_go_to_exit_pushes_through_to_jagged_pass():
    s = server()
    _load(s)
    r = s.call("mcp_go_to_exit", {"exit": "JAGGED PASS", "reasoning": "out"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and _location(s) == "JAGGED PASS", (r, _location(s))
    assert "STRENGTH" in r["summary"], r


def test_go_to_takes_the_boulder_route_too():
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Jagged Pass", "reasoning": "out"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and _location(s) == "JAGGED PASS", (r, _location(s))


def test_without_the_heat_badge_the_boulders_stay_walls():
    s = server()
    _load(s, heat_badge=False)
    r = s.call("mcp_navigate_to", {"x": 10, "y": 30, "variance": "none", "reason": "to the exit"})
    print("   ", r.get("summary") or r.get("error"))
    # Only boulders near the camera are loaded (field_control.loaded_boulders): none of
    # those may have left its map-file tile.
    assert _boulders(s) <= SPAWNS and not _flag(s, FLAG_SYS_USE_STRENGTH)
    snap = s.snapshot()
    assert (snap.x, snap.y) != (10, 30) and snap.y <= 21, snap


# --- heal_at_pokecenter from inside the dungeon --------------------------------------


def test_the_way_to_a_pokemon_center_goes_through_the_boulders():
    """The routes: from (5,21) the nearest Pokémon Center is through the boulders, JAGGED
    PASS and LAVARIDGE TOWN; with the boulders as walls at their map-file tiles there was
    none. Without the Heat Badge they are walls again."""
    from server import game_tools

    s = server()
    _load(s)
    with s.srv.memory_lock:
        route = game_tools.pokemon_center_route("MAGMA HIDEOUT 1F", (5, 21))
    print("   ", route)
    assert route and route[0][1] == "JAGGED PASS" and route[-1][1].endswith("POKEMON CENTER 1F"), route

    _load(s, heat_badge=False)
    with s.srv.memory_lock:
        route = game_tools.pokemon_center_route("MAGMA HIDEOUT 1F", (5, 21))
    print("   ", route)
    assert not route or route[0][1] != "JAGGED PASS", route


def test_pushed_boulders_are_read_where_they_stand():
    """Without Strength usable, boulders already pushed aside (RAM: as the live run left
    them, (4,22) and (5,23)) leave the way open: the route reads their live tiles."""
    from server import game_tools

    s = server()
    _load(s, heat_badge=False)
    _move_boulder(s, (5, 22), (4, 22))
    _move_boulder(s, (6, 23), (5, 23))
    with s.srv.memory_lock:
        route = game_tools.pokemon_center_route("MAGMA HIDEOUT 1F", (5, 21))
    print("   ", route)
    assert route and route[0][1] == "JAGGED PASS", route


def test_heal_in_a_dungeon_uses_the_escape_rope_first():
    s = server()
    _load(s)
    started = time.time()
    r = s.call("mcp_heal_at_pokecenter", {"reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"] is True, r
    assert "ESCAPE ROPE" in r["summary"] and _location(s).endswith("POKEMON CENTER 1F"), (r, _location(s))


def test_heal_in_a_dungeon_without_a_rope_walks_out_through_the_boulders():
    s = server()
    _load(s)
    _take_escape_ropes(s)
    started = time.time()
    r = s.call("mcp_heal_at_pokecenter", {"reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"] is True, r
    assert "JAGGED PASS" in r["summary"] and _location(s).endswith("POKEMON CENTER 1F"), (r, _location(s))


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
