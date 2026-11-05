"""Fly: fly_to, and go_to flying when that is shorter (server mcp_fly_to, _fly_to,
_go_to_fly_plan; game_tools fly_destinations / fly_plan; menu_state "Fly").

David (2026-11-04): "use TROPIUS + Fly instead of long walks".

Fixture test_fly_ready.state: LAVARIDGE TOWN (9,2), outdoors, walkable, just out of the
Pokémon Center; party NUMEL 20, AZUMARILL 22, SABLEYE 27, COMBUSKEN 30, BRELOOM 38,
TROPIUS 26 (FLY, RAZOR LEAF, STOMP, SWEET SCENT); badges Stone, Knuckle, Dynamo and
Feather. Visited (FLAG_VISITED_*, all set by the live game): every town but Pacifidlog,
Fortree, Lilycove, Mossdeep, Sootopolis and Ever Grande. Made from
test_lavaridge_pokecenter.state by the macros themselves: deposit_pokemon ELECTRIKE,
withdraw_pokemon TROPIUS (BOX2), teach_move HM02 to TROPIUS (it forgot GROWTH), go_to
"Lavaridge Town", env.save_state. Two RAM writes, both because the live game hadn't got
there yet: HM02 (item 340) put in the TMs & HMs pocket's first free slot (quantity 1,
XORed with the low half of the save's encryption key, as the bag stores it), and
FLAG_BADGE06_GET (0x86C) set. Seen while making it: gSaveBlock1Ptr moved during
teach_move (the bag closing reloads the field), so the flag write reads the pointer again.

RAM writes in the tests, documented where used: FLAG_BADGE06_GET and FLAG_VISITED_*
cleared.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_fly.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server
from test_party_rule import _write

HERE = Path(__file__).resolve().parent
FLY_READY = HERE / "test_fly_ready.state"
CENTER = HERE / "test_lavaridge_pokecenter.state"
FLAG_BADGE06_GET = 0x86C
FLAG_VISITED_PETALBURG_CITY = 0x876
FLAG_VISITED_MAUVILLE_CITY = 0x878
# The cells peers read off the fly map in pret (gRegionMapEntries + (1, 2)); fly_destinations
# reads them from ROM, so this list checks that reading.
CELLS = {
    "LITTLEROOT TOWN": (5, 13), "OLDALE TOWN": (5, 11), "DEWFORD TOWN": (3, 16), "LAVARIDGE TOWN": (6, 5),
    "FALLARBOR TOWN": (4, 2), "VERDANTURF TOWN": (5, 8), "PACIFIDLOG TOWN": (18, 12), "PETALBURG CITY": (2, 11),
    "SLATEPORT CITY": (9, 12), "MAUVILLE CITY": (9, 8), "RUSTBORO CITY": (1, 7), "FORTREE CITY": (13, 2),
    "LILYCOVE CITY": (19, 5), "MOSSDEEP CITY": (25, 7), "SOOTOPOLIS CITY": (22, 9), "EVER GRANDE CITY": (28, 11),
}
VISITED = {
    "LITTLEROOT TOWN", "OLDALE TOWN", "DEWFORD TOWN", "LAVARIDGE TOWN", "FALLARBOR TOWN", "VERDANTURF TOWN",
    "PETALBURG CITY", "SLATEPORT CITY", "MAUVILLE CITY", "RUSTBORO CITY",
}


def _load(s, path=FLY_READY):
    from server import game_tools

    s.load(path, wild_encounters=False)
    game_tools._field_moves_cache[0] = 0.0  # the planner's 1 s cache of usable field moves


def _set_flag(s, flag, on):
    """RAM: one event flag in SaveBlock1 (the pointer is read fresh: it moves)."""
    from server import game_tools

    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        address = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET + flag // 8
        byte = reader._read_u8(address)
    _write(s, 8, address, byte | (1 << flag % 8) if on else byte & ~(1 << flag % 8))
    game_tools._field_moves_cache[0] = 0.0
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader._is_event_flag_set(flag) == on


def _destinations(s):
    from server import game_tools

    with s.srv.memory_lock:
        return {d.town: d for d in game_tools.fly_destinations(s.srv.env.memory_reader)}


def _where(s):
    p = s.snapshot()
    return s.srv._location_name(), (p.x, p.y)


def test_fly_destinations_come_from_the_rom_and_the_flags():
    s = server()
    _load(s)
    d = _destinations(s)
    assert {town: dest.cell for town, dest in d.items()} == CELLS, d
    assert {town for town, dest in d.items() if dest.visited} == VISITED
    # Heal locations (pret src/data/heal_locations.json), each the town's own map.
    for town, tile in {
        "LILYCOVE CITY": (24, 15), "MOSSDEEP CITY": (28, 17), "SOOTOPOLIS CITY": (43, 32), "EVER GRANDE CITY": (27, 49),
        "MAUVILLE CITY": (22, 6), "PETALBURG CITY": (20, 17), "RUSTBORO CITY": (16, 39),
    }.items():
        assert (d[town].landing_map, d[town].landing) == (town, tile), d[town]
    # Littleroot lands outside the player's own house (CB_ExitFlyMap, by playerGender).
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        female = reader._read_u8(reader._read_u32(0x03005D90) + 8) == 1
    assert d["LITTLEROOT TOWN"].landing == ((14, 9) if female else (5, 9))


def test_fly_to_lands_at_the_heal_location_along_the_cursor_path():
    s = server()
    _load(s)
    started, frames = time.time(), s.srv._frames_run
    r = s.call("mcp_fly_to", {"town": "Mauville", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s, {s.srv._frames_run - frames} frames)")
    assert r["success"], r
    assert r["summary"].startswith("TROPIUS flew you to MAUVILLE CITY: now at (22,6) on MAUVILLE CITY"), r
    assert _where(s) == ("MAUVILLE CITY", (22, 6))
    assert s.srv._player_can_walk()
    # The fly map opened on Lavaridge (the player's mapsec), and every press moved the
    # cursor one cell: across, then down.
    assert s.srv.last_fly_cursor_path == [(6, 5), (7, 5), (8, 5), (9, 5), (9, 6), (9, 7), (9, 8)], s.srv.last_fly_cursor_path


def test_go_to_a_far_place_flies_then_walks():
    """On foot Petalburg City Gym is 14 maps from Lavaridge (Route 111's rocks smashed,
    Rusturf Tunnel, Rustboro, Petalburg Woods); Petalburg is visited, so go_to flies
    there and walks into the gym."""
    from server import game_tools

    s = server()
    _load(s)
    walk = game_tools.place_route("LAVARIDGE TOWN", (9, 2), "Petalburg City Gym")
    print("    on foot:", " → ".join(name for _, name in walk or []) or None)
    assert walk is None or len(walk) > 10, walk
    started = time.time()
    r = s.call("mcp_go_to", {"place": "Petalburg City Gym", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"], r
    # Not endswith: a POKéNAV call can ring on the way ("On the way: ..." follows; seen once).
    assert " Flew to PETALBURG CITY, then walked PETALBURG CITY GYM." in r["summary"], r
    assert r["flew_to"] == "PETALBURG CITY"
    assert s.srv._location_name() == "PETALBURG CITY GYM"


def test_go_to_flies_to_a_town_and_stops_there():
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Rustboro City", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    assert r["summary"].startswith("Arrived at Rustboro City: now at (16,39) on RUSTBORO CITY, facing DOWN. Flew to RUSTBORO CITY."), r


def test_go_to_walks_a_short_way_without_flying():
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Route 112", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and "Flew" not in r["summary"] and "flew_to" not in r, r
    assert s.srv._location_name() == "ROUTE 112"


def test_no_feather_badge_no_fly():
    s = server()
    _load(s)
    _set_flag(s, FLAG_BADGE06_GET, False)  # RAM: the Feather Badge taken away
    before = _where(s)
    assert s.srv._go_to_fly_plan("LAVARIDGE TOWN", (9, 2), "Petalburg City Gym") is None
    r = s.call("mcp_fly_to", {"town": "MAUVILLE CITY", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"] == "Can't fly: FLY outside battle needs the Feather Badge (Fortree City Gym).", r
    assert _where(s) == before and s.srv._player_can_walk()
    # With the badge this go_to flies (Mauville is visited, the walk is 5 maps); without, it walks.
    r = s.call("mcp_go_to", {"place": "Mauville City", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and "Flew" not in r["summary"] and "flew_to" not in r, r
    assert "Walked LAVARIDGE TOWN POKEMON CENTER 1F → LAVARIDGE TOWN → ROUTE 112 → ROUTE 111 → MAUVILLE CITY." in r["summary"], r


def test_go_to_flies_when_the_walk_is_longer():
    """The counterpart of test_no_feather_badge_no_fly's walk: the same go_to flies."""
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Mauville City", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and " Flew to MAUVILLE CITY." in r["summary"] and r["flew_to"] == "MAUVILLE CITY", r


def test_an_unvisited_town_is_never_chosen():
    s = server()
    _load(s)
    before = _where(s)
    r = s.call("mcp_fly_to", {"town": "Lilycove City", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"].startswith("You haven't been to LILYCOVE CITY yet"), r
    assert _where(s) == before and s.srv._player_can_walk()
    town, rest = s.srv._go_to_fly_plan("LAVARIDGE TOWN", (9, 2), "Petalburg City Gym")
    assert town.town == "PETALBURG CITY" and len(rest) == 1, (town, rest)
    # Without Petalburg the nearest visited town to its gym is Oldale (Route 102, 3 maps on).
    _set_flag(s, FLAG_VISITED_PETALBURG_CITY, False)  # RAM: Petalburg not visited
    town, rest = s.srv._go_to_fly_plan("LAVARIDGE TOWN", (9, 2), "Petalburg City Gym")
    assert town.town == "OLDALE TOWN" and [name for _, name in rest] == ["ROUTE 102", "PETALBURG CITY", "PETALBURG CITY GYM"], (town, rest)
    r = s.call("mcp_fly_to", {"town": "PETALBURG CITY", "reasoning": "test"})
    assert r["success"] is False and r["error"].startswith("You haven't been to PETALBURG CITY yet"), r
    # Past the refusals, _fly_to still checks the fly map's own mapSecType before A: told
    # (wrongly) that Mauville was visited, it stops on the cursor and backs out.
    _set_flag(s, FLAG_VISITED_MAUVILLE_CITY, False)  # RAM: Mauville not visited either
    from server import game_tools

    mauville = game_tools.FlyDestination(**{**_destinations(s)["MAUVILLE CITY"].__dict__, "visited": True})
    failed = asyncio_run(s.srv._fly_to(mauville, "test"))
    print("   ", failed)
    assert failed == "the fly map won't fly to MAUVILLE CITY (not visited)", failed
    assert _where(s) == before and s.srv._player_can_walk(), "the menus are still open"


def test_visit_triggers_read_from_the_map_scripts():
    """pret: Mossdeep sets FLAG_VISITED_MOSSDEEP_CITY only in
    MossdeepCity_EventScript_VisitedMossdeep, a coord event at four tiles; Dewford sets
    its flag in DewfordTown_OnTransition; Slateport through a call from its OnTransition
    (SlateportCity_EventScript_EnterSlateport). No emulator needed."""
    from server import game_tools

    assert game_tools.visit_triggers("MossdeepCity", "FLAG_VISITED_MOSSDEEP_CITY") == (
        False, ((25, 25), (26, 25), (32, 27), (33, 27))
    )
    assert game_tools.visit_triggers("DewfordTown", "FLAG_VISITED_DEWFORD_TOWN") == (True, ())
    assert game_tools.visit_triggers("SlateportCity", "FLAG_VISITED_SLATEPORT_CITY") == (True, ())
    # A flag the map never sets, and a map with no scripts.inc.
    assert game_tools.visit_triggers("MossdeepCity", "FLAG_VISITED_DEWFORD_TOWN") == (False, ())
    assert game_tools.visit_triggers("NoSuchMap", "FLAG_VISITED_MOSSDEEP_CITY") == (False, ())
    assert game_tools.visit_hint("MOSSDEEP CITY") == (
        "MOSSDEEP CITY counts as visited once you walk over (25,25), (26,25), (32,27) or (33,27) there."
    )
    assert game_tools.visit_hint("DEWFORD TOWN") == "DEWFORD TOWN counts as visited as soon as you arrive there."


def _trigger_tiles(map_dir, script):
    """The coord_events tiles in data/maps/<map_dir>/map.json that run `script`, row by row."""
    import json

    from pokemon_env.porymap_paths import get_porymap_root

    data = json.loads((Path(get_porymap_root()) / "data" / "maps" / map_dir / "map.json").read_text(encoding="utf-8"))
    return sorted(((e["x"], e["y"]) for e in data["coord_events"] if e["script"] == script), key=lambda t: (t[1], t[0]))


def test_an_unvisited_town_s_refusal_says_what_marks_it_visited():
    """Seen live: fly_to("MOSSDEEP CITY") was refused after the player had come in from
    Route 124's west edge, past none of the tiles that set its flag. Mossdeep and Ever
    Grande (both unvisited here) name their trigger tiles, read from map.json here;
    Lilycove's OnTransition sets its flag, so arriving is enough."""
    s = server()
    _load(s)
    before = _where(s)
    for town, map_dir, script in (
        ("MOSSDEEP CITY", "MossdeepCity", "MossdeepCity_EventScript_VisitedMossdeep"),
        ("EVER GRANDE CITY", "EverGrandeCity", "EverGrandeCity_EventScript_SetVisitedEverGrande"),
    ):
        tiles = [f"({x},{y})" for x, y in _trigger_tiles(map_dir, script)]
        assert len(tiles) >= 4, tiles
        r = s.call("mcp_fly_to", {"town": town, "reasoning": "test"})
        print("   ", r.get("error") or r)
        assert r["success"] is False and r["error"].startswith(f"You haven't been to {town} yet"), r
        assert r["error"].endswith(
            f" {town} counts as visited once you walk over {', '.join(tiles[:-1])} or {tiles[-1]} there."
        ), r
    r = s.call("mcp_fly_to", {"town": "Lilycove", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"].endswith(" LILYCOVE CITY counts as visited as soon as you arrive there."), r
    assert _where(s) == before and s.srv._player_can_walk()


def test_indoors_no_fly():
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Lavaridge Town Pokemon Center", "reasoning": "test"})
    assert r["success"], r
    before = _where(s)
    assert s.srv._go_to_fly_plan(before[0], before[1], "Petalburg City Gym") is None
    r = s.call("mcp_fly_to", {"town": "MAUVILLE CITY", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"] == "Can't fly: FLY can't be used in a building: go outside first.", r
    assert _where(s) == before and s.srv._player_can_walk()


def test_a_held_button_overshoots_and_the_cursor_move_corrects_it():
    """Holding RIGHT moves the fly map's cursor a cell every ~5 frames (pokebot-gen3's
    method overshoots this way); _move_fly_cursor reads where it really is and comes back."""
    from server import menu_state as ms
    from server.app import ActionRequest

    s = server()
    _load(s)
    srv = s.srv
    assert asyncio_run(srv._open_fly_map("test")) is None
    with srv.memory_lock:
        start = ms.fly_cursor(srv.env.memory_reader)
    assert (start.x, start.y) == (6, 5) and start.mapsec == 3  # MAPSEC_LAVARIDGE_TOWN

    async def hold_right():
        queued = await srv.take_action(ActionRequest(buttons=["RIGHT"], hold_frames=30, source="test"))
        await srv._await_sequence(queued.get("sequence_id"), 600)
        await srv._await_game(lambda: srv._mem(ms.fly_map_ready), 240)

    asyncio_run(hold_right())
    with srv.memory_lock:
        held = ms.fly_cursor(srv.env.memory_reader)
    print("    held RIGHT 30 frames: cursor", (held.x, held.y))
    assert held.x >= 9 and held.y == 5, held  # past Mauville's column (9) in one press
    assert asyncio_run(srv._move_fly_cursor((9, 8), "test")) is None
    path = srv.last_fly_cursor_path
    print("    corrected:", path)
    assert path[0] == (held.x, held.y) and path[-1] == (9, 8)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:], strict=False)), path
    with srv.memory_lock:
        cursor = ms.fly_cursor(srv.env.memory_reader)
    assert cursor.mapsec == _destinations(s)["MAUVILLE CITY"].mapsec and cursor.mapsec_type == ms.MAPSECTYPE_CITY_CANFLY
    assert asyncio_run(srv._leave_fly_map("test")) and _where(s) == ("LAVARIDGE TOWN", (9, 2))


def asyncio_run(coroutine):
    import asyncio

    return asyncio.run(coroutine)


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
