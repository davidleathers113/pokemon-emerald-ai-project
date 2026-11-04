"""Surf: walks and routes cross water when a party Pokémon knows SURF and the Balance
Badge is set, and stop at the shore otherwise.

Seen live (twice): with SURF on AZUMARILL and the Balance Badge, go_to("Route 119") from
ROUTE 118 said "Route 119 can't be reached on foot from ROUTE 118" and the AI stalled at
the river at (17,10). By hand: face the water, A, "Would you like to SURF?" YES, close
"AZUMARILL used SURF!", walk across.

Fixture test_surf_shore.state: ROUTE 118 (31,10), the river's east bank, facing the
water (30,10); party SABLEYE, AZUMARILL (knows SURF), CAMERUPT, MANECTRIC, COMBUSKEN,
BRELOOM; Balance Badge set. Made from a live save on ROUTE 120 (16,9) (in a trainer
battle with Parasol Lady Clarissa): the battle won with use_move, go_to("Route 118")
(Fortree City, Route 119, its trainers fought with use_move), navigate_to(31, 10), then
env.save_state. On foot the east bank has no way to Mauville City; the river is
elevation-1 POND_WATER between elevation-3 banks, (17,10) on the west.

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_surf.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server


SHORE = Path(__file__).resolve().parent / "test_surf_shore.state"
FLAG_BADGE05_GET = 0x86B  # pret include/constants/flags.h (SYSTEM_FLAGS + 0xB), the Balance Badge


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _surfing(s):
    from server.field_control import player_surfing

    with s.srv.memory_lock:
        return player_surfing(s.srv.env.memory_reader)


def _forget_field_moves():
    from server import game_tools

    game_tools._field_moves_cache[0] = 0.0  # the reader's answer is kept for a second


FLAG_HAS_MATCH_CALL = 0x12F  # pret include/constants/flags.h


def _clear_flag(s, flag):
    reader = s.srv.env.memory_reader
    addr = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET + flag // 8
    s.srv.env.core.memory.u8[addr] = reader._read_u8(addr) & ~(1 << (flag % 8))


def _load(s, balance_badge=True):
    """The fixture, with the Balance Badge cleared in RAM when balance_badge is False (as
    test_movement_exits.test_walks_smash_a_rock_the_party_can_clear does the Dynamo's).

    Kept from interrupting a walk, as wild_encounters=False does on land: the Repel lets
    through wild Pokémon of the lead's level or above (pret wild_encounter.c
    IsWildLevelAllowedByRepel) and ROUTE 118's water has Tentacool up to Lv35 over a Lv30
    SABLEYE, so BRELOOM (Lv42, slot 6) is swapped to the lead in RAM; and POKéNAV calls
    (seen in a run: a call rang mid-river and stopped the walk) are off, FLAG_HAS_MATCH_CALL
    cleared (pret match_call.c TryStartMatchCall)."""
    s.load(SHORE, wild_encounters=False)
    reader = s.srv.env.memory_reader
    memory = s.srv.env.core.memory
    with s.srv.memory_lock:
        base, size = reader.addresses.PARTY_BASE, reader.addresses.PARTY_POKEMON_SIZE
        lead = [memory.u8[base + i] for i in range(size)]
        last = [memory.u8[base + 5 * size + i] for i in range(size)]
        for i in range(size):
            memory.u8[base + i] = last[i]
            memory.u8[base + 5 * size + i] = lead[i]
        _clear_flag(s, FLAG_HAS_MATCH_CALL)
        if not balance_badge:
            _clear_flag(s, FLAG_BADGE05_GET)
        reader._invalidate_mem_cache()
        assert reader.read_party_pokemon()[0].species_name.upper() == "BRELOOM"
    _forget_field_moves()


def test_routes_join_the_shores_across_water_only_when_the_party_can_surf():
    """The area graph go_to, place_route and tile_route walk: the river's banks are one
    area with Surf, two without; the live failure's route (west bank to Route 119) exists
    only with Surf. Water under a bridge is passed under: ROUTE 120's pond south of its
    bridge (12,20) has no shore of its own and joins the pond north of it (16,14)."""
    from server import game_tools

    s = server()
    _load(s)
    with s.srv.memory_lock:
        assert game_tools.can_surf()
        areas = game_tools._route_areas("Route118")
        assert areas[(31, 10)] == areas[(17, 10)] == areas[(24, 10)], "Surf should join the banks"
        route = game_tools.place_route("ROUTE 118", (17, 10), "Route 119")
        assert route and route[0][0] == "north" and route[-1][1] == "ROUTE 119", route
        assert game_tools.place_route("ROUTE 118", (31, 10), "Mauville City") == [("west", "MAUVILLE CITY")]
        pond = game_tools._walk_areas_without("Route120", frozenset(), True, True)
        assert pond[(12, 20)] == pond[(16, 14)], "the water under ROUTE 120's bridge should join"
        assert (12, 20) not in game_tools._walk_areas_without("Route120", frozenset(), True, False)

    _load(s, balance_badge=False)
    with s.srv.memory_lock:
        assert not game_tools.can_surf()
        areas = game_tools._route_areas("Route118")
        assert (24, 10) not in areas and areas[(31, 10)] != areas[(17, 10)], "without Surf the river is a wall"
        assert game_tools.place_route("ROUTE 118", (17, 10), "Route 119") is None
        assert game_tools.place_route("ROUTE 118", (31, 10), "Mauville City") is None


def test_map_text_mentions_surf_only_when_the_party_can_surf():
    """The prompt's map: water shows as 'W' and the legend says the walks cross it, only
    when the party can Surf."""
    from utils.mapping.porymap_state import _format_porymap_info

    s = server()
    _load(s)
    with s.srv.memory_lock:
        text = "\n".join(_format_porymap_info("ROUTE 118", (31, 10), 5, s.srv.env.memory_reader).context_parts)
    assert "'W' = water (your party can Surf" in text, text[-1500:]
    _load(s, balance_badge=False)
    with s.srv.memory_lock:
        text = "\n".join(_format_porymap_info("ROUTE 118", (31, 10), 4, s.srv.env.memory_reader).context_parts)
    assert "Surf" not in text.split("(Legend:")[1].split(")")[0], text[-1500:]


def test_navigate_to_surfs_across_the_river():
    """The live stall, east to west: navigate_to a tile beyond the river Surfs from the
    bank, crosses, lands on the far bank and walks on."""
    s = server()
    _load(s)
    r = s.call("mcp_navigate_to", {"x": 14, "y": 10, "variance": "none", "reason": "across the river"})
    print("   ", r.get("summary"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == (14, 10), (snap, r)
    assert "SURF" in r["summary"], r
    assert not _surfing(s), "the landing should have ended Surf"


def test_go_to_crosses_water_to_the_next_map():
    """go_to("Mauville City") from the east bank: the only way is across the river."""
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "Mauville City", "reasoning": "across the river"})
    print("   ", r.get("summary"))
    assert r["success"] is True, r
    assert _location(s) == "MAUVILLE CITY", _location(s)
    assert "SURF" in r["summary"], r


def test_without_the_balance_badge_the_water_stays_a_wall():
    """Without the Balance Badge (RAM: FLAG_BADGE05_GET cleared) the game offers no Surf
    (pret GetInteractedWaterScript), so neither the walk nor the route goes on the water."""
    s = server()
    _load(s, balance_badge=False)
    r = s.call("mcp_navigate_to", {"x": 14, "y": 10, "variance": "none", "reason": "across the river"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert snap.x > 17 and not _surfing(s), (snap, r)  # still east of the river (17 is the west bank)
    assert "SURF" not in str(r.get("summary") or r.get("error")), r
    r = s.call("mcp_go_to", {"place": "Mauville City", "reasoning": "across the river"})
    print("   ", r.get("summary"))
    assert r["success"] is False and "can't be reached on foot from ROUTE 118" in r["summary"], r
    assert _location(s) == "ROUTE 118" and not _surfing(s)


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
