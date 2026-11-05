"""Drops through the floor: go_to, go_to_exit and navigate_to step onto a cracked floor to
fall to the floor below when that is the only way, and never otherwise.

Seen live (2026-11-05, build ef1a194): on SKY PILLAR 4F (11,2), come up from 3F (11,1),
go_to("SKY PILLAR 5F") said "SKY PILLAR 5F can't be reached on foot or by Surf from SKY
PILLAR 4F". Before Rayquaza wakes (VAR_SKY_PILLAR_STATE 0) the floors are drawn from
their CLEAN layouts (pret data/maps/SkyPillar_4F/scripts.inc OnTransition). The 5F
stairs (3,1) stand in a part of 4F reached only from 3F's stairs at (7,1), and those in a
pocket of 3F nothing walks into: the way in is a drop through 4F's cracked floor at (6,4)
or (7,4), which lands on 3F at the same tile (pret ScrCmd_warphole). By hand: RIGHT from
(5,4) landed on SKY PILLAR 3F (6,4); go_to_exit("SKY PILLAR 4F") came up at (7,2);
navigate_to(3,1) went up to 5F.

Fixture test_sky_pillar_4f_clean.state: the live checkpoint of that moment (run
20261105_205724, 18:15): SKY PILLAR 4F (11,2), VAR_SKY_PILLAR_STATE 0, Mach Bike in the
bag, on foot.

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_fall_through.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server


SKY_PILLAR_4F = Path(__file__).resolve().parent / "test_sky_pillar_4f_clean.state"
FLAG_HAS_MATCH_CALL = 0x12F  # pret include/constants/flags.h


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def _load(s):
    """The fixture, with POKéNAV calls off (FLAG_HAS_MATCH_CALL cleared, as test_surf does):
    a call rings by game timing and would stop a walk."""
    s.load(SKY_PILLAR_4F, wild_encounters=False)
    reader = s.srv.env.memory_reader
    with s.srv.memory_lock:
        sb1 = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
        addr = sb1 + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET + FLAG_HAS_MATCH_CALL // 8
        s.srv.env.core.memory.u8[addr] = reader._read_u8(addr) & ~(1 << (FLAG_HAS_MATCH_CALL % 8))
        reader._invalidate_mem_cache()
        assert reader.read_game_var(0x40CA) == 0  # VAR_SKY_PILLAR_STATE: the clean floors
    assert _location(s) == "SKY PILLAR 4F" and (s.snapshot().x, s.snapshot().y) == (11, 2)


def test_hole_warps_are_read_from_the_map_scripts():
    """Which floors drop the walker, and where to (pret setholewarp, setstepcallback
    STEP_CB_CRACKED_FLOOR); a floor without them drops nobody."""
    from utils.mapping.porymap_state import hole_warp

    assert hole_warp("SkyPillar_4F") == ("SkyPillar_3F", True)
    assert hole_warp("SkyPillar_2F") == ("SkyPillar_1F", True)
    assert hole_warp("GraniteCave_B1F") == ("GraniteCave_B2F", True)
    assert hole_warp("MirageTower_3F") == ("MirageTower_2F", True)
    assert hole_warp("SkyPillar_3F") == ("", False)
    assert hole_warp("Route101") == ("", False)


def test_routes_drop_only_when_no_walk_exists():
    """The route to 5F drops through (6,4) or (7,4); the route back to 3F takes the stairs
    the player came up, not a drop, though a drop also lands on 3F."""
    from server import game_tools

    s = server()
    _load(s)
    with s.srv.memory_lock:
        game_tools.note_live_layout(s.srv.env.memory_reader)
    route = game_tools.place_route("SKY PILLAR 4F", (11, 2), "SKY PILLAR 5F")
    assert route and route[0][0] in ("FALL THROUGH (6, 4)", "FALL THROUGH (7, 4)"), route
    assert [name for _, name in route] == ["SKY PILLAR 3F", "SKY PILLAR 4F", "SKY PILLAR 5F"], route
    back = game_tools.place_route("SKY PILLAR 4F", (11, 2), "SKY PILLAR 3F")
    assert back and not any(step.startswith(game_tools.FALL_STEP) for step, _ in back), back


def test_go_to_drops_through_the_cracked_floor_to_the_top():
    """The live stall: go_to from 4F (11,2) arrives at SKY PILLAR TOP, by the drop."""
    s = server()
    _load(s)
    r = s.call("mcp_go_to", {"place": "SKY PILLAR TOP", "reasoning": "up the pillar"})
    print("   ", r.get("summary"))
    assert r.get("success") is True and _location(s) == "SKY PILLAR TOP", r
    assert "dropped through the floor at (6,4)" in r["summary"] or "dropped through the floor at (7,4)" in r["summary"], r


def test_go_to_exit_reaches_stairs_beyond_a_drop():
    """go_to_exit("SKY PILLAR 5F") from 4F (11,2): the stairs are an exit of this map, out
    of reach on foot; the walk drops to 3F, comes back up and takes them."""
    s = server()
    _load(s)
    r = s.call("mcp_go_to_exit", {"exit": "SKY PILLAR 5F", "reasoning": "the stairs up"})
    print("   ", r.get("summary"))
    assert r.get("success") is True and _location(s) == "SKY PILLAR 5F", r


def test_walks_never_cross_cracked_floor():
    """(8,4) lies between the cracked (6,4)-(7,4) and (9,4): walking there from (5,4)
    would drop the player to 3F at (6,4). navigate_to(8, 4) stays on 4F and says the tile
    can't be reached."""
    s = server()
    _load(s)
    r = s.call("mcp_navigate_to", {"x": 5, "y": 4, "variance": "none", "reason": "beside the cracked floor"})
    assert (s.snapshot().x, s.snapshot().y) == (5, 4), r
    r = s.call("mcp_navigate_to", {"x": 8, "y": 4, "variance": "none", "reason": "past the cracked floor"})
    print("   ", r.get("summary") or r.get("error"))
    assert _location(s) == "SKY PILLAR 4F", r
    assert (s.snapshot().x, s.snapshot().y) != (8, 4), r


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
