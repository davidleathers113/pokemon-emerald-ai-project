"""train(method="fish"): training on what bites, standing facing water.

Asked for (2026-11-05): the team has the SUPER ROD. pret src/field_player_avatar.c
Task_Fishing: after USE on a rod, dots print (an A then loses the cast), "Oh! A bite!"
gives reelTimeouts[rod] frames (30 for the SUPER ROD) to press A, a SUPER ROD may want up
to six bites in a row, and "A POKéMON's on the hook!" waits for A before the battle. The
game loop presses those As itself (server/app.py _fishing_reel). The tests read outcomes
from game memory (the player's tile, battle species, party EXP).

Same in-process server harness as test_macros.py (real game loop thread, real savestates).

Fixtures:
  test_fish_route119.state     the live checkpoint of 2026-11-05 15:30 (test_rematch's
                               ROUTE 120 one) after go_to("ROUTE 119") and
                               navigate_to(25, 42), then DOWN: on the river's bank, facing
                               the water at (25,43); SUPER ROD in the KEY ITEMS
  test_rematch_route120.state  ROUTE 120 (20,8), facing a path, no water ahead

RAM writes (documented where used): the party's HP and PP to full (test_rematch._full_hp:
the live team came in weak, and the heal trip to FORTREE CITY is train's, not what is
tested here); the SUPER ROD's KEY ITEMS slot emptied for the no-rod refusal.

  python3 local_tools/run_suites.py fishing
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_macros import _party, _read, _write
from test_movement_feedback import server
from test_rematch import ROUTE120, _full_hp

HERE = Path(__file__).resolve().parent
RIVER = HERE / "test_fish_route119.state"
SUPER_ROD = 264  # pret include/constants/items.h ITEM_SUPER_ROD


def test_fishing_battles_what_bites_without_moving():
    """train(LOWEST, method="fish") casts the SUPER ROD over and over: every battle is
    with something the ROM's fishing table has for it here, the player never leaves the
    bank, and the lowest-level Pokémon (TROPIUS) gains EXP. Before, train ignored method
    and walked off into the tall grass."""
    from server import game_tools

    s = server()
    s.load(RIVER)
    _full_hp(s)
    start = s.snapshot()
    assert (start.x, start.y, start.facing) == (25, 42, "DOWN"), start
    exp_before = {m.species_name.upper(): m.experience for m in _party(s)}
    began = time.time()
    r = s.call("mcp_train", {"trainee": "LOWEST", "method": "fish", "max_minutes": 1, "reasoning": "test"})
    print(f"    ({time.time() - began:.0f} s) {r.get('summary') or r}")
    # A heal trip may still happen (seen under load: a hooked CARVANHA Lv40 hits hard), and
    # its walks can meet wild Pokémon in the grass: only hooked battles are checked
    # against the fishing table, and the spot only when no trip was made.
    hooked = [b for b in r["battles"] if b.get("hooked")]
    assert r["success"] is True and hooked, r
    end = s.snapshot()
    if "healed at" not in r["summary"]:
        assert (end.x, end.y, end.map_id) == (start.x, start.y, start.map_id), (start, end)
    area = _read(s, lambda rd: (game_tools.wild_table(rd), game_tools.fishing_wild("Route119", 2))[1])
    names = {_read(s, lambda rd, sp=sp: rd._battle_species_name(sp)).upper() for sp in area.species}
    for battle in hooked:
        species = battle["opponents"][0].rsplit(" Lv", 1)[0]
        assert species in names, (battle["opponents"], names)
    exp_after = {m.species_name.upper(): m.experience for m in _party(s)}
    assert exp_after["TROPIUS"] > exp_before["TROPIUS"], (exp_before, exp_after)
    assert r["casts"] >= len(hooked) and "with the SUPER ROD" in r["summary"], r


def test_fishing_refuses_without_water_ahead():
    s = server()
    s.load(ROUTE120)
    before = s.snapshot()
    r = s.call("mcp_train", {"trainee": "LOWEST", "method": "fish", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["stopped_because"] == "no_fishing", r
    assert r["error"].startswith("Not facing water: the player at (20,8)"), r
    after = s.snapshot()
    assert (after.x, after.y) == (before.x, before.y), (before, after)


def test_fishing_refuses_without_a_rod():
    """RAM: the SUPER ROD's KEY ITEMS slot emptied (item id 0: the bag lists a pocket up
    to its first empty slot, and the rod is the last key item)."""
    from server import menu_state as ms

    s = server()
    s.load(RIVER)
    ids = _read(s, lambda r: ms.bag_pocket_item_ids(r, ms.KEY_ITEMS_POCKET))
    assert ids[-1] == SUPER_ROD, ids
    slots = _read(s, lambda r: r._read_u32(ms.G_BAG_POCKETS + ms.KEY_ITEMS_POCKET * ms.BAG_POCKET_SIZE))
    _write(s, 16, slots + 4 * (len(ids) - 1), 0)
    r = s.call("mcp_train", {"trainee": "LOWEST", "method": "fish", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["stopped_because"] == "no_fishing", r
    assert "needs a fishing rod" in r["error"], r
    r = s.call("mcp_train", {"trainee": "LOWEST", "method": "net", "reasoning": "test"})
    assert r["success"] is False and "method must be one of grass, fish" in r["error"], r


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
