"""deposit_pokemon / withdraw_pokemon: the Pokémon Center PC in one call each (server
mcp_deposit_pokemon, mcp_withdraw_pokemon; menu_state "Pokémon Center PC").

David (2026-11-04): swap party members at a PC. Live plan: deposit CAMERUPT, withdraw
the TROPIUS a full party sent to the PC, then teach it HM02 Fly.

Fixture test_lavaridge_pokecenter.state: LAVARIDGE TOWN POKEMON CENTER 1F (7,4), facing
the nurse, walkable; party NUMEL 20, AZUMARILL 22, ELECTRIKE 20, SABLEYE 27, COMBUSKEN
30, BRELOOM 38 (full); current box BOX1 (empty); BOX2 holds CASTFORM in slot 1 and
TROPIUS in slot 8. Made from test_lavaridge_after_chimney.state (a live save) with
heal_at_pokecenter (which walked in and healed), then the two boxed Pokémon of
test_fortree_gym.state (a later live save: the AI's own CASTFORM and TROPIUS) copied byte
for byte, 80-byte BoxPokemon each, into BOX2 slots 1 and 8, and env.save_state. Slot 8
(second row) is not where the game would have put a second Pokémon; it is there so the
hand has to move down and across.

RAM writes in the tests, documented where used: party HP (fainted).

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_pc_storage.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server
from test_party_rule import CHIMNEY, PARTY, _names, _write

HERE = Path(__file__).resolve().parent
CENTER = HERE / "test_lavaridge_pokecenter.state"
WILD = HERE / "test_wild_shroomish.state"  # a wild battle, action menu open
TROPIUS_PID, CASTFORM_PID = 0xD700B676, 0xD09E9BE0


def _pids(s):
    from server import menu_state as ms

    with s.srv.memory_lock:
        return ms.party_personalities(s.srv.env.memory_reader)


def _boxed(s):
    from server import menu_state as ms

    with s.srv.memory_lock:
        return {m.personality: (m.box, m.slot) for m in ms.boxed_mons(s.srv.env.memory_reader)}


def _current_box(s):
    from server import menu_state as ms

    with s.srv.memory_lock:
        return ms.storage_current_box(s.srv.env.memory_reader)


def test_the_fixture_has_the_boxed_pokemon():
    s = server()
    s.load(CENTER, wild_encounters=False)
    assert _boxed(s) == {CASTFORM_PID: (1, 0), TROPIUS_PID: (1, 7)}, _boxed(s)
    assert _current_box(s) == 0
    assert len(_pids(s)) == 6 and s.srv._player_can_walk()


def test_deposit_then_withdraw_round_trip():
    s = server()
    s.load(CENTER, wild_encounters=False)
    numel = _pids(s)[0]
    started = time.time()
    r = s.call("mcp_deposit_pokemon", {"pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"], r
    # The current box, BOX1, is empty: its first slot.
    assert r["summary"].startswith("NUMEL deposited in BOX1 (slot 1). Party: 1. AZUMARILL,"), r
    assert numel not in _pids(s) and len(_pids(s)) == 5
    assert _boxed(s)[numel] == (0, 0)
    assert s.srv._player_can_walk(), "the PC is still open"
    started = time.time()
    r = s.call("mcp_withdraw_pokemon", {"pokemon": "numel", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"], r
    assert r["summary"].startswith("NUMEL withdrawn from BOX1, slot 1."), r
    assert _pids(s)[-1] == numel and len(_pids(s)) == 6
    assert numel not in _boxed(s)
    assert s.srv._player_can_walk(), "the PC is still open"


def test_withdraw_scrolls_to_another_box_and_moves_the_hand():
    """The live swap: deposit one, withdraw TROPIUS from BOX2 slot 8 while BOX1 is current."""
    s = server()
    s.load(CENTER, wild_encounters=False)
    r = s.call("mcp_deposit_pokemon", {"pokemon": "ELECTRIKE", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    assert _current_box(s) == 0
    started = time.time()
    r = s.call("mcp_withdraw_pokemon", {"pokemon": "TROPIUS", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"], r
    assert r["summary"].startswith("TROPIUS withdrawn from BOX2, slot 8."), r
    assert TROPIUS_PID in _pids(s) and TROPIUS_PID not in _boxed(s)
    assert _boxed(s)[CASTFORM_PID] == (1, 0)  # the other one stays
    assert "TROPIUS" in _names(s)
    assert s.srv._player_can_walk(), "the PC is still open"


def test_deposit_skips_a_full_box():
    """"Deposit in which BOX?" offers the last box deposited in (sDepositBoxId, BOX1 here);
    when it is full the game says so and asks again, so the macro moves the popup to the
    next box with room before pressing A."""
    from server import menu_state as ms

    s = server()
    s.load(CENTER, wild_encounters=False)
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        boxes = ms.storage_base(reader) + ms.STORAGE_BOXES
        castform = bytes(reader._read_bytes(boxes + ms.IN_BOX_COUNT * ms.BOX_MON_SIZE, ms.BOX_MON_SIZE))
    assert int.from_bytes(castform[:4], "little") == CASTFORM_PID
    for slot in range(ms.IN_BOX_COUNT):  # RAM: BOX1 filled with copies of CASTFORM
        for i, byte in enumerate(castform):
            _write(s, 8, boxes + slot * ms.BOX_MON_SIZE + i, byte)
    numel = _pids(s)[0]
    r = s.call("mcp_deposit_pokemon", {"pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    # BOX2 has CASTFORM in slot 1, so the first free slot is 2.
    assert r["summary"].startswith("NUMEL deposited in BOX2 (slot 2)."), r
    assert _boxed(s)[numel] == (1, 1)
    assert s.srv._player_can_walk(), "the PC is still open"


def test_withdraw_refuses_a_full_party_and_an_unknown_name():
    s = server()
    s.load(CENTER, wild_encounters=False)
    before = (_pids(s), _boxed(s), s.snapshot().__dict__)
    r = s.call("mcp_withdraw_pokemon", {"pokemon": "TROPIUS", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "TROPIUS is in BOX2 (slot 8), but your party is full (6)" in r["error"], r
    r = s.call("mcp_withdraw_pokemon", {"pokemon": "MEW", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False, r
    assert r["error"] == "There is no MEW in the PC boxes (in the PC: CASTFORM (BOX2), TROPIUS (BOX2)).", r
    r = s.call("mcp_deposit_pokemon", {"pokemon": "MEW", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"].startswith("MEW is not in your party (1. NUMEL,"), r
    # Nothing was pressed: same party, same boxes, same tile.
    assert (_pids(s), _boxed(s), s.snapshot().__dict__) == before
    assert s.srv._player_can_walk()


def test_deposit_refuses_the_last_pokemon_that_can_fight():
    s = server()
    s.load(CENTER, wild_encounters=False)
    for i in range(1, 6):
        _write(s, 16, PARTY + i * 100 + 0x56, 0)  # RAM: everyone but NUMEL fainted
    pids = _pids(s)
    r = s.call("mcp_deposit_pokemon", {"pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "NUMEL is the last Pokémon in your party that can fight" in r["error"], r
    assert _pids(s) == pids and s.srv._player_can_walk()
    # A fainted one can go: NUMEL still fights.
    r = s.call("mcp_deposit_pokemon", {"pokemon": "BRELOOM", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["summary"].startswith("BRELOOM deposited in BOX1 (slot 1)."), r
    assert s.srv._player_can_walk()


def test_both_refuse_outside_a_pokemon_center_and_in_battle():
    s = server()
    s.load(CHIMNEY, wild_encounters=False)  # MT CHIMNEY, outdoors
    before = s.snapshot().__dict__
    r = s.call("mcp_deposit_pokemon", {"pokemon": "BRELOOM", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "works at the PC in a Pokémon Center, and you are on MT CHIMNEY" in r["error"], r
    s.load(CHIMNEY, wild_encounters=False)
    r = s.call("mcp_withdraw_pokemon", {"pokemon": "MEW", "reasoning": "test"})
    assert r["success"] is False and "There is no MEW in the PC boxes" in r["error"], r
    assert s.snapshot().__dict__ == before and s.srv._player_can_walk()
    s.load(WILD)
    for endpoint in ("mcp_deposit_pokemon", "mcp_withdraw_pokemon"):
        r = s.call(endpoint, {"pokemon": "TORCHIC", "reasoning": "test"})
        print("   ", r.get("error") or r)
        assert r["success"] is False and "a battle is still going on" in r["error"], r


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
