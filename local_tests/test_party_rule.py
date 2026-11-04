"""David's party rule, kept up by the harness (server maintain_party, give_item).

David (2026-11-04, verbatim): "I want the pokemon with the lowest level to always be in
the first spot in the party so that it is the one that gets trained. That is, until we
have the exp share. Then the weakest pokemon in the party should always hold the exp
share. Then we will be able to level it up regardless of who's doing the fighting."

test_mt_chimney_busy_grunt.state (live): BRELOOM 37, COMBUSKEN 29, SABLEYE 26, AZUMARILL
18, ELECTRIKE 20, NUMEL 19, outside battle on MT CHIMNEY. RAM writes, documented where
used: the Exp. Share put in the bag (its quantity XOR'd with the save's key, as the game
stores it), a party member's HP (fainted) or level.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_party_rule.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent
CHIMNEY = HERE / "test_mt_chimney_busy_grunt.state"
ISAAC = HERE / "test_switch_question.state"  # a trainer battle, action menu open
PARTY = 0x020244EC  # gPlayerParty (100-byte mons; +0x54 level, +0x56 HP)


def _write(s, size, address, value):
    with s.srv.memory_lock:
        memory = s.srv.env.core.memory
        {8: memory.u8, 16: memory.u16, 32: memory.u32}[size][address] = value
        s.srv.env.memory_reader._invalidate_mem_cache()


def _party(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_party_pokemon()


def _names(s):
    return [p.species_name.upper() for p in _party(s)]


def _maintain(s):
    r = asyncio.run(s.srv.maintain_party({}))
    print("   ", r)
    return r


def _bag_ids(s, pocket=0):
    from server import menu_state as ms

    with s.srv.memory_lock:
        return ms.bag_pocket_item_ids(s.srv.env.memory_reader, pocket)


def _put_in_bag(s, item_id, quantity=1):
    """RAM: the item into the first empty slot of the ITEMS pocket."""
    from server import menu_state as ms

    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        slots = reader._read_u32(ms.G_BAG_POCKETS)
        capacity = reader._read_u8(ms.G_BAG_POCKETS + 4)
        key = reader._get_security_key() & 0xFFFF
        free = next(i for i in range(capacity) if reader._read_u16(slots + 4 * i) == 0)
    _write(s, 16, slots + 4 * free, item_id)
    _write(s, 16, slots + 4 * free + 2, quantity ^ key)


def test_the_lowest_level_pokemon_is_put_first():
    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    assert _names(s)[0] == "BRELOOM" and _names(s)[3] == "AZUMARILL", _names(s)
    r = _maintain(s)
    assert r["changed"] and r["summary"] == "Lead set to AZUMARILL (lowest level, Lv18).", r
    assert _names(s)[0] == "AZUMARILL", _names(s)
    assert s.srv._player_can_walk()
    again = _maintain(s)
    # Already right: nothing to do, and it says why (each "nothing" carries a reason).
    assert again["changed"] is False and again["summary"] == "" and "already leads" in again["why"], again
    feed = asyncio.run(s.srv.get_ai_feed())
    assert any(e["kind"] == "party" and "AZUMARILL" in e["text"] for e in feed["events"]), feed["events"]


def test_a_fainted_lowest_is_skipped_and_ties_go_to_the_one_needing_more_exp():
    from server import menu_state as ms

    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _write(s, 16, PARTY + 3 * 100 + 0x56, 0)  # RAM: AZUMARILL (slot 4) fainted
    r = _maintain(s)
    assert r["summary"] == "Lead set to NUMEL (lowest level, Lv19).", r
    # The rule itself, on the party as read: an egg never counts, a tie goes to the one
    # with more EXP still to gain for its next level.
    party = _party(s)
    import dataclasses

    numel = next(i for i, p in enumerate(party) if p.species_name.upper() == "NUMEL")
    electrike = next(i for i, p in enumerate(party) if p.species_name.upper() == "ELECTRIKE")
    eggy = [dataclasses.replace(p, is_egg=(i == numel)) for i, p in enumerate(party)]
    assert s.srv._party_rule_target(eggy) == electrike, "the egg was chosen"
    tied = [dataclasses.replace(p, level=20) if i in (numel, electrike) else p for i, p in enumerate(party)]
    with s.srv.memory_lock:
        needs = {i: ms.exp_to_next_level(s.srv.env.memory_reader, p.species_id, p.level, p.experience) for i, p in enumerate(tied)}
    assert needs[numel] != needs[electrike], needs
    want = numel if needs[numel] > needs[electrike] else electrike
    print("    EXP to Lv21 at Lv20:", {party[i].species_name: needs[i] for i in (numel, electrike)})
    assert s.srv._party_rule_target(tied) == want


def test_nothing_happens_in_a_battle_or_a_conversation():
    s = server()
    s.load(ISAAC, wild_encounters=False)
    order = _names(s)
    r = _maintain(s)
    assert r["changed"] is False and r["why"] == "in a battle" and _names(s) == order, r
    s.load(CHIMNEY, wild_encounters=False)
    s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "test"})  # the busy grunt's line opens
    deadline = time.time() + 5
    while s.srv._player_can_walk() and time.time() < deadline:
        time.sleep(0.1)
    assert not s.srv._player_can_walk()
    r = _maintain(s)
    assert r["changed"] is False and r["why"].startswith("the player can't walk") and _names(s)[0] == "BRELOOM", r


def test_with_the_exp_share_the_weakest_holds_it_and_it_moves_when_that_changes():
    from server import menu_state as ms

    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _put_in_bag(s, ms.ITEM_EXP_SHARE)  # RAM: as Mr. Stone gives it
    assert ms.ITEM_EXP_SHARE in _bag_ids(s)
    r = _maintain(s)
    held = {p.species_name.upper(): p.held_item for p in _party(s)}
    assert r["changed"] and r["summary"].startswith("Exp. Share given to AZUMARILL (lowest level, Lv18)"), r
    assert held["AZUMARILL"] == ms.ITEM_EXP_SHARE and ms.ITEM_EXP_SHARE not in _bag_ids(s), held
    assert _names(s)[0] == "BRELOOM", "with the Exp. Share the lead is no longer forced"
    assert _maintain(s)["changed"] is False
    # AZUMARILL grows past the rest (RAM level): NUMEL (Lv19) is the weakest now.
    azumarill = _names(s).index("AZUMARILL")
    _write(s, 8, PARTY + azumarill * 100 + 0x54, 40)
    r = _maintain(s)
    held = {p.species_name.upper(): p.held_item for p in _party(s)}
    assert r["summary"] == "Exp. Share given to NUMEL (lowest level, Lv19), from AZUMARILL.", r
    assert held["NUMEL"] == ms.ITEM_EXP_SHARE and held["AZUMARILL"] == 0, held
    assert s.srv._player_can_walk()


def test_give_item_swaps_what_the_pokemon_held_back_into_the_bag():
    from server import menu_state as ms

    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    balls_before = _bag_ids(s, ms.BALLS_POCKET)
    r = s.call("mcp_give_item", {"item": "POKE BALL", "pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"], r
    _put_in_bag(s, ms.ITEM_EXP_SHARE)
    r = s.call("mcp_give_item", {"item": "EXP SHARE", "pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    held = {p.species_name.upper(): p.held_item for p in _party(s)}
    assert r["success"] and held["NUMEL"] == ms.ITEM_EXP_SHARE, (r, held)
    assert "its POKE BALL went to the bag" in r["summary"], r["summary"]
    assert _bag_ids(s, ms.BALLS_POCKET) == balls_before  # the Poké Ball is back, nothing lost
    # (In this save NUMEL first held a RAWST BERRY: the first give put it in the bag.)
    r = s.call("mcp_give_item", {"item": "MASTER BALL", "pokemon": "NUMEL", "reasoning": "test"})
    assert r["success"] is False and "neither in the bag" in r["error"], r


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
