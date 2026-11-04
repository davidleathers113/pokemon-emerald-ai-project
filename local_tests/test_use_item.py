"""use_item: medicine from the bag outside battle (server mcp_use_item, _medicine_for).

emerald-new (2026-11-04), David's "Potions + Jagged Pass": train stopped with "the team
is weak (AZUMARILL 32/82 HP)" on Jagged Pass and walked to Lavaridge below the one-way
ledges, a 10-15 minute loop; a Super Potion would have done.

test_mt_chimney_busy_grunt.state (live): BRELOOM 37, COMBUSKEN 29, SABLEYE 26, AZUMARILL
18, ELECTRIKE 20, NUMEL 19, outside battle on MT CHIMNEY. RAM writes, documented where
used: medicine put in the ITEMS pocket (test_party_rule._put_in_bag), a party member's HP
or status.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_use_item.py
"""
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server
from test_party_rule import CHIMNEY, PARTY, _party, _put_in_bag, _write

POTION, ANTIDOTE, HYPER_POTION, SUPER_POTION, REVIVE = 13, 14, 21, 22, 24
STATUS1_POISON = 0x08  # struct Pokemon +0x50 status


def _index(s, name):
    return next(i for i, p in enumerate(_party(s)) if p.species_name.upper() == name)


def _counts(s):
    with s.srv.memory_lock:
        items = s.srv.env.memory_reader.read_full_inventory().get("items", [])
    counts = {}
    for item in items:
        counts[item["item_id"]] = counts.get(item["item_id"], 0) + item["quantity"]
    return counts


def test_the_cheapest_item_that_does_the_job_is_chosen():
    from server import app

    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    mon = _party(s)[0]
    bag = {POTION: 3, SUPER_POTION: 2, HYPER_POTION: 1, ANTIDOTE: 1, REVIVE: 1}
    at = lambda hp, status=0: dataclasses.replace(mon, current_hp=hp, status=type(mon.status)(status))
    assert app._medicine_for(at(mon.max_hp - 15), bag) == POTION
    assert app._medicine_for(at(mon.max_hp - 50), bag) == SUPER_POTION  # restores all 50
    assert app._medicine_for(at(mon.max_hp - 51), bag) == HYPER_POTION
    assert app._medicine_for(at(1), {POTION: 3, SUPER_POTION: 2}) == SUPER_POTION  # the most it can
    assert app._medicine_for(at(0), bag) == REVIVE
    assert app._medicine_for(at(mon.max_hp, STATUS1_POISON), bag) == ANTIDOTE
    assert app._medicine_for(at(mon.max_hp), bag) is None
    assert app._medicine_for(at(0), {POTION: 3}) is None  # a Potion can't revive
    assert app._medicine_for(at(1), {SUPER_POTION: 0}) is None


def test_use_item_heals_with_the_cheapest_and_says_how_many_are_left():
    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _put_in_bag(s, POTION, 5)
    _put_in_bag(s, SUPER_POTION, 10)
    i = _index(s, "AZUMARILL")
    top = _party(s)[i].max_hp
    _write(s, 16, PARTY + i * 100 + 0x56, top - 50)  # RAM: AZUMARILL 50 HP down
    before = _counts(s)
    started = time.time()
    r = s.call("mcp_use_item", {"pokemon": "AZUMARILL", "reasoning": "test"})
    print("   ", r.get("summary") or r, f"({time.time() - started:.1f}s)")
    assert r["success"], r
    left = before[SUPER_POTION] - 1  # the save had one already
    assert r["summary"] == f"Used SUPER POTION on AZUMARILL ({top - 50}→{top} HP); {left} SUPER POTIONs left.", r
    assert _party(s)[i].current_hp == top
    after = _counts(s)
    assert after[SUPER_POTION] == before[SUPER_POTION] - 1 and after[POTION] == before[POTION], (before, after)
    assert s.srv._player_can_walk()


def test_use_item_cures_a_status_and_refuses_what_would_do_nothing():
    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _put_in_bag(s, ANTIDOTE, 2)
    _put_in_bag(s, POTION, 3)
    i = _index(s, "NUMEL")
    _write(s, 32, PARTY + i * 100 + 0x50, STATUS1_POISON)  # RAM: NUMEL poisoned
    r = s.call("mcp_use_item", {"pokemon": "NUMEL", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["summary"].startswith("Used ANTIDOTE on NUMEL (POISON→OK)"), r
    assert int(_party(s)[i].status) == 0
    # A Potion on a Pokémon at full HP: the game says it won't have any effect.
    breloom = _index(s, "BRELOOM")
    assert _party(s)[breloom].current_hp == _party(s)[breloom].max_hp
    before = _counts(s)
    r = s.call("mcp_use_item", {"item": "POTION", "pokemon": "BRELOOM", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "was not used on BRELOOM" in r["error"], r
    assert _counts(s)[POTION] == before[POTION]
    assert s.srv._player_can_walk()
    r = s.call("mcp_use_item", {"pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "needs nothing" in r["error"], r
    r = s.call("mcp_use_item", {"item": "MAX REVIVE", "pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "There is no MAX REVIVE" in r["error"], r


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
