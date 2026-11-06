"""use_item: medicine from the bag outside battle (server mcp_use_item, _medicine_for).

emerald-new (2026-11-04), David's "Potions + Jagged Pass": train stopped with "the team
is weak (AZUMARILL 32/82 HP)" on Jagged Pass and walked to Lavaridge below the one-way
ledges, a 10-15 minute loop; a Super Potion would have done.

test_mt_chimney_busy_grunt.state (live): BRELOOM 37, COMBUSKEN 29, SABLEYE 26, AZUMARILL
18, ELECTRIKE 20, NUMEL 19, outside battle on MT CHIMNEY. RAM writes, documented where
used: medicine put in the ITEMS pocket (test_party_rule._put_in_bag), a party member's HP
or status.

Field items that take no Pokémon (Escape Rope, the Repels): test_magma_hideout_1f.state
(MAGMA HIDEOUT 1F (5,21), an ESCAPE ROPE in the bag; see test_strength.py), and the
Chimney save (allowEscaping FALSE) with the item put in the bag.

PP items, the arguments, and the outcome read from the game: test_league_lobby.state
(live, 2026-11-05, before the Elite Four; see test_macros.py): MANECTRIC first, its SPARK,
THUNDER WAVE, BITE and THUNDER at full PP (20, 20, 25, 10); ETHER x1, PP MAX x1, REVIVE x10,
HYPER POTION x8 in the bag. RAM writes: move PP (_set_pp), HP, more ETHERs.

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
from test_rematch import SUBSTRUCT_PLACES

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
    # A Potion on a Pokémon at full HP: the game would say it won't have any effect. It is
    # refused before the bag opens now, as in battle (_medicine_refusal); this used to
    # expect the bag walked to the game's no-effect text ("was not used on BRELOOM").
    breloom = _index(s, "BRELOOM")
    assert _party(s)[breloom].current_hp == _party(s)[breloom].max_hp
    before = _counts(s)
    r = s.call("mcp_use_item", {"item": "POTION", "pokemon": "BRELOOM", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "BRELOOM" in r["error"] and "needs nothing a POTION does" in r["error"], r
    assert _counts(s)[POTION] == before[POTION]
    assert s.srv._player_can_walk()
    r = s.call("mcp_use_item", {"pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "needs nothing" in r["error"], r
    r = s.call("mcp_use_item", {"item": "MAX REVIVE", "pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "There is no MAX REVIVE" in r["error"], r


HIDEOUT = Path(__file__).resolve().parent / "test_magma_hideout_1f.state"
SUPER_REPEL, MAX_REPEL, ESCAPE_ROPE, REPEL = 83, 84, 85, 86  # pret include/constants/items.h
VAR_REPEL_STEP_COUNT = 0x4021  # pret include/constants/vars.h
SAVE_BLOCK1_ESCAPE_WARP = 0x24  # pret include/global.h struct SaveBlock1 escapeWarp (s8 mapGroup, s8 mapNum)


def _repel_steps(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_game_var(VAR_REPEL_STEP_COUNT)


def _map_id(s):
    return s.snapshot().map_id


def test_escape_rope_climbs_out_of_a_dungeon_to_its_escape_warp():
    """Seen live in MAGMA HIDEOUT: use_item("ESCAPE ROPE") answered "pokemon is required".
    pret ItemUseOutOfBattle_EscapeRope: the bag closes, "<PLAYER> used the ESCAPE ROPE.",
    and the player warps to gSaveBlock1Ptr->escapeWarp (the last outdoor spot)."""
    s = server()
    s.load(HIDEOUT, wild_encounters=False)
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        sb1 = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
        escape = (reader._read_u8(sb1 + SAVE_BLOCK1_ESCAPE_WARP), reader._read_u8(sb1 + SAVE_BLOCK1_ESCAPE_WARP + 1))
    before = _counts(s)
    assert before.get(ESCAPE_ROPE) == 1, before
    r = s.call("mcp_use_item", {"item": "ESCAPE ROPE", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    assert _map_id(s) == escape, (_map_id(s), escape)
    with s.srv.memory_lock:
        place = s.srv.env.memory_reader.read_location()
    assert place == "JAGGED PASS" and "JAGGED PASS" in r["summary"], (place, r)
    assert _counts(s).get(ESCAPE_ROPE, 0) == 0
    assert s.srv._player_can_walk()


def test_escape_rope_is_refused_where_the_map_does_not_allow_escaping():
    """MT CHIMNEY's header has allowEscaping FALSE (map.json allow_escaping), so pret
    CanUseDigOrEscapeRopeOnCurMap refuses it: say so, and spend nothing."""
    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _put_in_bag(s, ESCAPE_ROPE, 2)  # the save holds one already
    where, before = _map_id(s), _counts(s)[ESCAPE_ROPE]
    r = s.call("mcp_use_item", {"item": "ESCAPE ROPE", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "caves and dungeons" in r["error"], r
    assert _counts(s)[ESCAPE_ROPE] == before and _map_id(s) == where
    assert s.srv._player_can_walk()


def test_repels_start_and_are_refused_while_one_still_works():
    """pret ItemUseOutOfBattle_Repel: with VAR_REPEL_STEP_COUNT at 0 the Repel sets it to
    its hold effect param (REPEL 100, SUPER REPEL 200, MAX REPEL 250); while it is not 0
    "But the effects of a REPEL lingered from earlier." and nothing is spent."""
    s = server()
    for item_id, name, steps in ((REPEL, "REPEL", 100), (SUPER_REPEL, "SUPER REPEL", 200), (MAX_REPEL, "MAX REPEL", 250)):
        s.load(CHIMNEY)  # wild encounters on: the harness's own Repel is not set
        assert _repel_steps(s) == 0
        _put_in_bag(s, item_id, 2)  # the save holds a REPEL already
        before = _counts(s)[item_id]
        r = s.call("mcp_use_item", {"item": name, "reasoning": "test"})
        print("   ", r.get("summary") or r)
        assert r["success"] and name in r["summary"], r
        assert _repel_steps(s) == steps, (name, _repel_steps(s))
        assert _counts(s)[item_id] == before - 1
        assert s.srv._player_can_walk()
        r = s.call("mcp_use_item", {"item": name, "reasoning": "test"})
        print("   ", r.get("error") or r)
        assert r["success"] is False and "still" in r["error"], r
        assert _counts(s)[item_id] == before - 1 and _repel_steps(s) == steps
        assert s.srv._player_can_walk()


def test_medicine_still_needs_a_pokemon():
    s = server()
    s.load(CHIMNEY, wild_encounters=False)
    _put_in_bag(s, POTION, 1)
    r = s.call("mcp_use_item", {"item": "POTION", "reasoning": "test"})
    assert r["success"] is False and "pokemon is required" in r["error"], r
    r = s.call("mcp_use_item", {"reasoning": "test"})
    assert r["success"] is False and "pokemon is required" in r["error"], r


# -- PP items, the arguments, and the outcome read from the game ------------------------

LEAGUE_LOBBY = Path(__file__).resolve().parent / "test_league_lobby.state"
ETHER, PP_MAX = 34, 71  # pret include/constants/items.h


def _set_pp(s, slot, pp):
    """RAM: a party Pokémon's move PP. The Attacks substruct (u16 moves[4], u8 pp[4]) is
    XOR'd with personality ^ otId and the box checksum (+0x1C) is the u16 sum of the 48
    decrypted bytes (pret CalculateBoxMonChecksum), so both are redone (test_rematch._full_pp)."""
    base = PARTY + 100 * slot
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        personality = reader._read_u32(base)
        key = personality ^ reader._read_u32(base + 4)
        words = [reader._read_u32(base + 0x20 + 4 * i) ^ key for i in range(12)]
    at = 3 * SUBSTRUCT_PLACES[personality % 24][1]
    attacks = b"".join(w.to_bytes(4, "little") for w in words[at:at + 3])[:8] + bytes(pp)
    words[at:at + 3] = [int.from_bytes(attacks[4 * i:4 * i + 4], "little") for i in range(3)]
    checksum = sum((w & 0xFFFF) + (w >> 16) for w in words) & 0xFFFF
    for i, word in enumerate(words):
        _write(s, 32, base + 0x20 + 4 * i, word ^ key)
    _write(s, 16, base + 0x1C, checksum)


def _pp(s, slot):
    return list(_party(s)[slot].move_pp)


def test_an_ether_restores_the_move_it_is_told():
    """Seen live (2026-11-05, Elite Four): use_item("ETHER", "MANECTRIC", move="THUNDER")
    restored SPARK 19→20 and left THUNDER at 7: the move was never read, and A on the party
    menu's "Restore which move?" list (pret ItemUseCB_PPRecovery) took its first move.
    RAM: SPARK 19 and THUNDER 7 PP, as live."""
    s = server()
    s.load(LEAGUE_LOBBY, wild_encounters=False)
    i = _index(s, "MANECTRIC")
    _set_pp(s, i, [19, 20, 25, 7])
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "move": "THUNDER", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    assert _pp(s, i) == [19, 20, 25, 10], _pp(s, i)
    assert r["summary"] == "Used ETHER on MANECTRIC (THUNDER 7→10 PP); 0 ETHERs left.", r
    assert _counts(s).get(ETHER, 0) == 0
    assert s.srv._player_can_walk()


def test_an_ether_with_no_move_goes_on_the_one_missing_the_most_pp():
    """No move named: the one missing the most PP, and the summary says so. A move the
    Pokémon doesn't know is refused, naming its moves; so is an ETHER nothing needs."""
    s = server()
    s.load(LEAGUE_LOBBY, wild_encounters=False)
    _put_in_bag(s, ETHER, 1)  # the save holds one: two
    i = _index(s, "MANECTRIC")
    _set_pp(s, i, [19, 20, 25, 7])
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "move": "SURF", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "MANECTRIC doesn't know SURF" in r["error"], r
    assert "SPARK 19/20 PP, THUNDER WAVE 20/20 PP, BITE 25/25 PP, THUNDER 7/10 PP" in r["error"], r
    assert _counts(s)[ETHER] == 2 and s.srv._player_can_walk()
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and "THUNDER 7→10 PP" in r["summary"] and "THUNDER was missing the most PP" in r["summary"], r
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "reasoning": "test"})
    assert r["success"] and "SPARK 19→20 PP" in r["summary"], r
    assert _pp(s, i) == [20, 20, 25, 10], _pp(s, i)
    _put_in_bag(s, ETHER, 1)
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "reasoning": "test"})
    assert r["success"] is False and "every move of MANECTRIC has full PP" in r["error"], r
    assert _counts(s)[ETHER] == 1 and s.srv._player_can_walk()


def test_pp_max_raises_the_named_move():
    """PP MAX takes the same move list (pret ItemUseCB_PPUp) and is for good: it needs a
    move named. pret ITEM5_PP_MAX: three PP Ups, THUNDER 10 → 16 max PP, the 6 added to
    its PP too."""
    s = server()
    s.load(LEAGUE_LOBBY, wild_encounters=False)
    i = _index(s, "MANECTRIC")
    r = s.call("mcp_use_item", {"item": "PP MAX", "pokemon": "MANECTRIC", "reasoning": "test"})
    assert r["success"] is False and "move is required for a PP MAX" in r["error"], r
    r = s.call("mcp_use_item", {"item": "PP MAX", "pokemon": "MANECTRIC", "move": "thunder", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and "THUNDER 10/10→16/16 PP" in r["summary"], r
    mon = _party(s)[i]
    assert list(mon.move_pp) == [20, 20, 25, 16] and (mon.pp_bonuses >> 6) & 3 == 3, (mon.move_pp, mon.pp_bonuses)
    assert _counts(s).get(PP_MAX, 0) == 0 and s.srv._player_can_walk()


def test_arguments_use_item_does_not_take_are_refused():
    """Seen live (2026-11-05): an argument nobody read was a silent wrong result (the
    ETHER's move). Refused now, naming the ones it takes, with nothing used; so is a move
    for an item that takes none."""
    s = server()
    s.load(LEAGUE_LOBBY, wild_encounters=False)
    i = _index(s, "MANECTRIC")
    _set_pp(s, i, [19, 20, 25, 7])
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "MANECTRIC", "moves": "THUNDER", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "doesn't take 'moves'" in r["error"] and "It takes: item, pokemon, move." in r["error"], r
    r = s.call("mcp_use_item", {"item": "HYPER POTION", "pokemon": "MANECTRIC", "move": "THUNDER", "reasoning": "test"})
    assert r["success"] is False and "a HYPER POTION takes none" in r["error"], r
    assert _counts(s)[ETHER] == 1 and _pp(s, i) == [19, 20, 25, 7] and s.srv._player_can_walk()


def test_medicine_on_a_fainted_pokemon_is_refused_or_used_as_the_game_says():
    """Seen live (2026-11-05, DRAKE's room): the REVIVE for a fainted MANECTRIC failed, and
    the HYPER POTION after it walked the bag to "it would have no effect" (pret
    ItemUseCB_Medicine: no HP item revives). It is refused before the bag opens now, as in
    battle; a REVIVE, then the HYPER POTION, work. RAM: MANECTRIC 0 HP."""
    s = server()
    s.load(LEAGUE_LOBBY, wild_encounters=False)
    i = _index(s, "MANECTRIC")
    top = _party(s)[i].max_hp
    _write(s, 16, PARTY + i * 100 + 0x56, 0)
    before = _counts(s)
    r = s.call("mcp_use_item", {"item": "HYPER POTION", "pokemon": "MANECTRIC", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and "MANECTRIC has fainted: only a REVIVE or MAX REVIVE" in r["error"], r
    assert _counts(s) == before and s.srv._player_can_walk()
    r = s.call("mcp_use_item", {"item": "REVIVE", "pokemon": "MANECTRIC", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["summary"].startswith(f"Used REVIVE on MANECTRIC (0→{top // 2} HP)"), r
    r = s.call("mcp_use_item", {"item": "HYPER POTION", "pokemon": "MANECTRIC", "reasoning": "test"})
    assert r["success"] and r["summary"].startswith(f"Used HYPER POTION on MANECTRIC ({top // 2}→{top} HP)"), r
    after = _counts(s)
    assert after[REVIVE] == before[REVIVE] - 1 and after[HYPER_POTION] == before[HYPER_POTION] - 1, (before, after)
    assert _party(s)[i].current_hp == top and s.srv._player_can_walk()


def test_the_outcome_is_the_games_when_the_choice_waits_in_the_queue():
    """The server has one button queue; another client's presses ahead of use_item's hold
    the A on the Pokémon past _macro_press's wait (_sequence_frames). use_item then read
    the party chooser still open and nothing spent, answered "it would have no effect",
    and the REVIVE went through once the queue got to that A. Whatever happens, the answer
    must be the game's: one fewer REVIVE and the Pokémon back up, or neither. The backlog
    here is a WAIT queued from "manual" (the /play page's source) just before that A. RAM:
    MANECTRIC 0 HP."""
    from server import app

    real = app.take_action
    for backlog in (app._sequence_frames(1) + 300, 2 * app._sequence_frames(1) + 3 * app.MACRO_MENU_FRAMES):
        s = server()
        s.load(LEAGUE_LOBBY, wild_encounters=False)
        i = _index(s, "MANECTRIC")
        _write(s, 16, PARTY + i * 100 + 0x56, 0)
        revives = _counts(s)[REVIVE]
        queued = []

        async def take_action(request, backlog=backlog, queued=queued):
            if not queued and request.buttons == ["A"] and request.source == "use_item" and app._mem(app._party_ready):
                queued.append(await real(app.ActionRequest(buttons=["WAIT"], release_frames=backlog, source="manual")))
            return await real(request)

        app.take_action = take_action
        try:
            r = s.call("mcp_use_item", {"item": "REVIVE", "pokemon": "MANECTRIC", "reasoning": "test"})
        finally:
            app.take_action = real
        print("   ", backlog, "frames:", r.get("summary") or r)
        assert queued, "the backlog was never queued"
        s.wait_idle()
        used = _counts(s)[REVIVE] == revives - 1
        assert r["success"] == used, (r, _counts(s)[REVIVE], revives)
        assert (_party(s)[i].current_hp > 0) == used, _party(s)[i].current_hp
        assert s.srv._player_can_walk()


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
