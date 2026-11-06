"""Double battles and the team's last resources: the foe to hit, the foes' boosts and the
moves they used, who replaces a fainted Pokémon, and medicine from the bag in battle.

Seen live (2026-11-05, 16:40-16:55 UTC, MOSSDEEP GYM, TATE & LIZA): XATU used CALM MIND
(+1 SpAtk +1 SpDef, then more) and swept AZUMARILL, BRELOOM and TROPIUS with PSYCHIC,
staying at 94/116 HP all fight, while every attack went into CLAYDOL (the foe the target
cursor starts on, healed by a HYPER POTION). "Answered for you" sent SABLEYE, BRELOOM,
BLAZIKEN, then MANECTRIC in ("the best matchup left; nobody wins the exchange"), and
3 REVIVEs and 5 SUPER POTIONs stayed in the bag; the battle state listed only Poké Balls.

Fixtures, all real states of that battle, replayed from the live gym checkpoint
(test_mossdeep_gym_statue.state, 2026-11-05): the walk to (23,8) fighting every gym
trainer on the way with use_move, use_item REVIVE + SUPER POTION on AZUMARILL and a
SUPER POTION on TROPIUS at (23,8), A on TATE AND LIZA, then the harness's own choices as
checked out at 59e79d5 (use_move with the view's best move, the target cursor's foe):
  test_tate_liza_calm_mind.state         turn 2, FIGHT/BAG/POKEMON/RUN for AZUMARILL
                                         74/144 (TROPIUS 106/106 beside it): CLAYDOL 52/112
                                         just used EARTHQUAKE, XATU 102/116 CALM MIND (+1
                                         SpAtk +1 SpDef). Bag: REVIVE x2, SUPER POTION x5,
                                         FULL RESTORE x1.
  test_tate_liza_tropius_fainted.state   from it: AZUMARILL DEFENSE CURL, TROPIUS STOMP;
                                         XATU's PSYCHIC knocked TROPIUS out. The game's
                                         replacement menu: SABLEYE, BLAZIKEN, BRELOOM,
                                         MANECTRIC to choose from.
  test_tate_liza_azumarill_fainted.state the next turn: SABLEYE (38/97) in, XATU's PSYCHIC
                                         knocked AZUMARILL out. BLAZIKEN, BRELOOM,
                                         MANECTRIC to choose from; the harness at 59e79d5
                                         chose BRELOOM, as live.
  test_tate_liza_two_left.state          later in a replay with this change (XATU aimed at
                                         and knocked out): BRELOOM and SABLEYE are the last
                                         two healthy, AZUMARILL, TROPIUS, BLAZIKEN and
                                         MANECTRIC fainted, LUNATONE left. FIGHT/BAG/POKEMON/RUN
                                         for BRELOOM.
A loaded save carries only each battler's last move (gLastMoves), so "used" here is
CLAYDOL's EARTHQUAKE and XATU's last move. No RAM writes but the ETHER test's (its docstring).

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_battle_doubles.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent
CALM_MIND = HERE / "test_tate_liza_calm_mind.state"
TROPIUS_FAINTED = HERE / "test_tate_liza_tropius_fainted.state"
AZUMARILL_FAINTED = HERE / "test_tate_liza_azumarill_fainted.state"
TWO_LEFT = HERE / "test_tate_liza_two_left.state"

# pret symbols (US): gBattleStruct (a pointer) -> struct BattleStruct moveTarget[4] at
# +0x0C, the target the game took for each battler's move. include/constants/items.h.
G_BATTLE_STRUCT_PTR = 0x0202449C
MOVE_TARGET = 0x0C
EARTHQUAKE = 89
SUPER_POTION, REVIVE = 22, 24
CLAYDOL, XATU = 1, 3  # their battlers (B_POSITION_OPPONENT_LEFT, _RIGHT)


def _load(state):
    s = server()
    s.load(state, wild_encounters=False)
    return s


def _view(s):
    return s.srv._read_battle_view()


def _lines(s):
    from utils.state_formatter import format_battle_view

    lines = format_battle_view(_view(s))
    print("    " + "\n    ".join(lines))
    return lines


def _move_target(s, battler):
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        return reader._read_u8(reader._read_u32(G_BATTLE_STRUCT_PTR) + MOVE_TARGET + battler)


def _party(s):
    with s.srv.memory_lock:
        return {p.species_name.upper(): p for p in s.srv.env.memory_reader.read_party_pokemon()}


def _bag(s, item_id):
    with s.srv.memory_lock:
        items = s.srv.env.memory_reader.read_full_inventory().get("items", [])
    return sum(int(i["quantity"]) for i in items if i["item_id"] == item_id)


# -- the foe to hit --------------------------------------------------------------------


def test_the_boosted_foe_is_the_target():
    """XATU (+1 SpAtk +1 SpDef, its PSYCHIC knocks AZUMARILL out) before CLAYDOL; the
    best move is named with it. Before, the target was the first foe standing: CLAYDOL."""
    s = _load(CALM_MIND)
    view = _view(s)
    assert view["is_double"] and view["menu"] == "action" and view["player"]["name"] == "Azumarill", view["player"]
    assert view["target"]["name"] == "Xatu", view["target"]["name"]
    lines = _lines(s)
    target = next((line for line in lines if line.startswith("Target: ")), "")
    assert target.startswith("Target: XATU (+1 SpAtk +1 SpDef; PSYCHIC ~"), target
    assert "a knock-out) before CLAYDOL (" in target, target
    assert any(line.startswith("Moves (") and " vs XATU PSYCHIC/FLYING" in line for line in lines), lines
    best = view["best_move"]["name"]
    assert any(f'call use_move("{best}", target="XATU")' in line for line in lines), lines


def test_the_stages_and_used_moves_show_in_the_summary():
    s = _load(CALM_MIND)
    first = _lines(s)[0]
    assert "CLAYDOL Lv41 (GROUND/PSYCHIC), HP 52/112, LEVITATE, used EARTHQUAKE;" in first, first
    assert "XATU Lv41 (PSYCHIC/FLYING), HP 102/116, SYNCHRONIZE, +1 SpAtk +1 SpDef, used CALM MIND" in first, first
    xatu = next(m for m in _view(s)["opponents"] if m["name"] == "Xatu")
    assert [tuple(c) for c in xatu["stage_changes"]] == [("SpAtk", 1), ("SpDef", 1)], xatu["stage_changes"]


def test_use_move_hits_the_suggested_target():
    """AZUMARILL's STRENGTH (one target) goes into XATU: the game's moveTarget for battler
    0 says so. Before, A confirmed the cursor's first foe, CLAYDOL. The call returns at
    TROPIUS's choice (the turn plays once both have chosen)."""
    s = _load(CALM_MIND)
    r = s.call("mcp_use_move", {"move": "STRENGTH", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["end"] == "action" and r.get("target") == "XATU", r
    assert "(Double battle: aimed at XATU.)" in r["summary"], r["summary"]
    assert "opponent XATU HP 102/116 before the hit" in r["summary"], r["summary"]
    assert _move_target(s, 0) == XATU, _move_target(s, 0)
    assert _view(s)["player"]["name"] == "Tropius"


def test_use_move_hits_the_foe_asked_for():
    s = _load(CALM_MIND)
    r = s.call("mcp_use_move", {"move": "STRENGTH", "target": "claydol", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r.get("target") == "CLAYDOL" and "aimed at CLAYDOL, as asked" in r["summary"], r
    assert _move_target(s, 0) == CLAYDOL
    s = _load(CALM_MIND)
    r = s.call("mcp_use_move", {"move": "STRENGTH", "target": "SOLROCK", "reasoning": "test"})
    assert r["success"] is False and "foes standing: CLAYDOL, XATU" in r["error"], r
    assert _view(s)["menu"] == "action"  # nothing pressed


# -- the foes' moves, and who comes in --------------------------------------------------


def test_the_moves_the_foes_used_are_remembered_per_battle():
    """The server's game loop records them after every frame (no battle view read here)."""
    s = _load(CALM_MIND)
    time.sleep(0.5)  # a few frames of the game loop's thread
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        claydol = reader._read_battle_mon(CLAYDOL)
        assert EARTHQUAKE in reader.battle_moves_used(claydol["personality"])
    s.load(HERE / "test_movement_route101.state")  # another save: forgotten
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.battle_moves_used(claydol["personality"]) == []


def test_a_replacement_avoids_the_mons_weak_to_a_seen_earthquake():
    """TROPIUS fainted; CLAYDOL used EARTHQUAKE, XATU PSYCHIC. BLAZIKEN and MANECTRIC are
    weak to EARTHQUAKE, BLAZIKEN and BRELOOM to PSYCHIC: SABLEYE comes in, and the
    answer says why, naming the moves."""
    s = _load(TROPIUS_FAINTED)
    assert s.srv._forced_party_menu()
    pick = s.srv._replacement()
    print("   ", pick)
    assert pick["name"] == "SABLEYE", pick
    assert "BLAZIKEN and MANECTRIC: weak to EARTHQUAKE (CLAYDOL used it)" in pick["why"], pick["why"]
    import asyncio

    played = asyncio.run(s.srv._play_out_battle("test"))
    report = s.srv._battle_report(played)
    print("   ", report)
    assert "Answered for you: sent in SABLEYE for the fainted Pokémon (" in report, report
    assert "weak to EARTHQUAKE (CLAYDOL used it)" in report, report
    with s.srv.memory_lock:
        out = s.srv.env.memory_reader._read_battle_mon(2)
    assert out["name"] == "Sableye", out["name"]


def test_a_replacement_avoids_the_one_a_seen_psychic_knocks_out():
    """AZUMARILL fainted with SABLEYE out: BLAZIKEN, BRELOOM, MANECTRIC left, each weak to
    a move the foes used. The harness chose BRELOOM, as live (it fainted to the next
    PSYCHIC); XATU's boosted PSYCHIC takes all of BRELOOM's HP in one hit, CLAYDOL's
    EARTHQUAKE doesn't take all of MANECTRIC's: MANECTRIC comes in."""
    s = _load(AZUMARILL_FAINTED)
    assert s.srv._forced_party_menu()
    pick = s.srv._replacement()
    print("   ", pick)
    assert pick["name"] == "MANECTRIC", pick
    assert "BLAZIKEN and BRELOOM: weak to PSYCHIC (XATU used it)" in pick["why"], pick["why"]


# -- medicine in battle -----------------------------------------------------------------


def test_the_item_line_names_a_potion_for_the_one_in_knock_out_range():
    s = _load(CALM_MIND)
    hint = _view(s)["item_hint"]
    assert hint and hint["uses"][0]["item"] == "SUPER POTION" and hint["uses"][0]["pokemon"] == "AZUMARILL", hint
    line = next((l for l in _lines(s) if l.startswith("ITEMS: ")), "")
    assert line.startswith('ITEMS: use_item("SUPER POTION", "AZUMARILL") (XATU\'s PSYCHIC (~'), line
    assert "It takes AZUMARILL's turn." in line and "REVIVE x2, SUPER POTION x5, FULL RESTORE x1" in line, line


def test_the_item_line_names_a_revive_when_two_healthy_are_left():
    s = _load(TWO_LEFT)
    healthy = [n for n, p in _party(s).items() if p.current_hp * 2 >= p.max_hp]
    assert sorted(healthy) == ["BRELOOM", "SABLEYE"], healthy
    line = next((l for l in _lines(s) if l.startswith("ITEMS: ")), "")
    # AZUMARILL (Lv40) is the highest level of the four that fainted.
    assert 'use_item("REVIVE", "AZUMARILL") (only 2 of your Pokémon can still fight well; AZUMARILL is fainted)' in line, line


def test_use_item_in_battle_heals_as_that_pokemons_turn():
    s = _load(CALM_MIND)
    before = _bag(s, SUPER_POTION)
    r = s.call("mcp_use_item", {"item": "SUPER POTION", "pokemon": "AZUMARILL", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["status"] == "USED" and r["end"] == "action", r
    assert r["summary"].startswith("Used SUPER POTION on AZUMARILL (74→124 HP) as AZUMARILL's turn;"), r["summary"]
    assert _party(s)["AZUMARILL"].current_hp == 124 and _bag(s, SUPER_POTION) == before - 1
    view = _view(s)
    assert view["menu"] == "action" and view["player"]["name"] == "Tropius", view["player"]  # the partner's turn


def test_use_item_in_battle_revives_a_fainted_one():
    """The battle's party menu lists the party in battle order (pret party_menu.c
    UpdatePartyToBattleOrder): the REVIVE must reach AZUMARILL, not who sits in its slot
    there."""
    s = _load(TWO_LEFT)
    before = _bag(s, REVIVE)
    r = s.call("mcp_use_item", {"item": "REVIVE", "pokemon": "AZUMARILL", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["summary"].startswith("Used REVIVE on AZUMARILL (0→72 HP) as BRELOOM's turn;"), r
    party = _party(s)
    assert party["AZUMARILL"].current_hp > 0 and _bag(s, REVIVE) == before - 1, party["AZUMARILL"]
    assert all(party[n].current_hp == 0 for n in ("TROPIUS", "BLAZIKEN", "MANECTRIC")), party


def test_use_item_in_battle_restores_the_named_moves_pp():
    """An ETHER in battle opens the battle's party menu and the same "Restore which move?"
    list as outside it (pret item_use.c ItemUseInBattle_PPRecovery, party_menu.c
    ItemUseCB_PPRecovery); the move named is the one restored. Seen live outside battle
    (2026-11-05): the ETHER meant for THUNDER went on the first move. RAM: an ETHER in the
    bag; AZUMARILL's first and last moves down to 1 PP, so the first in the list needs it
    too. The save holds an ETHER already."""
    from pokemon_env.enums import Move
    from test_party_rule import _put_in_bag
    from test_use_item import ETHER, _set_pp

    s = _load(CALM_MIND)
    with s.srv.memory_lock:
        party = s.srv.env.memory_reader.read_party_pokemon()
    slot = next(i for i, p in enumerate(party) if p.species_name.upper() == "AZUMARILL")
    pp = list(party[slot].move_pp)
    last = len([m for m in party[slot].moves if m and m != "NONE"]) - 1
    pp[0] = pp[last] = 1
    _set_pp(s, slot, pp)
    _put_in_bag(s, ETHER, 1)
    ethers = _bag(s, ETHER)
    move = party[slot].moves[last].replace("_", " ")
    with s.srv.memory_lock:  # max PP: the ROM's base PP (this team has no PP Ups)
        base = s.srv.env.memory_reader._rom_battle_move(Move[party[slot].moves[last]].value)["pp"]
    restored = min(1 + 10, base)  # an ETHER restores 10, up to max PP
    r = s.call("mcp_use_item", {"item": "ETHER", "pokemon": "AZUMARILL", "move": move, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] and r["summary"].startswith(f"Used ETHER on AZUMARILL ({move} 1→{restored} PP) as AZUMARILL's turn;"), r
    now = list(_party(s)["AZUMARILL"].move_pp)
    assert now[0] == 1 and now[last] == restored, now
    assert _bag(s, ETHER) == ethers - 1


def test_use_item_in_battle_refuses_what_would_do_nothing():
    s = _load(CALM_MIND)
    for item, mon, why in (
        ("SUPER POTION", "SABLEYE", "SABLEYE (97/97 HP) needs nothing a SUPER POTION does"),
        ("REVIVE", "TROPIUS", "TROPIUS hasn't fainted"),
        ("ESCAPE ROPE", "TROPIUS", "There is no ESCAPE ROPE"),
    ):
        r = s.call("mcp_use_item", {"item": item, "pokemon": mon, "reasoning": "test"})
        assert r["success"] is False and why in r["error"], r
        assert _view(s)["menu"] == "action" and _view(s)["action_cursor_name"] == "FIGHT"  # nothing pressed


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
