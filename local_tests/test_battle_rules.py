"""The harness's step hook (server harness_step): WILD_BATTLES, the DANGER switch, and
David's HM protocol in the party rule.

David (2026-11-04, verbatim): "I want it to run from the wild pokemon and focus on
fighting the trainers." Seen live the same day: TROPIUS (Lv27, the Fly carrier) led into a
Team Magma trainer's TORKOAL and fainted; the DANGER line said to switch to AZUMARILL and
the 8B model attacked anyway. So the harness acts itself, once per step, before the
model decides.

Fixtures (all real savestates):
  test_wild_shroomish.state        Petalburg Woods, wild SHROOMISH, FIGHT/BAG/POKEMON/RUN open
  test_wild_shroomish_intro.state  the same battle at "Wild SHROOMISH appeared!"
  test_battle_wurmple.state        Bug Catcher Lyle's WURMPLE: a trainer battle, move menu open
  test_switch_question.state       Isaac's WHISMUR (trainer), action menu; SABLEYE out,
                                   COMBUSKEN and BRELOOM benched (all Lv23-24)
  test_lavaridge_pokecenter.state  LAVARIDGE POKEMON CENTER, full party; TROPIUS boxed

RAM writes, documented where used: battler speed (gBattleMons +6) to force a failed
escape, battler HP (+0x28, +0x2C max), party HP (+0x56) and level (+0x54).

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_battle_rules.py [name-substring]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server
from test_party_rule import PARTY, _names, _party, _write

HERE = Path(__file__).resolve().parent
WILD = HERE / "test_wild_shroomish.state"
WILD_INTRO = HERE / "test_wild_shroomish_intro.state"
TRAINER = HERE / "test_battle_wurmple.state"
ISAAC = HERE / "test_switch_question.state"
CENTER = HERE / "test_lavaridge_pokecenter.state"

# pret pokeemerald (US): gBattleMons (0x58 bytes; +6 speed, +0x28 HP, +0x2C max HP),
# gBattleOutcome.
BATTLE_MONS = 0x02024084
FOE = BATTLE_MONS + 0x58
BATTLE_OUTCOME = 0x0202433A
B_OUTCOME_RAN = 4


def _read(s, fn):
    with s.srv.memory_lock:
        return fn(s.srv.env.memory_reader)


def _in_battle(s):
    return _read(s, lambda r: r.is_in_battle())


def _fresh(state, mode="run", wild_encounters=True):
    """Load a state with the hook's inputs as a state read would leave them: no training
    objective, no catch objective, the given WILD_BATTLES, nothing remembered."""
    s = server()
    s.load(state, wild_encounters=wild_encounters)
    srv = s.srv
    srv._training_objective_current = False
    srv._objective_guidance_seen = {}
    srv._battle_rules_done.clear()
    srv._train_running = False
    os.environ["WILD_BATTLES"] = mode
    return s


def _step(s):
    r = s.call("harness_step", {})
    print("   ", r)
    return r


# -- WILD_BATTLES ---------------------------------------------------------------


def test_run_mode_runs_from_a_wild_battle():
    s = _fresh(WILD)
    events = len(s.srv._pause_events)
    r = _step(s)
    assert r["rule"] == "wild_battles" and r["changed"] is True, r
    assert r["summary"].startswith("Ran from wild SHROOMISH Lv"), r
    assert "(WILD_BATTLES=run)" in r["summary"] and "Got away safely" in r["summary"], r
    assert not _in_battle(s)
    assert _read(s, lambda r: r._read_u8(BATTLE_OUTCOME)) == B_OUTCOME_RAN
    event = s.srv._pause_events[-1]
    assert len(s.srv._pause_events) == events + 1 and event["kind"] == "battle_rule", event
    assert event["source"] == "harness" and event["text"].startswith("Ran from wild SHROOMISH"), event


def test_run_mode_presses_through_the_intro_text():
    s = _fresh(WILD_INTRO)
    assert _read(s, lambda r: r.read_battle_view())["menu"] == "busy"
    r = _step(s)
    assert r["rule"] == "wild_battles" and r["changed"] is True, r
    assert not _in_battle(s)


def test_default_is_run():
    s = _fresh(WILD)
    del os.environ["WILD_BATTLES"]
    assert s.srv._wild_battles_mode() == "run"
    os.environ["WILD_BATTLES"] = " Fight "
    assert s.srv._wild_battles_mode() == "fight"
    os.environ["WILD_BATTLES"] = "nonsense"
    assert s.srv._wild_battles_mode() == "run"


def test_fight_mode_leaves_the_wild_battle_to_the_model():
    s = _fresh(WILD, mode="fight")
    r = _step(s)
    assert r["changed"] is False and r.get("rule") is None and "WILD_BATTLES=fight" in r["why"], r
    assert _in_battle(s) and _read(s, lambda r: r.read_battle_view())["menu"] == "action"
    assert s.srv._note_wild_battles({}) == ""  # and the state says nothing about running


def test_a_trainer_battle_is_never_run_from():
    s = _fresh(TRAINER)
    before = _read(s, lambda r: r.read_battle_view())
    r = _step(s)
    assert r["changed"] is False and r.get("rule") is None, r
    view = _read(s, lambda r: r.read_battle_view())
    assert _in_battle(s) and view["menu"] == before["menu"] == "move", view  # nothing pressed
    assert view["move_cursor_slot"] == before["move_cursor_slot"]
    assert s.srv._note_wild_battles({}) == ""
    # Isaac's battle is at FIGHT/BAG/POKEMON/RUN (and SABLEYE wins it: no DANGER).
    s = _fresh(ISAAC)
    r = _step(s)
    assert r["changed"] is False and r.get("rule") is None, r
    assert _in_battle(s) and _read(s, lambda r: r.read_battle_view())["menu"] == "action"
    assert s.srv._note_wild_battles({}) == ""


def test_no_run_while_a_training_objective_is_current():
    s = _fresh(WILD)
    s.srv._training_objective_current = True
    try:
        r = _step(s)
        assert r["changed"] is False and "training objective" in r["why"], r
        assert _in_battle(s)
        assert s.srv._note_wild_battles({}) == ""
    finally:
        s.srv._training_objective_current = False


def test_no_run_from_the_pokemon_a_catch_objective_wants():
    s = _fresh(WILD)
    s.srv._objective_guidance_seen = {"story": {"completion_condition": "party_has_shroomish"}}
    r = _step(s)
    assert r["changed"] is False and "wants this SHROOMISH caught" in r["why"], r
    assert _in_battle(s)
    # Another species wanted: this one is still run from (the hunt goes on).
    s.srv._objective_guidance_seen = {"story": {"completion_condition": "party_has_zubat"}}
    r = _step(s)
    assert r["rule"] == "wild_battles" and r["changed"] is True, r


def test_no_run_before_the_objectives_are_read():
    s = _fresh(WILD)
    s.srv._objective_guidance_seen = None
    r = _step(s)
    assert r["changed"] is False and "no objectives read yet" in r["why"], r
    assert _in_battle(s)


def test_no_run_while_train_runs():
    s = _fresh(WILD)
    s.srv._train_running = True
    try:
        r = _step(s)
        assert r["changed"] is False and r["why"].startswith("train() is running"), r
        assert _in_battle(s)
    finally:
        s.srv._train_running = False
    # train() clears it when it returns, a refusal included.
    s.call("mcp_train", {})
    assert s.srv._train_running is False


def test_a_failed_escape_is_reported_and_not_retried():
    """pret battle_util.c TryRunFromBattle: a slower Pokémon gets away when
    speed * 128 / foe speed + runTries * 30 > Random() & 0xFF. RAM: TORCHIC speed 1,
    SHROOMISH speed 500, so the first try is 0 > (0..255): it can't pass."""
    s = _fresh(WILD)
    assert s.srv._note_wild_battles({}).startswith("WILD BATTLE: the harness runs from wild battles")
    _write(s, 16, BATTLE_MONS + 6, 1)
    _write(s, 16, FOE + 6, 500)
    r = _step(s)
    assert r["rule"] == "wild_battles" and r["changed"] is False, r
    assert "couldn't" in r["summary"] and "won't try again" in r["summary"], r
    assert _in_battle(s) and _read(s, lambda r: r.read_battle_view())["menu"] == "action"
    assert s.srv._pause_events[-1]["text"].startswith("Couldn't run from wild SHROOMISH")
    # The next step leaves it alone: no second RUN, the state no longer says "run".
    r = _step(s)
    assert r["changed"] is False and r.get("rule") is None and "already failed" in r["why"], r
    assert _in_battle(s) and _read(s, lambda r: r.read_battle_view())["menu"] == "action"
    assert s.srv._note_wild_battles({}) == ""


# -- the DANGER switch -----------------------------------------------------------


def _isaac_danger(s):
    """As in test_battle_view's danger test: SABLEYE at 1 HP, Isaac's WHISMUR at 200/200,
    so WHISMUR knocks SABLEYE out before SABLEYE's five hits land."""
    _write(s, 16, BATTLE_MONS + 0x28, 1)
    _write(s, 16, FOE + 0x28, 200)
    _write(s, 16, FOE + 0x2C, 200)
    view = _read(s, lambda r: r.read_battle_view())
    assert view["danger"] and view["menu"] == "action", view
    return view


def _out_name(s):
    return str(_read(s, lambda r: r.read_battle_view())["player"]["name"]).upper()


def test_danger_switches_to_the_named_teammate_once_per_foe():
    s = _fresh(ISAAC)
    pick = _isaac_danger(s)["danger"]["switch"]
    assert pick and pick["name"] in ("COMBUSKEN", "BRELOOM"), pick
    r = _step(s)
    assert r["rule"] == "danger_switch" and r["changed"] is True, r
    assert r["summary"].startswith(f"Switched to {pick['name']}: SABLEYE loses to WHISMUR's"), r
    assert "(harness, DANGER)" in r["summary"], r
    assert _out_name(s) == pick["name"]
    assert s.srv._pause_events[-1]["kind"] == "battle_rule"
    # Same foe: never a second switch, whatever the view says now (no ping-pong).
    _write(s, 16, BATTLE_MONS + 0x28, 1)  # the new one in danger too
    r = _step(s)
    assert r["changed"] is False and r.get("rule") is None, r
    assert _out_name(s) == pick["name"]


def test_no_danger_switch_without_a_teammate_that_wins():
    s = _fresh(ISAAC)
    names = _names(s)
    for i, name in enumerate(names):
        if name != "SABLEYE":
            _write(s, 16, PARTY + i * 100 + 0x56, 0)  # party HP 0: fainted
    view = _isaac_danger(s)
    assert view["danger"]["switch"] is None, view["danger"]
    r = _step(s)
    assert r["changed"] is False and "no teammate wins" in r["why"], r
    assert _out_name(s) == "SABLEYE"


def test_hm_carriers_that_dont_fight_are_sent_in_only_when_nobody_else_wins():
    """pokemon_env/hm_carriers.py, patched for this save (it has no TROPIUS): with BRELOOM
    a non-fighting carrier, COMBUSKEN (which also wins) is named instead; with both, one
    of them is still named (nobody else wins), and the harness won't switch to it."""
    from pokemon_env import hm_carriers

    s = _fresh(ISAAC)
    saved = dict(hm_carriers.HM_CARRIER_FIGHTS)
    try:
        hm_carriers.HM_CARRIER_FIGHTS["BRELOOM"] = False
        assert _isaac_danger(s)["danger"]["switch"]["name"] == "COMBUSKEN"
        hm_carriers.HM_CARRIER_FIGHTS["COMBUSKEN"] = False
        pick = _read(s, lambda r: r.read_battle_view())["danger"]["switch"]
        assert pick and pick["name"] in ("COMBUSKEN", "BRELOOM"), pick
        r = _step(s)
        assert r["changed"] is False and "HM carrier that doesn't fight" in r["why"], r
        assert _out_name(s) == "SABLEYE"
    finally:
        hm_carriers.HM_CARRIER_FIGHTS.clear()
        hm_carriers.HM_CARRIER_FIGHTS.update(saved)


# -- trapped: no run, no switch (2026-11-05) ---------------------------------------
# Seen live (VICTORY ROAD, ef1a194): a wild GOLBAT used MEAN LOOK; run_from_battle chose
# RUN turn after turn, the game said "Can't escape!" each time and GOLBAT hit back: three
# faints. pret include/battle.h: gBattleMons[].status2 (+0x50) STATUS2_ESCAPE_PREVENTION,
# STATUS2_WRAPPED's turn count (bits 13-15); gDisableStructs[] (0x1C bytes) +0x14
# battlerPreventingEscape; gBattleStruct (a pointer) +0x04 wrappedMove[battler * 2];
# struct BattlePokemon +0x0C moves[4], +0x20 ability; gBattleResults +0x13
# battleTurnCounter (pokeemerald.sym).
STATUS2, ESCAPE_PREVENTION, WRAPPED_3_TURNS = 0x50, 0x04000000, 0x6000
DISABLE_STRUCTS, G_BATTLE_STRUCT = 0x020242BC, 0x0202449C
TURN_COUNTER = 0x03005D10 + 0x13
MEAN_LOOK, FIRE_SPIN, ABILITY_RUN_AWAY = 212, 83, 50
WILD_TWO = HERE / "test_wild_two_party.state"  # wild SILCOON; TORCHIC out, SHROOMISH benched


def _mean_look(s):
    """RAM, what wild GOLBAT did live: our battler 0 under STATUS2_ESCAPE_PREVENTION, set
    by battler 1 (gDisableStructs[0].battlerPreventingEscape), whose 4th move is MEAN LOOK."""
    status2 = _read(s, lambda r: r._read_u32(BATTLE_MONS + STATUS2))
    _write(s, 32, BATTLE_MONS + STATUS2, status2 | ESCAPE_PREVENTION)
    _write(s, 8, DISABLE_STRUCTS + 0x14, 1)
    _write(s, 16, FOE + 0x0C + 6, MEAN_LOOK)


def _fire_spin(s):
    """RAM, as test_train_macro's: our battler 0 wrapped for 3 turns by FIRE SPIN
    (gBattleStruct->wrappedMove[0]), what wild TORKOAL did live in MAGMA HIDEOUT."""
    status2 = _read(s, lambda r: r._read_u32(BATTLE_MONS + STATUS2))
    _write(s, 32, BATTLE_MONS + STATUS2, status2 | WRAPPED_3_TURNS)
    _write(s, 16, _read(s, lambda r: r._read_u32(G_BATTLE_STRUCT)) + 4, FIRE_SPIN)


def _record_presses(s):
    """Every button the server's macros press from now on (server _macro_press)."""
    presses = []
    real = s.srv._macro_press

    async def record(buttons, source):
        presses.append(list(buttons))
        await real(buttons, source)

    s.srv._macro_press = record
    return presses, real


def test_run_from_battle_refuses_while_trapped():
    """test_wild_shroomish.state at FIGHT/BAG/POKEMON/RUN; RAM: _mean_look. run_from_battle
    refuses up front with the trap and the best move named: no press, no turn spent (each
    "Can't escape!" was a free hit for the foe). With RUN AWAY (pret
    IsRunningFromBattleImpossible) the same trap still lets it run."""
    s = _fresh(WILD, mode="fight")
    _mean_look(s)
    view = _read(s, lambda r: r.read_battle_view())
    assert view["menu"] == "action", view["menu"]
    turns = _read(s, lambda r: r._read_u8(TURN_COUNTER))
    presses, real = _record_presses(s)
    try:
        r = s.call("mcp_run_from_battle", {"reasoning": "test"})
    finally:
        s.srv._macro_press = real
    print("   ", r.get("error"))
    best = view["best_move"]["name"]
    assert r["success"] is False and r.get("status") == "TRAPPED", r
    assert r["error"].startswith(f"Trapped by MEAN LOOK: you can't run. Fight: use_move(\"{best}\")"), r
    assert presses == [], presses
    assert _in_battle(s) and _read(s, lambda r: r.read_battle_view())["menu"] == "action"
    assert _read(s, lambda r: r._read_u8(TURN_COUNTER)) == turns
    assert view.get("trap") == {"why": "MEAN LOOK", "can_run": False}, view.get("trap")

    s = _fresh(WILD, mode="fight")
    _mean_look(s)
    _write(s, 8, BATTLE_MONS + 0x20, ABILITY_RUN_AWAY)
    r = s.call("mcp_run_from_battle", {"reasoning": "test"})
    assert r["status"] == "ESCAPED" and not _in_battle(s), r


def test_switch_pokemon_refuses_while_trapped():
    """test_wild_two_party.state (TORCHIC out, SHROOMISH benched). pret
    HandleTurnActionSelectionState, B_ACTION_SWITCH: STATUS2_ESCAPE_PREVENTION and
    STATUS2_WRAPPED both block a switch (Gen 3 has no Shed Shell), and RUN AWAY doesn't
    help. Each is refused up front: no press, no turn spent, TORCHIC still out."""
    for trap, why, run_away in ((_mean_look, "MEAN LOOK", False), (_fire_spin, "FIRE SPIN", False), (_mean_look, "MEAN LOOK", True)):
        s = _fresh(WILD_TWO, mode="fight")
        trap(s)
        if run_away:
            _write(s, 8, BATTLE_MONS + 0x20, ABILITY_RUN_AWAY)
        assert _read(s, lambda r: r.read_battle_view())["menu"] == "action"
        turns = _read(s, lambda r: r._read_u8(TURN_COUNTER))
        presses, real = _record_presses(s)
        try:
            r = s.call("mcp_switch_pokemon", {"pokemon": "SHROOMISH", "reasoning": "test"})
        finally:
            s.srv._macro_press = real
        print("   ", r.get("error"))
        assert r["success"] is False and r.get("status") == "TRAPPED", r
        assert r["error"].startswith(f"Trapped by {why}: you can't switch. Fight: use_move(\""), r
        assert presses == [] and _out_name(s) == "TORCHIC", presses
        assert _read(s, lambda r: r._read_u8(TURN_COUNTER)) == turns


def test_no_danger_switch_while_trapped():
    """The harness_step DANGER switch (_danger_switch_rule) never tries a switch the game
    refuses: Isaac's battle in danger (_isaac_danger), then MEAN LOOK or FIRE SPIN in RAM.
    Seen live (2026-11-05, MAGMA HIDEOUT): SHIFT under FIRE SPIN left the party menu open
    ("SABLEYE can't be switched out!")."""
    for trap, why in ((_mean_look, "MEAN LOOK"), (_fire_spin, "FIRE SPIN")):
        s = _fresh(ISAAC)
        assert _isaac_danger(s)["danger"]["switch"], "a teammate wins: the switch would happen untrapped"
        trap(s)
        r = _step(s)
        assert r["changed"] is False and r.get("rule") is None, r
        assert f"trapped by {why}, fighting" in r["why"].split("; "), r
        assert _out_name(s) == "SABLEYE"


# -- HM carriers never lead outside training ---------------------------------------


def _tropius_leads(s):
    """The live order: ELECTRIKE deposited, TROPIUS withdrawn (it lands in slot 6), then
    led with set_lead, as the AI did before Lavaridge."""
    assert s.call("mcp_deposit_pokemon", {"pokemon": "ELECTRIKE", "reasoning": "test"})["success"]
    assert s.call("mcp_withdraw_pokemon", {"pokemon": "TROPIUS", "reasoning": "test"})["success"]
    assert s.call("mcp_set_lead", {"pokemon": "TROPIUS", "reasoning": "test"})["success"]
    assert _names(s)[0] == "TROPIUS", _names(s)


def test_an_hm_carrier_that_doesnt_fight_is_moved_off_the_lead():
    s = _fresh(CENTER, wild_encounters=False)
    _tropius_leads(s)
    # The lowest-level fighter takes the lead, not the highest (this first asserted
    # BRELOOM Lv38; David doesn't want BRELOOM used for everything: "diversify").
    party = _party(s)
    lowest = min((p for p in party if p.species_name.upper() != "TROPIUS" and p.current_hp > 0), key=lambda p: p.level)
    want = lowest.species_name.upper()
    assert want != "BRELOOM", want
    r = _step(s)
    assert r["rule"] == "party_rule" and r["changed"] is True, r
    assert r["summary"].startswith(
        f"Lead set to {want} (Lv{lowest.level}, the lowest level able to battle): TROPIUS carries FLY, CUT, FLASH"
    ), r
    assert _names(s)[0] == want and "TROPIUS" in _names(s), _names(s)
    assert s.srv._pause_events[-1]["kind"] == "party"
    r = _step(s)  # the next step: nothing more to do about it
    assert r["changed"] is False and not r["summary"], r
    assert _names(s)[0] == want


def test_while_training_the_carrier_may_lead():
    """With a training objective current the party rule's own lead holds (the lowest
    level leads, to be trained). RAM: TROPIUS's level (party +0x54) set to 5, so it is the
    lowest and the training rule keeps it in front."""
    s = _fresh(CENTER, wild_encounters=False)
    _tropius_leads(s)
    _write(s, 8, PARTY + 0x54, 5)
    s.srv._training_objective_current = True
    try:
        r = _step(s)
        assert r["changed"] is False and "TROPIUS (lowest level, Lv5) already leads" in r["why"], r
        assert _names(s)[0] == "TROPIUS"
    finally:
        s.srv._training_objective_current = False


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
