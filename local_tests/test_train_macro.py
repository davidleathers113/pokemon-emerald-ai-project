"""train(): switch-training on wild battles without a model decision per battle.

David (2026-11-03): "Can we speed it up between battles? It sits for a while after the
battle is over." Each switch-training battle cost about three ~30 s model decisions
(walk into the grass, switch_pokemon, use_move); train runs that loop server-side. The
tests check outcomes from game memory (party EXP, level, HP, battle flags), not from the
macro's own report.

Same in-process server harness as test_macros.py (real game loop thread, real savestates).

Fixtures:
  test_train_woods.state           PETALBURG WOODS (9,8) in the tall grass, walking; party
                                   SHROOMISH Lv6 23/23 (lead) + TORCHIC Lv11 33/33. Made
                                   from test_wild_two_party.state: RAM speeds (Torchic 500,
                                   Silcoon 1) for a sure run_from_battle, then the two
                                   party structs swapped in RAM (each 100-byte struct
                                   Pokemon is self-contained) and both set to full HP and
                                   no status, saved with env.save_state
  test_wild_two_party.state        wild Silcoon at the action menu; Torchic out, Shroomish
  test_rustboro_pokecenter.state   RUSTBORO CITY POKEMON CENTER 1F, Torchic alone
  test_battle_wurmple.state        TRAINER battle (Bug Catcher Lyle), Torchic alone
  test_movement_route101.state     ROUTE 101, outdoors
  test_rusturf_tunnel.state        RUSTURF TUNNEL (6,10), a cave; SHROOMISH Lv10 + COMBUSKEN Lv17

RAM writes (documented where used): party HP for a weak team; the trainee's party level
byte one below its EXP's level, so the next EXP gain levels it up (the game recomputes
the level from EXP, pokeemerald CalculateMonStats); the battling Torchic at 1 HP and
poisoned with the wild Pokémon's HP raised, so the turn ends on "Use next POKéMON?";
the wild Pokémon at 1 HP for a sure KO (level-up panel); the wild Pokémon's battle
types, for the advantage rule.

Not covered: the learn-a-move and evolution stops (no fixture is one level-up away
from either; they reuse use_move's question detection), and a trainer spotting the
player mid-walk.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_train_macro.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_macros import BATTLE_MONS, PARTY, POKECENTER, TRAINER, WILD_TWO, _in_battle, _party, _read, _write
from test_movement_feedback import ROUTE101, server

HERE = Path(__file__).resolve().parent
WOODS = HERE / "test_train_woods.state"
TUNNEL = HERE / "test_rusturf_tunnel.state"
# struct Pokemon: +0x50 status, +0x54 level, +0x56 HP; struct BattlePokemon: +0x28 HP,
# +0x2C max HP, +0x4C status1.
STATUS1_POISON = 0x08


def _personalities(s):
    from server import menu_state as ms

    return _read(s, ms.party_personalities)


def _by_name(s, name):
    return next(p for p in _party(s) if p.species_name.upper() == name)


def _asking(s):
    from server import menu_state as ms

    flow = _read(s, ms.read_battle_flow)
    return flow.in_battle and (flow.yes_no_prompt or flow.script_question)


# -- where the grass is ---------------------------------------------------------


def test_encounter_tiles_from_the_map_data():
    from server import game_tools

    s = server()
    s.load(ROUTE101)
    with s.srv.memory_lock:
        spots = game_tools.wild_encounter_spots(s.srv.env)
        grid = game_tools.load_pathfinding_state(s.srv.env)[0]["map"]["porymap"]["grid"]
    print("   ", spots)
    assert spots["pairs"], spots
    for a, b in spots["pairs"]:
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)
        assert grid[a[1]][a[0]] == "~" and grid[b[1]][b[0]] == "~", (a, b)
    # Caves count on their floor; a town, and the Abandoned Ship's corridors (no land
    # Pokémon), don't.
    assert game_tools.map_has_wild_encounters("RusturfTunnel")
    assert game_tools.map_has_wild_encounters("GraniteCave_1F")
    assert game_tools.map_has_wild_encounters("PetalburgWoods")
    assert not game_tools.map_has_wild_encounters("RustboroCity")
    assert not game_tools.map_has_wild_encounters("AbandonedShip_Corridors_1F")


# -- refusals (nothing pressed) -------------------------------------------------


def test_train_refuses_without_grass_and_names_the_way():
    s = server()
    s.load(POKECENTER)
    before = s.snapshot()
    r = s.call("mcp_train", {"trainee": "TORCHIC", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is False and r["stopped_because"] == "no_grass", r
    assert "No tall grass here (RUSTBORO CITY POKEMON CENTER 1F)" in r["error"], r
    assert 'go_to_exit("RUSTBORO CITY")' in r["error"] and "ROUTE 116" in r["error"], r
    after = s.snapshot()
    assert (after.x, after.y) == (before.x, before.y), (before, after)


def test_train_only_picks_grass_it_can_walk_to():
    """Seen live from ROUTE 117 (0,6), a notch walled in on three sides: every train call
    stopped in 0.2 s, "could not get to the tall grass: No path found from (0, 6) to
    (11, 1)". The grid walk offered 6 pairs it could not reach "for the pathfinder to
    try"; all failed. test_route117_notch.state is that tile; test_verdanturf_east.state
    plus go_to_exit("east") lands at (0,8), on the open part of Route 117."""
    from server import game_tools

    s = server()
    s.load(HERE / "test_route117_notch.state", wild_encounters=False)
    with s.srv.memory_lock:
        spots = game_tools.wild_encounter_spots(s.srv.env)
    assert spots["pairs"] == [] and spots["unreachable"] is True, spots
    r = s.call("mcp_train", {"trainee": "MARILL", "until_level": 14, "reasoning": "test"})
    print("   ", r)
    assert r["success"] is False and r["stopped_because"] == "no_grass", r
    assert "ROUTE 117 has tall grass, but none of it can be walked to from (0,6)" in r["error"], r
    assert 'go_to_exit("west")' in r["error"] and "VERDANTURF TOWN → ROUTE 117" in r["error"], r
    s.load(HERE / "test_verdanturf_east.state", wild_encounters=False)
    assert s.call("mcp_go_to_exit", {"exit": "east", "reasoning": "test"})["success"]
    with s.srv.memory_lock:
        spots = game_tools.wild_encounter_spots(s.srv.env)
        state, _, _ = game_tools.load_pathfinding_state(s.srv.env)
    assert spots["pairs"] and not spots["unreachable"], spots
    snap = s.snapshot()
    ox, oy = spots["coord_offset"] or (0, 0)
    start = (snap.x - ox, snap.y - oy)
    for tile, _ in spots["pairs"]:
        assert game_tools.pathfinder.find_path(start, tile, state, consider_npcs=True, allow_partial=False), tile


def test_train_refuses_a_weak_team_and_says_where_to_heal():
    s = server()
    s.load(WOODS)
    before = s.snapshot()
    # RAM: TORCHIC (slot 2) down to 10/33, under half.
    _write(s, 16, PARTY + 100 + 0x56, 10)
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is False and r["stopped_because"] == "team_weak", r
    assert "the team is weak (TORCHIC 10/33 HP)" in r["error"] and "heal_at_pokecenter()" in r["error"], r
    after = s.snapshot()
    assert (after.x, after.y) == (before.x, before.y) and not _in_battle(s), (before, after)


def test_train_refuses_bad_arguments():
    s = server()
    s.load(WOODS)
    r = s.call("mcp_train", {"trainee": "LOTAD"})
    assert r["success"] is False and "LOTAD is not in your party" in r["error"] and "SHROOMISH Lv6" in r["error"], r
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "until_level": 6})
    assert r["success"] is False and "already Lv6" in r["error"], r
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "partner": "SHROOMISH"})
    assert r["success"] is False and "must not fight" in r["error"], r
    r = s.call("mcp_train", {"trainee": "TORCHIC", "partner": "LOTAD"})
    assert r["success"] is False and "LOTAD is not in your party" in r["error"], r


# -- grinding -------------------------------------------------------------------


def test_train_switch_trains_wild_battles_until_the_time_limit():
    """Walks to the grass pair, meets wild Pokémon, switches SHROOMISH out on turn 1 for
    TORCHIC, wins, and stops on its one-minute limit."""
    s = server()
    s.load(WOODS)
    shroomish_before = _by_name(s, "SHROOMISH")
    started = time.time()
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 1, "reasoning": "test"})
    took = time.time() - started
    print("   ", r["summary"])
    for battle in r["battles"]:
        print("      ", battle)
    assert r["success"] is True, r
    battles = r["battles"]
    assert battles, r
    # A learn-move question can end a run when SHROOMISH levels up; anything else
    # stops on the clock or the team's condition.
    # Encounters are random, so any of the designed stops can end a run: the partner
    # dropping under half HP ("no_partner") was seen on a re-run.
    assert r["stopped_because"] in ("time", "team_weak", "learn_move", "no_partner"), r
    assert took < 60 + 90 + 60, took  # the minute, the battle it was in, one last turn
    for battle in battles:
        assert battle["exp_gained"] > 0, battle
        if battle["trainee_fought"]:
            continue  # a foe weak to SHROOMISH (see the advantage tests); none live in the woods
        assert battle["switched"] is True and battle["partner"] == "TORCHIC", battle
        # The trainee never fought: still at full HP (it started full; a level-up raises
        # both numbers), and still EXP for having been sent out.
        assert battle["trainee_hp"] == battle["trainee_max_hp"] and battle["trainee_hp"] >= battle["trainee_hp_before"], battle
    assert all(b["outcome"] == "won" for b in battles[:-1]), battles
    shroomish_after = _by_name(s, "SHROOMISH")
    gained = shroomish_after.experience - shroomish_before.experience
    assert gained == sum(b["exp_gained"] for b in battles) and gained > 0, (gained, battles)
    assert _party(s)[0].species_name.upper() == "SHROOMISH", _party(s)  # still leads the next battle
    if r["stopped_because"] == "time":
        assert not _in_battle(s) and s.srv._player_can_walk(), r
    per_minute = len(battles) / (took / 60)
    print(f"    {len(battles)} battles, {r['steps']} steps in {took:.0f} s: {per_minute:.1f} battles/min")


def test_train_stops_when_the_trainee_reaches_until_level():
    s = server()
    s.load(WOODS)
    real = _by_name(s, "SHROOMISH").level
    # RAM: SHROOMISH's party level byte one lower than its EXP says; the next EXP gain
    # levels it back up (CalculateMonStats recomputes the level from EXP).
    _write(s, 8, PARTY + 0x54, real - 1)
    assert _by_name(s, "SHROOMISH").level == real - 1
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "until_level": real, "max_minutes": 3, "reasoning": "test"})
    print("   ", r["summary"])
    assert r["success"] is True and r["stopped_because"] == "until_level", r
    assert f"SHROOMISH reached Lv{real}" in r["summary"], r
    last = r["battles"][-1]
    assert (last["level_before"], last["level_after"]) == (real - 1, real), last
    assert _by_name(s, "SHROOMISH").level == real
    assert not _in_battle(s) and s.srv._player_can_walk()


def test_train_stops_at_a_battle_question():
    """Called mid-battle with the partner already out: it fights, and the partner fainting
    brings "Use next POKéMON?", which is the model's to answer."""
    s = server()
    s.load(WILD_TWO)
    # RAM: the battling TORCHIC at 1 HP and poisoned (poison makes it faint at the end of
    # the turn), and the wild SILCOON at 500/500 HP so EMBER can't end the battle first.
    _write(s, 16, BATTLE_MONS + 0x28, 1)
    _write(s, 32, BATTLE_MONS + 0x4C, STATUS1_POISON)
    _write(s, 16, BATTLE_MONS + 0x58 + 0x28, 500)
    _write(s, 16, BATTLE_MONS + 0x58 + 0x2C, 500)
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "reasoning": "test"})
    print("   ", r["summary"])
    assert r["success"] is True and r["stopped_because"] == "question", r
    assert "Use next POKéMON?" in r["summary"], r
    assert r["battles"][0]["outcome"] == "unfinished" and r["battles"][0]["switched"] is False, r
    assert _asking(s), "the question must still be up, unanswered"


def test_trainer_battle_is_not_fought():
    s = server()
    s.load(TRAINER)
    view_before = _read(s, lambda r: r.read_battle_view())
    # TORCHIC is alone: train refuses for lack of a partner before anything else.
    r = s.call("mcp_train", {"trainee": "TORCHIC", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is False and r["stopped_because"] == "no_partner", r
    # The battle step itself stops on a trainer battle without pressing anything.
    torchic = _personalities(s)[0]
    record, stop = asyncio.run(s.srv._train_battle(torchic, None, time.monotonic() + 60))
    print("   ", stop)
    assert stop[0] == "trainer_battle" and "WURMPLE" in stop[1], stop
    view = _read(s, lambda r: r.read_battle_view())
    assert (view["menu"], view["move_cursor_slot"]) == (view_before["menu"], view_before["move_cursor_slot"]), view


# -- the objective text points at train ------------------------------------------


def test_training_objective_recommends_train():
    s = server()
    s.load(WOODS)
    guidance = {"story": {"id": "t", "completion_condition": "level_shroomish_12"}}
    s.srv._note_conditions_met_in_memory(guidance)
    condition = guidance["story"]["completion_condition"]
    print("   ", condition)
    assert condition.startswith("level_shroomish_12: not yet (SHROOMISH is Lv6 of 12 needed).")
    assert 'train("SHROOMISH", until_level=12)' in condition, condition
    line = s.srv._switch_train_line(guidance)
    print("   ", line)
    assert line.startswith('TRAINING: call train("SHROOMISH", until_level=12)'), line
    # A Pokémon not in the party yet: catch one at that level (train stops for it).
    absent = {"story": {"id": "t", "completion_condition": "level_lotad_12"}}
    s.srv._note_conditions_met_in_memory(absent)
    text = absent["story"]["completion_condition"]
    assert "until_level" not in text, text
    assert 'catch="LOTAD", catch_min_level=12' in text, text
    assert s.srv._switch_train_line(absent) == ""
    # While the team is weak the heal-first line is the one to follow.
    _write(s, 16, PARTY + 100 + 0x56, 10)  # RAM: TORCHIC 10/33
    assert s.srv._switch_train_line(guidance) == ""


def test_train_in_a_cave_steps_on_open_floor():
    """Rusturf Tunnel: every floor tile can start a battle. Seen live: (6,10)→(6,11) was
    drawn as floor but its collision bit blocks it; the grid now shows it as a wall, and
    train must only use pairs it can actually step between."""
    from server import game_tools

    s = server()
    s.load(TUNNEL)
    with s.srv.memory_lock:
        spots = game_tools.wild_encounter_spots(s.srv.env)
    print("   ", spots["pairs"])
    assert spots["pairs"] and all((6, 11) not in pair for pair in spots["pairs"]), spots
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 0.75, "reasoning": "test"})
    print("   ", r["summary"])
    assert r["success"] is True and r["steps"] > 0, r
    assert r["stopped_because"] in ("time", "team_weak", "learn_move", "evolution"), r
    assert r["battles"], r  # Rusturf's encounter rate gives several in 45 s of stepping
    first = r["battles"][0]
    assert first["exp_gained"] > 0 and (first["switched"] or first["trainee_fought"]), first
    assert s.snapshot().map_id == (24, 4)  # still in the tunnel


# -- the level-up stat panel --------------------------------------------------------


def test_level_up_panel_is_pressed_through():
    """Seen live: after "SHROOMISH grew to LV. 10!" use_move and switch_pokemon ran out
    their 80 s budget: the stat panel (drawlvlupbox) waits for A without a text printer."""
    s = server()
    s.load(WILD_TWO)
    torchic = _by_name(s, "TORCHIC")
    # RAM: the wild SILCOON at 1 HP, so SCRATCH wins; TORCHIC's 50 EXP (972 → 1022)
    # crosses Lv12, a real level-up from EXP.
    _write(s, 16, BATTLE_MONS + 0x58 + 0x28, 1)
    started = time.time()
    r = s.call("mcp_use_move", {"move": "SCRATCH", "reasoning": "test"})
    took = time.time() - started
    print(f"    {took:.1f} s:", r["summary"])
    assert r["status"] == "BATTLE_OVER", r
    assert "TORCHIC grew to LV. 12!" in r["summary"], r
    assert took < 30, took  # was 80+ s (the whole frame budget) before
    assert _by_name(s, "TORCHIC").level == torchic.level + 1
    assert not _in_battle(s) and s.srv._player_can_walk()


# -- the trainee fights a foe it beats --------------------------------------------
# Types are pokeemerald TYPE_* ids, written into the wild Pokémon's gBattleMons types
# (+0x21, +0x22); damage and the battle view use those, not the species.
TYPE_NORMAL, TYPE_FLYING, TYPE_GROUND, TYPE_ROCK, TYPE_FIRE = 0, 2, 4, 5, 10


def _shroomish_out_against(s, types):
    s.load(WILD_TWO)
    r = s.call("mcp_switch_pokemon", {"pokemon": "SHROOMISH", "reasoning": "test"})
    assert r["success"] and r["end"] == "action", r
    _write(s, 8, BATTLE_MONS + 0x58 + 0x21, types[0])
    _write(s, 8, BATTLE_MONS + 0x58 + 0x22, types[1])


def test_trainee_with_the_advantage_is_told_to_fight():
    """David: "You don't need to switch train when the other pokemon is weak against shroomish"."""
    s = server()
    guidance = {"story": {"completion_condition": "level_shroomish_12: not yet (SHROOMISH is Lv6 of 12 needed)."}}
    _shroomish_out_against(s, (TYPE_ROCK, TYPE_GROUND))  # a GEODUDE's types: ABSORB 4x, nothing hits Grass 2x
    line = s.srv._switch_train_line(guidance)
    print("   ", line)
    assert line == (
        'SHROOMISH can beat SILCOON on its own (ABSORB 4x, no threat to it): fight with use_move("ABSORB") — no switch needed.'
    ), line
    _shroomish_out_against(s, (TYPE_FLYING, TYPE_FLYING))  # a TAILLOW's type: Flying hits Grass 2x
    line = s.srv._switch_train_line(guidance)
    print("   ", line)
    assert line.startswith('SWITCH-TRAINING: SHROOMISH') and 'switch_pokemon("TORCHIC")' in line, line
    # A WHISMUR's type, neutral both ways: it fights on its own too. David (2026-11-03):
    # "The pokemon are strong enough to fight on their own." (Before, this switched.)
    _shroomish_out_against(s, (TYPE_NORMAL, TYPE_NORMAL))
    line = s.srv._switch_train_line(guidance)
    assert line.startswith("SHROOMISH can beat SILCOON on its own (") and "1x" in line, line
    # A foe resisting its best move still brings the partner in: Fire takes Grass at 0.5x.
    _shroomish_out_against(s, (TYPE_FIRE, TYPE_FIRE))
    assert s.srv._switch_train_line(guidance).startswith("SWITCH-TRAINING: SHROOMISH")
    # Under half HP it switches even with the advantage. RAM: SHROOMISH (battler 0) 11/23.
    _shroomish_out_against(s, (TYPE_ROCK, TYPE_GROUND))
    _write(s, 16, BATTLE_MONS + 0x28, 11)
    assert s.srv._switch_train_line(guidance).startswith("SWITCH-TRAINING: SHROOMISH")


def test_train_lets_the_trainee_win_with_the_advantage():
    s = server()
    _shroomish_out_against(s, (TYPE_ROCK, TYPE_GROUND))
    exp_before = _by_name(s, "SHROOMISH").experience
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 0.2, "reasoning": "test"})
    print("   ", r["summary"])
    first = r["battles"][0]
    print("      ", first)
    assert first["trainee_fought"] is True and first["switched"] is False, first
    assert first["outcome"] == "won" and first["exp_gained"] > 0, first
    assert _by_name(s, "SHROOMISH").experience > exp_before


def test_trainee_not_leading_still_gets_the_exp():
    """Seen live: SABLEYE led, train("SHROOMISH") had SABLEYE fight every battle, and only
    SABLEYE levelled (only Pokémon that were out share the EXP). test_wild_two_party.state
    has TORCHIC out and SHROOMISH on the bench."""
    s = server()
    s.load(WILD_TWO)
    exp_before = _by_name(s, "SHROOMISH").experience
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 0.2, "reasoning": "test"})
    print("   ", r["summary"])
    first = r["battles"][0]
    print("      ", first)
    assert first.get("trainee_sent_in") is True and first["switched"] is True, first
    assert first["outcome"] == "won" and first["exp_gained"] > 0, first
    assert _by_name(s, "SHROOMISH").experience > exp_before


def test_train_switches_against_a_threat():
    s = server()
    _shroomish_out_against(s, (TYPE_FLYING, TYPE_FLYING))
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 0.2, "reasoning": "test"})
    print("   ", r["summary"])
    first = r["battles"][0]
    assert first["switched"] is True and first["trainee_fought"] is False and first["partner"] == "TORCHIC", first


def test_train_stops_for_the_pokemon_to_catch():
    """David (2026-11-03): "I want it to catch a level 10+ Sableye." train hunts the
    encounters and hands the battle back, before any move, when the wanted one appears;
    one below catch_min_level is fought as usual."""
    s = server()
    s.load(WILD_TWO)  # wild Silcoon at the action menu
    foe = s.srv._read_battle_view()["opponents"][0]
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "catch": "silcoon", "catch_min_level": foe["level"], "reasoning": "test"})
    print("   ", r["summary"])
    assert r["stopped_because"] == "catch_target", r
    assert f"SILCOON Lv{foe['level']}" in r["summary"] and "throw_ball(" in r["summary"], r["summary"]
    assert _in_battle(s) and r["battles"][0]["turns"] == 0 and not r["battles"][0]["switched"], r["battles"]

    s.load(WILD_TWO)
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "catch": "SILCOON", "catch_min_level": foe["level"] + 1,
                             "max_minutes": 0.2, "reasoning": "test"})
    print("   ", r["summary"])
    assert r["stopped_because"] != "catch_target" and r["battles"][0]["turns"] >= 1, r  # fought as usual


def test_agent_registers_train_and_outwaits_its_time_limit():
    """The agent's HTTP request must outlive max_minutes plus the battle train finishes,
    or the agent gives up on a call that is still pressing buttons."""
    import importlib

    from agents.tools.registry import build_tools_for_scaffold

    pa = importlib.import_module("agents.PokeAgent")  # the module, not the class
    s = server()
    assert "train" in pa.MACRO_TOOLS and "train" in pa.PRESSING_TOOLS
    assert "train" in {t["name"] for t in build_tools_for_scaffold("simple")}
    assert (pa.TRAIN_DEFAULT_MINUTES, pa.TRAIN_MAX_MINUTES) == (s.srv.TRAIN_DEFAULT_MINUTES, s.srv.TRAIN_MAX_MINUTES)
    finish = s.srv.TRAIN_FINISH_BATTLE_S
    assert pa.tool_timeout_s("train", {}) > s.srv.TRAIN_DEFAULT_MINUTES * 60 + finish
    assert pa.tool_timeout_s("train", {"max_minutes": 2}) > 2 * 60 + finish
    # Capped like the server caps max_minutes.
    assert pa.tool_timeout_s("train", {"max_minutes": 99}) == pa.tool_timeout_s("train", {"max_minutes": s.srv.TRAIN_MAX_MINUTES})
    assert pa.tool_timeout_s("use_move", {}) == pa.PRESSING_TOOL_TIMEOUT_S


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
