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


def _no_medicine(s):
    """RAM: every medicine in the ITEMS pocket down to 0 (its quantity XOR'd with the
    save's key, as the game stores it), so train can only heal at a Pokémon Center."""
    from server import app
    from server import menu_state as ms

    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        slots = reader._read_u32(ms.G_BAG_POCKETS)
        capacity = reader._read_u8(ms.G_BAG_POCKETS + 4)
        key = reader._get_security_key() & 0xFFFF
        medicine = [i for i in range(capacity) if reader._read_u16(slots + 4 * i) in app.MEDICINE]
    for i in medicine:
        _write(s, 16, slots + 4 * i + 2, key)


def _put_in_bag(s, item_id, quantity):
    from test_party_rule import _put_in_bag as put

    put(s, item_id, quantity)


def test_train_heals_from_the_bag_before_walking_to_a_pokemon_center():
    """emerald-new (2026-11-04), David's "Potions + Jagged Pass": train stopped with "the
    team is weak (AZUMARILL 32/82 HP)" and walked to Lavaridge below Jagged Pass's one-way
    ledges, a 10-15 minute loop. With medicine in the bag it heals on the spot and trains
    on. Same place as the next test; RAM: COMBUSKEN (slot 2) at 10 HP, no medicine but 3
    SUPER POTIONs (id 22)."""
    s = server()
    s.load(HERE / "test_verdanturf_east.state")
    assert s.call("mcp_go_to_exit", {"exit": "east", "reasoning": "test"})["success"]
    _no_medicine(s)
    _put_in_bag(s, 22, 3)
    _write(s, 16, PARTY + 100 + 0x56, 10)
    top = _by_name(s, "COMBUSKEN").max_hp
    r = s.call("mcp_train", {"trainee": "MARILL", "max_minutes": 1, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True, r
    want = f"On the way: used SUPER POTION on COMBUSKEN (10→{min(top, 60)} HP)"
    assert r["summary"].startswith(want), r["summary"]
    assert "healed at" not in r["summary"] and r["stopped_because"] != "team_weak", r
    assert "SUPER POTIONs left" in r["summary"] or "SUPER POTION left" in r["summary"], r


def test_train_heals_a_weak_team_itself_and_comes_back():
    """Seen live: train refused three times in a row ("the team is weak … call
    heal_at_pokecenter()"), the model alternated train and heal, and the stall watcher
    paused it. Now train makes one heal trip itself and comes back. (Until 2026-11-04 this
    test asserted the refusal.) RAM: TORCHIC (slot 2) down to 10/33, under half. Since
    train heals from the bag first (2026-11-04), the bag's medicine is emptied here: this
    is the trip it makes when nothing in the bag helps."""
    s = server()
    # ROUTE 117 (0,8), next to VERDANTURF TOWN: test_verdanturf_east.state then east. (From
    # test_marill_in_party.state's (18,14) the heal walk can cross the sight of an unbeaten
    # trainer at (8,12), depending on where he faces.) RAM: COMBUSKEN (slot 2) at 10 HP.
    s.load(HERE / "test_verdanturf_east.state")
    assert s.call("mcp_go_to_exit", {"exit": "east", "reasoning": "test"})["success"]
    _no_medicine(s)
    _write(s, 16, PARTY + 100 + 0x56, 10)
    r = s.call("mcp_train", {"trainee": "MARILL", "max_minutes": 1, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True, r
    assert r["summary"].startswith("On the way: healed at ") and "came back to ROUTE 117" in r["summary"], r
    # Training went on after the heal, not "team weak" again. (What stops it next varies:
    # the fixture carries a Repel from how it was made, and unbeaten Route 117 trainers
    # may spot the player on the way to the grass, depending on where they face.)
    assert r["stopped_because"] != "team_weak", r
    combusken = _by_name(s, "COMBUSKEN")
    assert combusken.current_hp * 2 > combusken.max_hp, combusken  # healed (a battle since may cost a little)
    # When the heal trip itself is stopped (from PETALBURG WOODS a ROUTE 104 trainer spots
    # the player on the way), train hands back and says why.
    s.load(WOODS)
    _no_medicine(s)
    _write(s, 16, PARTY + 100 + 0x56, 10)
    r = s.call("mcp_train", {"trainee": "SHROOMISH", "reasoning": "test"})
    print("   ", r.get("error"))
    assert r["success"] is False and r["stopped_because"] == "team_weak", r
    assert "tried to heal first: Did not heal: a trainer battle started on the way" in r["error"], r


def test_train_refuses_a_weak_team_in_a_battle():
    """In a battle there is no heal trip to make: the old refusal stands. RAM: in
    test_wild_two_party.state's wild battle, SHROOMISH (the partner for a TORCHIC trainee)
    down to 5/23."""
    s = server()
    s.load(WILD_TWO)
    _write(s, 16, PARTY + 100 + 0x56, 5)
    r = s.call("mcp_train", {"trainee": "TORCHIC", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is False and r["stopped_because"] == "team_weak", r
    assert "the team is weak" in r["error"] and "heal_at_pokecenter()" in r["error"], r


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


def test_a_hint_that_names_train_says_it_walks_itself():
    """Seen live: a hint said "In Route 112 grass call train(…)" and the model tried
    navigate_to into the grass first."""
    s = server()
    note = s.srv.TRAIN_WALKS_NOTE
    guidance = {
        "story": {"description": "Train Numel", "navigation_hint": 'In Route 112 grass call train("NUMEL", until_level=19).'},
        "battling_group": [{"description": "Beat the Route 112 trainers", "navigation_hint": "Walk north."}],
    }
    s.srv._note_train_walks_itself(guidance)
    assert guidance["story"]["navigation_hint"].endswith(note), guidance["story"]
    assert note not in guidance["battling_group"][0]["navigation_hint"], guidance["battling_group"][0]
    s.srv._note_train_walks_itself(guidance)  # never twice
    assert guidance["story"]["navigation_hint"].count(note) == 1, guidance["story"]


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
    """David: "You don't need to switch train when the other pokemon is weak against
    shroomish". Since 2026-11-04 "the advantage" is the damage race (estimated HP): the
    trainee fights alone when it knocks the foe out in at most 2 hits and sooner than the
    foe could knock it out. Seen live, the level/type rule it replaces let a Lv13 MARILL
    fight Lv13-14 wilds alone, and it fainted. SILCOON knows only HARDEN; foe moves below
    are RAM writes into gBattleMons[1].moves."""
    from pokemon_env.enums import Move

    s = server()
    guidance = {"story": {"completion_condition": "level_shroomish_12: not yet (SHROOMISH is Lv6 of 12 needed)."}}
    _shroomish_out_against(s, (TYPE_ROCK, TYPE_GROUND))  # GEODUDE's types: ABSORB 4x, ~17 HP of 20
    line = s.srv._switch_train_line(guidance)
    print("   ", line)
    assert line == (
        'SHROOMISH can beat SILCOON on its own (ABSORB 4x, ~17 HP a hit vs 20 HP left): '
        'fight with use_move("ABSORB") — no switch needed.'
    ), line
    # Neutral both ways, ABSORB does ~4 of 20 HP: five turns alone, so the partner (one
    # hit) is faster. (The old rule let it fight; David asked for faster training.)
    _shroomish_out_against(s, (TYPE_NORMAL, TYPE_NORMAL))
    assert s.srv._switch_train_line(guidance).startswith("SWITCH-TRAINING: SHROOMISH")
    # A foe that hits back: Flying with GUST (2x on Grass).
    _shroomish_out_against(s, (TYPE_FLYING, TYPE_FLYING))
    _write(s, 16, BATTLE_MONS + 0x58 + 0x0C, Move["GUST"].value)
    _write(s, 8, BATTLE_MONS + 0x58 + 0x24, 20)
    line = s.srv._switch_train_line(guidance)
    assert line.startswith('SWITCH-TRAINING: SHROOMISH') and 'switch_pokemon("TORCHIC")' in line, line
    # Two hits to win, but the foe's ROCK THROW knocks a 6-HP SHROOMISH out in two as well:
    # the faster side lands its second hit first. (Until 2026-11-04 a tie always switched;
    # seen live, AZUMARILL then left NUMEL, 4x weak to it, to BRELOOM.) RAM: struct
    # BattlePokemon speed (+6).
    _shroomish_out_against(s, (TYPE_ROCK, TYPE_GROUND))
    _write(s, 16, BATTLE_MONS + 0x58 + 0x0C, Move["ROCK_THROW"].value)
    _write(s, 8, BATTLE_MONS + 0x58 + 0x24, 15)
    _write(s, 16, BATTLE_MONS + 0x28, 6)
    _write(s, 16, BATTLE_MONS + 6, 10)
    _write(s, 16, BATTLE_MONS + 0x58 + 6, 50)
    assert s.srv._switch_train_line(guidance).startswith("SWITCH-TRAINING: SHROOMISH")
    _write(s, 16, BATTLE_MONS + 0x58 + 6, 5)
    line = s.srv._switch_train_line(guidance)
    assert line.startswith("SHROOMISH can beat SILCOON on its own"), line


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


def test_train_makes_the_trainee_lead_before_its_battles():
    """Seen live (2026-11-04, train("AZUMARILL") on Jagged Pass with NUMEL leading by the
    party rule): every battle switched AZUMARILL in, the foe hit it on the switch (82→65
    →50→32 HP, then "the team is weak"), and NUMEL took a share of the EXP. train now
    makes the trainee the lead first. test_train_woods.state: SHROOMISH leads, TORCHIC 2nd."""
    from server import menu_state as ms

    s = server()
    s.load(WOODS)
    assert _party(s)[0].species_name.upper() == "SHROOMISH"
    r = s.call("mcp_train", {"trainee": "TORCHIC", "max_minutes": 0.2, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"], r
    assert "made TORCHIC the lead" in r["summary"], r["summary"]
    assert _party(s)[0].species_name.upper() == "TORCHIC"
    for battle in r["battles"]:
        assert not battle.get("trainee_sent_in"), battle  # it started out, nobody switched it in
    # Holding the Exp. Share it earns from the bench: the lead is left alone. RAM: the
    # trainee's held item (struct Pokemon, encrypted substruct0) is written through the
    # give_item macro instead, from the bag (test_party_rule._put_in_bag).
    from test_party_rule import _put_in_bag

    s.load(WOODS)
    _put_in_bag(s, ms.ITEM_EXP_SHARE)
    assert s.call("mcp_give_item", {"item": "EXP SHARE", "pokemon": "TORCHIC", "reasoning": "test"})["success"]
    r = s.call("mcp_train", {"trainee": "TORCHIC", "max_minutes": 0.1, "reasoning": "test"})
    assert "made TORCHIC the lead" not in (r.get("summary") or ""), r
    assert _party(s)[0].species_name.upper() == "SHROOMISH"


def test_train_lowest_follows_the_party_rule_battle_by_battle():
    """train("LOWEST"): David (2026-11-04) "the lowest-level Pokémon is the one trained";
    the trainee is maintain_party's pick before every battle, and until_level means all
    of them. test_jagged_pass_top.state (live): NUMEL 19, AZUMARILL 20, ELECTRIKE 20,
    SABLEYE 26, COMBUSKEN 29, BRELOOM 38 (fainted), at the grass. RAM: every HP full;
    NUMEL's and AZUMARILL's level bytes at 18, under their EXP's levels (19, 20), so the
    first EXP either gains puts it back at its real level (the game recomputes the level
    from EXP) and the other one becomes the lowest."""
    s = server()
    s.load(HERE / "test_jagged_pass_top.state", wild_encounters=True)
    for i, mon in enumerate(_party(s)):
        _write(s, 16, PARTY + i * 100 + 0x56, mon.max_hp)
    r = s.call("mcp_train", {"trainee": "LOWEST", "until_level": 19, "reasoning": "test"})
    assert r["success"] is False and "Every Pokémon able to battle is Lv19 or higher already" in r["error"], r
    numel, azumarill = (next(i for i, p in enumerate(_party(s)) if p.species_name.upper() == n) for n in ("NUMEL", "AZUMARILL"))
    _write(s, 8, PARTY + numel * 100 + 0x54, 18)
    _write(s, 8, PARTY + azumarill * 100 + 0x54, 18)
    r = s.call("mcp_train", {"trainee": "LOWEST", "until_level": 19, "max_minutes": 4, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    for battle in r.get("battles", []):
        print("      ", battle)
    assert r["success"], r
    trainees = [b["trainee"] for b in r["battles"]]
    assert set(trainees[:2]) == {"NUMEL", "AZUMARILL"}, trainees  # one battle each, then the other
    assert f"{trainees[0]} then {trainees[1]} trained" in r["summary"], r["summary"]
    assert r["stopped_because"] == "until_level", r
    assert "every Pokémon able to battle is Lv19 or higher" in r["summary"], r["summary"]
    assert "Levels: " in r["summary"] and "NUMEL Lv18→19" in r["summary"] and "AZUMARILL Lv18→" in r["summary"], r["summary"]
    assert _party(s)[0].species_name.upper() in ("NUMEL", "AZUMARILL")


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


def test_a_trainee_holding_the_exp_share_stays_on_the_bench():
    """David (2026-11-04): once we have the Exp. Share the weakest holds it, and it levels
    "regardless of who's doing the fighting". train no longer brings such a trainee in.
    test_wild_two_party.state (TORCHIC out, SHROOMISH benched), with SHROOMISH read as
    holding the Exp. Share (the party read is wrapped: its held item is in the encrypted
    part of the Pokémon data)."""
    import dataclasses

    from server import menu_state as ms

    s = server()
    s.load(WILD_TWO)
    real = s.srv._train_party

    def holding():
        return [(pid, dataclasses.replace(mon, held_item=ms.ITEM_EXP_SHARE) if mon.species_name.upper() == "SHROOMISH" else mon)
                for pid, mon in real()]

    s.srv._train_party = holding
    try:
        r = s.call("mcp_train", {"trainee": "SHROOMISH", "max_minutes": 0.2, "reasoning": "test"})
    finally:
        s.srv._train_party = real
    print("   ", r["summary"])
    first = r["battles"][0]
    print("      ", first)
    assert not first.get("trainee_sent_in") and first["switched"] is False, first


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
