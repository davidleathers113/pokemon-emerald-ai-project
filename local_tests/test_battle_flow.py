"""Battle flow and team condition.

Problem 1 (David: "taking 30 seconds between each click"). Seen live in Rustboro Gym:
after each use_move the model spent a separate ~30 s decision pressing A through
"TORCHIC used EMBER!", "It's not very effective…", "Foe GEODUDE fainted!", "DAVID got
$160 for winning!". use_move (and the battle macros, and the agent's pre-decision
auto-advance) now press through battle text that only waits for A and stop at the next
decision: the FIGHT/BAG/POKEMON/RUN menu, the battle's end, or a question.

Problem 2 (David: "It should go to the pokemon center between trainers if the HP or PP
are low"). A team-condition check from memory puts a heal-first line, with the way to
the nearest Pokémon Center 1F, at the front of the story objective and on TRAINERS.

Every outcome is checked from game memory (gMain.inBattle, gBattleOutcome, the battle
controller, gPlayerParty), not from the tool's own words.

Fixtures (see test_macros.py for how they were made):
  test_battle_wurmple.state        Bug Catcher Lyle's only WURMPLE (Lv3, 15/15), move menu
                                   open on EMBER; Torchic Lv11 24/33
  test_wild_shroomish_intro.state  "Wild SHROOMISH appeared!" waiting for A
  test_wild_two_party.state        wild Silcoon at the action menu; Torchic + Shroomish
  test_rustboro_pokecenter.state   RUSTBORO CITY POKEMON CENTER 1F, Torchic 35/35
  test_rustboro_gym.state          RUSTBORO CITY GYM, mid-battle with Josh
  test_switch_question.state       ROUTE 117, Breeder Isaac (6 Pokémon) at the first
                                   FIGHT menu: SABLEYE, COMBUSKEN, BRELOOM vs WHISMUR Lv11.
                                   Made from test_route117_blocked.state; a Repel is
                                   running in it (the walk there used wild_encounters=False)

RAM writes (documented where used): the battle Torchic's HP 1 and poison, so the end of
the turn makes it faint (that reaches "Use next POKéMON?" with a second Pokémon in the
party); party HP for a weak team.

Not covered by a fixture: the learn-a-move question and evolution.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_battle_flow.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "test_battle_wurmple.state"
WILD_INTRO = HERE / "test_wild_shroomish_intro.state"
WILD_TWO = HERE / "test_wild_two_party.state"
POKECENTER = HERE / "test_rustboro_pokecenter.state"
GYM = HERE / "test_rustboro_gym.state"

# pret pokeemerald symbols (US): gPlayerParty (+0x56 HP), gBattleMons (0x58 bytes; +0x28
# HP, +0x4C status1), gBattleOutcome, gBattlerControllerFuncs[0].
PARTY = 0x020244EC
BATTLE_MONS = 0x02024084
BATTLE_OUTCOME = 0x0202433A
STATUS1_POISON = 0x08


def _reader(s):
    return s.srv.env.memory_reader


def _write(s, size, address, value):
    with s.srv.memory_lock:
        memory = s.srv.env.core.memory
        {8: memory.u8, 16: memory.u16, 32: memory.u32}[size][address] = value
        _reader(s)._invalidate_mem_cache()  # the next read must see the write


def _read(s, fn):
    with s.srv.memory_lock:
        return fn(_reader(s))


def _flow(s):
    from server import menu_state as ms

    return _read(s, ms.read_battle_flow)


def _view(s):
    return _read(s, lambda r: r.read_battle_view())


def _in_battle(s):
    return _read(s, lambda r: r.is_in_battle())


def _lyle_beaten(s):
    from utils.mapping.porymap_state import _map_trainers

    trainer_id = next(t[3] for t in _map_trainers("PetalburgWoods") if t[2] == "LYLE")
    return _read(s, lambda r: r._is_event_flag_set(0x500 + trainer_id))


def _in_order(text, parts):
    at = 0
    for part in parts:
        found = text.find(part, at)
        assert found != -1, f"{part!r} missing or out of order in: {text}"
        at = found + len(part)


def _asking(s):
    flow = _flow(s)
    return flow.in_battle and (flow.yes_no_prompt or flow.script_question)


def _faint_torchic_this_turn(s):
    # RAM: the battling Torchic at 1 HP and poisoned, so poison (or the foe's hit)
    # makes it faint before the turn ends.
    _write(s, 16, BATTLE_MONS + 0x28, 1)
    _write(s, 32, BATTLE_MONS + 0x4C, STATUS1_POISON)


# -- use_move plays the turn out ------------------------------------------------


def test_use_move_returns_at_the_battle_end_with_the_turn_quoted():
    """EMBER knocks out Lyle's only WURMPLE: one call, no extra A, battle over."""
    s = server()
    s.load(TRAINER)
    assert not _lyle_beaten(s)
    r = s.call("mcp_use_move", {"move": "EMBER", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "BATTLE_OVER", r
    summary = r["summary"]
    assert summary.startswith("Used EMBER (FIRE, 2x vs BUG). Torchic HP 24/33; opponent WURMPLE HP 15/15"), summary
    # The turn's messages in order (a critical hit may add a line), and nothing from the
    # move menu (PP/TYPE lines) in front of them.
    assert 'What happened: "TORCHIC used EMBER! · ' in summary, summary
    _in_order(
        summary,
        ["It’s super effective!", "Foe WURMPLE fainted!", "TORCHIC gained 34 EXP. Points!",
         "Player defeated BUG CATCHER LYLE!", "DAVID got $48 for winning!"],
    )
    assert "The battle is over (you won)" in summary and "you are on the map at" in summary, summary
    # Memory: the battle is over, won, Lyle's flag is set and the player can walk.
    assert not _in_battle(s)
    assert _read(s, lambda r: r._read_u8(BATTLE_OUTCOME)) == 1  # B_OUTCOME_WON
    assert _lyle_beaten(s)
    assert s.srv._player_can_walk(), "no A should be left to press after use_move"


def test_use_move_returns_at_the_next_action_menu_with_both_hp():
    """GROWL doesn't hurt: WURMPLE attacks back and the next turn's menu opens."""
    s = server()
    s.load(TRAINER)
    r = s.call("mcp_use_move", {"move": "GROWL", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "YOUR_TURN", r
    view = _view(s)
    assert view is not None and view["menu"] == "action", view  # memory: FIGHT/BAG/POKEMON/RUN
    player, foe = view["player"], view["target"]
    summary = r["summary"]
    assert '"TORCHIC used GROWL!' in summary and "Foe WURMPLE used" in summary, summary
    assert f"Now: TORCHIC HP {player['hp']}/{player['max_hp']}; foe WURMPLE HP {foe['hp']}/{foe['max_hp']}." in summary, summary
    assert foe["hp"] == 15, foe
    assert "Your turn: FIGHT/BAG/POKEMON/RUN is open" in summary, summary
    assert "What will" not in summary, summary  # the menu prompt is not a battle message


def test_use_move_while_text_plays_reads_it_first():
    """Called during another turn's text, it presses through to the menu and attacks."""
    s = server()
    s.load(TRAINER)
    # The move cursor is on EMBER (slot 4); UP, A uses GROWL (slot 2) by hand.
    s.call("mcp_press_buttons", {"buttons": ["UP", "A"], "reasoning": "test"})
    assert _view(s)["menu"] == "busy"
    r = s.call("mcp_use_move", {"move": "EMBER", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "BATTLE_OVER", r
    pp = _read(s, lambda r: [r._read_u8(BATTLE_MONS + 0x24 + i) for i in range(4)])
    assert pp[1] == 39 and pp[3] == 24, pp  # GROWL by hand, then EMBER by use_move
    assert _lyle_beaten(s) and not _in_battle(s)


def test_the_counter_to_the_next_trainer_pokemon_is_chosen():
    """David: "think about who the next pokemon is and then send out the pokemon that best
    counters that pokemon." At "Will DAVID change POKéMON?" the trainer has just said who
    comes next; the clear counter in the party is sent in. test_double_battle.state has
    the real team: SABLEYE (out, 22/57), COMBUSKEN, BRELOOM."""
    s = server()
    s.load(HERE / "test_double_battle.state")
    assert s.srv._next_foe_from(["MAY sent out LOMBRE!", "MAY is about to use MARSHTOMP.", "Will DAVID change POKéMON?"]) == "MARSHTOMP"
    assert s.srv._next_foe_from(["Wild ZUBAT appeared!"]) == ""
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        assert reader.species_types("MARSHTOMP") == [11, 4] and reader.species_types("NOT A POKEMON") == []
        assert reader.move_type_and_power("MEGA_DRAIN") == (12, 40)
    choice = s.srv._best_counter("MARSHTOMP")
    print("    MARSHTOMP ->", choice and (choice[0].species_name, choice[2], choice[3]))
    assert choice is not None and choice[0].species_name.upper() == "BRELOOM", choice  # Grass 4x, resists Water and Ground
    slugma = s.srv._best_counter("SLUGMA")
    assert slugma is None or slugma[0].species_name.upper() != "BRELOOM", slugma  # Fire hits Grass 2x
    assert s.srv._best_counter("NOT A POKEMON") is None


def test_will_player_change_pokemon_is_answered_not_handed_back():
    """Seen live on Route 117: use_move returned 'The battle asks a YES/NO question
    ("PkMn BREEDER ISAAC is about to use POOCHYENA. Will DAVID change POKéMON?"): answer
    it yourself'; the model opened the START menu instead. The question is the battle
    script's yesnobox (script_question), not the controller's YES/NO. test_switch_question
    .state is that battle: SABLEYE Lv24 at the first menu vs ISAAC's WHISMUR Lv11 (of 6)."""
    s = server()
    s.load(HERE / "test_switch_question.state")
    best = _view(s)["best_move"]["name"]
    r = s.call("mcp_use_move", {"move": best, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and r.get("end") == "action", r
    summary = r["summary"]
    assert "about to use" in summary and "answer it yourself" not in summary, summary
    assert "kept the same Pokémon in" in summary or "switched to" in summary, summary
    view = _view(s)
    assert view["menu"] == "action" and view["target"]["name"] != "Whismur", view["target"]


def test_double_battle_turns_are_played_through():
    """Seen live: against PLUSLE and MINUN (a double battle) use_move stopped at the target
    menu, and after a target it waited for battler 0 while battler 2 was choosing; both
    foes stayed at 40/40 for seven minutes. test_double_battle.state is SABLEYE with the
    target cursor up and COMBUSKEN beside it."""
    s = server()
    s.load(HERE / "test_double_battle.state")
    knocked_out = False
    for _ in range(8):
        view = s.srv._read_battle_view()
        if view is None:
            break
        best = (view.get("best_move") or {}).get("name") or 1
        r = s.call("mcp_use_move", {"move": best, "reasoning": "test"})
        print("   ", view["player"]["name"], best, "->", r.get("end"))
        assert r["success"] and r.get("end") in ("action", "over"), r
        after = s.srv._read_battle_view()
        knocked_out = knocked_out or after is None or any(o["hp"] == 0 for o in after["opponents"]) or "fainted" in r["summary"]
    assert knocked_out, "turns ran but no foe ever went down"


def test_a_message_long_enough_to_scroll_is_pressed_through():
    """Seen in a train run: TORCHIC's SCRATCH set off the wild SHROOMISH's EFFECT SPORE;
    "...paralyzed TORCHIC! It may be unable to move!" scrolls, its printer sat in
    RENDER_STATE_SCROLL_START (arrow up, waiting for a key), and use_move timed out.
    EFFECT SPORE fires 1 time in 10, so the RNG (gRngValue, RAM only) is set until it does."""
    s = server()
    state = HERE / "test_wild_shroomish.state"
    for seed in range(1, 400, 7):
        s.load(state)
        with s.srv.memory_lock:
            s.srv.env.core.memory.u32[0x03005D80] = seed * 2654435761 & 0xFFFFFFFF
        r = s.call("mcp_use_move", {"move": "SCRATCH", "reasoning": "test"})
        said = str(r.get("summary") or "")  # the turn's messages are quoted here
        if "EFFECT SPORE" in said or r.get("end") == "timeout":
            break
    else:
        raise AssertionError("EFFECT SPORE never fired: no seed reached the case")
    print("    seed", seed, r.get("summary"))
    assert r["end"] in ("action", "over"), r
    assert "EFFECT SPORE" in said, said


def test_question_stops_the_turn_and_is_quoted():
    """With a second Pokémon in the party, a fainted Torchic brings "Use next POKéMON?"."""
    s = server()
    s.load(WILD_TWO)
    _faint_torchic_this_turn(s)
    r = s.call("mcp_use_move", {"move": "GROWL", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "STOPPED", r
    assert "TORCHIC fainted!" in r["summary"], r
    assert 'YES/NO question ("Use next POKéMON?")' in r["summary"], r
    assert _asking(s), _flow(s)  # memory: the script's yesnobox is still up, unanswered


def _to_the_replacement_menu(s):
    """WILD_TWO with Torchic fainted and YES to "Use next POKéMON?": the game's own party
    menu, cursor on the fainted TORCHIC, offering SEND OUT."""
    from server import menu_state as ms

    _faint_torchic_this_turn(s)
    s.call("mcp_use_move", {"move": "GROWL", "reasoning": "test"})
    assert _asking(s), _flow(s)
    s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "test: YES"})
    deadline = time.time() + 10
    while time.time() < deadline and not s.srv._forced_party_menu():
        time.sleep(0.1)
    assert s.srv._forced_party_menu(), _flow(s)
    assert _read(s, ms.party_menu_slot) == 0  # on the fainted TORCHIC, as seen live


def test_switch_pokemon_works_in_the_replacement_menu():
    """Seen live: MARILL fainted in a wild battle; for 30 minutes switch_pokemon said "pick
    one with press_buttons" and A on the fainted one said "no energy left to battle". The
    game's menu offers SEND OUT, not SHIFT."""
    s = server()
    s.load(WILD_TWO)
    _to_the_replacement_menu(s)
    r = s.call("mcp_switch_pokemon", {"pokemon": "SHROOMISH", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "SWITCHED", r
    assert "TORCHIC fainted and SHROOMISH is in battle now" in r["summary"], r
    assert _view(s)["player"]["name"] == "Shroomish", _view(s)["player"]


def test_the_play_out_sends_in_a_replacement_itself():
    """Any Pokémon able to fight is a right answer to the replacement menu, so the
    play-out (use_move, train, ...) sends in the best counter and says so."""
    import asyncio

    s = server()
    s.load(WILD_TWO)
    _to_the_replacement_menu(s)
    played = asyncio.run(s.srv._play_out_battle("test"))
    print("   ", played["end"], played["notes"])
    assert played["end"] == "action", played["end"]
    # The note says why, by the fighter rule (pick_fighter, 2026-11-04); until then it read
    # "(the best of the rest here)". SHROOMISH is the only one left to send in.
    assert len(played["notes"]) == 1 and played["notes"][0].startswith("sent in SHROOMISH for the fainted Pokémon (wins in "), played["notes"]
    assert "Answered for you: sent in SHROOMISH" in s.srv._battle_report(played)
    assert _view(s)["player"]["name"] == "Shroomish"


# -- the agent's auto-advance ----------------------------------------------------


def test_auto_advance_presses_through_battle_text_to_the_menu():
    s = server()
    s.load(WILD_INTRO)
    assert _flow(s).text_waiting
    r = s.call("mcp_auto_advance_dialogue", {})
    print("   ", r)
    assert r["presses"] >= 1 and r["final_status"] == "battle_action", r
    # Only message-window text: not the healthbox name or the "What will TORCHIC do?" prompt.
    assert r["texts"] == ["Wild SHROOMISH appeared!", "Go! TORCHIC!"], r
    assert r["summary"].startswith("Battle text read (auto-advanced with"), r
    assert _view(s)["menu"] == "action"


def test_auto_advance_never_answers_a_battle_yes_no():
    s = server()
    s.load(WILD_TWO)
    _faint_torchic_this_turn(s)
    s.call("mcp_use_move", {"move": "GROWL", "reasoning": "test"})
    assert _asking(s), _flow(s)
    r = s.call("mcp_auto_advance_dialogue", {})
    print("   ", r)
    assert r["presses"] == 0, r
    time.sleep(0.5)
    assert _asking(s), _flow(s)  # still asking


def test_auto_advance_does_nothing_at_the_action_menu():
    s = server()
    s.load(WILD_TWO)
    r = s.call("mcp_auto_advance_dialogue", {})
    assert r["presses"] == 0, r
    assert _view(s)["menu"] == "action"


# -- team condition ----------------------------------------------------------------


def _mon(name, hp, max_hp, moves, pp):
    from pokemon_env.enums import StatusCondition
    from pokemon_env.types import PokemonData

    return PokemonData(
        species_id=0, species_name=name, current_hp=hp, max_hp=max_hp, level=10,
        status=StatusCondition(0), type1=None, type2=None, moves=moves, move_pp=pp, trainer_id=0,
    )


def test_assess_team_condition_rules():
    from pokemon_env.memory_reader import assess_team_condition

    table = {"EMBER": {"power": 40, "pp": 25}, "SCRATCH": {"power": 40, "pp": 35}, "GROWL": {"power": 0, "pp": 40}}
    data = table.get
    moves = ["SCRATCH", "GROWL", "EMBER", "NONE"]
    healthy = assess_team_condition([_mon("Torchic", 18, 35, moves, [30, 40, 20, 0])], data)
    assert healthy == {"weak": False, "reasons": [], "lead": "TORCHIC"}, healthy  # 18/35 is over half
    hurt = assess_team_condition([_mon("Torchic", 17, 35, moves, [30, 40, 20, 0])], data)
    assert hurt["weak"] and hurt["reasons"] == ["TORCHIC 17/35 HP"], hurt
    # Damaging moves all at <= 25% of base PP (GROWL's PP doesn't matter).
    tired = assess_team_condition([_mon("Torchic", 35, 35, moves, [8, 40, 6, 0])], data)
    assert tired["reasons"] == ["SCRATCH 8/35 PP", "EMBER 6/25 PP"], tired
    one_left = assess_team_condition([_mon("Torchic", 35, 35, moves, [9, 0, 6, 0])], data)
    assert not one_left["weak"], one_left  # SCRATCH 9/35 still has uses
    fainted = assess_team_condition(
        [_mon("Torchic", 0, 35, moves, [0, 0, 0, 0]), _mon("Shroomish", 20, 20, ["TACKLE"], [35])],
        {**table, "TACKLE": {"power": 35, "pp": 35}}.get,
    )
    assert fainted["reasons"] == ["TORCHIC fainted"] and fainted["lead"] == "SHROOMISH", fainted


def test_team_condition_from_memory_healthy_then_low():
    s = server()
    s.load(POKECENTER)
    healthy = _read(s, lambda r: r.read_team_condition())
    print("   ", healthy)
    assert healthy["weak"] is False and healthy["lead"] == "TORCHIC", healthy
    _write(s, 16, PARTY + 0x56, 15)  # RAM: Torchic 15/35
    low = _read(s, lambda r: r.read_team_condition())
    assert low["weak"] is True and low["reasons"] == ["TORCHIC 15/35 HP"], low


def _story_battle():
    return {
        "story": {
            "id": "rustboro_033", "description": "Battle Gym Leader Roxanne for Stone Badge",
            "action_type": "battle", "target_location": "Rustboro Gym",
            "navigation_hint": "Roxanne: Geodude Lv12.", "completion_condition": "stone_badge_obtained",
        }
    }


def test_weak_team_line_gives_the_way_to_the_pokemon_center():
    s = server()
    s.load(POKECENTER)
    assert s.call("mcp_go_to_exit", {"exit": "RUSTBORO CITY", "reasoning": "test"})["success"]
    _write(s, 16, PARTY + 0x56, 15)
    guidance = _story_battle()
    line = s.srv._note_team_condition(guidance)
    print("   ", line)
    story = guidance["story"]
    # heal_at_pokecenter walks there itself now (the model was given the way and didn't
    # walk it), so the line names the one call.
    assert story["navigation_hint"].startswith(
        "TEAM IS WEAK (TORCHIC 15/35 HP): heal before any trainer — call heal_at_pokecenter() now: "
        "it walks to RUSTBORO CITY POKEMON CENTER 1F and heals."
    ), story
    assert story["navigation_hint"].endswith("Roxanne: Geodude Lv12."), story
    assert story["description"].startswith("HEAL FIRST (team is weak), then: Battle Gym Leader Roxanne"), story


def test_weak_team_line_inside_the_center_says_heal_now():
    s = server()
    s.load(POKECENTER)
    _write(s, 16, PARTY + 0x56, 15)
    guidance = _story_battle()
    s.srv._note_team_condition(guidance)
    hint = guidance["story"]["navigation_hint"]
    print("   ", hint)
    assert "you are inside RUSTBORO CITY POKEMON CENTER 1F: call heal_at_pokecenter() now." in hint, hint


def test_healthy_team_adds_nothing():
    s = server()
    s.load(POKECENTER)
    guidance = _story_battle()
    assert s.srv._note_team_condition(guidance) is None
    assert guidance == _story_battle()


def test_from_the_gym_the_way_goes_through_rustboro_city():
    """test_rustboro_gym.state is mid-battle; the way is checked from its tile."""
    from server import game_tools

    s = server()
    s.load(GYM)
    snap = s.snapshot()
    way = game_tools.way_to_pokemon_center("RUSTBORO CITY GYM", (snap.x, snap.y))
    print("   ", way)
    assert way == (
        "call heal_at_pokecenter(): it walks to RUSTBORO CITY POKEMON CENTER 1F "
        "(RUSTBORO CITY → RUSTBORO CITY POKEMON CENTER 1F) and heals."
    ), way


def test_no_nagging_mid_battle_but_trainers_line_warns():
    from utils.mapping.porymap_state import _format_trainers

    s = server()
    s.load(GYM)
    _write(s, 16, PARTY + 0x56, 1)  # RAM: party Torchic 1 HP
    assert _in_battle(s)
    guidance = _story_battle()
    assert s.srv._note_team_condition(guidance) is None
    assert guidance == _story_battle()
    trainers = _read(s, lambda r: _format_trainers("RustboroCity_Gym", r))
    print("   ", trainers)
    assert trainers[0].endswith(" (heal first: team is weak)"), trainers


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
