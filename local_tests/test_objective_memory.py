"""An objective whose completion condition the game has already recorded says so.

Seen live: Birch handed over the POKéDEX inside a cutscene that dialogue auto-advance
read through; the model never saw it and pressed A at May for minutes to "receive the
Pokedex". test_objective_pokedex.state is that moment (FLAG_SYS_POKEDEX_GET set, no
badges).

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_objective_memory.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

POKEDEX_STATE = Path(__file__).resolve().parent / "test_objective_pokedex.state"


def _guidance():
    return {
        "story": {"id": "early_016", "completion_condition": "pokedex_received"},
        "battling_group": [{"id": "battle_009", "completion_condition": "stone_badge_obtained"}],
        "dynamics": {"id": "dyn_000", "completion_condition": "returned_to_lab"},
    }


def test_condition_already_true_names_the_call():
    s = server()
    s.load(POKEDEX_STATE)
    guidance = _guidance()
    s.srv._note_conditions_met_in_memory(guidance)
    story = guidance["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("pokedex_received: ALREADY TRUE in game memory"), story
    assert 'complete_direct_objective(objective_id="early_016")' in story, story


def test_condition_not_yet_true_is_unchanged():
    s = server()
    s.load(POKEDEX_STATE)
    guidance = _guidance()
    s.srv._note_conditions_met_in_memory(guidance)
    assert guidance["battling_group"][0]["completion_condition"] == "stone_badge_obtained"  # no badge yet
    assert guidance["dynamics"]["completion_condition"] == "returned_to_lab"  # no flag known for it


AFTER_WOODS_GRUNT = Path(__file__).resolve().parent / "test_after_woods_grunt.state"


def test_story_battle_already_won_says_so():
    """Seen live: petalburg_026 "Battle Team Aqua Grunt" stayed open after the win, and the
    model drifted back to the walkthrough's opening chapter. The fixture is that moment."""
    s = server()
    s.load(AFTER_WOODS_GRUNT)
    guidance = {
        "story": {"id": "petalburg_026", "completion_condition": "aqua_grunt_woods_defeated"},
        "battling_group": [
            {"id": "early_014", "completion_condition": "rival_battle_1_won"},  # one of six variants set
            {"id": "rustboro_038", "completion_condition": "aqua_grunt_tunnel_defeated"},
            {"id": "mossdeep_157", "completion_condition": "maxie_tabitha_defeated"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    story = guidance["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("aqua_grunt_woods_defeated: ALREADY TRUE in game memory"), story
    assert 'complete_direct_objective(objective_id="petalburg_026")' in story, story
    group = [g["completion_condition"] for g in guidance["battling_group"]]
    assert "your rival on Route 103" in group[0], group
    assert group[1:] == ["aqua_grunt_tunnel_defeated", "maxie_tabitha_defeated"], group


def test_item_already_in_the_bag_says_so():
    """Seen live: "Receive Great Ball from Devon Researcher" stayed open with the ball in
    the bag. The fixture is after the hand-over, before Roxanne."""
    s = server()
    s.load(AFTER_WOODS_GRUNT)
    guidance = {
        "story": {"id": "petalburg_027", "completion_condition": "great_ball_received"},
        "battling_group": [
            {"id": "early_017", "completion_condition": "pokeballs_received"},
            {"id": "rustboro_034", "completion_condition": "tm39_received"},
            {"id": "mauville_069", "completion_condition": "bicycle_received"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    story = guidance["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("great_ball_received: ALREADY TRUE in game memory (your bag has a GREAT BALL)"), story
    group = [g["completion_condition"] for g in guidance["battling_group"]]
    assert "your bag has POKé BALLS" in group[0], group
    assert group[1:] == ["tm39_received", "bicycle_received"], group


def test_unbeaten_trainers_are_listed_with_their_tiles():
    """Seen live: "Defeat Gym trainers" in Rustboro Gym; the model crept up one tile per
    decision not knowing where anyone stood."""
    from utils.mapping.porymap_state import _format_trainers, _map_trainers

    # Roxanne battles from her script (TRAINER_TYPE_NONE) and must still count.
    assert [t[2] for t in _map_trainers("RustboroCity_Gym")] == ["ROXANNE", "JOSH", "TOMMY", "MARC"]
    s = server()
    s.load(AFTER_WOODS_GRUNT)  # Lyle beaten, James never fought
    with s.srv.memory_lock:
        woods = _format_trainers("PetalburgWoods", s.srv.env.memory_reader)
    print("   ", woods)
    assert woods[0] == "- JAMES at (4, 14)", woods
    s.load(Path(__file__).resolve().parent / "test_rustboro_gym.state")  # mid-battle with Josh
    with s.srv.memory_lock:
        gym = _format_trainers("RustboroCity_Gym", s.srv.env.memory_reader)
    assert gym[0] == "- ROXANNE at (5, 2); JOSH at (5, 13); TOMMY at (3, 9); MARC at (1, 6)", gym


def test_lead_level_condition_reads_the_party():
    """David, 2026-11-03: "train Torchic to 16 on Route 116, then Roxanne"."""
    s = server()
    s.load(AFTER_WOODS_GRUNT)
    with s.srv.memory_lock:
        level = s.srv.env.memory_reader.read_party_pokemon()[0].level
    guidance = {
        "story": {"id": "train", "completion_condition": f"lead_level_{level + 1}"},
        "battling_group": [{"id": "low", "completion_condition": f"lead_level_{level}"}],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    print("   ", guidance["story"]["completion_condition"])
    story = guidance["story"]["completion_condition"]
    assert story == f"lead_level_{level + 1}: not yet (TORCHIC is Lv{level} of {level + 1} needed).", story
    assert guidance["battling_group"][0]["completion_condition"].startswith(f"lead_level_{level}: ALREADY TRUE")


def test_party_conditions_for_team_building():
    """David, after whiting out to Roxanne: "we need more pokemon"."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_wild_two_party.state")  # Torchic + a caught Shroomish
    guidance = {
        "story": {"id": "catch", "completion_condition": "party_has_shroomish"},
        "battling_group": [
            {"id": "train", "completion_condition": "level_shroomish_12"},
            {"id": "evolve", "completion_condition": "level_combusken_16"},
            {"id": "other", "completion_condition": "party_has_lotad"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    print("   ", guidance["story"]["completion_condition"])
    assert guidance["story"]["completion_condition"].startswith("party_has_shroomish: ALREADY TRUE")
    group = [g["completion_condition"] for g in guidance["battling_group"]]
    assert group[0].startswith("level_shroomish_12: not yet (SHROOMISH is Lv"), group
    assert group[1].startswith("level_combusken_16: not yet (no COMBUSKEN in the party yet"), group
    assert group[2].startswith("party_has_lotad: not yet"), group


def test_switch_training_says_switch_on_the_first_turn():
    """Seen live: the model attacked with the Lv6 SHROOMISH it was training and switched
    only after it took damage."""
    here = Path(__file__).resolve().parent
    s = server()
    s.load(here / "test_battle_wurmple.state")  # TORCHIC Lv11 in battle, alone in the party
    trainee = {"story": {"completion_condition": "level_torchic_15: not yet (TORCHIC is Lv11 of 15 needed)."}}
    # David: "You don't need to switch train when the other pokemon is weak against
    # shroomish": EMBER is 2x on the Bug-type WURMPLE and Bug isn't 2x on Fire, so
    # TORCHIC (24/33) fights. (This asserted "" before that rule: no partner to switch to.)
    line = s.srv._switch_train_line(trainee)
    assert line.startswith("TORCHIC can beat WURMPLE on its own (EMBER 2x"), line
    s.load(here / "test_wild_two_party.state")  # TORCHIC out, SHROOMISH in the party
    # RAM: the wild SILCOON made Water (gBattleMons[1] +0x21/+0x22, EMBER 0.5x) with 60 HP
    # (+0x28/+0x2C), so TORCHIC would need ~6 hits: switch-training applies. (Since
    # 2026-11-04 the trainee fights alone by the damage race, not by types: Water alone,
    # at 20 HP, is EMBER 2 hits against a SILCOON that only knows HARDEN, so it fights.)
    with s.srv.memory_lock:
        memory = s.srv.env.core.memory
        memory.u8[0x02024084 + 0x58 + 0x21] = 11
        memory.u8[0x02024084 + 0x58 + 0x22] = 11
        memory.u16[0x02024084 + 0x58 + 0x28] = 60
        memory.u16[0x02024084 + 0x58 + 0x2C] = 60
        s.srv.env.memory_reader._invalidate_mem_cache()
    line = s.srv._switch_train_line(trainee)
    print("   ", line)
    with s.srv.memory_lock:
        partner = [p for p in s.srv.env.memory_reader.read_party_pokemon() if p.species_name.upper() != "TORCHIC"][0]
    if partner.current_hp * 2 > partner.max_hp:
        assert line.startswith('SWITCH-TRAINING: TORCHIC is being trained') and 'switch_pokemon("SHROOMISH")' in line, line
    else:
        assert line == "", line  # partner below half HP is not sent in
    assert s.srv._switch_train_line({"story": {"completion_condition": "level_shroomish_15: not yet"}}) == ""  # trainee not out


def test_grass_is_walked_while_training():
    """David: "make sure it doesn't avoid tall grass now. I want it to battle pokemon"."""
    from server import game_tools
    from utils.mapping.pathfinding import GRASS_TILE_COST_MULTIPLIER

    s = server()
    s.srv._set_grass_cost_for({"story": {"completion_condition": "level_shroomish_15: not yet (SHROOMISH is Lv5 of 15 needed)."}})
    assert game_tools.pathfinder.grass_cost == 1.0
    s.srv._set_grass_cost_for({"story": {"completion_condition": "level_shroomish_15: ALREADY TRUE in game memory"}})
    assert game_tools.pathfinder.grass_cost == GRASS_TILE_COST_MULTIPLIER
    s.srv._set_grass_cost_for({"story": {"completion_condition": "stone_badge_obtained"}})
    assert game_tools.pathfinder.grass_cost == GRASS_TILE_COST_MULTIPLIER


def test_party_healed_condition():
    here = Path(__file__).resolve().parent
    s = server()
    s.load(here / "test_rustboro_pokecenter.state")  # Torchic 35/35
    assert s.srv._party_condition("party_healed")[0] is True
    s.load(here / "test_battle_wurmple.state")  # Torchic 24/33
    met, detail = s.srv._party_condition("party_healed")
    assert met is False and "TORCHIC 24/33" in detail.upper(), detail


def test_battle_against_the_wanted_pokemon_says_catch_it():
    """The battle view's "Best move now" would have Combusken knock out the Shroomish
    it was sent to catch."""
    here = Path(__file__).resolve().parent
    s = server()
    s.load(here / "test_wild_shroomish.state")
    want = {"story": {"id": "c", "completion_condition": "party_has_shroomish: not yet (party: TORCHIC Lv11)."}}
    line = s.srv._catch_target_line(want)
    print("   ", line)
    # The fixture's bag has a GREAT BALL, the better ball (x1.5).
    assert line.startswith("CATCH THIS ONE: the wild SHROOMISH") and 'throw_ball("Great Ball")' in line, line
    assert s.srv._catch_target_line({"story": {"id": "c", "completion_condition": "party_has_lotad"}}) == ""
    s.load(here / "test_battle_wurmple.state")  # trainer battle: can't catch
    assert s.srv._catch_target_line({"story": {"id": "c", "completion_condition": "party_has_wurmple"}}) == ""


def test_catch_plan_weakens_a_healthy_foe_without_risking_a_knockout():
    """Seen live: seven GREAT BALLs at a full-HP ELECTRIKE, leaving one GREAT BALL and four
    POKé BALLs for the NUMEL still to catch. Above half HP, the plan names one attack
    whose highest roll can't knock it out (here SCRATCH, not EMBER); else balls at once.
    RAM: the wild SHROOMISH's HP in gBattleMons[1] (+0x28 hp, +0x2C max)."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_wild_shroomish.state")
    want = {"story": {"id": "c", "completion_condition": "party_has_shroomish: not yet"}}
    line = s.srv._catch_target_line(want)
    print("   ", line)
    assert 'first use_move("SCRATCH") once to weaken it' in line and "never a knockout" in line, line
    with s.srv.memory_lock:
        view = s.srv.env.memory_reader.read_battle_view()
        scratch = next(m for m in view["moves"] if m["name"] == "SCRATCH")
    assert scratch["damage"] / 0.925 < view["target"]["hp"], (scratch, view["target"])

    def foe_hp(hp, max_hp):
        with s.srv.memory_lock:
            s.srv.env.core.memory.u16[0x02024084 + 0x58 + 0x28] = hp
            s.srv.env.core.memory.u16[0x02024084 + 0x58 + 0x2C] = max_hp
            s.srv.env.memory_reader._invalidate_mem_cache()

    foe_hp(10, 23)  # under half: straight to balls
    assert "weaken" not in s.srv._catch_target_line(want) and "Do NOT use damaging moves" in s.srv._catch_target_line(want)
    foe_hp(6, 6)  # full HP, but every attack could knock it out
    assert "weaken" not in s.srv._catch_target_line(want)


def test_catch_at_a_level_names_the_one_and_turns_down_the_rest():
    """"Catch a level 10+ Sableye" (David): level_<species>_<n> with none in the party.
    test_wild_shroomish.state is a wild SHROOMISH Lv6 against TORCHIC alone."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_wild_shroomish.state")
    line = s.srv._catch_target_line({"story": {"id": "c", "completion_condition": "level_shroomish_6"}})
    print("   ", line)
    assert line.startswith("CATCH THIS ONE: the wild SHROOMISH Lv6") and "throw_ball(" in line, line
    assert "First call use_move" not in line, line  # TORCHIC has no sleep or paralysis move
    low = s.srv._catch_target_line({"story": {"id": "c", "completion_condition": "level_shroomish_7"}})
    print("   ", low)
    assert low.startswith("NOT THIS ONE: this wild SHROOMISH is Lv6, below the Lv7") and "run_from_battle" in low, low
    # Once one is in the party, the level step is about training it, not catching.
    s.load(Path(__file__).resolve().parent / "test_wild_two_party.state")  # SHROOMISH in the party
    assert s.srv._catch_wanted({"story": {"id": "c", "completion_condition": "level_shroomish_30"}}) == {}


def test_a_pokemon_named_before_it_evolved_is_still_found():
    """Objective "Train Shroomish to Lv23" (it becomes BRELOOM at 23): read by species name,
    the evolved one looked gone, the step could never be met, and the catch logic would
    have sent the AI to catch a new Shroomish. test_rusturf_tunnel.state has COMBUSKEN
    Lv17 (it was a TORCHIC) and SHROOMISH Lv10."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_rusturf_tunnel.state")
    reader = s.srv.env.memory_reader
    assert reader.species_family("shroomish") == ["SHROOMISH", "BRELOOM"]
    assert reader.species_family("TORCHIC") == ["TORCHIC", "COMBUSKEN", "BLAZIKEN"]
    assert reader.species_family("COMBUSKEN") == ["COMBUSKEN", "BLAZIKEN"]  # never backwards
    met, detail = s.srv._party_condition("level_torchic_16")
    print("   ", detail)
    assert met and detail == "TORCHIC (now COMBUSKEN) is Lv17 of 16 needed", detail
    assert s.srv._party_condition("level_torchic_20") == (False, "TORCHIC (now COMBUSKEN) is Lv17 of 20 needed")
    assert s.srv._party_condition("party_has_torchic")[0]
    assert s.srv._catch_wanted({"story": {"id": "c", "completion_condition": "level_torchic_20"}}) == {}
    party = s.srv._train_party()
    assert party[s.srv._train_member(party, "TORCHIC")][1].species_name.upper() == "COMBUSKEN"
    assert s.srv._train_member(party, "BLAZIKEN") is None  # a later form is not this one


def test_a_pokemon_from_outside_the_hoenn_dex_is_named():
    """Seen live: a caught MARILL Lv13 read as "Species_183" in the party, so "Train Marill
    to Lv18" saw no Marill and would have become "catch another". The local name table
    lists only Hoenn-dex species. test_marill_in_party.state is the live party: BRELOOM,
    COMBUSKEN, SABLEYE, MARILL Lv13 (slot 4), on ROUTE 117 (18,14)."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_marill_in_party.state")
    with s.srv.memory_lock:
        names = [p.species_name for p in s.srv.env.memory_reader.read_party_pokemon()]
    print("   ", names)
    assert names[3] == "Marill" and not any(n.startswith("Species_") for n in names), names
    assert s.srv._party_condition("level_marill_18") == (False, "MARILL is Lv13 of 18 needed")
    assert s.srv._catch_wanted({"story": {"id": "m", "completion_condition": "level_marill_18"}}) == {}
    party = s.srv._train_party()
    assert s.srv._train_member(party, "MARILL") == 3
    assert s.srv._train_member(party, "marill") == 3
    import asyncio

    text = asyncio.run(s.srv.mcp_get_game_state())["state_text"]
    assert "Marill" in text and "Species_" not in text, text


def test_party_abilities_are_shown():
    """David wants a HUGE POWER Marill (Huge Power doubles Attack; the other is Thick Fat),
    so a second one is caught if this one isn't. The ability is the species' abilities[
    abilityNum] (pret GetAbilityBySpecies); test_battle_view checks that against the game."""
    import asyncio

    s = server()
    s.load(Path(__file__).resolve().parent / "test_marill_in_party.state")
    with s.srv.memory_lock:
        party = s.srv.env.memory_reader.read_party_pokemon()
    print("   ", [(p.species_name, p.ability_num, p.ability) for p in party])
    marill = party[3]
    assert marill.ability == ("HUGE POWER" if marill.ability_num else "THICK FAT"), marill
    assert [p.ability for p in party[:3]] == ["EFFECT SPORE", "BLAZE", "KEEN EYE"], party
    text = asyncio.run(s.srv.mcp_get_game_state())["state_text"]
    assert f"Marill (Lv.13, {marill.ability})" in text, text


def test_learn_move_advice():
    """Seen live: at Lv28 BRELOOM learned COUNTER and the model forgot MACH PUNCH (its STAB
    Fighting move); the learn-move prompt came with no advice. test_marill_in_party.state:
    BRELOOM knows MACH PUNCH, HEADBUTT, MEGA DRAIN, LEECH SEED."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_marill_in_party.state")
    texts = ["BRELOOM grew to LV. 28!", "BRELOOM is trying to learn COUNTER.",
             "But BRELOOM can’t learn more than four moves.", "Delete a move to make room for COUNTER?"]
    advice = s.srv._learn_move_advice(texts)
    print("   ", advice)
    assert advice.startswith("Advice: don't learn COUNTER (it does no damage of its own): press B (NO), then A (YES)"), advice
    advice = s.srv._learn_move_advice(["BRELOOM is trying to learn SKY UPPERCUT."])
    print("   ", advice)
    assert advice.startswith("Advice: learn SKY UPPERCUT and forget LEECH SEED (move 4"), advice
    assert s.srv._learn_move_advice(["Wild ZUBAT appeared!"]) == ""
    # Party types come from the ROM (the party struct has none): counter_score's STAB.
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        breloom = reader.read_party_pokemon()[0]
        assert reader.party_mon_types(breloom) == [12, 1], reader.party_mon_types(breloom)  # GRASS/FIGHTING
        assert reader.move_worth("MACH_PUNCH", [12, 1]) == 60 and reader.move_worth("COUNTER", [12, 1]) == 0


def test_insert_objective_goes_before_the_current_one_and_persists():
    from agents.objectives.direct_objectives import DirectObjectiveManager

    m = DirectObjectiveManager()
    m.load_categorized_full_game_sequence(start_story_index=33)
    roxanne = m.story_sequence[33].id
    r = m.insert_objective("story", 33, {
        "id": "custom_train_16", "description": "Train to Lv16 on Route 116",
        "action_type": "battle", "target_location": "Route 116", "completion_condition": "lead_level_16",
    })
    assert r["success"], r
    assert m.story_index == 33 and m.story_sequence[33].id == "custom_train_16"
    assert m.story_sequence[34].id == roxanne  # Roxanne comes next, nothing overwritten
    assert not m.insert_objective("story", 32, {"id": "x", "description": "x"})["success"]  # before current
    assert not m.insert_objective("story", 33, {"id": "custom_train_16", "description": "dup"})["success"]
    again = DirectObjectiveManager()
    again.restore_from_state(m.serialize_full_state())
    assert [o.id for o in again.story_sequence[33:35]] == ["custom_train_16", roxanne]


def test_navigate_objective_for_the_current_map_says_so():
    s = server()
    s.load(POKEDEX_STATE)  # standing in Birch's lab
    guidance = {
        "story": {"id": "early_015", "action_type": "navigate", "target_location": "Birch's Lab",
                  "completion_condition": "returned_to_lab"},
        "battling_group": [
            {"id": "petalburg_023", "action_type": "navigate", "target_location": "Route 104 South",
             "completion_condition": "route104_south_reached"},
            # Names the place but is about more than being there.
            {"id": "mauville_073", "action_type": "navigate", "target_location": "Birch's Lab",
             "completion_condition": "gym_puzzle_solved"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    story = guidance["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("returned_to_lab: ALREADY TRUE (you are on LITTLEROOT TOWN PROFESSOR BIRCHS LAB"), story
    assert 'complete_direct_objective(objective_id="early_015")' in story, story
    assert guidance["battling_group"][0]["completion_condition"] == "route104_south_reached"
    assert guidance["battling_group"][1]["completion_condition"] == "gym_puzzle_solved"


def test_game_vars_read_from_saveblock1():
    """SaveBlock1.vars sits at +0x139C (pret include/global.h). Each fixture's vars agree
    with its story: Torchic was the starter, the POKéDEX moves Birch's lab from 3 to 5,
    and the Woods' Devon researcher leaving after the Grunt sets the woods to 1."""
    here = Path(__file__).resolve().parent
    s = server()
    s.load(AFTER_WOODS_GRUNT)
    reader = s.srv.env.memory_reader
    with s.srv.memory_lock:
        assert reader.read_game_var(0x4023) == 1  # VAR_STARTER_MON: 1 = Torchic
        assert reader.read_game_var(0x4084) == 5  # VAR_BIRCH_LAB_STATE: POKéDEX received
        assert reader.read_game_var(0x4098) == 1  # VAR_PETALBURG_WOODS_STATE: Grunt beaten
        assert reader.read_game_var(0x8000) is None  # special vars are not in the save
    s.load(here / "test_battle_wurmple.state")  # earlier in the woods, Grunt not met yet
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.read_game_var(0x4098) == 0
    s.load(here / "test_movement_route101.state")  # starter in hand, no POKéDEX yet
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.read_game_var(0x4084) == 3


def _set_in_saveblock1(s, offset, value):
    """Write one byte of SaveBlock1 in the running game (RAM only; the fixture is untouched)."""
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        base = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR)
        s.srv.env.core.memory.u8[base + offset] = value
        reader._invalidate_mem_cache()


def test_story_scenes_recorded_in_flags_and_vars():
    """Seen live: "Exit Gym - Devon Researcher reports theft" stayed open with
    FLAG_DEVON_GOODS_STOLEN set; the agent wandered. test_rusturf_tunnel.state is before
    the theft, so nothing is true until the game's own bytes say so."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_rusturf_tunnel.state")
    guidance = {
        "story": {"id": "rustboro_036", "completion_condition": "theft_event_triggered"},
        "battling_group": [
            {"id": "rustboro_041", "completion_condition": "mr_stone_met"},
            {"id": "dewford_044", "completion_condition": "sailed_to_dewford"},
            {"id": "mossdeep_157", "completion_condition": "maxie_tabitha_defeated"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    assert guidance["story"]["completion_condition"] == "theft_event_triggered"
    assert [g["completion_condition"] for g in guidance["battling_group"]] == [
        "mr_stone_met", "sailed_to_dewford", "maxie_tabitha_defeated"]

    flags, variables = 0x1270, 0x139C
    _set_in_saveblock1(s, flags + 0x8E // 8, 1 << (0x8E % 8))  # FLAG_DEVON_GOODS_STOLEN
    _set_in_saveblock1(s, variables + 2 * (0x408F - 0x4000), 1)  # VAR_DEVON_CORP_3F_STATE = 1
    _set_in_saveblock1(s, variables + 2 * (0x409F - 0x4000), 2)  # VAR_MOSSDEEP_SPACE_CENTER_STATE = 2
    s.srv._note_conditions_met_in_memory(guidance)
    story = guidance["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("theft_event_triggered: ALREADY TRUE in game memory"), story
    assert 'complete_direct_objective(objective_id="rustboro_036")' in story, story
    group = [g["completion_condition"] for g in guidance["battling_group"]]
    assert group[0].startswith("mr_stone_met: ALREADY TRUE in game memory (you have met Mr. Stone)"), group
    assert group[1] == "sailed_to_dewford", group  # a different flag
    assert group[2] == "maxie_tabitha_defeated", group  # needs 3: 2 is only the grunts upstairs


def test_place_reached_earlier_is_read_from_its_first_visit_flag():
    """Seen live: "Travel to Mr. Briney's cottage" stayed open on the boat to Dewford.
    test_after_briney.state is that moment (Route 105, FLAG_LANDMARK_MR_BRINEY_HOUSE set,
    Dewford not reached); test_rusturf_tunnel.state is before the cottage."""
    from server.game_tools import first_visit_flag

    assert first_visit_flag("Mr. Briney's Cottage") == 0x860 + 0x3D  # FLAG_LANDMARK_MR_BRINEY_HOUSE
    assert first_visit_flag("Dewford Town") == 0x860 + 0x11  # FLAG_VISITED_DEWFORD_TOWN
    assert first_visit_flag("Route 104 South") is None

    def guidance():
        return {
            "story": {"id": "dewford_043", "action_type": "navigate", "target_location": "Mr. Briney's Cottage",
                      "completion_condition": "briney_cottage_reached"},
            "battling_group": [
                {"id": "dewford_045", "action_type": "navigate", "target_location": "Dewford Town",
                 "completion_condition": "dewford_reached"},
                {"id": "x_001", "action_type": "navigate", "target_location": "Mr. Briney's Cottage",
                 "completion_condition": "returned_to_briney"},
            ],
        }

    s = server()
    here = Path(__file__).resolve().parent
    s.load(here / "test_rusturf_tunnel.state")
    before = guidance()
    s.srv._note_conditions_met_in_memory(before)
    assert before["story"]["completion_condition"] == "briney_cottage_reached", before

    s.load(here / "test_after_briney.state")
    after = guidance()
    s.srv._note_conditions_met_in_memory(after)
    story = after["story"]["completion_condition"]
    print("   ", story)
    assert story.startswith("briney_cottage_reached: ALREADY TRUE in game memory (you have been to"), story
    assert 'complete_direct_objective(objective_id="dewford_043")' in story, story
    group = [g["completion_condition"] for g in after["battling_group"]]
    assert group[0] == "dewford_reached", group  # still at sea
    assert group[1] == "returned_to_briney", group  # a revisit: the flag says nothing about now


def test_heal_steps_and_hm_steps_read_the_party():
    here = Path(__file__).resolve().parent
    s = server()
    s.load(here / "test_rustboro_pokecenter.state")  # Torchic 35/35
    guidance = {
        "story": {"id": "dewford_051", "completion_condition": "healed_dewford"},
        "battling_group": [
            {"id": "battle_050", "completion_condition": "elite_four_final_heal"},
            {"id": "mauville_077", "completion_condition": "rock_smash_taught"},
        ],
    }
    s.srv._note_conditions_met_in_memory(guidance)
    assert guidance["story"]["completion_condition"].startswith("healed_dewford: ALREADY TRUE"), guidance
    group = [g["completion_condition"] for g in guidance["battling_group"]]
    assert group[0].startswith("elite_four_final_heal: ALREADY TRUE"), group
    assert group[1] == "rock_smash_taught", group  # Torchic knows no HM
    s.load(here / "test_battle_wurmple.state")  # Torchic 24/33
    met, detail = s.srv._party_condition("healed_petalburg_pre_norman")
    assert met is False and "TORCHIC 24/33" in detail.upper(), detail
    # The move check itself, with a move Torchic does know.
    with s.srv.memory_lock:
        known = [m for m in s.srv.env.memory_reader.read_party_pokemon()[0].moves if m][0]
    s.srv._CONDITION_MOVES["test_taught"] = known
    try:
        probe = {"story": {"id": "t", "completion_condition": "test_taught"}}
        s.srv._note_conditions_met_in_memory(probe)
    finally:
        del s.srv._CONDITION_MOVES["test_taught"]
    assert probe["story"]["completion_condition"].startswith(
        f"test_taught: ALREADY TRUE in game memory (TORCHIC knows {known.upper()})"), probe


def test_every_mapped_condition_is_an_objective():
    """A typo in a table key would silently never match."""
    from agents.objectives.all_obj_categorized import BATTLING_OBJECTIVES, STORY_OBJECTIVES

    srv = server().srv
    conditions = {o.completion_condition for o in [*STORY_OBJECTIVES, *BATTLING_OBJECTIVES]}
    tables = {
        "flags": srv._CONDITION_FLAGS, "vars": srv._CONDITION_VARS, "trainers": srv._CONDITION_TRAINERS,
        "items": srv._CONDITION_ITEMS, "moves": srv._CONDITION_MOVES,
    }
    for name, table in tables.items():
        missing = sorted(set(table) - conditions)
        assert not missing, f"{name}: {missing}"
    keys = [k for table in tables.values() for k in table]
    assert len(keys) == len(set(keys)), "a condition is in two tables; only the first would be read"


def test_arrived_at_tells_the_halves_of_a_route_apart():
    """Seen live: on Route 104 with "Travel west to Route 104 South", the model went
    looking for a south exit. The halves are separate walkable areas joined only through
    Petalburg Woods; the woods' south door (y=38 of 80) is still in the south half."""
    from server.game_tools import arrived_at

    for pos in [(39, 62), (10, 39), (32, 43)]:
        assert arrived_at("ROUTE 104", pos, "Route 104 South"), pos
        assert not arrived_at("ROUTE 104", pos, "Route 104 North"), pos
    for pos in [(10, 29), (15, 5)]:
        assert arrived_at("ROUTE 104", pos, "Route 104 North"), pos
        assert not arrived_at("ROUTE 104", pos, "Route 104 South"), pos
    assert not arrived_at("ROUTE 104", None, "Route 104 South")  # half unknown without a position
    assert arrived_at("PETALBURG CITY", None, "Petalburg City")
    assert not arrived_at("PETALBURG CITY GYM", None, "Petalburg City")
    assert not arrived_at("ROUTE 110", None, "Route 11")


def _run_all():
    import test_movement_feedback as harness

    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
