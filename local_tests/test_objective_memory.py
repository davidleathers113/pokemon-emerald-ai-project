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
