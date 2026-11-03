"""go_to_exit walks through map edges, doors, stairs and exit mats; navigate_to reports
unwalkable or unreachable targets instead of silently stopping short.

Same in-process server harness as test_movement_feedback.py.

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_movement_exits.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import HOUSE, ROUTE101, server


BATTLE = Path(__file__).resolve().parent / "test_battle_wurmple.state"  # Petalburg Woods, move menu vs Lyle's Wurmple


def _location(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_location()


def test_walking_tools_refuse_during_a_battle():
    """Seen live: between two of Bug Catcher Lyle's Wurmple the model called navigate_to
    and go_to_exit; the presses moved the battle cursor and the results said only "did
    not leave the map"."""
    s = server()
    s.load(BATTLE)
    for endpoint, payload in (
        ("mcp_go_to_exit", {"exit": "ROUTE 104", "reasoning": ""}),
        ("mcp_navigate_to", {"x": 16, "y": 39, "variance": "none", "reason": ""}),
    ):
        r = s.call(endpoint, payload)
        print("   ", r)
        assert r["success"] is False and "battle is still going on" in r["error"], r
        assert "buttons_queued" not in r, r
    with s.srv.memory_lock:
        assert s.srv.env.memory_reader.is_in_battle()
    # The battle itself still plays: A on the highlighted EMBER finishes this Wurmple.
    for _ in range(4):
        assert s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": ""})["success"]


def test_go_to_exit_north_edge_from_route101_reaches_oldale():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_go_to_exit", {"exit": "north", "reasoning": "to Oldale"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "OLDALE TOWN", _location(s)
    assert "OLDALE TOWN" in r["summary"], r


def test_go_to_exit_by_destination_name():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_go_to_exit", {"exit": "Oldale Town", "reasoning": "to Oldale"})
    assert r["success"] is True and _location(s) == "OLDALE TOWN", r


def test_go_to_exit_south_from_a_corner_of_oldale_uses_the_real_opening():
    """Seen live: from Oldale (15,19) go_to_exit("south") pressed DOWN into Route 101's
    trees 20+ times. Oldale's whole south row is open ground, but Route 101 continues
    only under x=8-11, so the exit list and the route must both use that opening."""
    s = server()
    s.load(ROUTE101)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_navigate_to", {"x": 15, "y": 19, "variance": "none", "reason": "corner"})
    assert r["success"] is True, r
    r = s.call("mcp_go_to_exit", {"exit": "south", "reasoning": "back to Littleroot"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "ROUTE 101", _location(s)


def test_go_to_exit_through_a_building_door():
    """Route 101 -> Oldale, then into the POKEMON CENTER through its front door."""
    s = server()
    s.load(ROUTE101)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_go_to_exit", {"exit": "POKEMON CENTER", "reasoning": "heal"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "OLDALE TOWN POKEMON CENTER 1F", _location(s)


def test_go_to_exit_through_house_exit_mat():
    """House exits are arrow warps: they fire when you walk off the mat, not onto it."""
    s = server()
    s.load(HOUSE)
    r = s.call("mcp_go_to_exit", {"exit": "LITTLEROOT TOWN", "reasoning": "go outside"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "LITTLEROOT TOWN", _location(s)


def test_go_to_exit_up_the_stairs():
    s = server()
    s.load(HOUSE)
    r = s.call("mcp_go_to_exit", {"exit": "Brendans House 2F", "reasoning": "upstairs"})
    print("   ", r)
    assert r["success"] is True, r
    assert _location(s) == "LITTLEROOT TOWN BRENDANS HOUSE 2F", _location(s)


def test_go_to_exit_unknown_name_lists_the_exits():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_go_to_exit", {"exit": "west", "reasoning": ""})
    assert r["success"] is False, r
    assert "north edge -> OLDALE TOWN" in r["error"] and "south edge -> LITTLEROOT TOWN" in r["error"], r


def test_navigate_to_unwalkable_target_says_so():
    s = server()
    s.load(ROUTE101)
    # (13,0) is a tree and every tile around it is too: refuse with a reason.
    r = s.call("mcp_navigate_to", {"x": 13, "y": 0, "variance": "none", "reason": ""})
    print("   ", r)
    assert r["success"] is False and "not walkable" in r["error"], r
    # (12,1) is a tree; the nearest reachable tile is (12,2): walk there and say so.
    r = s.call("mcp_navigate_to", {"x": 12, "y": 1, "variance": "none", "reason": ""})
    print("   ", r)
    assert "not walkable" in r.get("note", ""), r
    assert r["summary"].startswith("Did not reach (12,1)."), r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (12, 2), snap
    # Off the map: point at go_to_exit instead of clamping to the edge.
    r = s.call("mcp_navigate_to", {"x": 9, "y": -1, "variance": "none", "reason": ""})
    assert r["success"] is False and "go_to_exit" in r["error"], r


def test_navigate_to_a_door_reports_the_new_map():
    """Seen live: navigate_to onto a shop door said "Did not reach", and the model walked
    back out and in again."""
    s = server()
    s.load(ROUTE101)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    r = s.call("mcp_navigate_to", {"x": 6, "y": 16, "variance": "none", "reason": "Pokemon Center door"})
    print("   ", r)
    assert _location(s) == "OLDALE TOWN POKEMON CENTER 1F", _location(s)
    assert r["summary"].startswith("Went through a door/warp: now at") and "POKEMON CENTER 1F" in r["summary"], r


def test_navigate_to_across_an_exit_mat_reports_the_new_map():
    """Seen live: in the Oldale Mart, navigate_to(4,8) from (3,7) stepped on the exit mat
    and the warp cut the walk short. The result said "STOPPED ... on OLDALE TOWN MART: an
    event is running" although the player was already outside."""
    s = server()
    s.load(ROUTE101)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    assert s.call("mcp_go_to_exit", {"exit": "MART", "reasoning": ""})["success"]
    assert _location(s) == "OLDALE TOWN MART", _location(s)
    start = s.snapshot()
    r = s.call("mcp_navigate_to", {"x": start.x + 1, "y": start.y + 1, "variance": "none", "reason": "leave"})
    print("   ", r)
    assert _location(s) == "OLDALE TOWN", _location(s)
    assert r.get("status") != "STOPPED", r
    assert r["summary"].startswith("Went through a door/warp: now at") and "OLDALE TOWN" in r["summary"], r


def test_navigate_to_toward_a_neighbouring_map_names_the_exit_call():
    """Seen live: "Destination: Route 101 (south edge)" in the reason, then one-tile
    navigate_to calls wobbling against Oldale's walls for minutes."""
    from server.game_tools import exit_hint_for_text

    assert 'go_to_exit("south")' in exit_hint_for_text("OLDALE TOWN", "Destination: Route101 (south edge)")
    assert exit_hint_for_text("OLDALE TOWN", "heal at the Pokemon Center") == ""
    assert exit_hint_for_text("OLDALE TOWN", "toward ROUTE 1010") == ""  # no prefix matches
    # Seen live on Route 101: "heading east toward the exit to Littleroot" (no "Town").
    assert 'go_to_exit("south")' in exit_hint_for_text("ROUTE 101", "toward the exit to Littleroot")
    # A copied route names both neighbours: list both, don't guess.
    both = exit_hint_for_text("ROUTE 101", "Route 103 → Oldale → Route 101 → Littleroot → Birch's Lab")
    assert 'OLDALE TOWN use go_to_exit("north")' in both and 'LITTLEROOT TOWN use go_to_exit("south")' in both, both

    s = server()
    s.load(ROUTE101)
    assert s.call("mcp_go_to_exit", {"exit": "north", "reasoning": ""})["success"]
    start = s.snapshot()
    r = s.call("mcp_navigate_to", {
        "x": start.x, "y": start.y - 1, "variance": "none",
        "reason": "Destination: Route 101 (south edge). Walk south.",
    })
    print("   ", r)
    assert 'go_to_exit("south")' in r["summary"], r
    r = s.call("mcp_navigate_to", {"x": start.x, "y": start.y, "variance": "none", "reason": "step back"})
    assert "go_to_exit" not in r["summary"], r


def test_exit_list_says_when_an_edge_cannot_be_walked():
    """Seen live: Route 104's "south edge of the map -> ROUTE 105" (all sea) sent the
    model south from the very spot where its objective was already done."""
    from utils.mapping.porymap_state import _format_porymap_info

    text = str(_format_porymap_info("ROUTE 104", (39, 62)))
    assert "south edge of the map (no way across on foot: water or cliff all along it) -> ROUTE 105" in text
    assert "east edge of the map (walk off at x=39, y=62-63) -> PETALBURG CITY" in text


def test_way_to_an_objective_target():
    """Seen live: objective target "Rustboro Gym"; the model left the gym and searched the
    city tile by tile for its trainers."""
    from server.game_tools import way_to_place

    gym = way_to_place("RUSTBORO CITY", (13, 21), "Rustboro Gym")
    print("   ", gym)
    assert gym == 'RUSTBORO CITY GYM is through an exit of this map: go_to_exit("RUSTBORO CITY GYM").', gym
    lab = way_to_place("LITTLEROOT TOWN", (7, 17), "Birch's Lab")
    assert 'go_to_exit("LITTLEROOT TOWN PROFESSOR BIRCHS LAB")' in lab, lab
    far = way_to_place("ROUTE 104", (39, 62), "Rustboro City")
    assert "PETALBURG WOODS → ROUTE 104 → RUSTBORO CITY" in far, far
    assert way_to_place("RUSTBORO CITY GYM", (5, 19), "Rustboro Gym") == ""  # already there
    assert way_to_place("ROUTE 104", (39, 62), "Route 104 North") == ""  # on the route; halves are arrived_at's job
    assert way_to_place("RUSTBORO CITY", (13, 21), "Mr. Briney's Cottage") == ""  # no map by that name
    # Seen live: inside the gym, "In Rustboro City Gym" in the reason produced
    # go_to_exit("RUSTBORO CITY") and pulled the model out of the gym.
    from server.game_tools import exit_hint_for_text

    for text in ("In Rustboro City Gym, facing UP", "Rustboro Gym trainers", "back in Rustboro"):
        assert exit_hint_for_text("RUSTBORO CITY GYM", text, (5, 17)) == "", text
    # Naming another map still works; from the gym the city door is the right first step.
    out = exit_hint_for_text("RUSTBORO CITY GYM", "then Route 104", (5, 17))
    assert "RUSTBORO CITY → ROUTE 104" in out and 'go_to_exit("RUSTBORO CITY")' in out, out


def test_navigate_to_toward_a_map_further_away_names_the_first_exit():
    """Seen live in Littleroot: "travel west through Route 102 to Petalburg City", then
    twenty minutes of one-tile walks. Route 102 is reached north via Route 101 and Oldale."""
    from server.game_tools import exit_hint_for_text

    hint = exit_hint_for_text("LITTLEROOT TOWN", "travel west through Route 102 to Petalburg City")
    print("   ", hint)
    assert 'go_to_exit("north")' in hint, hint
    assert "ROUTE 101 → OLDALE TOWN → ROUTE 102 → PETALBURG CITY" in hint, hint
    assert hint.count("go_to_exit") == 1, hint  # both destinations start through the same exit
    # From Oldale, Petalburg lies west and Littleroot south: two exits, two lines.
    split = exit_hint_for_text("OLDALE TOWN", "Petalburg City or back to Littleroot")
    assert 'PETALBURG CITY is not next' in split and 'go_to_exit("west")' in split, split
    assert 'LITTLEROOT TOWN is not next' in split and 'go_to_exit("south")' in split, split
    # This map's own name is not a destination.
    assert exit_hint_for_text("LITTLEROOT TOWN", "explore Littleroot Town") == ""
    # Indoors the first step is the door.
    lab = exit_hint_for_text("LITTLEROOT TOWN PROFESSOR BIRCHS LAB", "go to Route 102")
    assert lab.startswith("ROUTE 102 is not next") and 'go_to_exit("LITTLEROOT TOWN")' in lab, lab
    # The north half of Route 104 is reached through Petalburg Woods, not across the trees.
    woods = exit_hint_for_text("PETALBURG CITY", "on to Rustboro City")
    assert "PETALBURG WOODS → ROUTE 104 → RUSTBORO CITY" in woods and 'go_to_exit("west")' in woods, woods
    # On foot there is no way from Oldale to Lilycove (it needs Surf), so no route is offered
    # rather than one that crosses the sea.
    assert exit_hint_for_text("OLDALE TOWN", "Lilycove City", (8, 10)) == ""


def test_navigate_to_walkable_target_reports_arrival():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_navigate_to", {"x": 9, "y": 0, "variance": "none", "reason": ""})
    print("   ", r)
    assert r["success"] is True and r["summary"].startswith("Reached (9,0)."), r
    snap = s.snapshot()
    assert (snap.x, snap.y) == (9, 0), snap


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
