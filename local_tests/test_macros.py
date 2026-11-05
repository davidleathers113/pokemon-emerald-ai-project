"""Closed-loop macros: heal_at_pokecenter, run_from_battle, throw_ball, switch_pokemon,
buy_item. Each test checks the outcome from game memory, not from the macro's own
report, and each macro has at least one refusal case.

Same in-process server harness as test_movement_feedback.py (real game loop thread,
real savestates).

Fixtures:
  test_rustboro_pokecenter.state   RUSTBORO CITY POKEMON CENTER 1F (7,4), Torchic 35/35, $2784
  test_rustboro_mart.state         RUSTBORO CITY MART entrance (3,7), $2784; made from the
                                   Pokémon Center fixture with go_to_exit("RUSTBORO CITY"),
                                   go_to_exit("MART")
  test_wild_shroomish_intro.state  Petalburg Woods, wild Shroomish Lv6, "Wild SHROOMISH
                                   appeared!" waiting for A; made from
                                   test_after_woods_grunt.state by stepping in the tall grass
  test_wild_shroomish.state        the same battle at FIGHT/BAG/POKEMON/RUN; Torchic Lv11
                                   18/33, 5 Poké Balls + 1 Great Ball
  test_wild_two_party.state        Petalburg Woods, wild Silcoon, action menu; party Torchic +
                                   Shroomish. The Shroomish was caught from
                                   test_wild_shroomish.state after setting its HP to 1 and
                                   putting it to sleep in RAM (a guaranteed catch), then the
                                   player stepped in the grass again
  test_battle_wurmple.state        TRAINER battle (Bug Catcher Lyle), move menu open
  test_movement_route101.state     ROUTE 101, outdoors

RAM writes (documented where used): party HP/status before healing, battle speeds to
force a failed escape, the wild Pokémon's HP and sleep for a guaranteed catch, and a
party member's HP 0 for the fainted refusal.

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_macros.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import ROUTE101, server

HERE = Path(__file__).resolve().parent
POKECENTER = HERE / "test_rustboro_pokecenter.state"
MART = HERE / "test_rustboro_mart.state"
WILD_INTRO = HERE / "test_wild_shroomish_intro.state"
WILD = HERE / "test_wild_shroomish.state"
WILD_TWO = HERE / "test_wild_two_party.state"
TRAINER = HERE / "test_battle_wurmple.state"

# pret pokeemerald symbols (US): gPlayerParty (100-byte mons; +0x50 status, +0x56 HP),
# gPlayerPartyCount, gBattleMons (0x58 bytes; +6 speed, +0x28 HP, +0x48 personality,
# +0x4C status1), gBattleOutcome, gMain.inBattle.
PARTY = 0x020244EC
PARTY_COUNT = 0x020244E9
BATTLE_MONS = 0x02024084
BATTLE_OUTCOME = 0x0202433A


def _reader(s):
    return s.srv.env.memory_reader


def _write(s, size, address, value):
    with s.srv.memory_lock:
        memory = s.srv.env.core.memory
        {8: memory.u8, 16: memory.u16, 32: memory.u32}[size][address] = value
        # The reader caches memory blocks until the next frame; drop them so the
        # very next read (by the test or the macro) sees the write.
        _reader(s)._invalidate_mem_cache()


def _read(s, fn):
    with s.srv.memory_lock:
        return fn(_reader(s))


def _party(s):
    return _read(s, lambda r: r.read_party_pokemon())


def _in_battle(s):
    return _read(s, lambda r: r.is_in_battle())


def _balls(s):
    return {b["name"]: b["quantity"] for b in _read(s, lambda r: r.read_full_inventory())["poke_balls"]}


def _wait(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


# -- heal_at_pokecenter -------------------------------------------------------


def test_heal_restores_hp_and_status():
    s = server()
    s.load(POKECENTER)
    # RAM: Torchic down to 10/35 and poisoned, so the heal has something to restore.
    _write(s, 16, PARTY + 0x56, 10)
    _write(s, 32, PARTY + 0x50, 0x08)
    before = _party(s)[0]
    assert (before.current_hp, int(before.status.value)) == (10, 8), before
    # Step away first so the macro has to walk back to the counter.
    s.call("mcp_press_buttons", {"buttons": ["DOWN", "DOWN", "LEFT"], "reasoning": "test"})
    r = s.call("mcp_heal_at_pokecenter", {})
    print("   ", r)
    assert r["success"] is True and r["status"] == "HEALED", r
    after = _party(s)[0]
    assert after.current_hp == after.max_hp == 35, after
    assert int(after.status.value) == 0, after
    assert "TORCHIC 10/35" in r["summary"] and "TORCHIC 35/35" in r["summary"], r["summary"]
    snap = s.snapshot()
    assert (snap.x, snap.y, snap.facing) == (7, 4, "UP"), snap
    assert s.srv._player_can_walk(), "the player should be free to walk after healing"


def test_heal_from_outside_walks_into_the_pokemon_center():
    """Seen live: told "TEAM IS WEAK … go_to_exit("south") takes the first step", the model
    called train and navigate_to on its own tile while poison drained SHROOMISH.
    heal_at_pokecenter now makes the trip itself. (It used to refuse outside a center.)"""
    s = server()
    s.load(POKECENTER)
    assert s.call("mcp_go_to_exit", {"exit": "RUSTBORO CITY", "reasoning": "test"})["success"]
    _write(s, 16, PARTY + 0x56, 10)  # RAM: TORCHIC 10/35
    r = s.call("mcp_heal_at_pokecenter", {})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and r["status"] == "HEALED", r
    assert r["summary"].startswith("Walked RUSTBORO CITY POKEMON CENTER 1F. Healed"), r["summary"]
    assert _party(s)[0].current_hp == 35


def test_talk_to_walks_up_faces_and_starts_the_battle():
    """Seen live: told to talk to Wally (8,6) in front of Mauville Gym, the model shuffled
    around him for many minutes. test_mauville_wally.state is the player at (10,7);
    talking to Wally starts his battle."""
    s = server()
    s.load(HERE / "test_mauville_wally.state")
    r = s.call("mcp_talk_to", {"x": 8, "y": 6, "reasoning": "test"})
    print("   ", r.get("summary")[-200:])
    # The whole scene is read (Wally and his uncle, then Wally to the player), and it
    # stops at his question: a YES/NO is the model's to answer.
    assert r["success"] is True and "will you have a battle with me?" in r["summary"], r
    assert "A choice (YES/NO or a menu) is open now" in r["summary"], r
    assert s.srv._field_status() == "choice"
    s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "test: YES"})
    # His lines after YES are read the way the agent's auto-dialogue reads them. A
    # deadline, not a count of calls: 6 calls 0.5 s apart failed 2 of 3 full macros runs
    # on 2026-11-04 (L13) and never alone or after the same earlier tests; the failure now
    # says what the screen showed.
    deadline, reads = time.time() + 20, []
    while not _in_battle(s) and time.time() < deadline:
        reads.append(s.call("mcp_auto_advance_dialogue", {}).get("final_status"))
        time.sleep(0.5)
    assert _in_battle(s), f"YES to Wally starts the battle; auto-advance said {reads}, now {s.srv._field_status()}"
    s.load(HERE / "test_mauville_wally.state")
    r = s.call("mcp_talk_to", {"x": "eight", "y": 6})
    assert r["success"] is False and "x and y are required" in r["error"], r


def test_talking_to_someone_who_is_not_a_trainer_names_the_trainers_left():
    """Seen live on MT CHIMNEY (23,16), 14:52-14:57 UTC: objective "Battle through Team
    Magma Grunts"; the grunt at (23,17) only says "METEORITES pack amazing power!", and the
    model pressed A at him eight times while the real trainers stood at (13,16) and
    (9,16). The read now says he isn't one and names who is; the second time, that he
    only repeats himself. test_mt_chimney_busy_grunt.state is a live save facing him."""
    s = server()
    s.load(HERE / "test_mt_chimney_busy_grunt.state", wild_encounters=False)
    said = []
    for _ in range(2):
        s.call("mcp_press_buttons", {"buttons": ["A"], "reasoning": "test"})
        r = s.call("mcp_auto_advance_dialogue", {})
        print("   ", r.get("summary"))
        assert not _in_battle(s)
        said.append(r["summary"])
    assert "The person at (23,17) is not a trainer to battle." in said[0], said[0]
    assert "Trainers still to battle on this map:" in said[0] and "talk_to(9, 16)" in said[0], said[0]
    assert "talking to them again only repeats the same words" in said[1], said[1]


def test_navigate_to_a_persons_tile_walks_next_to_them():
    """Seen live on MT CHIMNEY (12,16), 15:14-15:16 UTC: navigate_to(13,16) four times "to
    get closer"; a beaten grunt stands there, and each walk ended "BLOCKED … (13,16) is
    occupied by someone". A person's tile as the target now means the free tile next to
    them, with the talk_to call to use."""
    from server.field_control import npc_at

    s = server()
    s.load(HERE / "test_mt_chimney_busy_grunt.state", wild_encounters=False)
    r = s.call("mcp_navigate_to", {"x": 23, "y": 17, "variance": "none", "reason": "test"})
    print("   ", r.get("summary"))
    assert r["success"] is True and "already next to them at (23,16)" in r["summary"], r
    assert "talk_to(23, 17)" in r["summary"], r
    with s.srv.memory_lock:
        assert npc_at(_reader(s), 21, 19)
    r = s.call("mcp_navigate_to", {"x": 21, "y": 19, "variance": "none", "reason": "test"})
    print("   ", r.get("summary"))
    here = s.snapshot()
    assert abs(here.x - 21) + abs(here.y - 19) == 1, (here, r)
    assert "(21,19) is where someone stands" in r["summary"] and "talk_to(21, 19)" in r["summary"], r


def test_a_trainer_who_spots_the_player_on_the_way_is_reported_as_a_battle():
    """Seen live: walking to Verdanturf's Pokémon Center to restore BRELOOM's PP, a trainer
    spotted the player; the walk reported "something has the player's attention on the
    way", and the model fought him with no damaging PP. The spotting scene ("!", the walk
    up, "Listen, could I get you to battle…") reads as neither plain nor busy text, and
    the battle starts by itself ~2 s later. From test_route117_blocked.state the way west
    passes Breeder Isaac."""
    s = server()
    s.load(HERE / "test_route117_blocked.state", wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "Verdanturf Town", "reasoning": "test"})
    print("   ", r.get("error") or r.get("summary"))
    assert r["success"] is False, r
    assert "a trainer battle started on the way: fight it with use_move, then" in r["error"], r
    assert _in_battle(s)


def test_go_to_walks_map_after_map():
    """David (2026-11-04): "One call per objective." ROUTE 101 to OLDALE TOWN's Pokémon
    Center is a map edge and a door (wild grass on the way is run from); from inside
    ROUTE 117's Day Care, "Route 117" means the route outside."""
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_go_to", {"place": "Oldale Town Pokemon Center", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and r["status"] == "ARRIVED", r
    assert s.srv._location_name() == "OLDALE TOWN POKEMON CENTER 1F", s.srv._location_name()
    assert "Walked OLDALE TOWN → OLDALE TOWN POKEMON CENTER 1F" in r["summary"], r
    s.load(HERE / "test_route117_daycare.state")
    r = s.call("mcp_go_to", {"place": "Route 117", "reasoning": "test"})
    assert r["success"] is True and s.srv._location_name() == "ROUTE 117", (s.srv._location_name(), r)
    r = s.call("mcp_go_to", {"place": "Route 117", "reasoning": "test"})
    assert r["status"] == "ARRIVED" and "already" in r["summary"], r
    # Seen live from ROUTE 117: "Mauville City Poke Mart can't be reached on foot".
    r = s.call("mcp_go_to", {"place": "Mauville City Poke Mart", "reasoning": "test"})
    assert r["success"] is True and s.srv._location_name() == "MAUVILLE CITY MART", (s.srv._location_name(), r)
    r = s.call("mcp_go_to", {"place": "Nowhere Town", "reasoning": "test"})
    assert r["success"] is False and "Did not reach Nowhere Town" in r["error"], r


def test_go_to_from_south_route111_goes_round_the_desert():
    """Seen live (12:48-12:54 UTC): go_to("Route 113") from ROUTE 111 (0,66), south of the
    desert, bounced ROUTE 111 → ROUTE 112 → ROUTE 111 … for fifteen maps. Route 111's west
    side joins both ROUTE 113 (top) and ROUTE 112, and "west" took the nearest, 112; and
    the desert between counted as walkable, though without Go-Goggles its sandstorm
    pushes the player back (pret Route111_EventScript_ViciousSandstormTrigger). The way
    on foot is round by Fiery Path. test_route111_south_to_113.state is a live save from
    that bounce, at ROUTE 112 (39,46), the far side of the 111/112 edge."""
    s = server()
    s.load(HERE / "test_route111_south_to_113.state", wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "Route 113", "reasoning": "test"})
    print("   ", r.get("error") or r.get("summary"))
    where = s.srv._location_name()
    if r["success"]:
        assert where == "ROUTE 113", (where, r)
    else:
        # The way passes unbeaten trainers; a battle with one is a fair stop.
        assert "a trainer battle started on the way" in r["error"], r
        assert _in_battle(s)
    walked = r.get("summary") or r.get("error")
    assert "ROUTE 112 → ROUTE 111 → ROUTE 112" not in walked and "ROUTE 111 → ROUTE 112 → ROUTE 111" not in walked, r


def test_go_to_jagged_pass_rides_the_cable_car_to_its_top():
    """emerald-new (2026-11-04): at 18:03 from the Lavaridge Pokémon Center, go_to("Jagged
    Pass") walked into its bottom (14,40), below one-way ledges, where the tall grass
    can't be reached. From the top (Mt. Chimney, by Cable Car) the ledges lead everywhere.
    A way into a map is dropped when another way in reaches all of its part and more;
    Route 104's halves, joined only through Petalburg Woods, are not like that. Hints for
    the model (way_to_place) never name the ride: only go_to can take it.
    test_mt_chimney_busy_grunt.state: MT CHIMNEY before Maxie (grunts stand on row 39, so
    only the ride down is tried there). test_lavaridge_after_chimney.state: the live save
    from 18:03, LAVARIDGE TOWN (6,7) outside the Pokémon Center, after Maxie."""
    from server import game_tools

    route = game_tools.place_route("LAVARIDGE TOWN", (12, 15), "Jagged Pass")
    print("    from Lavaridge:", route)
    assert route and route[-1] == ("JAGGED PASS", "JAGGED PASS"), route
    assert (game_tools.RIDE_EXIT, "MT CHIMNEY CABLE CAR STATION") in route, route
    assert game_tools.place_route("PETALBURG CITY", (2, 15), "Route 104")[0][1] == "ROUTE 104"
    assert "CABLE CAR\"" not in game_tools.way_to_place("LAVARIDGE TOWN", (12, 15), "Jagged Pass")

    s = server()
    s.load(HERE / "test_mt_chimney_busy_grunt.state", wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "Route 112 Cable Car Station", "reasoning": "test"})
    print("   ", r.get("error") or r.get("summary"))
    assert r["success"] and s.srv._location_name() == "ROUTE 112 CABLE CAR STATION", r
    assert "MT CHIMNEY CABLE CAR STATION → ROUTE 112 CABLE CAR STATION" in r["summary"], r
    s.load(HERE / "test_lavaridge_after_chimney.state", wild_encounters=False)
    r = s.call("mcp_go_to", {"place": "Jagged Pass", "reasoning": "test"})
    print("   ", r.get("error") or r.get("summary"))
    snap = s.snapshot()
    assert r["success"] and s.srv._location_name() == "JAGGED PASS", r
    assert snap.y < 23, f"arrived below the grass at ({snap.x},{snap.y})"


def test_route_plans_know_which_part_of_route111_they_start_in():
    """From ROUTE 111's north part (0,31) Route 113 is the west edge's top span, named by
    its map because ROUTE 112 is on the west side too (seen live: "west" walked into 112
    from there). From the south part the plan goes round by Fiery Path, unless the bag
    holds the Go-Goggles, which open the desert (the gate's own checkitem)."""
    from server import game_tools

    s = server()
    s.load(HERE / "test_route111_south_to_113.state", wild_encounters=False)
    assert game_tools.place_route("ROUTE 111", (0, 31), "Route 113") == [("ROUTE 113", "ROUTE 113")]
    south = [name for _, name in game_tools.place_route("ROUTE 111", (0, 66), "Route 113")]
    print("    from the south:", " → ".join(south))
    assert "FIERY PATH" in south and south[-1] == "ROUTE 113", south
    real = game_tools.items_reader
    try:
        game_tools.items_reader = lambda: real() | {"GOGOGGLES"}
        game_tools._items_cache[0] = 0.0
        assert game_tools.place_route("ROUTE 111", (0, 66), "Route 113") == [("ROUTE 113", "ROUTE 113")]
    finally:
        game_tools.items_reader = real
        game_tools._items_cache[0] = 0.0


def test_a_walk_that_loops_between_two_maps_stops():
    """Seen live: a wrong plan bounced ROUTE 111 ↔ ROUTE 112 until go_to's fifteen-map
    budget ran out (six minutes). A plan that sends the walk off the same map by the same
    exit twice stops there, saying so. The plan here is made to loop on purpose: north
    from ROUTE 101, south from OLDALE TOWN, toward a goal never reached."""
    import asyncio

    s = server()
    s.load(ROUTE101, wild_encounters=False)

    def looping(location, position):
        return [("north", "OLDALE TOWN")] if location == "ROUTE 101" else [("south", "ROUTE 101")]

    entered, why = asyncio.run(s.srv._walk_along(looping, lambda: False, "Nowhere", "test", "go_to again", 15))
    print("   ", entered, why)
    assert entered == ["OLDALE TOWN", "ROUTE 101"], entered
    assert why and "the way loops" in why and 'go_to_exit("north") from ROUTE 101' in why, why


def test_heal_from_a_route_walks_to_the_next_town():
    """ROUTE 101 to OLDALE TOWN's center: a map edge, a door, and wild grass on the way
    (a wild battle there is run from)."""
    s = server()
    s.load(ROUTE101)
    _write(s, 16, PARTY + 0x56, 5)  # RAM: the starter at 5 HP
    r = s.call("mcp_heal_at_pokecenter", {})
    print("   ", r.get("summary") or r)
    assert r["success"] is True, r
    assert "OLDALE TOWN POKEMON CENTER 1F" in r["summary"], r["summary"]
    lead = _party(s)[0]
    assert lead.current_hp == lead.max_hp, lead


# -- run_from_battle ----------------------------------------------------------


def test_run_escapes_a_wild_battle():
    s = server()
    s.load(WILD)
    assert _in_battle(s)
    r = s.call("mcp_run_from_battle", {})
    print("   ", r)
    assert r["success"] is True and r["status"] == "ESCAPED", r
    assert "Got away safely from wild SHROOMISH" in r["summary"], r
    assert not _in_battle(s)
    assert _read(s, lambda r: r._read_u8(BATTLE_OUTCOME)) == 4  # B_OUTCOME_RAN
    assert s.srv._player_can_walk()


def test_run_presses_through_the_intro_text_first():
    """'Wild SHROOMISH appeared!' waits for A; the macro reads that and presses it."""
    s = server()
    s.load(WILD_INTRO)
    assert _read(s, lambda r: r.read_battle_view())["menu"] == "busy"
    r = s.call("mcp_run_from_battle", {})
    print("   ", r)
    assert r["success"] is True and not _in_battle(s), r


def test_run_that_fails_says_the_battle_continues():
    s = server()
    s.load(WILD)
    # RAM: Torchic speed 1, Shroomish speed 500, so the escape check can't pass.
    _write(s, 16, BATTLE_MONS + 6, 1)
    _write(s, 16, BATTLE_MONS + 0x58 + 6, 500)
    r = s.call("mcp_run_from_battle", {})
    print("   ", r)
    assert r["success"] is False and r["status"] == "STILL_IN_BATTLE", r
    assert "Couldn't get away" in r["summary"] and "continues" in r["summary"], r
    assert _in_battle(s)
    assert _read(s, lambda r: r.read_battle_view())["menu"] == "action"


def test_run_refuses_a_trainer_battle():
    s = server()
    s.load(TRAINER)
    r = s.call("mcp_run_from_battle", {})
    print("   ", r)
    assert r["success"] is False and "can't run from a trainer battle" in r["error"], r
    view = _read(s, lambda r: r.read_battle_view())
    assert view["menu"] == "move" and view["move_cursor_slot"] == 4, view  # nothing pressed


# -- throw_ball ---------------------------------------------------------------


def test_throw_ball_uses_one_ball_and_reports_the_result():
    s = server()
    s.load(WILD)
    r = s.call("mcp_throw_ball", {"ball": "Poke Ball"})
    print("   ", r)
    assert r["success"] is True, r
    assert _balls(s)["Poke Ball"] == 4, _balls(s)
    assert (r["balls_before"], r["balls_after"]) == (5, 4), r
    if r["caught"]:
        assert not _in_battle(s) and len(_party(s)) == 2, r
    else:
        assert r["status"] == "BROKE_FREE" and _in_battle(s), r
        assert _read(s, lambda r: r.read_battle_view())["menu"] == "action"
        assert len(_party(s)) == 1


def test_throw_great_ball_catch_declines_the_nickname():
    s = server()
    s.load(WILD)
    # RAM: wild Shroomish at 1 HP and asleep makes the catch certain (catch rate 255).
    _write(s, 16, BATTLE_MONS + 0x58 + 0x28, 1)
    _write(s, 32, BATTLE_MONS + 0x58 + 0x4C, 3)
    r = s.call("mcp_throw_ball", {"ball": "great"})
    print("   ", r)
    assert r["success"] is True and r["caught"] is True and r["status"] == "CAUGHT", r
    assert "nickname" in r["summary"] and "joined your party" in r["summary"], r
    assert not _in_battle(s)
    assert _read(s, lambda r: r._read_u8(BATTLE_OUTCOME)) == 7  # B_OUTCOME_CAUGHT
    party = _party(s)
    assert [p.species_name for p in party] == ["Torchic", "Shroomish"], party
    assert party[1].nickname == "SHROOMISH", party[1]  # NO to the nickname question
    assert "Great Ball" not in _balls(s), _balls(s)


def test_throw_ball_refuses_a_ball_you_dont_have():
    s = server()
    s.load(WILD)
    r = s.call("mcp_throw_ball", {"ball": "Ultra Ball"})
    print("   ", r)
    assert r["success"] is False and "Poke Ball x5" in r["error"] and "Great Ball x1" in r["error"], r
    assert _read(s, lambda r: r.read_battle_view())["menu"] == "action"


def test_throw_ball_refuses_a_trainer_battle():
    s = server()
    s.load(TRAINER)
    r = s.call("mcp_throw_ball", {})
    print("   ", r)
    assert r["success"] is False and "trainer" in r["error"], r
    assert _balls(s)["Poke Ball"] == 5


# -- switch_pokemon -----------------------------------------------------------


def _active_personality(s):
    return _read(s, lambda r: r._read_u32(BATTLE_MONS + 0x48))


def _party_personality(s, index):
    return _read(s, lambda r: r._read_u32(PARTY + 100 * index))


def test_switch_sends_out_the_named_pokemon():
    s = server()
    s.load(WILD_TWO)
    assert _active_personality(s) == _party_personality(s, 0)  # Torchic leads
    r = s.call("mcp_switch_pokemon", {"pokemon": "Shroomish"})
    print("   ", r)
    assert r["success"] is True and r["status"] == "SWITCHED", r
    assert _active_personality(s) == _party_personality(s, 1), "Shroomish should be in battle"
    assert _read(s, lambda r: r._read_u16(BATTLE_MONS)) == 306  # SPECIES_SHROOMISH
    assert "TORCHIC came back and SHROOMISH" in r["summary"], r


def test_switch_by_slot_number():
    s = server()
    s.load(WILD_TWO)
    r = s.call("mcp_switch_pokemon", {"pokemon": 2})
    assert r["success"] is True, r
    assert _active_personality(s) == _party_personality(s, 1)


def test_switch_refuses_the_active_pokemon():
    s = server()
    s.load(WILD_TWO)
    r = s.call("mcp_switch_pokemon", {"pokemon": "TORCHIC"})
    print("   ", r)
    assert r["success"] is False and "already in battle" in r["error"], r
    assert _read(s, lambda r: r.read_battle_view())["menu"] == "action"


def test_switch_refuses_a_fainted_pokemon():
    s = server()
    s.load(WILD_TWO)
    _write(s, 16, PARTY + 100 + 0x56, 0)  # RAM: Shroomish fainted
    r = s.call("mcp_switch_pokemon", {"pokemon": "Shroomish"})
    print("   ", r)
    assert r["success"] is False and "has fainted" in r["error"], r
    assert _active_personality(s) == _party_personality(s, 0)


# -- buy_item -----------------------------------------------------------------


def _money(s):
    return _read(s, lambda r: r.read_money())


def _bag_count(s, name):
    inventory = _read(s, lambda r: r.read_full_inventory())
    return sum(e["quantity"] for pocket in inventory.values() for e in pocket if e["name"] == name)


def test_buy_potions():
    s = server()
    s.load(MART)
    assert _money(s) == 2784 and _bag_count(s, "Potion") == 0
    r = s.call("mcp_buy_item", {"item": "Potion", "count": 3})
    print("   ", r)
    assert r["success"] is True and r["status"] == "BOUGHT", r
    assert _money(s) == 2784 - 3 * 300, _money(s)
    assert _bag_count(s, "Potion") == 3
    assert "money $2784 -> $1884" in r["summary"] and "Potion in bag 0 -> 3" in r["summary"], r
    assert s.srv._player_can_walk(), "the shop should be closed"


def test_buy_twelve_poke_balls_counts_by_tens():
    """12 = RIGHT (+10) then UP, UP; 10+ Poké Balls also bring a free Premier Ball."""
    s = server()
    s.load(MART)
    r = s.call("mcp_buy_item", {"item": "POKé BALL", "count": 12})
    print("   ", r)
    assert r["success"] is True, r
    assert _bag_count(s, "Poke Ball") == 5 + 12
    assert _money(s) == 2784 - 12 * 200
    assert "Premier Ball x1" in r["summary"], r


def test_buy_refuses_an_item_not_sold_and_lists_the_stock():
    s = server()
    s.load(MART)
    r = s.call("mcp_buy_item", {"item": "Ultra Ball", "count": 1})
    print("   ", r)
    assert r["success"] is False and "doesn't sell Ultra Ball" in r["error"], r
    assert "Potion $300" in r["error"] and "Poke Ball $200" in r["error"], r
    assert _money(s) == 2784
    assert _wait(s.srv._player_can_walk), "the shop should be closed"


def test_buy_refuses_more_than_you_can_afford():
    s = server()
    s.load(MART)
    r = s.call("mcp_buy_item", {"item": "Potion", "count": 20})
    print("   ", r)
    assert r["success"] is False and "you can afford 9" in r["error"], r
    assert _money(s) == 2784 and _bag_count(s, "Potion") == 0


def test_buy_refuses_outside_a_mart():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_buy_item", {"item": "Potion", "count": 1})
    print("   ", r)
    assert r["success"] is False and "no shop clerk here" in r["error"], r


def _give_hm06(s):
    """RAM: HM06 (item 344) into the first empty TMs & HMs slot of SaveBlock1; bag
    quantities are XORed with the save's key (pret: gSaveBlock2Ptr->encryptionKey)."""
    from pokemon_env.emerald_utils import SAVE_BLOCK_1_OFFSETS

    reader = _reader(s)
    with s.srv.memory_lock:
        pocket = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + SAVE_BLOCK_1_OFFSETS["bagPocket_TMHM"]
        slot = next(i for i in range(64) if reader._read_u16(pocket + 4 * i) == 0)
        key = reader._get_security_key() & 0xFFFF
    _write(s, 16, pocket + 4 * slot, 344)
    _write(s, 16, pocket + 4 * slot + 2, 1 ^ key)


def test_set_lead_puts_a_party_pokemon_first():
    """David (2026-11-04): BRELOOM, always first, took every battle's EXP; the trainees
    should lead. set_lead goes START → POKéMON → the Pokémon → SWITCH → slot 1; the order
    is read back from gPlayerParty. RAM: gPartyMenu.slotId (+9) left on slot 3, so the
    party cursor doesn't start on the lead."""
    from server import menu_state as ms

    s = server()
    s.load(HERE / "test_marill_in_party.state", wild_encounters=False)
    before = _read(s, lambda rd: [p.species_name.upper() for p in rd.read_party_pokemon()])
    print("    before:", before)
    assert before[0] == "BRELOOM" and "MARILL" in before[1:], before
    _write(s, 8, ms.G_PARTY_MENU + 9, 2)
    r = s.call("mcp_set_lead", {"pokemon": "MARILL", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    after = _read(s, lambda rd: [p.species_name.upper() for p in rd.read_party_pokemon()])
    assert r["success"] is True and after[0] == "MARILL", (after, r)
    assert sorted(after) == sorted(before), after  # a swap, nobody lost
    assert s.srv._player_can_walk()
    r = s.call("mcp_set_lead", {"pokemon": "MARILL", "reasoning": "test"})
    assert r["success"] is True and "already leads" in r["summary"], r
    r = s.call("mcp_set_lead", {"pokemon": "PIKACHU", "reasoning": "test"})
    assert r["success"] is False and "not in your party" in r["error"], r


def test_teach_move_teaches_an_hm_and_forgets_the_named_move():
    """Seen live: told to teach Rock Smash, the model pressed B over and over. By hand the
    party screen's cursor started on slot 2, and A at once picked COMBUSKEN. RAM: HM06 in
    the bag (test_marill_in_party.state has none) and gPartyMenu.slotId (+9) left on 1."""
    from server import menu_state as ms

    s = server()
    s.load(HERE / "test_marill_in_party.state", wild_encounters=False)
    _give_hm06(s)
    _write(s, 8, ms.G_PARTY_MENU + 9, 1)
    r = s.call("mcp_teach_move", {"item": "HM06", "pokemon": "BRELOOM", "forget": "LEECH SEED", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["summary"].startswith("BRELOOM learned ROCK SMASH (forgot LEECH SEED)"), r
    moves = _read(s, lambda rd: [p.moves for p in rd.read_party_pokemon()])
    assert moves[0] == ["MACH_PUNCH", "HEADBUTT", "MEGA_DRAIN", "ROCK_SMASH"], moves[0]
    assert "ROCK_SMASH" not in moves[1], moves[1]  # COMBUSKEN untouched
    assert s.srv._player_can_walk()
    # Refusals: knows it already; a Pokémon the HM's ROM bit says is NOT ABLE; not in the bag.
    r = s.call("mcp_teach_move", {"item": "ROCK SMASH", "pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "already knows ROCK SMASH" in r["error"], r
    party = _read(s, lambda rd: rd.read_party_pokemon())
    # HM05 (FLASH, item 343) is in this bag already.
    unable = [p for p in party if not _read(s, lambda rd: ms.can_learn_tmhm(rd, p.species_id, 343))]
    print("    NOT ABLE for HM05:", [p.species_name for p in unable])
    assert unable, "expected a party Pokémon that can't learn FLASH"
    for p in unable:
        r = s.call("mcp_teach_move", {"item": "HM05", "pokemon": p.species_name, "reasoning": "test"})
        assert r["success"] is False and "NOT ABLE" in r["error"], r
    r = s.call("mcp_teach_move", {"item": "TM26", "pokemon": "BRELOOM", "reasoning": "test"})
    assert r["success"] is False and "not in your TMs & HMs pocket" in r["error"], r


def test_teach_move_backs_out_cleanly_when_the_forget_cursor_misses():
    """Seen live (WALLY'S HOUSE, 2026-11-04): teach_move("HM03", AZUMARILL, forget DEFENSE
    CURL) answered "Could not put the forget cursor…; Could not close every menu" and left
    the forget list open; the model mashed A/DOWN/B and SURF went in over BUBBLE BEAM. B on
    that list asks "Stop trying to teach…?", where B answers NO and reopens it. Here the
    cursor step is made to miss (the forget step's _step_cursor does nothing): the macro
    must answer YES to stop, close every menu, and forget nothing."""
    from server import menu_state as ms

    s = server()
    s.load(HERE / "test_marill_in_party.state", wild_encounters=False)
    _give_hm06(s)
    _write(s, 8, ms.G_PARTY_MENU + 9, 1)
    before = _read(s, lambda rd: [p.moves for p in rd.read_party_pokemon()])

    async def stuck(read, goal, source):
        return False

    real = s.srv._step_cursor
    s.srv._step_cursor = stuck
    try:
        r = s.call("mcp_teach_move", {"item": "HM06", "pokemon": "BRELOOM", "forget": "LEECH SEED", "reasoning": "test"})
    finally:
        s.srv._step_cursor = real
    print("   ", r)
    assert r["success"] is False and "nothing was forgotten" in r["error"], r
    assert "Could not close every menu" not in r["error"], r
    assert s.srv._player_can_walk(), "a menu was left open"
    assert _read(s, lambda rd: [p.moves for p in rd.read_party_pokemon()]) == before, "a move was forgotten"


def test_teach_move_keeps_a_table_hm_on_its_carrier():
    """David's HM protocol (pokemon_env/hm_carriers.py): FLASH goes on TROPIUS, never on a
    fighter. test_fly_ready.state (see test_fly.py) has HM05 (FLASH) and HM06 (ROCK SMASH,
    not in the table) in the bag, TROPIUS in the party, and SABLEYE and BRELOOM able to
    learn FLASH; test_fortree_gym.state has HM05 too and no TROPIUS. No RAM writes."""
    from pokemon_env import hm_carriers

    assert hm_carriers.HM_CARRIERS["FLASH"] == "TROPIUS"
    s = server()
    s.load(HERE / "test_fly_ready.state", wild_encounters=False)
    before = _read(s, lambda rd: [p.moves for p in rd.read_party_pokemon()])
    for fighter in ("SABLEYE", "BRELOOM"):
        r = s.call("mcp_teach_move", {"item": "HM05", "pokemon": fighter, "reasoning": "test"})
        print("   ", r.get("error") or r)
        assert r["success"] is False, r
        assert r["error"].startswith(f"HM05 (FLASH) goes on TROPIUS, not {fighter}"), r
        assert "keeps HMs on their carrier and off the fighters" in r["error"], r
        assert "not in your party" not in r["error"], r
    assert s.srv._player_can_walk(), "a menu was opened"
    assert _read(s, lambda rd: [p.moves for p in rd.read_party_pokemon()]) == before
    # The guard leaves TMs and HMs outside the table alone (they go on to the menus as before).
    party = _party(s)
    assert s.srv._hm_carrier_refusal({"item_id": 344, "name": "HM06"}, "ROCK SMASH", "SABLEYE", party) is None
    assert s.srv._hm_carrier_refusal({"item_id": 327, "name": "TM39"}, "ROCK TOMB", "SABLEYE", party) is None
    # The carrier itself is taught: the whole menu flow runs and FLASH replaces SWEET SCENT.
    r = s.call("mcp_teach_move", {"item": "HM05", "pokemon": "TROPIUS", "forget": "SWEET SCENT", "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and r["summary"].startswith("TROPIUS learned FLASH (forgot SWEET SCENT)"), r
    tropius = next(p for p in _party(s) if p.species_name.upper() == "TROPIUS")
    assert "FLASH" in tropius.moves and "SWEET_SCENT" not in tropius.moves, tropius.moves
    assert s.srv._player_can_walk()
    # Without the carrier in the party, the refusal says where it is.
    s.load(HERE / "test_fortree_gym.state", wild_encounters=False)
    r = s.call("mcp_teach_move", {"item": "HM05", "pokemon": "MANECTRIC", "reasoning": "test"})
    print("   ", r.get("error") or r)
    assert r["success"] is False and r["error"].startswith("HM05 (FLASH) goes on TROPIUS, not MANECTRIC"), r
    assert r["error"].endswith("TROPIUS is not in your party: withdraw it from the PC first."), r


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
