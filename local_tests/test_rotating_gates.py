"""Rotating gates (pret src/rotating_gate.c): FORTREE CITY GYM's puzzle is planned push
by push, and navigate_to / talk_to reach Winona and every gym trainer through it.

Before (the pre-change code on this fixture): navigate_to(15,3) answered "BLOCKED moving
UP at (2,19) on FORTREE CITY GYM: (2,18) is not walkable"; (2,18) is floor, and the
gate at (3,19) can't turn there (test_a_gate_whose_sweep_meets_a_wall_blocks_the_step).

The model (utils/mapping/rotating_gates.py) is checked against the game itself, step by
step, on the fixture: every step of a planned walk must turn exactly the gate the model
says (orientations read from VAR_TEMP_0 on), and a step the model says a gate stops must
be stopped.

Fixture test_fortree_gym.state: FORTREE CITY GYM (2,19), walkable, party BLAZIKEN 36,
AZUMARILL 34, CAMERUPT 33, SABLEYE 35, MANECTRIC 33, BRELOOM 42; Winona and her six
trainers not beaten; gates [1, 2, 2, 1, 0, 3, 1, 2] (the player already turned three on
the way in). A live save from the AI's run.

Trainers battle when they spot the player; the full-route tests mark their trainer flags
beaten in RAM (as test_warp_routes.py does for Flannery's gym), and one test leaves them
and checks the battle stopping the walk is reported.

  docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD":/app \
      -v ~/continual-harness/Emerald-GBAdvance:/app/Emerald-GBAdvance:ro -w /app \
      emerald-harness:local local_tests/test_rotating_gates.py [name-substring]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import server

FORTREE_GYM = Path(__file__).resolve().parent / "test_fortree_gym.state"
GYM = "FortreeCity_Gym"
WINONA = (15, 2)
BELOW_WINONA = (15, 3)
# pret FortreeCity_Gym map.json warp 0: the gym door; RotatingGate_InitPuzzle resets the
# gates to the config's orientations whenever the player comes in through it.
ENTRANCE = (15, 23)
FIXTURE_GATES = (1, 2, 2, 1, 0, 3, 1, 2)
DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def _beside(x, y):
    return [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]


def _read(s, fn):
    with s.srv.memory_lock:
        return fn(s.srv.env.memory_reader)


def _gates(s):
    from server import game_tools

    return _read(s, lambda r: game_tools.live_gate_orientations(s.srv.env, GYM))


def _plan_state(s):
    from server import game_tools

    with s.srv.memory_lock:
        state, offset, _ = game_tools.load_pathfinding_state(s.srv.env)
    assert offset is None, offset
    return state


def _route(state, start, goal, gates=None):
    """(buttons, gates after) of a whole planned walk, or (None, None)."""
    from utils.mapping.pathfinding import Pathfinder

    porymap = state["map"]["porymap"]
    if gates is not None:
        porymap = {**porymap, "gate_orientations": tuple(gates)}
        state = {**state, "map": {**state["map"], "porymap": porymap}}
    from server.field_control import end_of_path

    finder = Pathfinder()
    buttons = finder.find_path(start, goal, state, consider_npcs=True, allow_partial=False)
    # A goal that can't be stood on is swapped for a reachable tile near it: not a route.
    if not buttons or end_of_path(start, buttons) != goal:
        return None, None
    return buttons, finder.last_gate_orientations


def _trainers():
    from utils.mapping.porymap_state import _map_trainers

    return [t for t in _map_trainers(GYM) if t[2] != "WINONA"]


def _load(s, beaten=True):
    """The fixture; with beaten, every gym trainer but Winona marked beaten in RAM
    (TRAINER_FLAGS_START + id in SaveBlock1, pret include/constants/flags.h)."""
    from utils.mapping.porymap_state import _TRAINER_FLAGS_START

    s.load(FORTREE_GYM, wild_encounters=False)
    if beaten:
        with s.srv.memory_lock:
            reader = s.srv.env.memory_reader
            flags = reader._read_u32(reader.addresses.SAVE_BLOCK1_PTR) + reader.addresses.SAVE_BLOCK1_FLAGS_OFFSET
            for _, _, _, trainer_id in _trainers():
                flag = _TRAINER_FLAGS_START + trainer_id
                s.srv.env.core.memory.u8[flags + flag // 8] |= 1 << (flag % 8)
            reader._invalidate_mem_cache()
    snap = s.snapshot()
    assert (snap.x, snap.y) == (2, 19) and s.srv._location_name() == "FORTREE CITY GYM", snap


def _press(s, button):
    r = s.call("mcp_press_buttons", {"buttons": [button], "reasoning": "test"})
    return r


# --- the model and the planner (fixture's gates, no walking) -----------------------------


def test_fixture_is_the_gym_with_its_trainers_not_beaten():
    """The fixture is what the tests assume: in the gym, gates as listed, nobody beaten."""
    from utils.mapping.porymap_state import _TRAINER_FLAGS_START, _map_trainers

    s = server()
    _load(s, beaten=False)
    assert _gates(s) == FIXTURE_GATES, _gates(s)
    trainers = _map_trainers(GYM)
    assert {t[2] for t in trainers} == {"WINONA", "JARED", "FLINT", "ASHLEY", "EDWARDO", "HUMBERTO", "DARIUS"}, trainers
    beaten = _read(s, lambda r: [r._is_event_flag_set(_TRAINER_FLAGS_START + t[3]) for t in trainers])
    assert not any(beaten), beaten


def test_planner_reaches_winona_and_every_trainer_through_the_gates():
    """From the fixture's tile and gates, and from the door with the gates as entering
    resets them: a whole walk to the tile below Winona and to a tile beside each gym
    trainer, every one turning gates on the way. Without the gate model the planner
    walked straight through the arms (no orientations, nothing turned)."""
    s = server()
    _load(s)
    state = _plan_state(s)
    porymap = state["map"]["porymap"]
    puzzle = porymap.get("rotating_gates")
    assert puzzle is not None and tuple(porymap.get("gate_orientations")) == FIXTURE_GATES, porymap.get("gate_orientations")
    for start, gates in (((2, 19), FIXTURE_GATES), (ENTRANCE, puzzle.initial)):
        buttons, after = _route(state, start, BELOW_WINONA, gates)
        print(f"    {start} gates {list(gates)} → below Winona: {len(buttons or [])} steps, gates after {after}")
        assert buttons and after is not None and after != tuple(gates), (start, buttons, after)
        for x, y, name, _ in _trainers():
            routes = [(_route(state, start, tile, gates), tile) for tile in _beside(x, y)]
            found = [(len(b), tile) for (b, _), tile in routes if b]
            print(f"    {start} → beside {name} ({x},{y}): {min(found) if found else None}")
            assert found, (start, name)


def test_a_gate_whose_sweep_meets_a_wall_blocks_the_step():
    """pret RotatingGate_CanRotate: the arm is pushed but a wall is in the sweep, so the
    step is blocked and nothing turns; pushing the same gate the other way turns it."""
    from utils.mapping import rotating_gates

    s = server()
    _load(s)
    state = _plan_state(s)
    puzzle = state["map"]["porymap"]["rotating_gates"]
    stops = [
        (gate_id, button)
        for gate_id, gate in enumerate(puzzle.gates)
        for button, table in rotating_gates.ROTATION_INFO.items()
        for (cx, cy), (_, arm, long) in table.items()
        if gate.has_arm(FIXTURE_GATES[gate_id], arm, long)
        and puzzle.step(FIXTURE_GATES, button, (gate.x + cx - 2, gate.y + cy - 2)) is None
    ]
    assert stops, "no gate in the fixture is stopped by a wall in any direction"
    # Worked by hand from pret's tables: gate 2 at (3,19) is T2 (north short+long, east,
    # south) at orientation 2, so its arms point south (long), west and north. From
    # (2,19) UP onto (2,18) is box tile (1,1) of sRotatingGate_RotationInfoNorth:
    # GATE_ROT_CW(GATE_ARM_WEST, 0), and the west arm is there. Turning clockwise, the
    # shape's south arm (world north now) sweeps sRotatingGate_ArmPositionsClockwiseRotation
    # [0] = (0,-1) from (3,19): (3,18), collision 1 in the layout. Blocked.
    gate = puzzle.gates[2]
    assert (gate.x, gate.y) == (3, 19) and puzzle.collision_at(3, 18) == 1
    assert puzzle.step(FIXTURE_GATES, "UP", (2, 18)) is None
    # From (2,19) RIGHT onto (3,19) crosses its south arm (box tile (2,2) of
    # sRotatingGate_RotationInfoEast: GATE_ROT_ACW(GATE_ARM_SOUTH, 0)): anticlockwise, 2 → 1.
    turned = puzzle.step(FIXTURE_GATES, "RIGHT", (3, 19))
    assert turned == FIXTURE_GATES[:2] + (1,) + FIXTURE_GATES[3:], turned


# --- live: the game agrees with the model --------------------------------------------


def test_every_step_of_a_planned_walk_turns_the_gates_the_model_says():
    """The walk to below Winona, one press per call: after each step the player is on
    the next tile and the gates read from VAR_TEMP_0 on are exactly the model's."""
    from utils.mapping import rotating_gates

    s = server()
    _load(s)
    state = _plan_state(s)
    puzzle = state["map"]["porymap"]["rotating_gates"]
    buttons, after = _route(state, (2, 19), BELOW_WINONA, FIXTURE_GATES)
    assert buttons, buttons
    gates, turns = FIXTURE_GATES, 0
    for i, button in enumerate(buttons):
        snap = s.snapshot()
        dx, dy = rotating_gates.DELTAS[button]
        dest = (snap.x + dx, snap.y + dy)
        expected = puzzle.step(gates, button, dest)
        assert expected is not None, (i, button, snap, gates)
        for _ in range(2):  # the first press may only turn the player to face that way
            _press(s, button)
            now = s.snapshot()
            if (now.x, now.y) == dest:
                break
        live = _gates(s)
        assert (now.x, now.y) == dest, (i, button, snap, now)
        assert live == expected, (i, button, dest, gates, expected, live)
        turns += live != gates
        gates = live
    print(f"    {len(buttons)} steps, {turns} gate turns, gates now {list(gates)}")
    assert turns >= 3 and gates == after, (turns, gates, after)


def test_a_push_the_model_says_is_stopped_is_stopped_and_reported():
    """Walk to a tile from which the model says a press is stopped by a gate, press it:
    the player stays, no gate turns, and BLOCKED names the gate, not the floor."""
    s = server()
    _load(s)
    state = _plan_state(s)
    puzzle = state["map"]["porymap"]["rotating_gates"]
    grid = state["map"]["porymap"]["grid"]
    raw = state["map"]["porymap"]["raw_tiles"]
    # Every (tile, press) a gate stops, nearest first, where the tile is floor.
    candidates = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            for button, (dx, dy) in DELTAS.items():
                dest = (x + dx, y + dy)
                if cell != "." or not (0 <= dest[1] < len(raw) and 0 <= dest[0] < len(raw[0])):
                    continue
                if raw[dest[1]][dest[0]][2] != 0 or raw[y][x][3] != raw[dest[1]][dest[0]][3]:
                    continue
                if puzzle.step(FIXTURE_GATES, button, dest) is None:
                    candidates.append((abs(x - 2) + abs(y - 19), (x, y), button))
    candidates.sort()
    assert candidates, "no gate-stopped press in the fixture"
    for _, tile, button in candidates:
        if tile == (2, 19):  # already there, gates as they are
            break
        buttons, after = _route(state, (2, 19), tile, FIXTURE_GATES)
        if buttons and after is not None and puzzle.step(after, button, (tile[0] + DELTAS[button][0], tile[1] + DELTAS[button][1])) is None:
            break
    else:
        raise AssertionError(f"no gate-stopped press reachable: {candidates[:5]}")
    if tile != (2, 19):
        r = s.call("mcp_navigate_to", {"x": tile[0], "y": tile[1], "variance": "none", "reason": "test"})
    snap = s.snapshot()
    assert (snap.x, snap.y) == tile, snap
    gates = _gates(s)
    print(f"    at {tile} gates {list(gates)}, pressing {button}")
    r = {}
    for _ in range(2):  # the first press may only turn the player
        r = _press(s, button)
        if r.get("status") == "BLOCKED":
            break
    print("   ", r.get("summary"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == tile and _gates(s) == gates, (snap, _gates(s), gates)
    assert r.get("status") == "BLOCKED" and "rotating gate" in r["summary"], r


def test_navigate_after_turning_a_gate_by_hand_plans_from_the_new_gates():
    """One press into a gate's arm turns it (as the model says); the next navigate_to
    plans from the turned gates and still reaches the tile below Winona."""
    from utils.mapping import rotating_gates

    s = server()
    _load(s)
    state = _plan_state(s)
    puzzle = state["map"]["porymap"]["rotating_gates"]
    pushes = [
        (button, (2 + dx, 19 + dy), puzzle.step(FIXTURE_GATES, button, (2 + dx, 19 + dy)))
        for button, (dx, dy) in rotating_gates.DELTAS.items()
    ]
    pushes = [p for p in pushes if p[2] is not None and p[2] != FIXTURE_GATES]
    assert pushes, "no press from (2,19) turns a gate"
    button, dest, expected = pushes[0]
    for _ in range(2):
        _press(s, button)
        if (s.snapshot().x, s.snapshot().y) == dest:
            break
    turned = _gates(s)
    print(f"    {button} onto {dest}: gates {list(FIXTURE_GATES)} → {list(turned)}")
    assert turned == expected, (turned, expected)
    fresh = _plan_state(s)
    assert tuple(fresh["map"]["porymap"]["gate_orientations"]) == turned
    r = s.call("mcp_navigate_to", {"x": BELOW_WINONA[0], "y": BELOW_WINONA[1], "variance": "none", "reason": "test"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert (snap.x, snap.y) == BELOW_WINONA and "Reached (15,3)" in r["summary"], (r, snap)


def test_a_walk_that_goes_off_its_plan_is_planned_again_from_the_game():
    """A plan made from gates that are no longer so (as when a trainer's scene ate a
    press, or a gate turned after the plan): the walk ends off the plan, navigate_to sees
    the game's position and gates differ, plans again from them and arrives."""
    from server import game_tools

    s = server()
    _load(s)
    for _ in range(2):  # turn gate (3,19) by hand: RIGHT onto (3,19)
        _press(s, "RIGHT")
        if (s.snapshot().x, s.snapshot().y) == (3, 19):
            break
    assert _gates(s) != FIXTURE_GATES, _gates(s)
    real = game_tools.rotating_gate_state
    stale = {"left": 1}

    def stale_once(env, map_dir, raw_tiles, offset=None):
        puzzle, gates = real(env, map_dir, raw_tiles, offset)
        if puzzle is not None and stale["left"]:
            stale["left"] -= 1
            return puzzle, FIXTURE_GATES
        return puzzle, gates

    game_tools.rotating_gate_state = stale_once
    try:
        r = s.call("mcp_navigate_to", {"x": BELOW_WINONA[0], "y": BELOW_WINONA[1], "variance": "none", "reason": "test"})
    finally:
        game_tools.rotating_gate_state = real
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert "did not go as planned" in r["summary"] and "planned again" in r["summary"], r
    assert (snap.x, snap.y) == BELOW_WINONA and "Reached (15,3)" in r["summary"], (r, snap)


def test_talk_to_winona_through_the_gates_when_her_trainers_are_beaten():
    """The whole crossing, end to end, with the gym's trainers marked beaten: talk_to
    walks the planned pushes, faces Winona and her battle starts."""
    s = server()
    _load(s)
    r = s.call("mcp_talk_to", {"x": WINONA[0], "y": WINONA[1], "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] and r["status"] in ("BATTLE", "TALKED"), r
    snap = s.snapshot()
    in_battle = _read(s, lambda m: m.is_in_battle())
    assert in_battle or (snap.x, snap.y) in _beside(*WINONA), (snap, r)
    assert "Talked to" in r["summary"] and "(15,2)" in r["summary"], r


def test_talk_to_reaches_every_gym_trainer_through_the_gates():
    """Each gym trainer in turn, from wherever the last talk left the player: talk_to
    reaches a tile beside them (they are marked beaten, so each only talks)."""
    s = server()
    _load(s)
    for x, y, name, _ in _trainers():
        r = s.call("mcp_talk_to", {"x": x, "y": y, "reasoning": "test"})
        print(f"    {name} ({x},{y}):", (r.get("summary") or r.get("error"))[:160])
        snap = s.snapshot()
        assert r["success"] and r["status"] == "TALKED", (name, r)
        assert (snap.x, snap.y) in _beside(x, y), (name, snap)
        for _ in range(4):  # close what they said before walking on
            from server.field_control import DIALOGUE_NONE

            if s.srv._field_status() == DIALOGUE_NONE:
                break
            _press(s, "B")


def test_go_to_exit_leaves_the_gym_through_the_gates():
    """go_to_exit plans with the same gate-aware planner: from the fixture's tile, out of
    the gym door to FORTREE CITY, turning gates (6,23) / (12,22) on the way. (On the
    pre-change code this one walk happened to work too, its straight path pushing the
    gates the right way; the planner check fails there.)"""
    s = server()
    _load(s)
    state = _plan_state(s)
    buttons, after = _route(state, (2, 19), ENTRANCE, FIXTURE_GATES)
    assert buttons and after != FIXTURE_GATES, (buttons, after)  # the way out turns gates
    r = s.call("mcp_go_to_exit", {"exit": "FORTREE CITY", "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    assert r["success"] and s.srv._location_name() == "FORTREE CITY", r


def _move_person(s, spawn, to):
    """Put the loaded person whose map-file tile is spawn at to (RAM: ObjectEvent
    currentCoords and previousCoords, +0x10 / +0x14, pret include/global.fieldmap.h), as a
    trainer who walked over to battle is left."""
    from server.field_control import G_OBJECT_EVENTS, MAP_OFFSET, OBJECT_EVENT_COUNT, OBJECT_EVENT_SIZE

    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        for i in range(OBJECT_EVENT_COUNT):
            base = G_OBJECT_EVENTS + i * OBJECT_EVENT_SIZE
            if not reader._read_u8(base) & 1:
                continue
            if (reader._read_s16(base + 0x0C) - MAP_OFFSET, reader._read_s16(base + 0x0E) - MAP_OFFSET) != spawn:
                continue
            for offset in (0x10, 0x14):
                s.srv.env.core.memory.u16[base + offset] = to[0] + MAP_OFFSET
                s.srv.env.core.memory.u16[base + offset + 2] = to[1] + MAP_OFFSET
            reader._invalidate_mem_cache()
            return
    raise AssertionError(f"nobody loaded with map-file tile {spawn}")


def test_people_are_taken_where_they_stand_not_at_their_map_file_tile():
    """Seen live: JARED walked over to battle and stayed at (4,16) while the map still put
    him at (4,14). The planner leaves (4,14) free and steps round (4,16); talk_to(4,14)
    talks to him where he stands."""
    from server.field_control import moved_people

    s = server()
    _load(s)
    _move_person(s, (4, 14), (4, 16))
    assert _read(s, moved_people) == [((4, 14), (4, 16))], _read(s, moved_people)
    state = _plan_state(s)
    porymap = state["map"]["porymap"]
    assert (4, 14) not in {(o.get("x"), o.get("y")) for o in porymap["objects"]}, porymap["objects"]
    # Each live person also carries their elevation now (who they stop on a bridge).
    live = {(o.get("current_x"), o.get("current_y")) for o in porymap["runtime_objects"]}
    assert (4, 16) in live, porymap["runtime_objects"]
    assert _route(state, (2, 19), (4, 15))[0] is None  # only through him
    r = s.call("mcp_talk_to", {"x": 4, "y": 14, "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    snap = s.snapshot()
    assert r["success"] and r["status"] == "TALKED", r
    assert "stands at (4,16)" in r["summary"] and (snap.x, snap.y) in _beside(4, 16), (snap, r)


def test_a_trainer_battle_stopping_the_walk_is_reported():
    """Trainers not beaten: the walk toward Winona is stopped by one who spots the
    player, and talk_to says a battle started rather than that the tile can't be reached."""
    s = server()
    _load(s, beaten=False)
    r = s.call("mcp_talk_to", {"x": WINONA[0], "y": WINONA[1], "reasoning": "test"})
    print("   ", r.get("summary") or r.get("error"))
    deadline = time.time() + 20
    while time.time() < deadline and not _read(s, lambda m: m.is_in_battle()):
        time.sleep(0.2)
    assert _read(s, lambda m: m.is_in_battle()), r
    text = r.get("summary") or r.get("error") or ""
    assert "A trainer battle (a trainer spotted you) started on the way to (15,2)" in text, r
    assert "talk_to(15, 2) again" in text, r
    assert "Could not reach" not in text and "No path" not in text, r


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
