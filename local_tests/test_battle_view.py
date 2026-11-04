"""Battle view and use_move: the prompt names the opponent and the best move, and
use_move steers the battle menus to it.

Seen live: the battle section said only "Opponent detection disabled", every battle
opens with the move cursor on slot 1, and the model mashed A, so a Lv11 Torchic used
SCRATCH on a Lv5 Poochyena for 12+ minutes with the super-effective EMBER in slot 4.
Bug Catcher Lyle's trainer battle also showed as wild (wrong gBattleTypeFlags address).

Same in-process server harness as test_movement_feedback.py.

Fixtures:
  test_battle_wurmple.state     Petalburg Woods, Bug Catcher Lyle's Wurmple, move menu
                                open with the cursor on EMBER (slot 4)
  test_after_woods_grunt.state  overworld after a trainer battle
  test_movement_route101.state  overworld

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_battle_view.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import ROUTE101, server

HERE = Path(__file__).resolve().parent
BATTLE = HERE / "test_battle_wurmple.state"
AFTER_GRUNT = HERE / "test_after_woods_grunt.state"

# gBattleMons[0].pp[4] (pret struct BattlePokemon +0x24): the player's live PP.
PLAYER_PP = 0x02024084 + 0x24


def _view(s):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader.read_battle_view()


def _pp(s, slot):
    with s.srv.memory_lock:
        return s.srv.env.memory_reader._read_u8(PLAYER_PP + slot - 1)


def _wait_for_pp(s, slot, expected, timeout=30.0):
    """Let the turn play out (the server's game loop runs on its own thread)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _pp(s, slot) == expected:
            return True
        time.sleep(0.1)
    return False


def test_battle_view_on_lyle_wurmple():
    from utils.state_formatter import format_battle_view

    s = server()
    s.load(BATTLE)
    view = _view(s)
    lines = format_battle_view(view)
    print("    " + "\n    ".join(lines))
    assert view["battle_type"] == "trainer" and view["is_trainer"] and not view["is_double"], view
    assert view["menu"] == "move" and view["move_cursor_slot"] == 4, view
    opp = view["target"]
    assert (opp["name"], opp["level"], opp["types"], opp["hp"], opp["max_hp"]) == ("Wurmple", 3, ["BUG"], 15, 15), opp
    player = view["player"]
    assert (player["name"], player["level"], player["hp"], player["max_hp"]) == ("Torchic", 11, 24, 33), player
    by_name = {m["name"]: m for m in view["moves"]}
    assert list(by_name) == ["SCRATCH", "GROWL", "FOCUS ENERGY", "EMBER"], by_name
    ember = by_name["EMBER"]
    assert (ember["slot"], ember["type"], ember["power"], ember["pp"]) == (4, "FIRE", 40, 25), ember
    assert ember["effectiveness"] == 2.0 and ember["stab"] and ember["expected"] == 120.0, ember
    assert by_name["SCRATCH"]["effectiveness"] == 1.0 and by_name["GROWL"]["expected"] == 0, by_name
    assert view["best_move"]["name"] == "EMBER", view["best_move"]
    assert 'Best move now: EMBER (slot 4) — call use_move("EMBER").' in lines, lines
    assert any("Trainer battle vs WURMPLE Lv3 (BUG), HP 15/15" in line for line in lines), lines
    # The older fields other code reads are filled from the same structs now.
    with s.srv.memory_lock:
        info = s.srv.env.memory_reader.read_comprehensive_battle_info()
    assert info["is_trainer_battle"] and info["battle_type"] == "trainer", info
    assert info["opponent_pokemon"]["species"] == "Wurmple", info["opponent_pokemon"]
    assert info["player_pokemon"]["moves"] == ["SCRATCH", "GROWL", "FOCUS ENERGY", "EMBER"], info["player_pokemon"]


def test_the_agent_prompt_shows_the_battle_view():
    import asyncio

    s = server()
    s.load(BATTLE)
    text = asyncio.run(s.srv.mcp_get_game_state())["state_text"]
    assert "Trainer battle vs WURMPLE Lv3 (BUG), HP 15/15" in text, text
    assert 'call use_move("EMBER")' in text, text
    assert "Opponent detection disabled" not in text, text


def test_type_chart_from_rom():
    """Immunities after the 0xFE separator still apply, and two types multiply."""
    s = server()
    with s.srv.memory_lock:
        reader = s.srv.env.memory_reader
        assert reader.type_effectiveness(0, [7]) == 0.0      # NORMAL vs GHOST
        assert reader.type_effectiveness(4, [2, 5]) == 0.0   # GROUND vs FLYING/ROCK
        assert reader.type_effectiveness(11, [4, 5]) == 4.0  # WATER vs GROUND/ROCK
        assert reader.type_effectiveness(10, [6, 6]) == 2.0  # a single type counts once


def test_not_in_battle_on_the_overworld_after_a_battle():
    """gBattleTypeFlags stays set after a battle, so it must not decide is_in_battle."""
    s = server()
    for fixture in (AFTER_GRUNT, ROUTE101):
        s.load(fixture)
        with s.srv.memory_lock:
            reader = s.srv.env.memory_reader
            flags = reader._read_u32(reader.addresses.BATTLE_TYPE_FLAGS)
            print(f"    {fixture.name}: gBattleTypeFlags=0x{flags:X}")
            assert not reader.is_in_battle(), fixture.name
            assert reader.read_battle_view() is None, fixture.name


def test_grid_presses():
    grid = server().srv._grid_presses
    assert grid(3, 0) == ["LEFT", "UP"]
    assert grid(0, 0) == []
    assert grid(1, 2, [True, True, True, True]) == ["LEFT", "DOWN"]
    # Slot 4 empty: from slot 2 to slot 3 the cursor can't pass through slot 4.
    assert grid(1, 2, [True, True, True, False]) == ["LEFT", "DOWN"]
    assert grid(2, 1, [True, True, True, False]) == ["UP", "RIGHT"]


def test_use_move_from_the_move_menu():
    """The cursor sits on EMBER (slot 4); SCRATCH (slot 1) needs LEFT, UP, A."""
    s = server()
    s.load(BATTLE)
    assert _pp(s, 1) == 27
    r = s.call("mcp_use_move", {"move": "Scratch", "reasoning": "test"})
    print("   ", r)
    assert r["success"] is True and r["move"] == "SCRATCH", r
    assert r["summary"].startswith("Used SCRATCH (NORMAL, 1x vs BUG). Torchic HP 24/33; opponent WURMPLE HP 15/15"), r
    assert _wait_for_pp(s, 1, 26), f"SCRATCH PP is {_pp(s, 1)}, expected 26"
    assert _pp(s, 4) == 25


def test_use_move_from_the_action_menu():
    """B goes back to FIGHT/BAG/POKEMON/RUN; RIGHT puts the cursor on BAG first."""
    s = server()
    s.load(BATTLE)
    assert s.call("mcp_press_buttons", {"buttons": ["B", "RIGHT"], "reasoning": ""})["success"]
    view = _view(s)
    assert view["menu"] == "action" and view["action_cursor_name"] == "BAG", view
    r = s.call("mcp_use_move", {"move": "EMBER", "reasoning": "super effective"})
    print("   ", r)
    assert r["success"] is True, r
    assert "Used EMBER (FIRE, 2x vs BUG)" in r["summary"], r
    assert _wait_for_pp(s, 4, 24), f"EMBER PP is {_pp(s, 4)}, expected 24"
    assert _pp(s, 1) == 27


def test_use_move_by_slot_number():
    s = server()
    s.load(BATTLE)
    r = s.call("mcp_use_move", {"move": 4, "reasoning": ""})
    assert r["success"] is True and r["move"] == "EMBER", r
    assert _wait_for_pp(s, 4, 24)


def test_use_move_leaves_no_text_playing():
    """It used to return mid-turn and refuse a second call with "press A to continue";
    now it returns only after the turn (test_battle_flow.py covers the details)."""
    s = server()
    s.load(BATTLE)
    r = s.call("mcp_use_move", {"move": "EMBER", "reasoning": ""})
    print("   ", r)
    assert r["success"] is True, r
    view = _view(s)
    assert view is None or view["menu"] != "busy", view


def test_use_move_unknown_move_lists_the_moves():
    s = server()
    s.load(BATTLE)
    r = s.call("mcp_use_move", {"move": "FLY", "reasoning": ""})
    print("   ", r)
    assert r["success"] is False, r
    for name in ("SCRATCH (slot 1", "GROWL (slot 2", "FOCUS ENERGY (slot 3", "EMBER (slot 4"):
        assert name in r["error"], r
    view = _view(s)
    assert view["menu"] == "move" and view["move_cursor_slot"] == 4, view  # nothing pressed


def test_use_move_refuses_outside_a_battle():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_use_move", {"move": "EMBER", "reasoning": ""})
    assert r["success"] is False and "No battle is going on" in r["error"], r


ISAAC = HERE / "test_switch_question.state"  # SABLEYE out vs WHISMUR; COMBUSKEN, BRELOOM benched
FOE = 0x02024084 + 0x58  # gBattleMons[1]


def _poke(s, size, address, value):
    with s.srv.memory_lock:
        memory = s.srv.env.core.memory
        {8: memory.u8, 16: memory.u16, 32: memory.u32}[size][address] = value
        s.srv.env.memory_reader._invalidate_mem_cache()


def test_no_damaging_pp_names_the_switch():
    """Seen live: BRELOOM's three attacks were at 0 PP; the view said "switch Pokémon, use
    an item, or run", and the model used LEECH SEED five times instead. RAM: SABLEYE's
    damaging moves at 0 PP."""
    from utils.state_formatter import format_battle_view

    s = server()
    s.load(ISAAC)
    for m in _view(s)["moves"]:
        if m["power"]:
            _poke(s, 8, PLAYER_PP + m["slot"] - 1, 0)
    view = _view(s)
    lines = format_battle_view(view)
    print("    " + "\n    ".join(lines))
    assert view["best_move"] is None and view["switch_hint"]["name"] in ("COMBUSKEN", "BRELOOM"), view["switch_hint"]
    name = view["switch_hint"]["name"]
    assert any(f'No damaging move with PP left: call switch_pokemon("{name}")' in line for line in lines), lines
    # With damaging PP there is no switch hint.
    s.load(ISAAC)
    assert _view(s)["switch_hint"] is None


def test_use_move_refuses_a_status_move_that_cannot_work():
    """Seen live: LEECH SEED on a WINGULL already seeded, five times, each "missed". RAM
    writes put a status move in SABLEYE's slot 1 and change the foe (pret Cmd_setseeded,
    Cmd_trysetsleep family)."""
    from pokemon_env.enums import Move

    s = server()
    cases = [
        ("LEECH_SEED", lambda: _poke(s, 32, 0x020242AC + 4, 1 << 2), "already seeded"),  # gStatuses3[1]
        ("LEECH_SEED", lambda: (_poke(s, 8, FOE + 0x21, 12), _poke(s, 8, FOE + 0x22, 12)), "Grass type"),
        ("THUNDER_WAVE", lambda: (_poke(s, 8, FOE + 0x21, 4), _poke(s, 8, FOE + 0x22, 4)), "immune to Electric"),
        ("HYPNOSIS", lambda: _poke(s, 32, FOE + 0x4C, 2), "already has SLEEP"),  # status1: asleep
        ("TOXIC", lambda: (_poke(s, 8, FOE + 0x21, 8), _poke(s, 8, FOE + 0x22, 8)), "can't be poisoned"),
    ]
    for move, change_foe, why in cases:
        s.load(ISAAC)
        _poke(s, 16, 0x02024084 + 0x0C, Move[move].value)  # gBattleMons[0].moves[0]
        _poke(s, 8, PLAYER_PP, 10)
        name = move.replace("_", " ")
        with s.srv.memory_lock:
            assert s.srv.env.memory_reader.status_move_failure(Move[move].value, _view(s)["target"]) is None, move
        change_foe()
        menu_before = _view(s)["menu"]
        r = s.call("mcp_use_move", {"move": name, "reasoning": "test"})
        print("   ", r.get("error"))
        assert r["success"] is False and f"Not using {name}" in r["error"] and why in r["error"], r
        assert _view(s)["menu"] == menu_before  # nothing pressed


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
