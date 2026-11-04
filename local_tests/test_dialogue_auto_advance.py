"""Auto-advance presses A through plain dialogue and never answers a YES/NO or menu.

Same in-process server harness as test_movement_feedback.py, real savestates:
  test_dialogue_pokecenter.state  Oldale POKEMON CENTER 1F, player at (7,4) facing the
                                  nurse across the counter (made from
                                  test_movement_route101.state with go_to_exit north,
                                  go_to_exit "POKEMON CENTER", navigate_to(7,4), UP)
  test_phase23_overworld.state    Brendan's house 1F, Mom at (2,6)

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_dialogue_auto_advance.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_movement_feedback import HOUSE, ROUTE101, server

POKECENTER = Path(__file__).resolve().parent / "test_dialogue_pokecenter.state"


def _status(s):
    from server.field_control import dialogue_status, read_field_snapshot

    with s.srv.memory_lock:
        return dialogue_status(read_field_snapshot(s.srv.env.memory_reader))


def _press(s, buttons):
    return s.call("mcp_press_buttons", {"buttons": buttons, "reasoning": "test"})


def test_stops_at_the_nurse_yes_no_without_answering():
    s = server()
    s.load(POKECENTER)
    _press(s, ["A"])  # talk to the nurse
    r = s.call("mcp_auto_advance_dialogue", {})
    print("   ", r)
    assert r["final_status"] == "choice", r
    assert "Would you like to rest your" in " ".join(r["texts"]), r
    assert "answer it yourself" in r["summary"], r
    # The YES/NO is still waiting: auto-advance did not pick YES (no healing text yet).
    assert _status(s) == "choice"
    assert not any("restored your" in t for t in r["texts"]), r
    # Called again in front of the YES/NO it presses nothing.
    again = s.call("mcp_auto_advance_dialogue", {})
    assert again["presses"] == 0 and again["final_status"] == "choice", again


def test_reads_a_multi_box_dialogue_to_the_end():
    """After the model answers YES, the nurse's lines (with the healing pause) run to the end."""
    s = server()
    s.load(POKECENTER)
    _press(s, ["A"])
    s.call("mcp_auto_advance_dialogue", {})
    assert _status(s) == "choice"
    # The model answers YES itself, after its ~30 s think. (An A in the very first
    # frames of the YES/NO box is ignored by the game, so don't answer instantly.)
    time.sleep(1.0)
    _press(s, ["A"])
    r = s.call("mcp_auto_advance_dialogue", {})
    print("   ", r)
    said = " ".join(r["texts"])
    assert "take your" in said and "restored your" in said and "see you again" in said, r
    assert r["final_status"] == "none", r
    assert r["presses"] >= 3 and not r["capped"], r


def test_single_box_dialogue_is_read_and_closed():
    s = server()
    s.load(HOUSE)
    _press(s, ["LEFT", "LEFT", "DOWN"])  # stand above Mom, facing her
    _press(s, ["A"])
    assert _status(s) == "plain"
    r = s.call("mcp_auto_advance_dialogue", {})
    print("   ", r)
    assert r["final_status"] == "none" and r["presses"] >= 1, r
    assert r["texts"] == ["MOM: See you, honey!"], r
    # Control is back: the player can walk again.
    _press(s, ["RIGHT"])
    snap = s.snapshot()
    assert (snap.x, snap.y) == (3, 5), snap


def test_pokenav_call_with_the_callers_icon_is_read_through():
    """Seen live: Dad's POKéNAV call on Mr. Briney's boat draws the caller's icon in a
    second window; it read as a YES/NO, nothing pressed A, and the boat sat at sea.
    test_boat_peeko_box.state is that call, waiting on its first box."""
    s = server()
    s.load(Path(__file__).resolve().parent / "test_boat_peeko_box.state")
    assert _status(s) == "plain"
    said = []
    for _ in range(4):  # each "… … …" pause is a box of its own: about 40 presses in all
        r = s.call("mcp_auto_advance_dialogue", {"max_presses": 15})
        said += r["texts"]
        if not r["capped"]:
            break
    print("   ", r)
    assert any("Registered DAD NORMAN" in t for t in said), said  # hung up, then the POKéNAV note
    from server.menu_state import phone_call_active

    with s.srv.memory_lock:
        assert not phone_call_active(s.srv.env.memory_reader)
        assert tuple(s.srv.env.memory_reader._read_current_map()) != (0, 20), "the boat should have sailed on"


def test_nothing_happens_without_a_text_box():
    s = server()
    s.load(ROUTE101)
    r = s.call("mcp_auto_advance_dialogue", {})
    assert r["presses"] == 0 and r["final_status"] == "none", r


def test_never_presses_a_into_the_start_menu():
    s = server()
    s.load(ROUTE101)
    _press(s, ["START"])
    assert _status(s) == "choice"
    r = s.call("mcp_auto_advance_dialogue", {})
    assert r["presses"] == 0 and r["final_status"] == "choice", r
    _press(s, ["B"])


def test_cap_limits_presses():
    from server.field_control import DIALOGUE_PLAIN, auto_advance_dialogue

    presses = []
    result = auto_advance_dialogue(
        status=lambda: DIALOGUE_PLAIN,
        read_text=lambda: "endless",
        press_a=lambda: presses.append(1),
        settle=lambda: None,
        max_presses=7,
    )
    assert result.presses == 7 and len(presses) == 7 and result.capped


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
