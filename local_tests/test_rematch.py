"""Match Call rematches: the state's REMATCH READY line, the registration hint, and
rematch_loop (fight the trainers who want a rematch, walk the step counter up, re-enter).

Asked for (2026-11-05): train through trainer rematches (trainer battles pay 1.5x EXP,
pret Cmd_getexp). The tests read outcomes from game memory (trainer flags,
gSaveBlock1Ptr->trainerRematches and its step counter, party EXP), not from the macro's
own report.

Same in-process server harness as test_macros.py (real game loop thread, real savestates).

Fixtures:
  test_rematch_route120.state  the live checkpoint of 2026-11-05 15:30 (run ai-local-
                               20261105-1355): ROUTE 120 (20,8), 6 badges, Match Call, 24
                               trainers registered, 10 waiting for a rematch, ROBERT
                               (ROUTE 120, (32,14)) among them, the step counter at 255
  test_rematch_route102.state  the same save after go_to("ROUTE 102") (it flew to OLDALE
                               TOWN and walked): ROUTE 102 (49,11). CALVIN (33,14) was
                               beaten before the POKéNAV and is not registered
  (the live team's Repel var is restored to the save's own 0 after the walks)

RAM writes (documented where used): the party's HP to full (the live team came in weak,
and a heal trip to FORTREE CITY each test is minutes of walking that isn't under test);
gSaveBlock1Ptr->trainerRematches[] and trainerRematchStepCounter, to set who is ready and
how far the counter is; FLAG_HAS_MATCH_CALL cleared for the no-POKéNAV refusal.

Not covered: the 31% roll itself is the game's (Random() at the map load), so the
counter test asserts the walk and the re-entry, and checks the rest only for consistency;
double-battle rematches (left to the model); gym leader rematches (post-game).

  python3 local_tools/run_suites.py rematch
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_macros import PARTY, _party, _read, _write
from test_movement_feedback import server

HERE = Path(__file__).resolve().parent
ROUTE120 = HERE / "test_rematch_route120.state"
ROUTE102 = HERE / "test_rematch_route102.state"
# pret include/global.h struct SaveBlock1: /*0x9C8*/ trainerRematchStepCounter,
# /*0x9CA*/ trainerRematches[]; flags at 0x1270. include/constants/flags.h:
# TRAINER_FLAGS_START 0x500, TRAINER_REGISTERED_FLAGS_START 0x15C, FLAG_HAS_MATCH_CALL
# 0x12F. include/constants/rematches.h: REMATCH_CALVIN 34, REMATCH_ROBERT 44.
# include/constants/opponents.h: TRAINER_ROBERT_2 409.
SAVE_BLOCK1_PTR = 0x03005D8C
STEP_COUNTER, REMATCHES, FLAGS = 0x9C8, 0x9CA, 0x1270
TRAINER_FLAGS_START, REGISTERED_FLAGS_START, FLAG_HAS_MATCH_CALL = 0x500, 0x15C, 0x12F
REMATCH_CALVIN, REMATCH_ROBERT = 34, 44
TRAINER_ROBERT_2 = 409


def _sb1(s):
    return _read(s, lambda r: r._read_u32(SAVE_BLOCK1_PTR))


def _flag(s, flag):
    return bool(_read(s, lambda r: r._read_u8(_sb1(s) + FLAGS + flag // 8)) & (1 << (flag % 8)))


def _ready(s, index):
    return _read(s, lambda r: r._read_u8(_sb1(s) + REMATCHES + index))


def _counter(s):
    return _read(s, lambda r: r._read_u16(_sb1(s) + STEP_COUNTER))


# pret src/pokemon.c GetSubstruct: personality % 24 orders the four 12-byte substructs
# (Growth, Attacks, EVs, Misc); this is where each one sits (pokemon_env/emerald_utils.py
# parse_box_pokemon's table).
SUBSTRUCT_PLACES = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 1, 2], [0, 2, 3, 1], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [2, 0, 1, 3], [3, 0, 1, 2], [2, 0, 3, 1], [3, 0, 2, 1],
    [1, 2, 0, 3], [1, 3, 0, 2], [2, 1, 0, 3], [3, 1, 0, 2], [2, 3, 0, 1], [3, 2, 0, 1],
    [1, 2, 3, 0], [1, 3, 2, 0], [2, 1, 3, 0], [3, 1, 2, 0], [2, 3, 1, 0], [3, 2, 1, 0],
]


def _full_pp(s, slot):
    """RAM: a party Pokémon's moves to their base PP (gBattleMoves .pp, from the ROM). The
    Attacks substruct (u16 moves[4], u8 pp[4]) is XOR'd with personality ^ otId, and the
    box checksum (+0x1C) is the u16 sum of the 48 decrypted bytes (pret
    CalculateBoxMonChecksum), so both are redone."""
    base = PARTY + 100 * slot
    personality = _read(s, lambda r: r._read_u32(base))
    key = personality ^ _read(s, lambda r: r._read_u32(base + 4))
    words = [_read(s, lambda r, i=i: r._read_u32(base + 0x20 + 4 * i)) ^ key for i in range(12)]
    at = 3 * SUBSTRUCT_PLACES[personality % 24][1]
    attacks = b"".join(w.to_bytes(4, "little") for w in words[at:at + 3])
    moves = [int.from_bytes(attacks[2 * i:2 * i + 2], "little") for i in range(4)]
    pp = bytes(_read(s, lambda r, m=m: r._rom_battle_move(m)["pp"]) if m else 0 for m in moves)
    attacks = attacks[:8] + pp
    words[at:at + 3] = [int.from_bytes(attacks[4 * i:4 * i + 4], "little") for i in range(3)]
    checksum = sum((w & 0xFFFF) + (w >> 16) for w in words) & 0xFFFF
    for i, word in enumerate(words):
        _write(s, 32, base + 0x20 + 4 * i, word ^ key)
    _write(s, 16, base + 0x1C, checksum)


def _full_hp(s):
    """RAM: every party Pokémon to full HP, no status and full PP (+0x56 HP, +0x58 max
    HP, +0x50 status; _full_pp), so the team isn't weak and no heal trip runs: the live
    team came in with BLAZIKEN's SLASH and DOUBLE KICK at 0 PP."""
    for i, mon in enumerate(_party(s)):
        _write(s, 16, PARTY + 100 * i + 0x56, mon.max_hp)
        _write(s, 32, PARTY + 100 * i + 0x50, 0)
        _full_pp(s, i)


def rematch_entries_of(index):
    """A reader function: the gRematchTable entry at `index` (server.rematch)."""
    from server import rematch

    return lambda r: rematch.rematch_entries(r)[index]


def _state_text(s):
    return asyncio.run(s.srv.mcp_get_game_state()).get("state_text") or ""


def _line(text, start):
    return next((line for line in text.splitlines() if line.startswith(start)), "")


# -- the table and the state line ------------------------------------------------


def test_rematch_table_is_read_from_the_rom():
    """gRematchTable's entries are 16 bytes in the ROM (agbcc pads the 14-byte struct),
    and each trainer stands where pret's map data puts their object."""
    from server import rematch

    s = server()
    s.load(ROUTE120)
    entries = _read(s, rematch.rematch_entries)
    assert len(entries) == 78, len(entries)
    by_name = {e.name: e for e in entries}
    rose = entries[0]
    assert (rose.name, rose.map_name, rose.trainer_ids) == ("ROSE", "ROUTE 118", (37, 40, 41, 42, 43)), rose
    fernando = by_name["FERNANDO"]
    assert (fernando.map_dir, fernando.spots, fernando.rolls) == ("Route123", ((66, 16),), True), fernando
    robert = entries[REMATCH_ROBERT]
    assert (robert.name, robert.map_name, robert.spots, robert.trainer_ids[1]) == ("ROBERT", "ROUTE 120", ((32, 14),), TRAINER_ROBERT_2)
    # Two trainers, one double battle; the early routes' scripts offer Match Call when
    # talked to (IsTrainerRegistered); gym leaders are not rolled by the step counter.
    assert by_name["ANNA AND MEG"].double and len(by_name["ANNA AND MEG"].spots) == 2
    assert entries[REMATCH_CALVIN].registers_on_talk and not fernando.registers_on_talk
    assert not by_name["WINONA"].rolls and by_name["WALLY VR"].rolls


def test_state_says_who_wants_a_rematch_and_where():
    s = server()
    s.load(ROUTE120)
    text = _state_text(s)
    line = _line(text, "REMATCH READY: ")
    print("    " + line)
    # The map the player is on comes first.
    assert line.startswith("REMATCH READY: ROUTE 120: ROBERT at (32, 14) | "), line
    # Every trainer RAM says is ready (trainerRematches[i] != 0, read here directly) is named.
    from server import rematch

    entries = _read(s, rematch.rematch_entries)
    ready = [e.name for e in entries if e.rolls and _ready(s, e.index)]
    assert len(ready) == 10 and all(f"{name} at (" in line for name in ready), (ready, line)
    # This map's other rematch trainer is not beaten yet: beating him registers him.
    hint = _line(text, "MATCH CALL REGISTRATION on ROUTE 120: ")
    assert "not beaten yet: JEFFREY at (19, 80); beat them once" in hint, hint
    # Nobody ready (RAM: every trainerRematches[] byte to 0): one short line.
    for e in entries:
        _write(s, 8, _sb1(s) + REMATCHES + e.index, 0)
    line = _line(_state_text(s), "REMATCHES: ")
    print("    " + line)
    assert line == "REMATCHES: none ready (24 registered; step counter 255/255).", line


# -- registration -------------------------------------------------------------------


def test_rematch_loop_says_how_to_register_and_talking_registers():
    """CALVIN (ROUTE 102) was beaten before the POKéNAV: pret registers a trainer right
    after the first win (CB2_EndTrainerBattle → RegisterTrainerInMatchCall) only with
    Match Call, and Route102_EventScript_Calvin asks IsTrainerRegistered when talked to
    later and offers it. rematch_loop says so; talk_to(33, 14) then registers him. Wild
    encounters off (a Repel): seen in a run, a wild battle in ROUTE 102's grass stopped
    the walk to him."""
    s = server()
    s.load(ROUTE102, wild_encounters=False)
    assert not _flag(s, REGISTERED_FLAGS_START + REMATCH_CALVIN)
    hint = _line(_state_text(s), "MATCH CALL REGISTRATION on ROUTE 102: ")
    assert "CALVIN (talk_to(33, 14)); talked to, they offer to register" in hint, hint
    before = s.snapshot()
    r = s.call("mcp_rematch_loop", {"map": "ROUTE 102", "reasoning": "test"})
    print("   ", r.get("error"))
    assert r["success"] is False and r["stopped_because"] == "not_registered", r
    assert "CALVIN (talk_to(33, 14)); talked to, they offer to register" in r["error"], r
    after = s.snapshot()
    assert (after.x, after.y) == (before.x, before.y), (before, after)  # refused before any walk
    r = s.call("mcp_talk_to", {"x": 33, "y": 14, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert _flag(s, REGISTERED_FLAGS_START + REMATCH_CALVIN), r
    assert "MATCH CALL REGISTRATION" not in _state_text(s)


# -- rematch_loop -------------------------------------------------------------------


def test_rematch_loop_fights_a_trainer_who_wants_a_rematch():
    """ROBERT wants a rematch (trainerRematches[REMATCH_ROBERT] = 1, his second team,
    TRAINER_ROBERT_2). rematch_loop walks up, talks to him, and wins with train's rules:
    the lowest-level Pokémon (TROPIUS, Lv45) is the trainee and gains EXP. Wild
    encounters off (a Repel, test_movement_feedback) so the walk is only to him."""
    s = server()
    s.load(ROUTE120, wild_encounters=False)
    _full_hp(s)
    assert _ready(s, REMATCH_ROBERT) == 1 and not _flag(s, TRAINER_FLAGS_START + TRAINER_ROBERT_2)
    exp_before = {m.species_name.upper(): m.experience for m in _party(s)}
    started = time.time()
    r = s.call("mcp_rematch_loop", {"map": "ROUTE 120", "max_minutes": 1, "reasoning": "test"})
    print(f"    ({time.time() - started:.0f} s) {r.get('summary') or r}")
    assert r["success"] is True and "healed at" not in r["summary"], r
    robert = [b for b in r["battles"] if b.get("trainer") == "ROBERT"]
    assert robert and robert[0]["rematch_tier"] == 1 and robert[0]["outcome"] == "won", r["battles"]
    # From memory: his second team is beaten. The win cleared his rematch byte; the loop
    # then re-enters the map, and a 31% roll may make him ready again (seen in a suite
    # run), in which case it names his next team, the first not beaten yet (pret
    # SetRematchIdForTrainer): the tiers rise.
    assert _flag(s, TRAINER_FLAGS_START + TRAINER_ROBERT_2)
    tiers = _read(s, rematch_entries_of(REMATCH_ROBERT)).trainer_ids
    unbeaten = next((i for i, t in enumerate(tiers) if i and not _flag(s, TRAINER_FLAGS_START + t)), len(tiers) - 1)
    assert _ready(s, REMATCH_ROBERT) in (0, unbeaten), (_ready(s, REMATCH_ROBERT), unbeaten)
    exp_after = {m.species_name.upper(): m.experience for m in _party(s)}
    assert exp_after["TROPIUS"] > exp_before["TROPIUS"], (exp_before, exp_after)
    assert "ROBERT (rematch 1): won; " in r["summary"] and "TROPIUS" in robert[0]["exp"], r["summary"]


def test_rematch_loop_walks_the_counter_up_and_reenters_the_map():
    """Nobody on ROUTE 120 ready (RAM: ROBERT's trainerRematches byte to 0) and the step
    counter at 250 (RAM): rematch_loop walks the 5 steps to 255 on plain tiles, then
    leaves the map and comes back, which rolls 31% for ROBERT (pret
    UpdateRandomTrainerRematches at the map load). The roll is the game's own, so: the
    walk and the re-entry are asserted, and the rest only for consistency (counter back
    to 0 means ROBERT became ready, and then he was fought)."""
    s = server()
    s.load(ROUTE120, wild_encounters=False)
    _full_hp(s)
    _write(s, 8, _sb1(s) + REMATCHES + REMATCH_ROBERT, 0)
    _write(s, 16, _sb1(s) + STEP_COUNTER, 250)
    r = s.call("mcp_rematch_loop", {"map": "ROUTE 120", "max_minutes": 1, "reasoning": "test"})
    print("   ", r.get("summary") or r)
    assert r["success"] is True and "healed at" not in r["summary"], r
    first = r["loops"][0]
    assert first["steps"] >= 5 and first["reentries"] >= 1, first
    fought = any(b.get("trainer") == "ROBERT" for b in r["battles"])
    assert fought == _flag(s, TRAINER_FLAGS_START + TRAINER_ROBERT_2), r["battles"]
    if not fought:
        assert _counter(s) == 255 or _ready(s, REMATCH_ROBERT), (_counter(s), _ready(s, REMATCH_ROBERT))


def test_rematch_loop_refusals():
    s = server()
    s.load(ROUTE120)
    before = s.snapshot()
    r = s.call("mcp_rematch_loop", {"map": "ROUTE 999", "reasoning": "test"})
    assert r["success"] is False and 'Unknown map "ROUTE 999"' in r["error"], r
    # FORTREE CITY's only rematch is WINONA, a gym leader's (post-game, not rolled).
    r = s.call("mcp_rematch_loop", {"map": "Fortree City", "reasoning": "test"})
    print("   ", r.get("error"))
    assert r["success"] is False and r["stopped_because"] == "no_rematches", r
    assert r["error"].startswith("No trainer on FORTREE CITY battles again") and "ROUTE 120" in r["error"], r
    _write(s, 8, _sb1(s) + FLAGS + FLAG_HAS_MATCH_CALL // 8,
           _read(s, lambda rd: rd._read_u8(_sb1(s) + FLAGS + FLAG_HAS_MATCH_CALL // 8)) & ~(1 << (FLAG_HAS_MATCH_CALL % 8)))
    r = s.call("mcp_rematch_loop", {"map": "ROUTE 120", "reasoning": "test"})
    assert r["success"] is False and "no Match Call" in r["error"], r
    after = s.snapshot()
    assert (after.x, after.y) == (before.x, before.y), (before, after)


def test_agent_registers_rematch_loop_and_outwaits_it():
    import importlib

    from agents.tools.registry import build_tools_for_scaffold

    pa = importlib.import_module("agents.PokeAgent")
    s = server()
    assert "rematch_loop" in pa.MACRO_TOOLS and "rematch_loop" in pa.PRESSING_TOOLS
    assert "rematch_loop" in {t["name"] for t in build_tools_for_scaffold("simple")}
    assert (pa.REMATCH_DEFAULT_MINUTES, pa.REMATCH_MAX_MINUTES) == (s.srv.REMATCH_DEFAULT_MINUTES, s.srv.REMATCH_MAX_MINUTES)
    finish = s.srv.TRAIN_FINISH_BATTLE_S
    assert pa.tool_timeout_s("rematch_loop", {}) > s.srv.REMATCH_DEFAULT_MINUTES * 60 + finish
    assert pa.tool_timeout_s("rematch_loop", {"max_minutes": 99}) == pa.tool_timeout_s(
        "rematch_loop", {"max_minutes": s.srv.REMATCH_MAX_MINUTES}
    )


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
