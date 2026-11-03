"""In-game save survives checkpoint -> new emulator -> soft reset -> CONTINUE.

Exercises pokemon_env.emulator.EmeraldEmulator.save_state/load_state, which embed
the cartridge flash in the state file (mGBA savestate extdata):

  A  load an overworld state, save in-game from the START menu, save_state();
     a NEW emulator (blank flash) load_state()s it, soft-resets, CONTINUEs from the
     title screen and must stand where the in-game save was made. A control run
     loads only the raw-state part (what checkpoints used to be) and must NOT be
     able to continue there, proving the check is meaningful.
  B  a state without embedded flash still loads (warning, flash untouched); truncated,
     cut-off and corrupt states raise SaveStateError and leave the emulator unchanged.
  C  mGBA's own loader (mCoreLoadStateNamed) reads the flash back out of our file.

Run (repo mounted read-write; the test works in a temp cwd so it writes nothing there):
  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_phase23_save_persistence.py [--state <raw overworld state>]
Exit 0 = all checks passed.
"""
import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen3_save import report
from save_cycle_test import KEYS, SAVE_BLOCK1_PTR, saved_location, title_to_overworld

FAILURES = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


class Pad:
    """The press/run/shot interface save_cycle_test's helpers drive, on an EmeraldEmulator."""

    def __init__(self, emu):
        self.emu = emu

    def run(self, frames):
        self.emu.tick(frames)

    def press(self, key, hold=8, after=30):
        self.emu.core.add_keys(KEYS[key])
        self.run(hold)
        self.emu.core.clear_keys(KEYS[key])
        self.run(after)

    def shot(self, name):
        pass


def save_in_game(pad, before):
    """START > SAVE, then A until the flash holds a newer valid save. Returns its number.

    The start menu wraps and always ends SAVE, OPTION, EXIT, so UP x3 from the top
    reaches SAVE whether or not POKEDEX/POKEMON/POKENAV are unlocked yet.
    """
    pad.press("START", after=60)
    for _ in range(3):
        pad.press("UP")
    for _ in range(6):
        pad.press("A", after=120)
        for _ in range(10):  # the write takes several hundred frames
            pad.run(60)
            newest_now = report(pad.emu._clone_savedata())["newest_valid"]
            if newest_now is not None and (before is None or newest_now > before):
                pad.run(120)
                pad.press("B", after=60)
                pad.run(300)  # keep running past the first post-save writes
                return newest_now
    raise AssertionError("game never reported a new save")


def location(emu):
    import struct
    sb1 = emu.core.memory.u32[SAVE_BLOCK1_PTR]
    raw = bytes(emu.core.memory.u8[sb1 + i] for i in range(8))
    return struct.unpack("<hhbbb", raw[:7])[:4]  # x, y, mapGroup, mapNum


def new_emu(rom):
    from pokemon_env.emulator import EmeraldEmulator
    emu = EmeraldEmulator(rom_path=str(rom))
    emu.initialize()
    return emu


def newest(save):
    return report(save)["newest_valid"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rom", type=Path, default=REPO / "Emerald-GBAdvance" / "rom.gba")
    # Player free in the house on 1F right after the opening TV scene (raw state, no flash).
    p.add_argument("--state", type=Path, default=REPO / "local_tests" / "test_phase23_overworld.state")
    a = p.parse_args()
    rom, start_state = a.rom.resolve(), a.state.resolve()
    work = Path(tempfile.mkdtemp())
    os.chdir(work)  # EmeraldEmulator writes .pokeagent_cache/ relative to cwd
    logging.basicConfig(level=logging.WARNING, format="  log %(levelname)s %(name)s: %(message)s")

    from pokemon_env.emulator import SAVESTATE_SAVEDATA, SaveStateError, _extract_savedata
    from mgba._pylib import lib

    # ---- A: in-game save, checkpoint, restore into a new emulator, CONTINUE ----
    emu1 = new_emu(rom)
    state_size = emu1._state_size()
    check(newest(emu1._clone_savedata()) is None, "fresh emulator starts with a blank flash")
    emu1.load_state(str(start_state))  # an old-style raw state: must load (Test B part 1)
    emu1.tick(120)
    at_save = location(emu1)
    print(f"     start state loaded, player at {at_save}", flush=True)
    pad1 = Pad(emu1)
    save_no = save_in_game(pad1, None)
    print(f"     in-game save #{save_no} written", flush=True)
    flash1 = emu1._clone_savedata()
    check(saved_location(flash1)[:4] == at_save, f"in-game save stores the player location {at_save}")

    ckpt = work / "ckpt.state"
    data = emu1.save_state(str(ckpt))
    on_disk = ckpt.read_bytes()
    check(on_disk == data, "save_state returns exactly the bytes it wrote")
    check(len(on_disk) == state_size + 2 * 16 + len(flash1),
          f"state file = raw state {state_size} + extdata headers 32 + flash {len(flash1)} = {len(on_disk)} bytes")
    check(_extract_savedata(on_disk, state_size) == flash1, "embedded flash equals the live flash at save time")
    check(not list(work.glob("ckpt.state.tmp*")), "no temp file left behind by the atomic write")
    pos_at_ckpt = location(emu1)
    emu1.stop()
    del emu1, pad1

    emu2 = new_emu(rom)
    check(newest(emu2._clone_savedata()) is None, "second emulator starts with a blank flash")
    emu2.load_state(str(ckpt))
    check(emu2._clone_savedata() == flash1, "load_state restored the flash into the new emulator")
    check(location(emu2) == pos_at_ckpt, f"load_state restored the position {pos_at_ckpt}")
    emu2.core.reset()
    title_to_overworld(Pad(emu2))
    continued = location(emu2)
    check(continued == at_save, f"after soft reset + CONTINUE player is at {continued}, saved at {at_save}")
    emu2.stop()

    # Control: the raw-state part alone (what checkpoints used to be) cannot CONTINUE.
    emu3 = new_emu(rom)
    emu3.load_state(state_bytes=on_disk[:state_size])
    check(newest(emu3._clone_savedata()) is None, "control: raw state alone leaves the flash blank")
    emu3.core.reset()
    title_to_overworld(Pad(emu3))
    check(location(emu3) != at_save, f"control: without the flash CONTINUE does not reach {at_save} (got {location(emu3)})")
    emu3.stop()

    # ---- B: old-style, truncated, cut-off and corrupt states ----
    old = work / "old.state"
    old.write_bytes(on_disk[:state_size])
    emu4 = new_emu(rom)
    emu4.load_state(str(ckpt))
    before_flash = emu4._clone_savedata()
    emu4.load_state(str(old))
    check(emu4._clone_savedata() == before_flash, "old-style state loads and leaves the current flash alone")

    def expect_error(name, blob, needle):
        bad = work / f"{name}.state"
        bad.write_bytes(blob)
        emu4.load_state(str(ckpt))
        flash_before, pos_before = emu4._clone_savedata(), location(emu4)
        try:
            emu4.load_state(str(bad))
        except SaveStateError as e:
            check(needle in str(e), f"{name}: SaveStateError mentions '{needle}': {e}")
        else:
            check(False, f"{name}: load_state raised nothing")
        check(emu4._clone_savedata() == flash_before and location(emu4) == pos_before,
              f"{name}: emulator unchanged after the failed load")

    expect_error("truncated", on_disk[:1000], "truncated")
    expect_error("cut_in_flash", on_disk[:-1000], "out of range")
    expect_error("cut_in_headers", on_disk[:state_size + 8], "truncated")
    # Zero the version magic: GBADeserialize rejects it. Pair it with a different flash
    # so a missing rollback would be visible.
    other_flash = bytes(b ^ 0xFF for b in flash1)
    corrupt = bytearray(on_disk)
    corrupt[0:4] = b"\0\0\0\0"
    off = len(corrupt) - len(flash1)
    corrupt[off:] = other_flash
    expect_error("bad_magic", bytes(corrupt), "rejected")
    try:
        emu4.load_state(str(work / "missing.state"))
        check(False, "missing file: load_state raised nothing")
    except SaveStateError as e:
        check("Cannot read" in str(e), f"missing file: {e}")
    emu4.stop()

    # ---- C: mGBA itself reads the embedded flash from our file ----
    emu5 = new_emu(rom)
    vf = lib.VFileOpen(str(ckpt).encode(), os.O_RDONLY)
    ok = lib.mCoreLoadStateNamed(emu5.core._core, vf, SAVESTATE_SAVEDATA)
    vf.close(vf)
    check(bool(ok) and emu5._clone_savedata() == flash1, "mCoreLoadStateNamed restores the same flash from our file")
    emu5.stop()

    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
