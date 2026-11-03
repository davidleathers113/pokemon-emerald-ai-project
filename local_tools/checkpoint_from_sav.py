"""Turn an in-game Emerald save (.sav) into a harness checkpoint the AI can resume.

Boots the ROM with the save's flash, picks CONTINUE on the title screen, and writes
a checkpoint (raw state + the flash embedded as mGBA savestate extdata, see
EmeraldEmulator.save_state). The result is verified before it is kept:
  - the .sav has a valid Gen 3 save slot (pret/pokeemerald checksums),
  - after CONTINUE the player stands where the newest save slot says,
  - the written checkpoint loads in a fresh emulator to that same spot.

Usage (inside the image, repo at /app, cwd writable):
  python /app/local_tools/checkpoint_from_sav.py <rom.gba> <save.sav> <out/checkpoint.state>
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "local_tests"))

from gen3_save import report
from save_cycle_test import TITLE_TO_CONTINUE, saved_location
from pokemon_env.emulator import EmeraldEmulator

FLASH_SIZE = 131072  # Emerald: 128 KiB flash; mGBA may append a 16-byte RTC trailer
SAVE_BLOCK1_PTR = 0x03005D8C


def player_location(emu: EmeraldEmulator):
    mem = emu.core.memory
    sb1 = mem.u32[SAVE_BLOCK1_PTR]
    raw = bytes(mem.u8[sb1 + i] for i in range(7))
    x, y = int.from_bytes(raw[0:2], "little", signed=True), int.from_bytes(raw[2:4], "little", signed=True)
    return (x, y, int.from_bytes(raw[4:5], "little", signed=True), int.from_bytes(raw[5:6], "little", signed=True))


def boot(rom: str) -> EmeraldEmulator:
    emu = EmeraldEmulator(rom, headless=True, sound=False)
    emu.initialize()
    return emu


def main(rom: str, sav_path: str, out_path: str) -> int:
    sav = Path(sav_path).read_bytes()
    r = report(sav)
    if r["newest_valid"] is None:
        print(f"{sav_path}: no valid save slot", r["slots"])
        return 1
    expected = saved_location(sav)[:4]
    print(f"save #{r['newest_valid']}: player saved at x,y,group,map = {expected}")

    emu = boot(rom)
    if not emu._restore_savedata(sav[:FLASH_SIZE]):
        print("mGBA refused the save data")
        return 1
    emu.core.reset()
    emu.tick(400)
    for key in TITLE_TO_CONTINUE:
        emu.press_buttons([key], hold_frames=8, release_frames=90)
    emu.tick(240)
    at = player_location(emu)
    print(f"after CONTINUE: {at}")
    if at != expected:
        print("FAIL: CONTINUE did not put the player where the save says")
        return 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    emu.save_state(out_path)

    check = boot(rom)
    check.load_state(out_path)
    check.tick(30)
    reloaded = player_location(check)
    print(f"checkpoint reloads to: {reloaded} ({os.path.getsize(out_path)} bytes)")
    if reloaded != expected:
        os.remove(out_path)
        print("FAIL: written checkpoint does not reload to the saved spot; removed it")
        return 1
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    paths = [os.path.abspath(p) for p in sys.argv[1:]]  # before chdir, so relative paths work
    os.chdir(tempfile.mkdtemp())  # the emulator writes milestone/map files into cwd
    sys.exit(main(*paths))
