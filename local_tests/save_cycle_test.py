"""End-to-end in-game save test for Emerald on the libmgba the bindings load.

Phase A (save): boot the ROM with a flash .sav set up like the harness does,
open START > SAVE and confirm, pressing A only until the game reports a new
save (detected by validating the .sav on disk the way pokeemerald does).
Phase B (reload): boot a fresh core on the written .sav, CONTINUE from the title
screen and check that the player's position and map in RAM equal the ones stored
in the newest valid save slot.

Modes for the starting save:
  existing  copy <sav> (a complete Emerald save, no RTC trailer) next to the ROM
  empty     start with no .sav at all (what the harness does on every launch),
            then load <state> (a raw savestate taken in the overworld)

Usage: save_cycle_test.py <rom> <out_dir> --mode existing --sav <file> [--resets 1|2]
       save_cycle_test.py <rom> <out_dir> --mode empty --state <file> [--resets 1|2]
       save_cycle_test.py <rom> <out_dir> --make-state <file> --sav <file>
Exit 0 = pass, 1 = check failed, crash = signal exit (139 SIGSEGV / 135 SIGBUS).
"""
import argparse
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import mgba.core
import mgba.image
import mgba.log

sys.path.insert(0, str(Path(__file__).parent))
from gen3_save import SECTOR_SIZE, SECTORS_PER_SLOT, report

KEYS = {"A": 0, "B": 1, "SELECT": 2, "START": 3, "RIGHT": 4, "LEFT": 5, "UP": 6, "DOWN": 7}
SAVE_BLOCK1_PTR = 0x03005D8C  # gSaveBlock1Ptr in pokeemerald (US)
TITLE_TO_CONTINUE = ["START"] * 6 + ["A"]  # skip intro/title, pick CONTINUE


class Emu:
    def __init__(self, rom: Path, work: Path, resets: int, out: Path, tag: str):
        shutil.copy(rom, work / "rom.gba")
        self.out, self.tag, self.shots = out, tag, 0
        self.core = mgba.core.load_path(str(work / "rom.gba"))
        self.core.autoload_save()
        self.core.reset()
        w, h = self.core.desired_video_dimensions()
        self.buf = mgba.image.Image(w, h)
        self.core.set_video_buffer(self.buf)
        if resets == 2:
            self.core.reset()  # pokemon_env/emulator.py resets again after attaching video

    def run(self, frames):
        for _ in range(frames):
            self.core.run_frame()

    def press(self, key, hold=8, after=30):
        self.core.add_keys(KEYS[key])  # takes key indices, builds the bitmask itself
        self.run(hold)
        self.core.clear_keys(KEYS[key])
        self.run(after)

    def shot(self, name):
        self.shots += 1
        self.buf.to_pil().convert("RGB").save(self.out / f"{self.tag}{self.shots}_{name}.png")

    def u32(self, addr):
        return self.core.memory.u32[addr]

    def player_location(self):
        sb1 = self.u32(SAVE_BLOCK1_PTR)
        raw = bytes(self.core.memory.u8[sb1 + i] for i in range(8))
        return struct.unpack("<hhbbb", raw[:7])  # x, y, mapGroup, mapNum, warpId


def saved_location(save: bytes):
    """Player x, y, mapGroup, mapNum, warpId from SaveBlock1 in the newest valid slot."""
    r = report(save)
    slot = next(i for i, s in enumerate(r["slots"]) if s["valid"] and s["counter"] == r["newest_valid"])
    for n in range(slot * SECTORS_PER_SLOT, (slot + 1) * SECTORS_PER_SLOT):
        sector = save[n * SECTOR_SIZE:(n + 1) * SECTOR_SIZE]
        if struct.unpack_from("<H", sector, 0xFF4)[0] == 1:  # SECTOR_ID_SAVEBLOCK1_START
            return struct.unpack_from("<hhbbb", sector, 0)
    raise ValueError("no SaveBlock1 sector")


def title_to_overworld(emu: Emu):
    emu.run(400)
    for key in TITLE_TO_CONTINUE:
        emu.press(key, after=90)
    emu.run(240)


def save_from_menu(emu: Emu, sav: Path, before) -> int:
    """START > SAVE, then A until a newer valid save is on disk. Returns its number."""
    emu.press("START", after=60)
    for _ in range(5):  # POKEDEX, POKEMON, BAG, POKENAV, <name> -> SAVE
        emu.press("DOWN")
    emu.shot("menu")
    for attempt in range(6):
        emu.press("A", after=120)
        emu.shot(f"confirm{attempt}")
        for _ in range(10):  # the write takes several hundred frames
            emu.run(60)
            newest = report(sav.read_bytes())["newest_valid"]
            if newest is not None and (before is None or newest > before):
                emu.run(120)  # let mGBA's idle sync (RTC trailer remap) happen
                emu.shot("saved")
                emu.press("B", after=60)
                emu.run(300)  # keep running past the first post-save writes
                return newest
    raise AssertionError("game never reported a new save")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rom", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=["existing", "empty"])
    p.add_argument("--sav", type=Path)
    p.add_argument("--state", type=Path)
    p.add_argument("--resets", type=int, choices=[1, 2], default=2)
    p.add_argument("--make-state", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mgba.log.silence()

    if a.make_state:  # helper: overworld savestate from a complete save
        work = Path(tempfile.mkdtemp())
        shutil.copy(a.sav, work / "rom.sav")
        emu = Emu(a.rom, work, 2, a.out, "make")
        title_to_overworld(emu)
        emu.shot("overworld")
        a.make_state.write_bytes(bytes(emu.core.save_raw_state()))
        print("wrote", a.make_state, "at", emu.player_location())
        return 0

    work = Path(tempfile.mkdtemp())
    sav = work / "rom.sav"
    if a.mode == "existing":
        shutil.copy(a.sav, sav)
    before = report(sav.read_bytes())["newest_valid"] if sav.exists() else None
    print(f"mode={a.mode} resets={a.resets} save before: #{before}", flush=True)

    emu = Emu(a.rom, work, a.resets, a.out, "A")
    if a.mode == "existing":
        title_to_overworld(emu)
    else:
        emu.run(60)
        if not emu.core.load_raw_state(a.state.read_bytes()):
            raise AssertionError("load_raw_state rejected the state")
        emu.run(120)
    emu.shot("start")
    at_save = emu.player_location()
    newest = save_from_menu(emu, sav, before)
    written = sav.read_bytes()
    r = report(written)
    print(f"save after: {len(written)} bytes, newest valid #{newest}, slots:",
          [s["counter"] if s["valid"] else "invalid" for s in r["slots"]], flush=True)
    stored = saved_location(written)
    del emu

    work2 = Path(tempfile.mkdtemp())
    (work2 / "rom.sav").write_bytes(written)
    emu2 = Emu(a.rom, work2, a.resets, a.out, "B")
    title_to_overworld(emu2)
    emu2.shot("continued")
    loaded = emu2.player_location()
    print(f"player at save time {at_save}, stored in save {stored}, after CONTINUE {loaded}", flush=True)

    ok = len(written) == 131088 and stored[:4] == at_save[:4] and loaded[:4] == stored[:4]
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
