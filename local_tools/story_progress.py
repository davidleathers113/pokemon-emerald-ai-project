"""Report early-game story progress stored in an Emerald checkpoint.

Reads event flags and script variables from SaveBlock1 the way the game does
(pret/pokeemerald include/global.h: flags at +0x1270, vars at +0x139C;
constants from include/constants/flags.h and vars.h). Use it to check where a
save really is in the story, e.g. before choosing --direct-objectives-start.

Usage (inside the image, repo at /app): python story_progress.py <rom.gba> <checkpoint.state>
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pokemon_env.emulator import EmeraldEmulator

SAVE_BLOCK1_PTR = 0x03005D8C
FLAGS_OFFSET, VARS_OFFSET, VARS_START = 0x1270, 0x139C, 0x4000
SYSTEM_FLAGS = 0x860

FLAGS = {
    "FLAG_SYS_POKEMON_GET (has a Pokemon)": SYSTEM_FLAGS + 0x0,
    "FLAG_RESCUED_BIRCH": 0x52,
    "FLAG_MET_RIVAL_MOM": 0x57,
    "FLAG_DEFEATED_RIVAL_ROUTE103": 0x82,
    "FLAG_ADVENTURE_STARTED (received Pokedex)": 0x74,
    "FLAG_SYS_POKEDEX_GET": SYSTEM_FLAGS + 0x1,
    "FLAG_RECEIVED_RUNNING_SHOES": 0x112,
    "FLAG_SYS_B_DASH (running shoes usable)": SYSTEM_FLAGS + 0x60,
}
VARS = {
    "VAR_LITTLEROOT_INTRO_STATE": 0x4092,
    "VAR_LITTLEROOT_TOWN_STATE": 0x4050,
    "VAR_LITTLEROOT_RIVAL_STATE": 0x408D,
    "VAR_BIRCH_LAB_STATE": 0x4084,
    "VAR_ROUTE101_STATE": 0x4060,
    "VAR_OLDALE_TOWN_STATE": 0x4051,
}


def main(rom, state):
    rom, state = os.path.abspath(rom), os.path.abspath(state)  # before chdir, so relative paths work
    os.chdir(tempfile.mkdtemp())  # the emulator writes milestone/map files into cwd
    emu = EmeraldEmulator(rom, headless=True, sound=False)
    emu.initialize()
    emu.load_state(state)
    emu.tick(10)
    mem = emu.core.memory
    sb1 = mem.u32[SAVE_BLOCK1_PTR]
    party = emu.memory_reader.read_party_pokemon() or []
    print(f"location: {emu.memory_reader.read_location()}  party: "
          f"{[(getattr(p, 'species_name', '?'), getattr(p, 'level', '?')) for p in party]}")
    for name, flag in FLAGS.items():
        byte = mem.u8[sb1 + FLAGS_OFFSET + (flag >> 3)]
        print(f"  {name:45s} {'SET' if byte & (1 << (flag & 7)) else '-'}")
    for name, var in VARS.items():
        addr = sb1 + VARS_OFFSET + 2 * (var - VARS_START)
        print(f"  {name:45s} {mem.u8[addr] | (mem.u8[addr + 1] << 8)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
