"""The harness must report the map the player is actually on.

Regression test for the memory reader using sWarpDestination (0x020322E4) as the
current map: right after CONTINUE from an in-game save no warp has happened, it
reads 0,0 and the agent was told it was in Petalburg City while standing in
Littleroot Town. Each case loads a checkpoint and compares read_location() with
the map stored in gSaveBlock1Ptr->location and with the map the screen showed
when the fixture was made.

Usage: python test_map_location.py <rom> <state>=<EXPECTED LOCATION NAME> [...]
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pokemon_env.emulator import EmeraldEmulator


def main(rom, cases):
    rom = os.path.abspath(rom)  # resolve before chdir, so relative paths work
    cases = [(os.path.abspath(c.split("=", 1)[0]), c.split("=", 1)[1]) for c in cases]
    os.chdir(tempfile.mkdtemp())  # the emulator writes milestone/map files into cwd
    failures = 0
    for state, expected in cases:
        emu = EmeraldEmulator(rom, headless=True, sound=False)
        emu.initialize()
        emu.load_state(state)
        emu.tick(30)
        reader = emu.memory_reader
        warp = (reader._read_u8(reader.addresses.MAP_BANK), reader._read_u8(reader.addresses.MAP_NUMBER))
        current = reader._read_current_map()
        location = reader.read_location()
        ok = location == expected
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {Path(state).name}: read_location={location!r} expected={expected!r} "
              f"current_map={current} warp_destination={warp}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
