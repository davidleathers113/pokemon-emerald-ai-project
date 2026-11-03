"""Run the emulator for a fixed wall-clock time with random button input.

Meant to run under AddressSanitizer to surface memory errors outside the save
path. Starts from a raw savestate (overworld), presses a random button every
few frames, and periodically saves/loads raw states the way the harness's
checkpoints do. Exit 0 if it runs the whole duration.

Usage: soak_frames.py <rom> <state> <seconds> [seed]
"""
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mgba.core
import mgba.image
import mgba.log

rom, state, seconds = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 1)

work = Path(tempfile.mkdtemp())
shutil.copy(rom, work / "rom.gba")
mgba.log.silence()
core = mgba.core.load_path(str(work / "rom.gba"))
core.autoload_save()
core.reset()
buf = mgba.image.Image(*core.desired_video_dimensions())
core.set_video_buffer(buf)
core.reset()
if not core.load_raw_state(state.read_bytes()):
    sys.exit("load_raw_state rejected the state")

end, frames, states = time.monotonic() + seconds, 0, 0
while time.monotonic() < end:
    key = rng.choice([0, 1, 4, 5, 6, 7, 7, 6, 4, 5, 0])  # A B and d-pad, START rarely
    if rng.random() < 0.01:
        key = 3
    core.add_keys(key)
    for _ in range(rng.randint(4, 16)):
        core.run_frame()
    core.clear_keys(key)
    for _ in range(rng.randint(2, 30)):
        core.run_frame()
    frames += 1
    if frames % 500 == 0:
        snap = bytes(core.save_raw_state())
        if not core.load_raw_state(snap):
            sys.exit("round-trip load_raw_state failed")
        states += 1
        buf.to_pil()
print(f"soak OK: {frames} inputs, {states} state round-trips, {seconds:.0f}s")
