# Incident: in-game SAVE crashes the game server (2026-11-02)

## Summary
Choosing **START → SAVE** in Emerald killed the harness's game server with
`Fatal Python error: Bus error` (sometimes `Segmentation fault`) inside
`mgba/core.py run_frame`. The container kept showing "Up", the agent kept waiting,
and about 750 steps of human play since the last checkpoint were lost.

**Root cause:** a bug in mGBA's **Python bindings**, not in libmgba and not in the game.
`mCorePythonCallbackCreate` (mGBA `src/platform/python/core.c`, compiled into the PyPI
`mgba==0.10.2` wheel's `_pylib.abi3.so`, still unfixed on mGBA master as of 2026-11-02)
allocates `struct mCoreCallbacks` with `malloc()` and fills in only 5 of its 8 function
pointers. `shutdown`, `savedataUpdated` and `alarm` keep whatever bytes `malloc` returned.
`GBAFrameEnded` (`src/gba/gba.c:920`) calls `savedataUpdated` at exactly one moment: when
an in-game save finishes writing flash. If the garbage happens to be zero, the call is
skipped; otherwise the emulator jumps to a random address. That explains why only SAVE
crashed, why it was intermittent, and why it came in bursts (it depends on heap contents).

**Fix:** `local_patches/fix_mgba_callbacks.py`, run while building the image
(`Dockerfile.local`), sets the three pointers to NULL in `CoreCallbacks.__init__` before
the struct is copied into the core. The build fails if the patch doesn't apply or doesn't
take effect.

## Timeline (local time)
- 23:07:59: David saves in game, and the server dies with a Bus error. The container stays "Up".
- 23:13: the previous session rebuilds the image with libmgba **0.10.2**, believing a
  changelog entry ("crash when resizing flash save games for RTC data") fixed it.
- 23:15: handoff to this session; "fix built, not verified".
- 23:16–23:22: verification. The repro script had a button bug (`add_keys` takes key
  indices, not bitmasks), so its first "results" were meaningless. Once fixed, 0.10.2 and
  0.10.1 each crashed in some configurations and passed in others.
- 23:24: the cited changelog entry turns out to be from **0.6.2 (2018)**. The real RTC fix,
  `GBASavedataRTCWrite` leaving `currentBank` dangling, shipped in **0.10.3**. Built 0.10.5.
- 23:33: a 0.10.5 matrix test still crashes 38 of 40 runs. **The libmgba version was never the cause.**
- 23:35–23:41: the crash rate collapses to ~0 with nothing changed. Wall-clock (RTC) and
  disk hypotheses were tested and **refuted** (libfaketime: the same timestamp crashes once,
  then passes 3 times).
- 23:42: a core dump shows a jump into non-code from a per-frame callback in `GBAFrameEnded`.
- 23:46: an **AddressSanitizer** build shows `pc 0xbebebebebebebebe`, which is ASan's fill
  pattern for uninitialized `malloc` memory. Reading the source confirms 3 uninitialized pointers.
- 23:50: controlled proof under ASan, which fills `malloc` deterministically:
  **unpatched 5/5 crash, patched 10/10 pass, zero ASan errors.**
- 23:55: production image with the fix: **120/120 save cycles pass.** Each run validates
  every save sector checksum the way the game does, then reboots, CONTINUEs, and checks the
  player is back on the saved tile.

## What was wrong in earlier reasoning (so it isn't repeated)
1. **A changelog match was taken as proof.** The 0.10.2 claim was never checked against
   the code; the diff of `savedata.c` 0.10.1→0.10.2 contains no such fix.
2. **The repro was trusted before it was validated.** A wrong button encoding and a
   missing second `reset()` (all-black screenshots) made early results meaningless.
3. **A plausible mechanism was taken for the cause.** The `currentBank` bug is real and
   fixed upstream, but it wasn't what crashed us. Only a deterministic instrument (ASan's
   0xbe fill) plus an A/B proof settled it.

## Evidence (in this folder)
- `crashed-rom.sav`: the save from the crashed container. **David's save #1 is intact**
  (slot 2, all checksums valid). The game finished writing before the crash.
- `emerald-ai.log`: the full container log, including the fatal error.

## Also fixed in the same pass
- libmgba 0.10.5 (pinned by commit): upstream fixes for the `currentBank` remap bug and
  savestate GPIO/RTC restore. The build asserts the bindings load exactly that commit.
- Saves are kept across restarts and checkpoints carry the in-game save (see CHANGES.md).
- The server now fails loudly instead of leaving a zombie container; human play is
  auto-checkpointed (see CHANGES.md).

## Tests that guard this
- `local_tests/save_cycle_test.py`: save → validate → reboot → CONTINUE → position check.
- `local_tests/gen3_save.py`: Gen 3 save validator (format from pret/pokeemerald `save.c`).
- `local_tests/soak_frames.py`: long random-input run, used under ASan.
