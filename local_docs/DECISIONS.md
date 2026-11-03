# Decisions and their reasons

## Model: `qwen3-vl:8b-instruct` in Ollama
- `qwen3-vl:30b-a3b` needed about 22 GB, pushed the Mac into swap and took the disk down to 3 GB free.
  It was removed.
- `qwen3-vl:8b` (the default tag) is the thinking variant: 3,500+ thinking tokens and over
  3 minutes per move. It was removed.
- The 8B instruct model runs fully on the GPU (37/37 layers), at about 5–40 s per move.
- Sampling is temperature 0.7 / top_p 0.8, following the Qwen3-VL-8B-Instruct model card
  for vision tasks. `presence_penalty` 1.5 can't be passed through Ollama's
  `/v1/responses` endpoint.

## Ollama settings
These come from the Ollama FAQ.
- Flash attention plus a q8_0 KV cache halves the cache memory.
- One request and one model at a time.
- 32K context: prompts measured 7–9K tokens, peaking around 21.6K, and the agent now trims
  history to a 28K budget.
- Keep-alive is −1, so the model never unloads while David is playing by hand.

## Prompt budget: 14K tokens
- Ollama reads prompts at about 250 tokens/s on this Mac.
  - With the agent's default 28K budget, moves grew from 36 s to 64 s as history filled,
    and would reach about 110 s at the cap.
  - 14K keeps moves at about 25–55 s with roughly the last 10 steps of history.
- This is a speed-versus-memory trade-off, adjustable with `PROMPT_BUDGET`.
- Not done: reordering the prompt so Ollama can reuse its cache across moves. The harness
  starts each prompt with the step number and a sliding history window, so the cacheable
  prefix is only the system prompt. Fixing that means restructuring upstream's prompt
  builder, and that's left as a possible future change.

## Emulator
- The real crash fix is in the Python bindings (`local_patches/fix_mgba_callbacks.py`). It was
  proven with ASan.
- libmgba is pinned to 0.10.5 anyway. Its fixes are real, they change no ABI, and the
  build refuses any other library.
- The live `.sav` stays in the container's own `/tmp`, not on the macOS bind mount. A
  memory-mapped file on virtiofs is fragile. Copies go out in checkpoints, and are rescued
  on stop.
- The in-game save is embedded in the checkpoint rather than kept as a sidecar file. The
  harness's backup and restore only carry `checkpoint.state`, so a sidecar would be
  silently dropped.

## Agent
- Scaffold `simple`, plus walkthrough, wiki and navigation tools. An 8B model needs the help;
  the trade-off is no comparability with the paper's baseline.
- Objective roadmap `categorized_full_game` is on. For David's recovered save it starts at
  index 11, "Travel north through Route 101 to Oldale Town". That index came from the
  save's own story variables, not a guess:
  - Birch has been rescued.
  - The rival's mom has been met.
  - `VAR_BIRCH_LAB_STATE` = 3, which means "received starter in lab".
  - The town is Oldale, still not visited.
- The agent is told never to save in game. Checkpoints already include the save, and saving
  costs the AI about 10 moves each time. David can save whenever he likes.

## Recovery of David's play on 2026-11-02
- The crash happened while David was saving. The save it was writing (save #1, outside his
  house with Torchic) turned out to be complete and valid.
- It was turned into a verified checkpoint, so his manual progress was not lost, and that is
  what the AI resumed from.
- The earlier AI checkpoint (23:04, opening cutscenes) is still on disk.

## Deliberately not done
- **Docker Desktop VM memory (12 GB reserved, the container uses well under 1 GB).** It could
  be capped at about 4 GB, but that affects David's other projects. His call.
- **Host memory tuning (2026-11-03): David is handling it himself; don't change these.**
  Measured then:
  - swap 16.3 of 17.4 GB used;
  - llama-server about 13 GB, the Docker VM 12 GB resident;
  - a model call took a median of 31 s, p90 40 s (20–25 s earlier in the night).

  Options he has, ranked:
  1. Docker VM cap 12 GB → about 4 GB (frees about 8 GB).
  2. Ollama context 32K → 16K (frees about 2 GB). Measure the peak prompt tokens first.
  3. Close Chrome tabs and gopls while it plays.
- **Removing unused torch/CUDA packages (~4.3 GB) from the image.** This is size only, with
  no correctness impact, and it would mean diverging from the lockfile.
- **The same schema-flattening bug in the OpenRouter backend.** It's unused here and can't be
  tested without a key.
- **Reporting the bindings bug upstream (mGBA).** It's worth doing, but that's posting on
  David's behalf, so it's his decision.

## Code-review findings deliberately left as they are
- **Read-only endpoints (e.g. `/mcp/get_game_state`) read emulator memory without the lock.**
  Every call that *changes* emulator state is locked, or runs at startup. An unlocked read
  can only see a torn value during a manual `/load_state` or `/load_checkpoint`, which
  normal play never calls. Moving the lock into every emulator method would be a wide
  change with its own deadlock risk.
- **Async endpoints wait for the lock on the event loop.** A checkpoint holds it for well
  under a second (measured max 0.5 s under a heavy soak). If the emulator hung inside a
  load, `/health` would stop answering and callers would correctly see the server as
  unhealthy.
- **The agent's call guard can still be followed by a new request while an abandoned one
  runs.** It needs a single model call stuck past every HTTP timeout (3 × 240 s plus a
  30 s guard); requests close at their timeout and Ollama then cancels the generation.
- **The 28K default prompt budget in `PokeAgent.py` applies to hosted models too.** It is
  an environment setting; the launcher sets 14K for the local model.

## Open notes
- During the work, every scratch file outside the two helper agents' own folders was deleted.
  Neither agent did it, and the cause is unknown. Nothing in the repo or in David's files was
  affected, and the fixtures were recreated.
