# Local AI plays Pokémon Emerald: setup and daily use

Everything runs on this Mac. The model runs in Ollama on the host; the game
(mGBA emulator + harness) runs in a Docker container. No cloud keys are used.

## Everyday commands (run in `~/continual-harness`)
| Command | What it does |
|---|---|
| `./play-local.sh` | Resume the newest AI run (memory, objectives, game and in-game save). Opens the controller page. |
| `./play-local.sh stop` | Write a final checkpoint, copy the live in-game save out, stop the container. |
| `./play-local.sh new` | Start a brand-new game (the AI starts from the title screen). |

Pages:
- **http://127.0.0.1:8000/play**: big screen and controller. "Take control" pauses the AI;
  "hand back" resumes it. Keys: arrows = D-pad (hold to keep walking), X = A, Z = B,
  Return = START, Delete = SELECT. While you play, progress is checkpointed every minute,
  at every hand-over and when you leave the tab. A red banner means the game server is
  down; run `./play-local.sh` to resume.
- **http://127.0.0.1:8000/stream**: the AI's reasoning, step by step.

## Where things live
| Path | What |
|---|---|
| `Emerald-GBAdvance/rom.gba` | US Emerald ROM (SHA-1 f3ae088181bf583e55daf962a92bb46f4f1d07b7) |
| `.pokeagent_cache/<run>/checkpoint.state` | Checkpoint: emulator state **plus the in-game save embedded** (528,416 bytes). Older 397,312-byte checkpoints still load, without a save. |
| `.pokeagent_cache/<run>/objectives.json`, `memory.json` | The AI's objective progress and long-term memory, carried into the next launch. |
| `.pokeagent_cache/rescued_saves/<time>/rom.sav` | The live in-game save, copied out whenever a container is stopped or replaced. |
| `backups/resume/*.zip` | The previous run, packed for `--backup-state` (how a launch resumes). |
| `llm_logs/` | Every model call: prompt, reply, latency, tokens. |
| `local_docs/` | This documentation. |

David's trade saves (`~/Games/Pokemon Emerald Trade/`) and GB300 backups are separate
and never touched by any of this.

## Settings (environment variables for `play-local.sh`)
| Variable | Default | Notes |
|---|---|---|
| `MODEL` | `qwen3-vl:8b-instruct` | Must be pulled in Ollama. |
| `BACKEND` | `ollama` | Anything else (`openai`, `gemini`, …) uses real API keys. |
| `OBJECTIVES` | `categorized_full_game` | Story roadmap (212 story + 57 battle objectives). |
| `OBJECTIVES_START` | `0` | Only used when a run has no saved objectives yet. Check a save's real progress with `local_tools/story_progress.py` first. |
| `PROMPT_BUDGET` | `14000` | Tokens the AI's prompt is trimmed to (oldest history first). Ollama reads ~250 tokens/s here, so a move costs about budget ÷ 250 seconds. Raise it for more memory, lower it for speed. |
| `GAME_FPS` / `TURBO_FPS` | `60` / `480` | Game speed while the AI thinks / while the harness presses buttons (0 = unlimited). |
| `POKEAGENT_AUTO_DIALOGUE` | on | Set to `0` to make the AI read dialogue itself. |
| `PORT` | `8000` | |

Ollama is started (or restarted, if it is our own instance with old settings) with:
`OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=32768
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1`.
If a different Ollama (Ollama.app, brew services) is already running without these
settings, the launcher stops with instructions instead of touching it.

## Handing the AI one of your own saves
```
docker run --rm --entrypoint python -v ~/continual-harness:/app -v /path/to/saves:/s \
  emerald-harness:local /app/local_tools/checkpoint_from_sav.py \
  /app/Emerald-GBAdvance/rom.gba /s/your.sav /app/.pokeagent_cache/$(date -u +%Y%m%d_%H%M%S)_ai-local-from-save/checkpoint.state
docker run --rm --entrypoint python -v ~/continual-harness:/app emerald-harness:local \
  /app/local_tools/story_progress.py /app/Emerald-GBAdvance/rom.gba <that checkpoint>
OBJECTIVES_START=<matching story index> ./play-local.sh
```
The first tool refuses a save that doesn't validate or doesn't CONTINUE to the saved spot.
Work on a **copy** of any save you care about.

## Rebuilding the image
`docker build -f Dockerfile.local -t emerald-harness:local .`. The build fails loudly
if libmgba isn't the pinned 0.10.5 commit, if any library symbol is missing, or if the
bindings fix (`local_patches/fix_mgba_callbacks.py`) doesn't apply.

## Tests
Run inside the image (`docker run --rm --entrypoint python -v ~/continual-harness:/app emerald-harness:local …`):
| Test | Checks |
|---|---|
| `local_tests/save_cycle_test.py` | In-game SAVE works; save validates; reboot + CONTINUE returns to the saved tile. |
| `local_tests/test_phase23_save_persistence.py` | Checkpoints carry the in-game save; corrupt or truncated states are rejected safely. |
| `local_tests/test_phase23_health.py`, `test_phase23_server_soak.py` | Health endpoint; concurrent checkpoint/load/action soak. |
| `local_tests/test_phase4_backend.py`, `test_phase4_agent.py` | Model-call limits, retries, tool schema, malformed replies, prompt budget. |
| `local_tests/test_map_location.py` | The AI is told the map it is actually on. |
| `local_tests/test_macros.py` | heal_at_pokecenter, run_from_battle, throw_ball, switch_pokemon, buy_item: outcomes checked from memory, plus refusals. |
| `local_tests/gen3_save.py <file.sav>` | Validate any Emerald save the way the game does. |
