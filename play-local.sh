#!/usr/bin/env bash
# Start (or resume) the AI-played Pokémon Emerald run on this Mac.
#
#   ./play-local.sh          resume the newest AI run (new game if none exists)
#   ./play-local.sh new      start a brand-new game
#   ./play-local.sh stop     checkpoint, then stop the running game
#
# Defaults to the local Ollama model. To use a cloud model instead, set e.g.
#   BACKEND=gemini MODEL=gemini-2.5-flash GEMINI_API_KEY=... ./play-local.sh
# Works with macOS /bin/bash 3.2 (empty arrays are expanded with ${a[@]+...}).
set -euo pipefail

cd "$(dirname "$0")"
NAME=emerald-ai
IMAGE=emerald-harness:local
PORT="${PORT:-8000}"
# "ollama" = the harness's OpenAI backend pointed at Ollama on this Mac. Any other
# value is passed to the harness as-is (openai, gemini, ...) and uses real API keys,
# so an OPENAI_API_KEY exported for other work can't silently send the game to OpenAI.
BACKEND="${BACKEND:-ollama}"
# Non-thinking (instruct) variant: the default qwen3-vl:8b tag is the thinking model,
# which spent minutes per move reasoning. "simple" drops the built-in subagents.
MODEL="${MODEL:-qwen3-vl:8b-instruct}"
SCAFFOLD="${SCAFFOLD:-simple}"
# Story roadmap the agent works through (212 story + 57 battle objectives). It's
# restored with the run on resume; OBJECTIVES_START only applies when a run has no
# saved objectives yet (index into the story list, e.g. 5 = "Visit rival's house").
OBJECTIVES="${OBJECTIVES:-categorized_full_game}"
OBJECTIVES_START="${OBJECTIVES_START:-0}"
# Prompt size the agent trims history to. Ollama reads prompts at ~250 tokens/s on
# this Mac, so each move costs roughly budget/250 s: 14K keeps moves under a minute
# with the last ~10 steps of history; raise it for more memory at the cost of speed.
PROMPT_BUDGET="${PROMPT_BUDGET:-14000}"
# Game speed while the AI thinks / while the harness is pressing buttons (0 = unlimited).
GAME_FPS="${GAME_FPS:-60}"
TURBO_FPS="${TURBO_FPS:-480}"
RUN_TAG=ai-local
OLLAMA_URL=http://127.0.0.1:11434
# Settings our Ollama must run with (Ollama FAQ): flash attention + q8_0 KV cache
# halve cache memory; one model, one request at a time; 32K context (prompts measured
# ~7-9K tokens, peak ~21.6K); the model never unloads, so handing control back to
# the AI after playing for a while doesn't pay a cold reload.
OLLAMA_SETTINGS=(OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=32768
                 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1)

die() { echo "ERROR: $*" >&2; exit 1; }

healthy() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }

# docker inspect prints an empty line before failing on a missing container, so
# its output only counts when it succeeds.
container_state() {
    local state
    state=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null) || state=missing
    echo "$state"
}

# Copy the live in-game save (rom.sav in the emulator's temp dir) out of a container
# that is about to be removed, so a crash between checkpoints can't lose it.
rescue_save() {
    local dest=.pokeagent_cache/rescued_saves stamp
    stamp=$(date +%Y%m%d-%H%M%S)
    mkdir -p "$dest/$stamp"
    if docker cp "$NAME:/tmp" "$dest/$stamp/tmp" >/dev/null 2>&1; then
        find "$dest/$stamp/tmp" -name rom.sav -exec mv {} "$dest/$stamp/rom.sav" \; 2>/dev/null || true
        rm -rf "${dest:?}/${stamp:?}/tmp"
    fi
    if [ -f "$dest/$stamp/rom.sav" ]; then
        echo "Copied the in-game save out of the container: $dest/$stamp/rom.sav"
    else
        rmdir "$dest/$stamp" 2>/dev/null || true
    fi
}

checkpoint() {
    curl -sf -m 60 -X POST -H 'Content-Type: application/json' -d '{}' \
        "http://127.0.0.1:$PORT/checkpoint" >/dev/null
}

remove_container() {
    [ "$(container_state)" = missing ] && return 0
    rescue_save
    docker rm -f "$NAME" >/dev/null
}

if [ "${1:-}" = "stop" ]; then
    if [ "$(container_state)" = missing ]; then echo "Not running."; exit 0; fi
    if healthy && checkpoint; then
        echo "Checkpoint saved."
    else
        echo "WARNING: the game server didn't answer, so no final checkpoint was written." >&2
    fi
    remove_container
    echo "Stopped."
    exit 0
fi

case "$(container_state)" in
    running)
        if healthy; then
            echo "Already running. Watch at http://127.0.0.1:$PORT/play  (stop with: $0 stop)"
            open "http://127.0.0.1:$PORT/play"
            exit 0
        fi
        die "container $NAME is running but the game server isn't answering on port $PORT.
Last log lines:
$(docker logs --tail 30 "$NAME" 2>&1)
Restart it with: $0 stop && $0"
        ;;
    missing) ;;
    *)
        echo "Previous container exited ($(docker inspect -f '{{.State.ExitCode}}' "$NAME")). Last log lines:"
        docker logs --tail 15 "$NAME" 2>&1 | while IFS= read -r line; do printf '  %s\n' "$line"; done
        remove_container
        ;;
esac

ollama_pid() { lsof -ti "tcp:11434" -sTCP:LISTEN 2>/dev/null | head -1 || true; }

# Prints the required settings the running Ollama server lacks (empty = all set).
ollama_missing_settings() {
    local env setting
    env=$(ps eww -o command= -p "$1" 2>/dev/null | tr ' ' '\n')
    for setting in "${OLLAMA_SETTINGS[@]}"; do
        printf '%s\n' "$env" | grep -qxF "$setting" || printf '%s ' "$setting"
    done
}

start_ollama() {
    echo "Starting Ollama..."
    env "${OLLAMA_SETTINGS[@]}" nohup ollama serve >>"$HOME/Library/Logs/ollama-serve.log" 2>&1 &
    for _ in $(seq 1 30); do
        curl -sf -m 2 "$OLLAMA_URL/api/version" >/dev/null && return 0
        sleep 1
    done
    die "Ollama didn't start within 30 s; see ~/Library/Logs/ollama-serve.log"
}

env_args=()
harness_backend="$BACKEND"
if [ "$BACKEND" = "ollama" ]; then
    command -v ollama >/dev/null || die "ollama is not installed (brew install ollama)"
    harness_backend=openai
    pid=$(ollama_pid)
    if [ -z "$pid" ]; then
        start_ollama
    else
        missing=$(ollama_missing_settings "$pid")
        if [ -n "$missing" ]; then
            # Only restart an Ollama this script started (it carries our settings);
            # anything else (Ollama.app, brew services, another project) is left alone.
            if ps eww -o command= -p "$pid" | tr ' ' '\n' | grep -qxF OLLAMA_CONTEXT_LENGTH=32768; then
                echo "Restarting Ollama to apply: $missing"
                kill "$pid"
                for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
                start_ollama
            else
                die "an Ollama server is already running without the settings this game needs:
  missing: $missing
Quit it (Ollama.app menu > Quit, or 'brew services stop ollama'), then rerun $0."
            fi
        fi
    fi
    ollama list | awk 'NR > 1 {print $1}' | grep -qxF "$MODEL" \
        || die "model $MODEL is not pulled (ollama pull $MODEL)"
    # Load the model now so the first move doesn't wait for it (keep_alive -1 = stay loaded).
    curl -sf -m 300 "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"$MODEL\", \"keep_alive\": -1}" >/dev/null \
        || die "Ollama couldn't load $MODEL; see ~/Library/Logs/ollama-serve.log"
    # Sampling from the Qwen3-VL-8B-Instruct model card (VL tasks); without it
    # Ollama's /v1/responses defaults to temperature 1.0.
    env_args+=(-e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 -e OPENAI_API_KEY=ollama
               -e OPENAI_TEMPERATURE=0.7 -e OPENAI_TOP_P=0.8)
else
    for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY; do
        if [ -n "${!var:-}" ]; then env_args+=(-e "$var"); fi
    done
fi
# Speed-plan switches, passed through when set so each can be tried on its own, e.g.
# BRIEF_REASONING=1 ./play-local.sh (one-line rationale), SCREENSHOT_POLICY=needed (the
# frame only when the text state can't describe the screen), OPENAI_MAX_OUTPUT_TOKENS.
for var in BRIEF_REASONING SCREENSHOT_POLICY OPENAI_MAX_OUTPUT_TOKENS; do
    if [ -n "${!var:-}" ]; then env_args+=(-e "$var"); fi
done

state_args=()
if [ "${1:-}" != "new" ]; then
    # Resume the newest previous AI run that has a checkpoint. Each launch gets a new
    # run folder, so the old one is handed over as a backup zip (--backup-state, the
    # harness's documented way to resume): that restores the game state and in-game
    # save, plus the agent's memory, objective progress and history, not just the game.
    latest=""
    for f in .pokeagent_cache/*_"$RUN_TAG"*/checkpoint.state; do
        [ -e "$f" ] || continue
        if [ -z "$latest" ] || [ "$f" -nt "$latest" ]; then latest=$f; fi
    done
    if [ -n "$latest" ]; then
        run_dir=$(dirname "$latest")
        zip_path="backups/resume/$(basename "$run_dir").zip"
        mkdir -p backups/resume
        python3 - "$run_dir" "$zip_path" <<'PY'
import os, sys, zipfile

run_dir, zip_path = sys.argv[1].rstrip("/"), sys.argv[2]
tmp = zip_path + ".tmp"
# Same layout the harness's backup_manager writes: one top-level <run_id>/ folder.
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            path = os.path.join(root, name)
            zf.write(path, os.path.relpath(path, os.path.dirname(run_dir)))
os.replace(tmp, zip_path)
PY
        echo "Resuming from $run_dir"
        state_args=(--backup-state "$zip_path")
    else
        echo "No previous AI run found; starting a new game."
    fi
else
    echo "Starting a new game."
fi

# A fresh launch means the AI plays; a pause left over from the last session
# would otherwise keep it idle with no sign why.
rm -f .pokeagent_cache/AGENT_PAUSED

docker run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" \
    -v "$PWD:/app" -e PYTHONFAULTHANDLER=1 -e POKEAGENT_PROMPT_TOKEN_BUDGET="$PROMPT_BUDGET" \
    -e GAME_FPS="$GAME_FPS" -e TURBO_FPS="$TURBO_FPS" \
    ${env_args[@]+"${env_args[@]}"} "$IMAGE" \
    --game emerald --backend "$harness_backend" --model-name "$MODEL" --port 8000 \
    --agent-auto --scaffold "$SCAFFOLD" --headless \
    --direct-objectives "$OBJECTIVES" --direct-objectives-start "$OBJECTIVES_START" \
    --run-name "$RUN_TAG-$(date +%Y%m%d-%H%M)" ${state_args[@]+"${state_args[@]}"} >/dev/null

echo "Started ($MODEL, $SCAFFOLD). Waiting for the game server..."
for _ in $(seq 1 90); do
    healthy && break
    if [ "$(container_state)" != running ]; then
        die "the container stopped during startup. Last log lines:
$(docker logs --tail 50 "$NAME" 2>&1)"
    fi
    sleep 2
done
healthy || die "the game server didn't come up within 3 minutes. Last log lines:
$(docker logs --tail 50 "$NAME" 2>&1)"

# Seed Emerald controls/story hints into the agent's long-term memory. Titles
# already present (e.g. after a resume) are skipped.
python3 - "$PORT" hints/emerald_hints.json <<'PY'
import json, sys, urllib.request

port, path = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}/mcp"

def post(endpoint, body):
    req = urllib.request.Request(f"{base}/{endpoint}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)

added = 0
for hint in json.load(open(path)):
    found = post("search_memory", {"query": hint["title"]})
    titles = {r.get("title") for r in found.get("results", [])}
    if hint["title"] in titles:
        continue
    if post("add_memory", hint).get("success"):
        added += 1
print(f"Hints: {added} added")
PY

echo "  Play/controller: http://127.0.0.1:$PORT/play"
echo "  AI thinking:     http://127.0.0.1:$PORT/stream"
echo "  Stop with:       $0 stop"
open "http://127.0.0.1:$PORT/play"
