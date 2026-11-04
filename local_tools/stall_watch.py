"""Watch the AI's decisions and pause it the moment it looks stuck.

David (2026-11-04), after measuring that about half of two hours of walk/train calls
left the player on the same tile: stalls must be caught in about two minutes, not at the
next five-minute check. Each stall prints one line (for a watcher to act on) and pauses
the agent (POST /agent_pause), so no more time is spent until someone looks. It watches
again once the agent is running.

Stuck means any of:
  - SAME_TILE: the last 3 walking/training calls ended with the player on the same tile
    and map (battles are left out: a battle keeps the player still on purpose);
  - REPEAT: the same tool with the same arguments 4 times in a row;
  - DOWN: the game server did not answer 3 polls in a row.

  python3 local_tools/stall_watch.py [--interval 15] [--no-pause]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = "http://127.0.0.1:8000"
MOVE_TOOLS = {"navigate_to", "go_to_exit", "go_to", "train", "talk_to"}
SAME_TILE_RUN = 3
REPEAT_RUN = 4
DOWN_RUN = 3


def _get(path: str):
    with urllib.request.urlopen(SERVER + path, timeout=8) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _latest_decisions(limit: int = 12) -> list:
    """The newest run's last decisions, oldest first (one run dir per server start)."""
    runs = sorted(REPO.glob(".pokeagent_cache/*_ai-local-*/trajectory_history.jsonl"), key=lambda p: p.stat().st_mtime)
    if not runs:
        return []
    rows = []
    for line in runs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _call(row: dict):
    calls = (row.get("action") or {}).get("tool_calls") or []
    if not calls:
        return None, {}
    args = {k: v for k, v in (calls[0].get("args") or {}).items() if k not in ("reasoning", "reason")}
    return calls[0].get("name"), args


def _took(rows: list, i: int) -> float:
    """Seconds from decision i to the next one (how long its call ran), or 0."""
    from datetime import datetime

    if i + 1 >= len(rows):
        return 0.0
    try:
        a = datetime.fromisoformat(rows[i]["timestamp"])
        b = datetime.fromisoformat(rows[i + 1]["timestamp"])
    except (KeyError, ValueError, TypeError):
        return 0.0
    return (b - a).total_seconds()


# A walk/train call that ran this long did something (train stepping in the grass for
# minutes ends back on its own tile).
WORKED_S = 60.0


def stall_reason(rows: list) -> str:
    """Why these decisions look stuck, or '' (pure: tested in local_tests)."""
    moves = []
    for i, r in enumerate(rows):
        if _call(r)[0] not in MOVE_TOOLS:
            continue
        if _took(rows, i) > WORKED_S:
            moves = []  # progress: start counting again
            continue
        moves.append(r)
    tail = moves[-SAME_TILE_RUN:]
    if len(tail) == SAME_TILE_RUN:
        spots = {(r.get("location"), tuple(r.get("player_coords") or ())) for r in tail}
        if len(spots) == 1:
            loc, xy = next(iter(spots))
            names = ", ".join(_call(r)[0] for r in tail)
            return f"SAME_TILE: {SAME_TILE_RUN} walk/train calls left the player at {xy} on {loc} ({names})"
    calls = [_call(r) for r in rows[-REPEAT_RUN:]]
    if len(calls) == REPEAT_RUN and calls[0][0] and all(c == calls[0] for c in calls):
        return f"REPEAT: {calls[0][0]}({json.dumps(calls[0][1])}) {REPEAT_RUN} times in a row"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--no-pause", action="store_true", help="only report, never pause the agent")
    args = parser.parse_args()
    last_seen = None  # timestamp of the decision a stall was reported on
    down = 0
    while True:
        try:
            paused = bool(_get("/agent_pause").get("paused"))
            down = 0
        except Exception:
            down += 1
            if down == DOWN_RUN:
                print(f"DOWN: the game server did not answer {DOWN_RUN} polls ({args.interval * DOWN_RUN:.0f} s)", flush=True)
            time.sleep(args.interval)
            continue
        rows = _latest_decisions()
        newest = rows[-1].get("timestamp") if rows else None
        if not paused and newest and newest != last_seen:
            why = stall_reason(rows)
            if why:
                last_seen = newest
                if not args.no_pause:
                    try:
                        _post("/agent_pause", {"paused": True})
                        why += " — agent PAUSED"
                    except Exception as e:
                        why += f" — could not pause ({e})"
                print(f"STALL {time.strftime('%H:%M:%S')}: {why}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
