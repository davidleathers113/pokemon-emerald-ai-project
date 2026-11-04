"""Watch the AI's decisions and pause it the moment it looks stuck.

David (2026-11-04), after measuring that about half of two hours of walk/train calls
left the player on the same tile: stalls must be caught in about two minutes, not at the
next five-minute check. Each stall prints one line (for a watcher to act on) and pauses
the agent (POST /agent_pause), so no more time is spent until someone looks. It watches
again once the agent is running.

Stuck means any of:
  - SAME_TILE: the last 3 walking/training calls each ended with the player on the tile
    and map it started from (battles are left out: a battle keeps the player still on
    purpose);
  - REPEAT: the same tool with the same arguments 4 times in a row;
  - CYCLE: the last 6 calls repeat a 2- or 3-call pattern while nothing changes;
  - NO_PROGRESS: 8 decisions in a row with nothing changing (tile, battle and party
    lines of the prompt);
  - WANDER: 5 minutes among 2 or 3 maps, going back to one already left, with no
    objective completed and the party unchanged;
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
CYCLE_RUN = 6
NO_PROGRESS_RUN = 8
WANDER_S = 300.0
WANDER_MAPS = 3
WANDER_RETURNS = 2
DOWN_RUN = 3


def _get(path: str):
    with urllib.request.urlopen(SERVER + path, timeout=8) as r:
        return json.loads(r.read().decode())


def _post(path: str, body: dict):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


def _latest_decisions(limit: int = 30) -> list:
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


def _spot(row: dict) -> tuple:
    return row.get("location"), tuple(row.get("player_coords") or ())


def stall_reason(rows: list) -> str:
    """Why these decisions look stuck, or '' (pure: tested in local_tests).

    A row's location and player_coords are where the player stood when that decision
    was made, so where a call LEFT the player is the next row's. Seen live: navigate_to,
    navigate_to, go_to_exit("east") from (12,4) in VERDANTURF TOWN was reported stuck
    though the go_to_exit had reached ROUTE 117; each row's own (starting) tile was
    compared. The newest call has no next row yet and is judged once it does.
    """
    moves = []
    for i, r in enumerate(rows[:-1]):
        if _call(r)[0] not in MOVE_TOOLS:
            continue
        if _took(rows, i) > WORKED_S or _spot(rows[i + 1]) != _spot(r):
            moves = []  # progress: start counting again
            continue
        moves.append(r)
    tail = moves[-SAME_TILE_RUN:]
    if len(tail) == SAME_TILE_RUN:
        loc, xy = _spot(tail[-1])
        names = ", ".join(_call(r)[0] for r in tail)
        return f"SAME_TILE: {SAME_TILE_RUN} walk/train calls left the player at {xy} on {loc} ({names})"
    calls = [_call(r) for r in rows[-REPEAT_RUN:]]
    # Seen live 08:42: four throw_ball("Great Ball") at a full-HP ELECTRIKE (8 balls -> 4)
    # were a catch attempt, not a stall: the ball count changed every time.
    if (
        len(calls) == REPEAT_RUN and calls[0][0] and all(c == calls[0] for c in calls)
        and not _changed(rows[-REPEAT_RUN:])
    ):
        return f"REPEAT: {calls[0][0]}({json.dumps(calls[0][1])}) {REPEAT_RUN} times in a row"
    # Seen live 07:06-07:36: press_buttons(A) and switch_pokemon(...) in turn for 30
    # minutes in the game's replacement menu; neither SAME_TILE (no walking) nor REPEAT
    # (two different calls) saw it.
    tail = rows[-CYCLE_RUN:]
    if len(tail) == CYCLE_RUN and _unchanged(tail):
        calls = [_call(r) for r in tail]
        for period in (2, 3):
            if all(calls[i] == calls[i - period] for i in range(period, CYCLE_RUN)):
                names = " / ".join(f"{name}({json.dumps(args)})" for name, args in calls[:period])
                return f"CYCLE: {names} in turn, {CYCLE_RUN} calls with nothing changing"
    tail = rows[-NO_PROGRESS_RUN:]
    if len(tail) == NO_PROGRESS_RUN and _unchanged(tail):
        loc, xy = _spot(tail[-1])
        return f"NO_PROGRESS: {NO_PROGRESS_RUN} decisions at {xy} on {loc} with no change to the battle or party"
    return _wander_reason(rows)


def _time(row: dict):
    from datetime import datetime

    try:
        return datetime.fromisoformat(row["timestamp"])
    except (KeyError, ValueError, TypeError):
        return None


def _party(row: dict):
    """The prompt's party lines ("1. Breloom (Lv.27) HP: 72/75"), or None without a prompt."""
    prompt = row.get("llm_prompt")
    if not prompt:
        return None
    return tuple(
        s for s in (line.strip() for line in str(prompt).splitlines())
        if s[:1].isdigit() and ". " in s[:4] and "(Lv." in s and "HP" in s
    )


def _wander_reason(rows: list) -> str:
    """WANDER, or ''.

    Seen live 12:48-12:54 UTC: go_to, go_to_exit("west"), go_to_exit("north") in turn,
    three times, bouncing between ROUTE 111 and ROUTE 112 with the story objective
    unchanged. Every call changed the tile, so SAME_TILE, CYCLE and NO_PROGRESS stayed
    quiet. A party that changed (a fight, a heal, a level) is something happening, a
    trip through a map without coming back to it is travel, and one return is an errand
    (into the Mart and out): it takes WANDER_RETURNS returns.
    """
    newest = _time(rows[-1]) if rows else None
    if newest is None:
        return ""
    old = [i for i, r in enumerate(rows) if _time(r) is not None and (newest - _time(r)).total_seconds() >= WANDER_S]
    if not old:
        return ""  # the rows don't reach back far enough to judge
    window = rows[old[-1]:]
    if any((r.get("outcome") or {}).get("objectives_completed") for r in window):
        return ""
    parties = [_party(r) for r in window]
    if any(p is None or p != parties[0] for p in parties):
        return ""
    visits = []
    for r in window:
        if r.get("location") and (not visits or visits[-1] != r.get("location")):
            visits.append(r.get("location"))
    maps = list(dict.fromkeys(visits))
    if not 2 <= len(maps) <= WANDER_MAPS or len(visits) - len(maps) < WANDER_RETURNS:
        return ""
    minutes = (newest - _time(window[0])).total_seconds() / 60
    return f"WANDER: {minutes:.0f} min going {' → '.join(visits)} with no objective completed"


def _fingerprint(row: dict):
    """What a decision saw that progress would change: the tile, in battle or not, and
    the prompt's party lines ("1. Breloom (Lv.27) HP: 72/75") and battle lines ("…battle
    vs WINGULL Lv12 (WATER/FLYING), HP 30/30", "You: BRELOOM …"), and the bag's ball
    lines ("- Great Ball x8": a throw uses one). None without a prompt."""
    prompt = row.get("llm_prompt")
    if not prompt:
        return None
    lines = []
    for line in str(prompt).splitlines():
        s = line.strip()
        party = s[:1].isdigit() and ". " in s[:4] and "(Lv." in s and "HP" in s
        balls = s.startswith("- ") and " Ball x" in s
        if party or balls or "battle vs " in s or s.startswith("You: "):
            lines.append(s)
    pre = row.get("pre_state") or {}
    return _spot(row), bool(pre.get("is_in_battle")), tuple(dict.fromkeys(lines))


def _unchanged(rows: list) -> bool:
    prints = [_fingerprint(r) for r in rows]
    return prints[0] is not None and all(p == prints[0] for p in prints)


def _changed(rows: list) -> bool:
    """Something visibly changed between these rows (rows without a prompt show nothing)."""
    prints = [p for p in (_fingerprint(r) for r in rows) if p is not None]
    return any(p != prints[0] for p in prints)


def rows_after(rows: list, watch_from: str) -> list:
    """The rows decided after watch_from (an ISO time like the rows' timestamps).

    Seen live: after ./play-local.sh restarted the game, the new run's trajectory file
    began with the previous run's rows (their own timestamps kept), so the stall that
    had just been reported and handled paused the agent again right after the unpause.
    """
    from datetime import datetime

    start = datetime.fromisoformat(watch_from)
    kept = []
    for row in rows:
        try:
            if datetime.fromisoformat(row["timestamp"]) > start:
                kept.append(row)
        except (KeyError, ValueError, TypeError):
            continue
    return kept


def newest_time(rows: list) -> str:
    """The newest row's own timestamp, the cutoff for rows_after ("" sorts before all).

    Rows are stamped by the game container (UTC); the watcher runs on the Mac (local
    time, five hours behind). Seen live: a cutoff from the Mac's clock let every row of
    the last five hours through, so a stall already handled paused the agent again 11 s
    after the unpause. Taking the cutoff from the rows themselves needs no time zone.
    """
    from datetime import datetime

    stamps = []
    for row in rows:
        try:
            stamps.append(datetime.fromisoformat(row["timestamp"]))
        except (KeyError, ValueError, TypeError):
            continue
    return max(stamps).isoformat() if stamps else "0001-01-01T00:00:00"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--no-pause", action="store_true", help="only report, never pause the agent")
    args = parser.parse_args()
    last_seen = None  # timestamp of the decision a stall was reported on
    # Only decisions newer than the newest one at the watcher's start, then at each
    # unpause, count (whoever unpaused has dealt with what came before).
    watch_from = newest_time(_latest_decisions())
    was_paused = False
    down = 0
    while True:
        try:
            paused = bool(_get("/agent_pause").get("paused"))
            down = 0
            if was_paused and not paused:
                watch_from = newest_time(_latest_decisions())
            was_paused = paused
        except Exception:
            down += 1
            if down == DOWN_RUN:
                print(f"DOWN: the game server did not answer {DOWN_RUN} polls ({args.interval * DOWN_RUN:.0f} s)", flush=True)
            time.sleep(args.interval)
            continue
        rows = rows_after(_latest_decisions(), watch_from)
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
