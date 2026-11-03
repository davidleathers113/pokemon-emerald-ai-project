"""Soak the running game server's state endpoints against live button presses.

Concurrently, for --seconds: POST /checkpoint, POST /load_checkpoint, POST /save_state
+ /load_state (rotating files), POST /action presses, GET /state, and GET /health
every 0.5 s. Every response must succeed and /health must stay 200. After each
/checkpoint the file on disk is checked to be a complete state with its flash save;
every --keep-every'th one is copied aside and, at the end, each copy is loaded into a
fresh EmeraldEmulator. Run it where the server's cache dir is visible at the same
relative path, e.g. sharing the server container's network and repo mount:

  docker run --rm --network container:<server> -v <repo copy>:/app -w /app \
      --entrypoint python emerald-harness:local local_tests/test_phase23_server_soak.py \
      --port 8011 --seconds 330

Exit 0 = no failed request, no unhealthy /health, every kept checkpoint loads.
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

lock = threading.Lock()
stats = Counter()
errors = []
latency_max = Counter()


def record(name, ok, seconds, detail=""):
    with lock:
        stats[f"{name}.{'ok' if ok else 'fail'}"] += 1
        latency_max[name] = max(latency_max[name], seconds)
        if not ok and len(errors) < 50:
            errors.append(f"{time.strftime('%H:%M:%S')} {name}: {detail}")


def call(base, method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return resp.status, payload, time.monotonic() - start
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.monotonic() - start
    except Exception as e:  # connection refused/reset = server gone
        return None, str(e).encode(), time.monotonic() - start


def complete_state(blob, state_size):
    """True if blob is a raw state followed by mGBA extdata holding a flash save."""
    from pokemon_env.emulator import _extract_savedata
    if len(blob) < state_size:
        return False
    savedata = _extract_savedata(blob, state_size)
    return savedata is not None and len(savedata) == 131072


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--seconds", type=float, default=330)
    p.add_argument("--keep-every", type=int, default=10)
    a = p.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    deadline = time.monotonic() + a.seconds
    kept_dir = Path(tempfile.mkdtemp(prefix="soak_ckpts_"))
    rotating = [f".pokeagent_cache/soak_rot_{i}.state" for i in range(8)]
    written = set()
    state_size = 397312  # GBA state size in mGBA 0.10 (stateSize)

    def ok_json(status, payload, key, want):
        if status != 200:
            return False
        try:
            return json.loads(payload).get(key) == want
        except ValueError:
            return False

    def checkpoints():
        n = 0
        while time.monotonic() < deadline:
            n += 1
            st, body, dt = call(base, "POST", "/checkpoint", {"step_count": n})
            ok = ok_json(st, body, "status", "checkpoint_saved")
            record("checkpoint", ok, dt, f"{st} {body[:200]!r}")
            if ok:
                path = json.loads(body)["files"]["state"]
                blob = Path(path).read_bytes()
                good = complete_state(blob, state_size)
                record("checkpoint_file_complete", good, 0, f"{path}: {len(blob)} bytes")
                if n % a.keep_every == 0:
                    (kept_dir / f"ckpt_{n:05d}.state").write_bytes(blob)

    def load_checkpoints():
        while time.monotonic() < deadline:
            st, body, dt = call(base, "POST", "/load_checkpoint")
            ok = st == 200 and json.loads(body).get("status") in ("checkpoint_loaded", "no_checkpoint")
            record("load_checkpoint", ok, dt, f"{st} {body[:200]!r}")
            time.sleep(random.uniform(0, 0.3))

    def save_load():
        while time.monotonic() < deadline:
            target = random.choice(rotating)
            st, body, dt = call(base, "POST", "/save_state", {"filepath": target})
            ok = ok_json(st, body, "status", "success")
            record("save_state", ok, dt, f"{st} {body[:200]!r}")
            if ok:
                written.add(target)
            if written:
                source = random.choice(sorted(written))
                st, body, dt = call(base, "POST", "/load_state", {"filepath": source})
                record("load_state", ok_json(st, body, "status", "success"), dt, f"{st} {body[:200]!r}")

    def actions():
        buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "B"]
        while time.monotonic() < deadline:
            press = random.sample(buttons, random.randint(1, 3))
            st, body, dt = call(base, "POST", "/action", {"buttons": press})
            record("action", ok_json(st, body, "status", "success"), dt, f"{st} {body[:200]!r}")
            time.sleep(random.uniform(0.05, 0.3))

    def states():
        while time.monotonic() < deadline:
            st, body, dt = call(base, "GET", "/state")
            record("state", st == 200, dt, f"{st} {body[:200]!r}")
            time.sleep(random.uniform(0.1, 0.5))

    def health():
        while time.monotonic() < deadline:
            st, body, dt = call(base, "GET", "/health", timeout=10)
            record("health", st == 200, dt, f"{st} {body[:200]!r}")
            time.sleep(0.5)

    workers = [checkpoints, load_checkpoints, save_load, actions, actions, states, health]
    threads = [threading.Thread(target=w, name=w.__name__) for w in workers]
    started = time.monotonic()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            print(f"[{time.monotonic() - started:5.0f}s] {dict(sorted(stats.items()))}", flush=True)
    for t in threads:
        t.join()

    print("\nrequests:", dict(sorted(stats.items())))
    print("max latency s:", {k: round(v, 3) for k, v in sorted(latency_max.items())})
    for e in errors:
        print("  ERROR", e)

    # Every kept checkpoint (and every rotating save_state file) must load into a fresh emulator.
    from pokemon_env.emulator import EmeraldEmulator
    files = sorted(kept_dir.glob("*.state")) + [Path(r) for r in sorted(written)]
    rom = str(REPO / "Emerald-GBAdvance" / "rom.gba")
    copies = Path(tempfile.mkdtemp(prefix="soak_verify_"))
    for f in files:
        shutil.copy(f, copies / f.name)
    os.chdir(tempfile.mkdtemp())  # keep the emulator's cache writes out of the repo copy
    emu = EmeraldEmulator(rom_path=rom)
    emu.initialize()
    load_fail = 0
    for f in sorted(copies.glob("*.state")):
        try:
            emu.load_state(str(f))
        except Exception as e:
            load_fail += 1
            print(f"  LOAD FAIL {f.name}: {e}")
    print(f"verified loads: {len(files) - load_fail}/{len(files)} state files load into a fresh emulator")

    failed = sum(v for k, v in stats.items() if k.endswith(".fail"))
    passed = failed == 0 and load_fail == 0 and stats["checkpoint.ok"] > 0 and stats["health.ok"] > 0
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
