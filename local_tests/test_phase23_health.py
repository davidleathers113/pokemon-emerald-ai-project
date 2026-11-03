"""/health answers from the game loop's frame heartbeat, not just from a live web thread.

Calls server.app.get_health() directly with the heartbeat in each state:
no emulator, no frame yet, fresh frame, stale frame (> HEALTH_MAX_FRAME_AGE_S).

  docker run --rm --entrypoint python -v ~/continual-harness:/app -w /app emerald-harness:local \
      local_tests/test_phase23_health.py
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def status_of(response):
    if isinstance(response, dict):
        return 200, response
    return response.status_code, json.loads(response.body)


def main():
    os.chdir(tempfile.mkdtemp())  # importing the server creates cache dirs relative to cwd
    from server import app as srv

    failures = 0

    def check(name, expected_code):
        nonlocal failures
        code, body = status_of(asyncio.run(srv.get_health()))
        ok = code == expected_code
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: {code} {body}")

    srv.env = None
    check("no emulator", 503)
    srv.env = object()
    srv.last_frame_monotonic = None
    check("emulator up, no frame yet", 503)
    srv.last_frame_monotonic = time.monotonic()
    check("frame just emulated", 200)
    srv.last_frame_monotonic = time.monotonic() - (srv.HEALTH_MAX_FRAME_AGE_S + 1)
    check("no frame for longer than the limit", 503)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
