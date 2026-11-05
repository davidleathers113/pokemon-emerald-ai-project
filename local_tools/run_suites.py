"""Run the local test suites several at a time and print one table.

emerald-new (2026-11-04), for David: the suites ran one after another (~40 min before a
push) while the machine sat mostly idle. mGBA is single-threaded, so each suite uses
about one core (seen: ~62% of a CPU, 176 MB); the Docker VM has 8 CPUs. Each docker
suite runs in its own container (own network, and the in-process server works in a
temp cwd), so suites share no ports or files; the repo is mounted as by hand, and no
default suite writes to it. Ollama is never called (test_phase4_backend mocks it).

Not run by default: test_phase23_server_soak.py (needs a running server, --port) and
test_map_location.py (needs a ROM and states as arguments).

A suite that fails while others run is run again alone before the table is printed:
"FLAKY" means it failed under load and passed alone, so the result names both.

  python3 local_tools/run_suites.py                 # every default suite, 4 at a time
  python3 local_tools/run_suites.py -j 5 macros train_macro
"""
import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGE = "emerald-harness:local"
# Longest first (minutes, measured serially), so the pool doesn't end on one long suite.
DOCKER_SUITES = [
    "train_macro", "macros", "warp_routes", "rotating_gates", "movement_exits", "objective_memory", "battle_flow",
    "battle_view",
    "use_item", "party_rule", "dialogue_agent", "dialogue_auto_advance", "phase4_agent",
    "movement_feedback", "surf", "prompt_cache", "play_endpoints", "phase4_backend", "phase23_health",
    "phase23_save_persistence", "gemini_fallback",
]
HOST_PYTHON_SUITES = ["stall_watch", "speed_report"]  # plain Python, no emulator
NODE_SUITES = ["play_panels"]
TIMEOUT_S = 40 * 60


def command(suite: str) -> list:
    if suite in NODE_SUITES:
        return ["node", str(REPO / "local_tests" / f"test_{suite}.js")]
    if suite in HOST_PYTHON_SUITES:
        return [sys.executable, str(REPO / "local_tests" / f"test_{suite}.py")]
    return [
        "docker", "run", "--rm", "--entrypoint", "/opt/venv/bin/python",
        "-v", f"{REPO}:/app", "-w", "/app", IMAGE, f"local_tests/test_{suite}.py",
    ]


def summary_line(output: str) -> str:
    """The suite's own 'N/M passed' line, else its last line."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    passed = [line for line in lines if line.endswith(" passed")]
    return (passed or lines or ["(no output)"])[-1]


def run(suite: str, log_dir: Path) -> dict:
    started = time.time()
    try:
        done = subprocess.run(command(suite), cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S)
        output, code = done.stdout + done.stderr, done.returncode
    except subprocess.TimeoutExpired as e:
        output, code = f"{e.stdout or ''}{e.stderr or ''}\nTIMED OUT after {TIMEOUT_S} s", 124
    log = log_dir / f"{suite}.log"
    log.write_text(output, encoding="utf-8")
    failed = [line.strip() for line in output.splitlines() if line.startswith("FAIL")]
    return {"suite": suite, "code": code, "seconds": time.time() - started, "line": summary_line(output),
            "failed": failed, "log": log}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="*", help="suite names without test_ (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="suites at a time (default 4)")
    args = parser.parse_args()
    known = DOCKER_SUITES + HOST_PYTHON_SUITES + NODE_SUITES
    suites = [s[len("test_"):] if s.startswith("test_") else s for s in args.suites] or known
    unknown = [s for s in suites if s not in known]
    if unknown:
        print(f"Unknown suites: {', '.join(unknown)}. Known: {', '.join(known)}")
        return 2
    log_dir = Path(tempfile.mkdtemp(prefix="suites-"))
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda s: run(s, log_dir), suites))
    alone_dir = log_dir / "alone"
    alone_dir.mkdir()
    for result in [r for r in results if r["code"] != 0]:
        result["alone"] = run(result["suite"], alone_dir)  # nothing else running now
    print(f"\n{'suite':26} {'result':8} {'time':>7}  summary")
    failures = 0
    for r in results:
        alone = r.get("alone")
        if r["code"] == 0:
            verdict = "ok"
        elif alone is not None and alone["code"] == 0:
            verdict = "FLAKY"
        else:
            verdict = "FAIL"
            failures += 1
        print(f"{r['suite']:26} {verdict:8} {r['seconds']:6.0f}s  {r['line']}")
        # A real failure shows what failed alone; a flaky one, what failed under load.
        shown = alone if verdict == "FAIL" and alone is not None else r
        for line in shown["failed"]:
            print(f"{'':36}{line[:160]}")
    print(f"\n{len(results) - failures}/{len(results)} suites passed in {time.time() - started:.0f} s; logs in {log_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
