"""Where a model call's time goes: two real consecutive agent prompts sent to Ollama the
way the agent sends them, and again through Ollama's native API, which reports how many
prompt tokens it actually evaluated (the rest came from its cache) and for how long.

Speed plan follow-up (2026-11-04): window 1 measured ~29.4 s of model time per overworld
step against ~21.7 s expected from the tokens (3.2K uncached text at ~280 tok/s, the
frame's 1,082 tokens in 5.4 s, ~121 output tokens at ~25 tok/s). This shows whether the
"cached" count Ollama reports on /v1/responses is really skipped.

It uses Ollama's one model slot: pause the agent first. Run inside the harness image:

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 -e OPENAI_API_KEY=ollama \
      emerald-harness:local local_tools/cache_probe.py
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OLLAMA = "http://host.docker.internal:11434"
SERVER = "http://host.docker.internal:8000"


def _consecutive_prompts():
    """The last two prompts of adjacent agent steps from the newest llm_logs file."""
    newest = max(glob.glob(str(REPO / "llm_logs" / "llm_log_*.jsonl")), key=os.path.getmtime)
    rows = []
    for line in open(newest, encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "interaction" and isinstance(row.get("prompt"), str) and row.get("agent_step"):
            rows.append(row)
    for a, b in zip(reversed(rows[:-1]), reversed(rows[1:])):
        if b["agent_step"] == a["agent_step"] + 1:
            return a, b
    raise SystemExit("no two consecutive steps in " + newest)


def _frame():
    with urllib.request.urlopen(f"{SERVER}/api/frame", timeout=10) as r:
        return json.loads(r.read().decode())["frame"]


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    args = parser.parse_args()

    from PIL import Image

    from agents.prompts.paths import SIMPLE_PROMPT_PATH, render_prompt
    from agents.tools.registry import build_tools_for_scaffold
    from utils.agent_infrastructure.vlm_backends import OpenAIBackend

    system = render_prompt((REPO / SIMPLE_PROMPT_PATH).read_text(encoding="utf-8"))
    tools = build_tools_for_scaffold("simple")
    backend = OpenAIBackend(args.model, tools=tools, system_instruction=system)
    first, second = _consecutive_prompts()
    frame = _frame()
    image = Image.open(io.BytesIO(base64.b64decode(frame)))
    print(f"steps {first['agent_step']} and {second['agent_step']}: prompts of {len(first['prompt'])} and {len(second['prompt'])} chars")

    print("\n/v1/responses, as the agent calls it (first warms the cache, second reuses it):")
    for row in (first, second, second):
        started = time.time()
        response = backend._call_responses(system, backend._responses_input(image, row["prompt"]), tools=backend._tools_openai)
        usage = response.usage
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        print(f"  step {row['agent_step']}: {time.time() - started:.1f} s wall; input {usage.input_tokens}, "
              f"cached {cached}, output {usage.output_tokens}")

    print("\n/api/chat, same system, tools, prompt and frame (Ollama's own timings):")
    native_tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
                    for t in backend._tools_openai]
    for row in (first, second, second):
        started = time.time()
        reply = _post(f"{OLLAMA}/api/chat", {
            "model": args.model, "stream": False, "keep_alive": -1, "tools": native_tools,
            "options": {"num_predict": backend.max_output_tokens or 256, "temperature": 0.7, "top_p": 0.8},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": row["prompt"]},
                         {"role": "user", "content": "", "images": [frame]}],
        })
        ns = 1e9
        print(f"  step {row['agent_step']}: {time.time() - started:.1f} s wall; total {reply.get('total_duration', 0) / ns:.1f} s; "
              f"prompt evaluated {reply.get('prompt_eval_count')} tokens in {reply.get('prompt_eval_duration', 0) / ns:.1f} s; "
              f"output {reply.get('eval_count')} tokens in {reply.get('eval_duration', 0) / ns:.1f} s; "
              f"load {reply.get('load_duration', 0) / ns:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
