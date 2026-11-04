"""How many prompt tokens the screenshot costs: the same prompt to Ollama with and without
the frame, prompt_eval_count of each.

Speed plan, Phase 4 (David, 2026-11-04): Ollama pads qwen3-vl's images to a 1,024-token
floor, so the 240x160 frame (~40-70 tokens natively) would cost ~1K tokens every step.
This confirms the number before SCREENSHOT_POLICY=needed is tried live.

It uses Ollama's one model slot: run it while the agent is paused or stopped.

  python3 local_tools/image_token_probe.py [--model qwen3-vl:8b-instruct] [--frame frame.png]

Without --frame it takes the current game frame from the server (GET /api/frame).
"""
import argparse
import base64
import json
import sys
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
SERVER = "http://127.0.0.1:8000"
PROMPT = "Describe the game screen in one short sentence."


def _post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _frame(path):
    if path:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    with urllib.request.urlopen(f"{SERVER}/api/frame", timeout=10) as r:
        return json.loads(r.read().decode())["frame"]


def measure(model, frame_b64):
    """{"text": prompt tokens without the image, "image": with it, "image_tokens": the
    difference, and the prompt-eval seconds of each}."""
    out = {}
    for name, images in (("text", None), ("image", [frame_b64])):
        message = {"role": "user", "content": PROMPT}
        if images:
            message["images"] = images
        # num_predict 1: only the prompt's cost is measured. keep_alive -1 leaves the
        # model loaded for the agent.
        reply = _post(f"{OLLAMA}/api/chat", {
            "model": model, "messages": [message], "stream": False,
            "options": {"num_predict": 1}, "keep_alive": -1,
        })
        out[name] = reply.get("prompt_eval_count")
        out[f"{name}_eval_s"] = round((reply.get("prompt_eval_duration") or 0) / 1e9, 2)
    out["image_tokens"] = (out["image"] or 0) - (out["text"] or 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--frame", help="a PNG to send instead of the live game frame")
    args = parser.parse_args()
    result = measure(args.model, _frame(args.frame))
    print(json.dumps(result, indent=2))
    print(f"The frame costs {result['image_tokens']} prompt tokens ({result['image_eval_s']} s vs {result['text_eval_s']} s to read).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
