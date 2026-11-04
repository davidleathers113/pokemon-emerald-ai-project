"""BACKEND=gemini+ollama: Gemini's free tier first, the local model when it can't answer.

David (2026-11-04): play on Gemini's free tier and fall back to the local Qwen when the
daily quota is used up (FallbackBackend, classify_gemini_error). Both backends are fakes
here: errors are real google.api_core exception objects carrying the status details
Gemini sends (QuotaFailure quotaId, RetryInfo retryDelay, ErrorInfo reason).

  docker run --rm --entrypoint /opt/venv/bin/python -v ~/continual-harness:/app -w /app \
      emerald-harness:local local_tests/test_gemini_fallback.py
"""
import asyncio
import os
import sys
import tempfile
import warnings
from datetime import datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

warnings.filterwarnings("ignore")  # google.generativeai's end-of-support notice
from google.api_core import exceptions as gx
from PIL import Image

from utils.agent_infrastructure import vlm_backends as vb

PT = ZoneInfo("America/Los_Angeles")
DAILY_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE_ID = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def _quota(quota_id, retry="34s", limit=20):
    """A 429 as Gemini's REST API sends it: QuotaFailure (with the limit reached, seen live
    as quota_value 20 on gemini-3.6-flash) + RetryInfo details."""
    return gx.ResourceExhausted("You exceeded your current quota", details=[
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": quota_id, "quotaValue": str(limit)}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
    ])


def _bad_key():
    return gx.InvalidArgument("API key not valid. Please pass a valid API key.", details=[
        {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"},
    ])


# -- reading the error ------------------------------------------------------------


def test_the_limit_reached_is_read_from_the_error():
    """Google's docs no longer list free-tier numbers; the 429 carries them (seen live:
    quota_value 20 on gemini-3.6-flash), over REST as a string, over gRPC as a number."""
    assert vb.gemini_quota_value(_quota(DAILY_ID, limit=20)) == 20
    proto = gx.ResourceExhausted("quota", details=[
        SimpleNamespace(violations=[SimpleNamespace(quota_id=DAILY_ID, quota_value=250)]),
    ])
    assert vb.gemini_quota_value(proto) == 250
    assert vb.gemini_quota_value(gx.ResourceExhausted("429")) is None


def test_errors_are_classified_by_their_own_fields():
    assert vb.classify_gemini_error(_quota(DAILY_ID)) == (vb.GEMINI_DAILY, 34.0)
    assert vb.classify_gemini_error(_quota(MINUTE_ID, "7.5s")) == (vb.GEMINI_MINUTE, 7.5)
    # gRPC: protobuf messages, read by attribute (quota_id, retry_delay as a Duration).
    proto = gx.ResourceExhausted("quota", details=[
        SimpleNamespace(violations=[SimpleNamespace(quota_id=DAILY_ID)]),
        SimpleNamespace(retry_delay=SimpleNamespace(seconds=12, nanos=500_000_000)),
    ])
    assert vb.classify_gemini_error(proto) == (vb.GEMINI_DAILY, 12.5)
    # A 429 that doesn't name its quota is taken as per-minute (FallbackBackend escalates).
    assert vb.classify_gemini_error(gx.ResourceExhausted("429")) == (vb.GEMINI_MINUTE, None)
    assert vb.classify_gemini_error(_bad_key())[0] == vb.GEMINI_AUTH
    assert vb.classify_gemini_error(gx.PermissionDenied("denied"))[0] == vb.GEMINI_AUTH
    assert vb.classify_gemini_error(gx.ServiceUnavailable("overloaded"))[0] == vb.GEMINI_TRANSIENT
    assert vb.classify_gemini_error(gx.DeadlineExceeded("slow"))[0] == vb.GEMINI_TRANSIENT
    assert vb.classify_gemini_error(gx.InvalidArgument("bad schema")) == (None, None)
    assert vb.classify_gemini_error(ValueError("a bug")) == (None, None)


def test_the_reset_is_the_next_midnight_pacific():
    now = datetime(2026, 11, 4, 11, 46, tzinfo=PT).timestamp()
    assert datetime.fromtimestamp(vb.next_gemini_reset(now), PT) == datetime(2026, 11, 5, 0, 0, tzinfo=PT)
    # Across the spring-forward night (2026-03-08) it is still midnight local time.
    now = datetime(2026, 3, 7, 23, 30, tzinfo=PT).timestamp()
    assert datetime.fromtimestamp(vb.next_gemini_reset(now), PT) == datetime(2026, 3, 8, 0, 0, tzinfo=PT)


# -- GeminiBackend's own retries ------------------------------------------------------


def _gemini(errors, cap=None, raise_errors=False):
    """A GeminiBackend without the network: its model raises `errors` in turn, then answers."""
    backend = object.__new__(vb.GeminiBackend)
    backend.model_name, backend.tools, backend.max_retry_wait_s = "gemini-2.5-flash", [], cap
    backend.raise_errors = raise_errors
    queue = list(errors)
    calls = []

    def generate_content(parts, **kwargs):
        calls.append(parts)
        if queue:
            raise queue.pop(0)
        return "reply"

    backend.model = SimpleNamespace(generate_content=generate_content)
    return backend, calls


def test_gemini_raises_a_used_up_day_or_a_bad_key_at_once():
    for error in (_quota(DAILY_ID), _bad_key()):
        backend, calls = _gemini([error])
        with mock.patch.object(vb.time, "sleep") as slept:
            try:
                backend._call_generate_content(["x"])
                raise AssertionError("expected the error")
            except type(error):
                pass
        assert len(calls) == 1 and not slept.called, (calls, slept.call_args_list)


def test_gemini_waits_out_a_per_minute_limit_only_within_its_cap():
    backend, calls = _gemini([_quota(MINUTE_ID, "5s")], cap=30)
    with mock.patch.object(vb.time, "sleep") as slept:
        assert backend._call_generate_content(["x"]) == "reply"
    assert [c.args[0] for c in slept.call_args_list] == [5.0] and len(calls) == 2
    backend, calls = _gemini([_quota(MINUTE_ID, "40s")], cap=30)  # the hint alone is over the cap
    with mock.patch.object(vb.time, "sleep") as slept:
        try:
            backend._call_generate_content(["x"])
            raise AssertionError("expected the error")
        except gx.ResourceExhausted:
            pass
    assert not slept.called and len(calls) == 1


def test_the_cap_covers_gemini_s_own_time_not_only_the_waits():
    """Seen in the live check (2026-11-04): gemini-3.8-flash under "high demand" timed out
    at its 60 s request timeout three times; only the sleeps were counted against the
    30 s cap, so the step took ~4 minutes before the local model could play it. Now each
    request's timeout is what is left of the cap, and the clock includes the requests."""
    backend, _ = _gemini([], cap=30)
    timeouts = []
    clock = [1000.0]

    def slow(parts, **kwargs):
        timeouts.append(kwargs["request_options"]["timeout"])
        clock[0] += kwargs["request_options"]["timeout"]  # the request runs out its timeout
        raise gx.DeadlineExceeded("Deadline expired before operation could complete.")

    backend.model = SimpleNamespace(generate_content=slow)
    with mock.patch.object(vb.time, "monotonic", side_effect=lambda: clock[0]), \
            mock.patch.object(vb.time, "sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)):
        try:
            backend._call_generate_content(["x"])
            raise AssertionError("expected DeadlineExceeded")
        except gx.DeadlineExceeded:
            pass
    assert timeouts == [30], timeouts  # no time left for a retry after a 30 s request
    assert clock[0] - 1000.0 <= 30, clock


def test_a_quota_error_is_not_hidden_behind_a_canned_reply():
    """GeminiBackend.get_query retried text-only on any error, and get_text_query turned
    errors into "I encountered an error… press 'A'": a used-up quota never reached the
    fallback. The frame still goes in after the text ([text, image])."""
    backend, calls = _gemini([_quota(DAILY_ID)])
    try:
        backend.get_query(Image.new("RGB", (240, 160)), "the prompt", "test")
        raise AssertionError("expected the quota error")
    except gx.ResourceExhausted:
        pass
    assert len(calls) == 1 and calls[0][0] == "the prompt" and isinstance(calls[0][1], Image.Image)


# -- FallbackBackend ---------------------------------------------------------------


class _Fake:
    """Stands in for GeminiBackend / OpenAIBackend: raises queued errors, else answers."""

    def __init__(self, model_name, tools=None, system_instruction=None, **kwargs):
        self.model_name, self.kwargs, self.errors, self.calls = model_name, kwargs, [], 0

    def get_query(self, img, text, module_name="Unknown"):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return f"{self.model_name}: {text}"

    get_text_query = None


def _fallback(env=None):
    env = {"GEMINI_API_KEY": "test-key", **(env or {})}
    with mock.patch.dict(os.environ, env, clear=False), \
            mock.patch.object(vb, "GeminiBackend", _Fake), mock.patch.object(vb, "OpenAIBackend", _Fake):
        backend = vb.FallbackBackend("qwen3-vl:8b-instruct", tools=[], system_instruction="sys")
    said = []
    backend.on_switch.append(said.append)
    return backend, said


LITE = vb.DEFAULT_GEMINI_FALLBACK_MODEL


def _hop(backend, model):
    return next(h for h in backend.hops if h["model"] == model)


def test_the_chain_is_flash_then_flash_lite_then_local():
    """David (2026-11-04): "Have it fall back to flash-lite and then to qwen"."""
    backend, _ = _fallback()
    assert [h["model"] for h in backend.hops] == [vb.DEFAULT_GEMINI_MODEL, LITE]
    assert all(h["backend"].kwargs["raise_errors"] is True for h in backend.hops)
    backend, _ = _fallback({"GEMINI_FALLBACK_MODEL": "gemini-3.5-flash-lite", "GEMINI_MODEL": "gemini-3.6-flash"})
    assert [h["model"] for h in backend.hops] == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    backend, _ = _fallback({"GEMINI_FALLBACK_MODEL": "none"})
    assert [h["model"] for h in backend.hops] == [vb.DEFAULT_GEMINI_MODEL]


def test_a_per_minute_limit_or_outage_hands_that_one_step_to_the_next_hop():
    backend, said = _fallback()
    backend.primary.errors.append(_quota(MINUTE_ID))
    assert backend.get_query(None, "step 1") == f"{LITE}: step 1"  # this step only
    assert backend.current_model == LITE
    assert backend.get_query(None, "step 2") == f"{vb.DEFAULT_GEMINI_MODEL}: step 2"
    assert backend.current_model == vb.DEFAULT_GEMINI_MODEL and said == []
    # Both busy: Qwen is the floor (David: Qwen for that step).
    backend.primary.errors.append(gx.ServiceUnavailable("overloaded"))
    _hop(backend, LITE)["backend"].errors.append(gx.DeadlineExceeded("slow"))
    assert backend.get_query(None, "step 3").startswith("qwen3-vl") and said == []
    assert backend.current_model == "qwen3-vl:8b-instruct"


def test_the_step_s_budget_is_shared_across_hops():
    """One 30 s budget for the whole step: a hop that used 25 s leaves the next 5."""
    backend, _ = _fallback()
    clock = [100.0]

    def slow_primary(img, text, module_name="Unknown"):
        clock[0] += 25
        raise gx.DeadlineExceeded("slow")

    backend.primary.get_query = slow_primary
    with mock.patch.object(vb.time, "monotonic", side_effect=lambda: clock[0]):
        assert backend.get_query(None, "x") == f"{LITE}: x"
    assert 4.0 < _hop(backend, LITE)["backend"].max_retry_wait_s <= 5.0

    def slower_primary(img, text, module_name="Unknown"):
        clock[0] += 29.5
        raise gx.DeadlineExceeded("slow")

    backend.primary.get_query = slower_primary
    lite_calls = _hop(backend, LITE)["backend"].calls
    with mock.patch.object(vb.time, "monotonic", side_effect=lambda: clock[0]):
        assert backend.get_query(None, "y").startswith("qwen3-vl")  # no time left for the lite hop
    assert _hop(backend, LITE)["backend"].calls == lite_calls


def test_a_used_up_day_moves_down_the_chain_until_the_reset():
    backend, said = _fallback()
    now = datetime(2026, 11, 4, 15, 0, tzinfo=PT).timestamp()
    backend.primary.errors.append(_quota(DAILY_ID, limit=20))
    with mock.patch.object(vb.time, "time", return_value=now):
        # The step that hit the quota is answered by the next hop in the same call.
        assert backend.get_query(None, "step 1") == f"{LITE}: step 1"
        assert backend.get_query(None, "step 2") == f"{LITE}: step 2"
    assert backend.primary.calls == 1, "a used-up model must not be asked again before the reset"
    assert said == [
        f"Switched to {LITE}: the free daily quota for {vb.DEFAULT_GEMINI_MODEL} is used up (20 requests/day); "
        "back after 00:00 PT"
    ], said
    # The lite hop's day runs out too: Qwen until the reset, and the row says so.
    _hop(backend, LITE)["backend"].errors.append(_quota(DAILY_ID, limit=500))
    with mock.patch.object(vb.time, "time", return_value=now + 600):
        assert backend.get_query(None, "step 3").startswith("qwen3-vl")
    assert said[-1].startswith(f"Switched to qwen3-vl:8b-instruct: the free daily quota for {LITE} is used up (500"), said
    after = datetime(2026, 11, 5, 0, 1, tzinfo=PT).timestamp()
    with mock.patch.object(vb.time, "time", return_value=after):
        assert backend.get_query(None, "step 4") == f"{vb.DEFAULT_GEMINI_MODEL}: step 4"
    assert said[-1] == f"Switched back to {vb.DEFAULT_GEMINI_MODEL}: it answers again after the reset"
    assert _hop(backend, vb.DEFAULT_GEMINI_MODEL)["out_until"] is None


def test_still_used_up_after_the_reset_waits_another_day():
    backend, said = _fallback()
    backend.primary.errors += [_quota(DAILY_ID), _quota(DAILY_ID)]
    with mock.patch.object(vb.time, "time", return_value=datetime(2026, 11, 4, 15, 0, tzinfo=PT).timestamp()):
        backend.get_query(None, "a")
    with mock.patch.object(vb.time, "time", return_value=datetime(2026, 11, 5, 0, 1, tzinfo=PT).timestamp()):
        assert backend.get_query(None, "b") == f"{LITE}: b"
    out = _hop(backend, vb.DEFAULT_GEMINI_MODEL)["out_until"]
    assert datetime.fromtimestamp(out, PT) == datetime(2026, 11, 6, 0, 0, tzinfo=PT)


def test_per_minute_limits_for_ten_minutes_count_as_the_day():
    backend, said = _fallback()
    start = datetime(2026, 11, 4, 15, 0, tzinfo=PT).timestamp()
    for minute in range(0, 11):
        backend.primary.errors.append(gx.ResourceExhausted("429 with no quota named"))
        with mock.patch.object(vb.time, "time", return_value=start + 60 * minute):
            backend.get_query(None, f"step {minute}")
    assert len(said) == 1 and "daily quota" in said[0], said
    assert _hop(backend, vb.DEFAULT_GEMINI_MODEL)["out_until"] is not None


def test_a_rejected_key_takes_every_gemini_hop_out_for_the_run():
    backend, said = _fallback()
    backend.primary.errors.append(_bad_key())
    assert backend.get_query(None, "x").startswith("qwen3-vl")
    assert _hop(backend, LITE)["backend"].calls == 0, "one key: the lite hop would be refused too"
    assert said and "Gemini rejected the API key (InvalidArgument: API key not valid)" in said[0], said
    assert "test-key" not in said[0], said
    with mock.patch.object(vb.time, "time", return_value=4_000_000_000):  # long after any reset
        backend.get_query(None, "y")
    assert backend.primary.calls == 1


def test_a_model_this_key_cannot_use_is_skipped_for_the_run():
    """Seen in the live check (2026-11-04) with David's new key: "404 This model
    models/gemini-2.5-flash is no longer available to new users", though the model list
    named it. The old path turned it into the canned "press 'A'" reply; under the
    fallback Gemini raises every error (raise_errors) and a 404 skips that model."""
    error = gx.NotFound("This model models/gemini-2.5-flash is no longer available to new users. Please update")
    assert vb.classify_gemini_error(error)[0] == vb.GEMINI_GONE
    backend, said = _fallback()
    backend.primary.errors.append(error)
    assert backend.get_query(None, "x") == f"{LITE}: x"
    assert said and f"Switched to {LITE} for this run: {vb.DEFAULT_GEMINI_MODEL} is not served to this key" in said[0]
    assert "no longer available to new users" in said[0], said
    assert _hop(backend, vb.DEFAULT_GEMINI_MODEL)["out_until"] == float("inf")
    # The real GeminiBackend with raise_errors: any error comes out, never the canned reply.
    gemini, calls = _gemini([gx.InvalidArgument("a schema problem")], raise_errors=True)
    try:
        gemini.get_query(Image.new("RGB", (240, 160)), "p", "test")
        raise AssertionError("expected InvalidArgument")
    except gx.InvalidArgument:
        pass
    assert len(calls) == 1
    assert backend.primary.calls == 1


def test_another_error_is_raised_not_hidden():
    backend, _ = _fallback()
    backend.primary.errors.append(gx.InvalidArgument("bad tool schema"))
    try:
        backend.get_query(None, "x")
        raise AssertionError("expected InvalidArgument")
    except gx.InvalidArgument:
        pass
    assert backend.secondary.calls == 0


def test_without_a_key_the_local_model_plays_alone():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        with mock.patch.object(vb, "GeminiBackend", _Fake), mock.patch.object(vb, "OpenAIBackend", _Fake):
            backend = vb.FallbackBackend("qwen3-vl:8b-instruct", tools=[], system_instruction="sys")
    assert backend.primary is None and backend.get_query(None, "x").startswith("qwen3-vl")
    assert backend.current_model == "qwen3-vl:8b-instruct"
    assert vb.VLM.BACKENDS["gemini+ollama"] is vb.FallbackBackend


def test_a_switch_is_a_feed_row():
    """The agent posts each switch to /feed_event; /play shows it as a 🔁 row, and each
    step row names the model that chose it (trajectory "model")."""
    os.chdir(tempfile.mkdtemp())  # importing the server creates cache dirs relative to cwd
    from server import app as srv

    text = "Switched to qwen3-vl:8b-instruct: Gemini's free daily quota for gemini-2.5-flash is used up"
    assert asyncio.run(srv.add_feed_event({"kind": "backend", "text": text})) == {"ok": True}
    assert srv._pause_events[-1]["kind"] == "backend" and srv._pause_events[-1]["text"] == text
    refused = asyncio.run(srv.add_feed_event({"kind": "pause", "text": "x"}))
    assert refused.status_code == 400  # the agent can't fake a pause row


def _run_all():
    failures = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report every failure, not just the first
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
