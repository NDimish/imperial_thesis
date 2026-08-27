"""Cross-process-safe Gemini call throttle.

stage2_inference_judge.py and Modules/AI_checker.py each had their own MIN_INTERVAL_SECONDS
throttle built on a plain in-memory `_last_call_at` global. That only
protects calls within a single process -- a second process (a restart after
a crash mid-run, or a second script sharing the same GEMINI_API_KEY quota
running at the same time) starts its own fresh `_last_call_at = 0.0` and has
no idea the shared quota was just used a moment ago, so two processes can
both decide it's safe to call at once.

This replaces that with a real file lock (portalocker, already a project
dependency -- see requirements.txt) around a tiny JSON state file recording
the last call timestamp. The lock is held for the full wait, not just the
read/write, so two processes can't both observe the same stale timestamp and
both proceed -- the second one blocks until the first has both waited its
turn AND recorded its own call time.
"""
import json
import os
import time

import portalocker

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gemini_rate_limit_state.json")

# Same limit/margin stage2_inference_judge.py and Modules/AI_checker.py already used
# (gemini-3.6-flash's 5 RPM cap, with a 90s window instead of 60s for
# headroom) -- centralized here so there's one number to tune, not two
# copies that can drift apart.
RPM = 5
MIN_INTERVAL_SECONDS = 90 / RPM

# How long to wait to ACQUIRE the lock itself before giving up -- not how
# long a call can be throttled for. Generous on purpose: if several
# processes are queued up sharing this quota, each one only holds the lock
# for its own throttle wait (at most MIN_INTERVAL_SECONDS) before releasing
# it, so this would only be hit by an unrealistic pile-up of callers.
LOCK_ACQUIRE_TIMEOUT_SECONDS = 300


def throttle(min_interval_seconds=None):
    """Blocks until it's safe to make another Gemini call against the shared
    quota, then atomically records this call's timestamp -- safe to call
    from multiple processes/scripts at once, and safe across a crash/restart
    since the state lives on disk, not in memory."""
    interval = MIN_INTERVAL_SECONDS if min_interval_seconds is None else min_interval_seconds

    with portalocker.Lock(STATE_PATH, mode="a+", timeout=LOCK_ACQUIRE_TIMEOUT_SECONDS) as f:
        f.seek(0)
        raw = f.read().strip()
        try:
            state = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            state = {}
        last_call_at = state.get("last_call_at", 0.0)

        elapsed = time.time() - last_call_at
        if elapsed < interval:
            time.sleep(interval - elapsed)

        state["last_call_at"] = time.time()
        f.seek(0)
        f.truncate()
        f.write(json.dumps(state))
        f.flush()
