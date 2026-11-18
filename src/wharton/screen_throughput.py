"""Measured daily screening throughput: learn cache hit rate and cost per name, size the batch."""

import json
import math
import os
import tempfile
from pathlib import Path

from wharton.screen_budget import WORKING_CREDITS, quarantine_reserve

ALPHA, KEEP, SAFETY = 0.3, 14, 0.9
BOOTSTRAP_BATCH = 3000
GROWTH, MISS_FLOOR = 2, 0.25  # max batch growth per run; floor on expected miss rate
MAX_QUEUE = 100000
METRICS = ("hit_rate", "credits_per_refreshed", "seconds_per_name")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _observations(root, day):
    data = _read_json(Path(root) / "capacity-v2.json")
    rows = data.get("observations") if isinstance(data, dict) else None
    return [o for o in rows if isinstance(o, dict) and o.get("day") == day] if rows else []


def day_spend(root, day):
    """Observed non-news credits this UTC day (news is metered by its own runner)."""
    return sum(
        o.get("account_delta", 0) for o in _observations(root, day) if o.get("endpoint") != "news"
    )


def credits_remaining(acquisition, root, day):
    """Working cap minus all observed account usage today; no network."""
    working = acquisition.get(
        "daily_working_credit_limit",
        min(acquisition.get("daily_credit_limit", WORKING_CREDITS), WORKING_CREDITS),
    )
    seen = max((o.get("account_credits", 0) for o in _observations(root, day)), default=0)
    # a quarantined denied attempt may still be charged: its full possible cost stays reserved
    return max(0, working - seen - quarantine_reserve(root, day))


def load(root):
    """Valid history or None; a corrupt file never blocks a run."""
    data = _read_json(Path(root) / "throughput.json")
    if (
        isinstance(data, dict)
        and data.get("version") == 1
        and isinstance(data.get("ewma"), dict)
        and isinstance(data.get("runs"), list)
    ):
        return data
    return None


def _ewma(values):
    out = None
    for v in values:
        out = v if out is None else ALPHA * v + (1 - ALPHA) * out
    return out


def record(state_root, run_stats):
    """Append (or replace the same day) one completed-run observation and refresh the EWMA."""
    root = Path(state_root)
    history = load(root) or {"version": 1, "runs": [], "ewma": {}}
    s = run_stats
    done = s["cache_current"] + s["refreshed"]
    row = dict(s)
    row["hit_rate"] = s["cache_current"] / done if done else None
    row["credits_per_refreshed"] = s["credits_spent"] / s["refreshed"] if s["refreshed"] else None
    row["seconds_per_name"] = s["acquisition_seconds"] / done if done else None
    runs = [r for r in history["runs"] if r.get("day") != s["day"]] + [row]
    runs = runs[-KEEP:]
    ewma = {}
    for key in METRICS:
        value = _ewma([r[key] for r in runs if r.get(key) is not None])
        if value is not None:
            ewma[key] = value
    history = {"version": 1, "runs": runs, "ewma": ewma}
    with tempfile.NamedTemporaryFile("w", dir=root, delete=False) as out:
        json.dump(history, out, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(out.name, root / "throughput.json")
    return history


def estimate(config, history, *, universe_size, credits_remaining, seconds_available, protected=0):
    """Batch size bounded by universe, queue cap, credits and time; holdings/instruments floor."""
    ewma = (history or {}).get("ewma") or {}
    try:
        hit, cost, secs = (float(ewma[k]) for k in METRICS)
        valid = (
            all(map(math.isfinite, (hit, cost, secs))) and 0 <= hit <= 1 and cost > 0 and secs >= 0
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        batch, bound, basis = (
            BOOTSTRAP_BATCH,
            "bootstrap",
            "bootstrap: no usable throughput history",
        )
        measured = {}
    else:
        # Floor the miss rate: a cache-expiry shock must not let one run spend the whole cap.
        per_name = max(1 - hit, MISS_FLOOR) * cost
        limits = {
            "credits": int(SAFETY * credits_remaining / per_name) if per_name > 0 else MAX_QUEUE,
            "time": int(SAFETY * seconds_available / secs) if secs > 0 else MAX_QUEUE,
        }
        runs = (history or {}).get("runs") or []
        prior = runs[-1].get("queued") if runs and isinstance(runs[-1], dict) else None
        if type(prior) is int and prior > 0:
            limits["growth"] = max(GROWTH * prior, 1)
        bound = min(limits, key=limits.get)
        batch = limits[bound]
        measured = dict(hit_rate=hit, credits_per_refreshed=cost, seconds_per_name=secs)
        measured["expected_cost_per_name"] = per_name
        basis = f"measured ({len((history or {}).get('runs', []))} runs), {bound}-bound"
    batch = max(protected, 1, min(batch, universe_size, MAX_QUEUE))
    return dict(
        batch=batch,
        basis=basis,
        bound=bound,
        credits_remaining=credits_remaining,
        seconds_available=seconds_available,
        protected=protected,
        **measured,
    )
