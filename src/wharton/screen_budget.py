"""EODHD transport and conservative acquisition budget policy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

from wharton import coverage_contracts, eodhd, screening

WORKING_CREDITS = 75000  # project working cap per UTC day (user-approved 2026-11-04)
# Minimum gap between provider requests: <=800 req/min, under the EODHD limit of 1,000/min.
MIN_REQUEST_GAP = 0.075

COST_FLOORS = {
    "fundamentals": 10,
    "sec_annual": 10,
    "sec_quarterly": 10,
    "sec_financials": 10,
    "prices": 1,
    "fx": 1,
    "splits": 1,
    "earnings": 1,
    "news": 5,
}


_COVERAGE_LANE = {
    "bulk-fundamentals": "bulk_fundamentals",
    "eod-bulk-last-day": "bulk_market",
    "calendar/earnings": "calendar",
    "calendar/splits": "calendar",
    "calendar/dividends": "calendar",
}


def coverage_lane(endpoint, params=None):
    """Lane that gates a coverage route, or None. Single-symbol calendar/earnings stays legacy."""
    if endpoint == "calendar/earnings" and (params or {}).get("symbols"):
        return None
    route = endpoint if endpoint.startswith("calendar/") else endpoint.partition("/")[0]
    return _COVERAGE_LANE.get(route)


def _final(params):
    # the token is never bound; fmt=json is appended by eodhd.request AFTER reserve, so both
    # sides drop it (any other fmt stays bound)
    return {
        k: v
        for k, v in (params or {}).items()
        if k != "api_token" and not (k == "fmt" and v == "json")
    }


def endpoint_cost(endpoint, params=None, lanes=frozenset()):
    """Read-only data routes only; bulk and symbol-less calendar need an explicitly enabled lane."""
    if endpoint == "user":
        return 0
    lane = coverage_lane(endpoint, params)
    if lane is not None and lane in lanes:
        return coverage_contracts.route_cost(endpoint, params)  # the single cost authority
    if re.fullmatch(r"fundamentals/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", endpoint):
        return 10
    if re.fullmatch(r"sec-companies/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+/financials", endpoint):
        return 10
    if re.fullmatch(r"sec-filings/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+/10[kq]", endpoint):
        return 10
    if (
        re.fullmatch(r"(?:eod|splits)/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", endpoint)
        or endpoint == "calendar/earnings"
    ):
        return 1
    # Exchange inventories cost one call: https://eodhd.com/financial-apis/api-limits
    if re.fullmatch(r"exchange-symbol-list/[A-Za-z0-9_-]+", endpoint):
        return 1
    if endpoint == "news":
        return 5  # documented single-ticker floor; news_fetch tracks observed deltas
    raise ValueError("provider route is outside the read-only data allowlist")


class BudgetHold(ValueError):
    """A safe local stop; no provider purchase or quota top-up is attempted."""


def account_usage(account, job_date, headroom, *, server_day=None):
    """Only base quota counts; yesterday's lazy-reset counter can be effective zero."""
    safe = {k: account.get(k) for k in ("apiRequests", "dailyRateLimit", "apiRequestsDate")}

    def counter(value):
        if type(value) is int:
            return value
        return int(value) if isinstance(value, str) and re.fullmatch(r"[0-9]+", value) else None

    count, quota = (counter(safe[k]) for k in ("apiRequests", "dailyRateLimit"))
    usage_date = screening.day(safe["apiRequestsDate"])
    if (
        count is None
        or quota is None
        or count < 0
        or quota <= 0
        or usage_date is None
        or str(usage_date) != safe["apiRequestsDate"]
        or usage_date > job_date
    ):
        raise ValueError("account quota/date could not be verified safely")
    if server_day is not None and server_day != job_date:
        raise BudgetHold("provider UTC day differs from job day; acquisition stopped")
    if usage_date < job_date and server_day is None:
        now = datetime.now(UTC)
        if now.hour == 0 and now.minute == 0:
            raise BudgetHold("lazy-reset counter awaits the first UTC minute without a server date")
    effective = count if usage_date == job_date else 0
    return safe, effective, quota - effective - headroom


def validate_budget(config):
    if config.get("accounting_mode", "legacy") not in {"legacy", "account"}:
        raise ValueError("unknown accounting_mode")
    for key in ("initial_credit_limit", "daily_credit_limit"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"acquisition requires positive explicit {key}")
    costs = config.get("request_credit_cost")
    if isinstance(costs, dict):
        costs = [costs.get(key) for key in ("fundamentals", "prices")]
    else:
        costs = [costs]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in costs):
        raise ValueError("request_credit_cost must explicitly price fundamentals and prices")
    if config["initial_credit_limit"] > 40000:
        raise ValueError("initial_credit_limit cannot exceed 40000")
    # User-authorized exception (extra headroom, 23:25 UTC 2026-11-04): working/hard 99000 with a
    # 1000 BASE reserve on that UTC day ONLY; it expires at UTC midnight and defaults never change
    # (every other day stays 75000 / 25000).
    temporary = config.get("temporary_limit_date") == "2026-11-04" and (
        datetime.now(UTC).date().isoformat() == "2026-11-04"
    )
    cap, minimum_headroom = (99000, 1000) if temporary else (75000, 25000)
    if config["daily_credit_limit"] > cap:
        raise ValueError(f"daily_credit_limit cannot exceed {cap}")
    headroom = config.get("account_headroom", 25000)
    if type(headroom) is not int or headroom < 0:
        raise ValueError("account_headroom must be a nonnegative integer")
    if headroom < minimum_headroom:
        raise ValueError(f"acquisition requires at least {minimum_headroom} base-quota headroom")
    working = config.get(
        "daily_working_credit_limit", min(config["daily_credit_limit"], WORKING_CREDITS)
    )
    if type(working) is not int or not 1 <= working <= min(config["daily_credit_limit"], cap):
        raise ValueError(
            "daily_working_credit_limit must be positive and at most 75000 and the daily limit"
        )
    configured_costs = config["request_credit_cost"]
    if isinstance(configured_costs, dict):
        if set(configured_costs) - set(COST_FLOORS):
            raise ValueError("unknown provider endpoint cost key")
        if any(
            type(configured_costs[k]) is not int or configured_costs[k] < COST_FLOORS[k]
            for k in configured_costs
        ):
            raise ValueError("request cost is below the documented endpoint floor")
    elif configured_costs < 10:
        raise ValueError("scalar request cost must cover the 10-credit fundamental floor")
    if config.get("fresh_sources"):
        required = {"sec_annual", "sec_quarterly", "splits", "earnings", "fx"}
        if not isinstance(config.get("request_credit_cost"), dict) or not required <= set(
            config["request_credit_cost"]
        ):
            raise ValueError("fresh_sources requires explicit SEC, splits, earnings and FX costs")
        for key in required:
            if (
                type(config["request_credit_cost"][key]) is not int
                or config["request_credit_cost"][key] <= 0
            ):
                raise ValueError("fresh source request costs must be positive integers")
        if any(
            config["request_credit_cost"][key] < 10
            for key in ("fundamentals", "sec_annual", "sec_quarterly")
        ):
            raise ValueError("fundamentals and SEC requests require at least 10 credits")
    if config.get("mode", "initial") not in {"initial", "daily"}:
        raise ValueError("acquisition mode must be initial or daily")
    public = config.get("sec_public", {})
    if public.get("enabled"):
        if (
            type(public.get("max_requests_per_day")) is not int
            or not 1 <= public["max_requests_per_day"] <= 1500
        ):
            raise ValueError("SEC public max_requests_per_day must be explicit 1..1500 attempts")
        if not public.get("user_agent"):
            raise ValueError("SEC public requires a truthful explicit user_agent")


_DISCRETIONARY_PURPOSES = frozenset(
    {
        "missing_identity",
        "missing_accounting",
        "missing_price_history",
        "peer_cohort",
        "material_event",
        "held_instrument",
    }
)


def reservation_allowed(config, *, used_today, initial_total, cost, purpose="routine_refresh"):
    """Check a conservative reservation; the caller must atomically persist it."""
    validate_budget(config)
    if any(type(v) is not int or v < 0 for v in (used_today, initial_total)):
        return False
    if type(cost) is not int or cost <= 0:
        return False
    total = used_today + cost
    hard = config["daily_credit_limit"]
    working = config.get("daily_working_credit_limit", min(hard, WORKING_CREDITS))
    if total > hard or (total > working and purpose not in _DISCRETIONARY_PURPOSES):
        return False
    return (
        config.get("mode", "initial") != "initial"
        or initial_total + cost <= config["initial_credit_limit"]
    )


class CapacityLimit(BudgetHold):
    """An observed account limit prevents a paid attempt."""


class CoverageUnresolved(BudgetHold):
    """A coverage attempt charged less than reserved (or the counter regressed): fail closed.

    The capacity `pending` marker stays set and the response is not returned as data; the
    evidence record (coverage-attempts.jsonl) and `reason` survive. Only reconcile_full_cost can
    ever clear it, and only for a proven full charge."""

    def __init__(self, reason, attempt, http_status=None):
        super().__init__(f"accounting hold: coverage attempt unresolved ({reason})")
        self.reason, self.attempt, self.http_status = reason, attempt, http_status
        self.accounting_unknown = True


def _pending_hold(pending):
    """Hold for an unresolved pending: the usual message plus non-secret facts for run status."""
    hold = BudgetHold("accounting hold: unresolved paid attempt")
    hold.accounting_unknown = True
    pending = pending if isinstance(pending, dict) else {}
    hold.pending = {k: pending.get(k) for k in ("day", "endpoint", "possible_cost")}
    return hold


class SourceUnavailable(ValueError):
    """A documented missing-symbol response is coverage, not an authentication failure."""


class StopSession:
    """Existing redacted HTTP boundary, but authentication/rate errors never retry."""

    def __init__(self, session):
        self.session = session

    def checked(self, response):
        if response.status_code == 404:
            raise SourceUnavailable("EODHD HTTP 404: source coverage unavailable")
        if response.status_code == 402:
            raise BudgetHold("EODHD HTTP 402: acquisition stopped; no extra calls purchased")
        if response.status_code in {401, 403, 429}:
            raise ValueError(f"EODHD HTTP {response.status_code}: acquisition stopped")
        return response

    def get(self, *args, **kwargs):
        return self.checked(self.session.get(*args, **kwargs))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
        name = out.name
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


# ---- coverage evidence, lane holds and the one sanctioned reconciliation --------------------

ATTEMPTS_FILE = "coverage-attempts.jsonl"
LANES_FILE = "coverage-lanes.json"
_BODY_CAP = 4096


def _now():
    return datetime.now(UTC).isoformat()


def _read_jsonl(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except ValueError:
        raise BudgetHold("accounting hold: invalid coverage attempts log") from None


def _append_jsonl(path, record):
    """Append-only and fsynced: evidence is durable before any classification can fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as out:
        out.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


_DENIED = (401, 402, 403)


def quarantine_of(root, pending):
    """Facts of a DENIED coverage attempt whose `pending` is quarantined, else None.

    A coverage request answered 401/402/403 (entitlement/auth) is not retried: its lane is held
    with the exact reason and the original pending (possible_cost, before, fingerprint) is KEPT
    unchanged, so the legacy runtime stays blocked and the cost is never settled as zero. The new
    guard may continue the SUPPORTED lanes while the pending is quarantined, counting the full
    possible_cost as a conservative reserve in every cap/headroom check. Quarantine holds only when
    the retained attempt record (same endpoint, fingerprint, reserved cost, day and pre-counter),
    the held lane (same attempt seq) and the pending agree; anything else (5xx, lag, regression,
    an in-flight marker, missing evidence) stays a full accounting hold."""
    if not isinstance(pending, dict) or "inflight" in pending:
        return None
    try:
        keys = ("day", "endpoint", "possible_cost", "before", "fingerprint")
        day, endpoint, cost, before, fingerprint = (pending[k] for k in keys)
        if type(cost) is not int or cost <= 0 or type(before) is not int:
            return None
        attempt = next(
            (
                r
                for r in reversed(_read_jsonl(Path(root) / ATTEMPTS_FILE))
                if r.get("kind") == "attempt" and r.get("fingerprint") == fingerprint
            ),
            None,
        )
        if (
            attempt is None
            or attempt.get("endpoint") != endpoint
            or attempt.get("day") != day
            or attempt.get("reserved_cost") != cost
            or attempt.get("http_status") not in _DENIED
            or (attempt.get("pre_snapshot") or {}).get("count") != before
        ):
            return None
        seq, status = attempt.get("seq"), attempt["http_status"]
        reason = f"http_{status}_entitlement_or_auth"
        held = any(  # the EXACT denial reason: a regression/undercharge hold never qualifies
            isinstance(v, dict)
            and v.get("attempt_seq") == seq
            and v.get("status") == "held"
            and v.get("reason") == reason
            for v in _lanes(root).values()
        )
        after = next(
            (
                r
                for r in reversed(_read_jsonl(Path(root) / ATTEMPTS_FILE))
                if r.get("kind") == "snapshot" and r.get("attempt_seq") == seq
            ),
            None,
        )
        # a post-attempt snapshot must exist and the counter must NOT have regressed
        if (
            not held
            or after is None
            or type(after.get("count")) is not int
            or after["count"] < before
        ):
            return None
        return dict(
            reserve=cost,
            day=day,
            endpoint=endpoint,
            attempt_seq=attempt.get("seq"),
            http_status=attempt["http_status"],
            reason=f"http_{attempt['http_status']}_entitlement_or_auth",
        )
    except (KeyError, TypeError, ValueError, OSError, BudgetHold):
        return None


def blocking_pending(capacity, root):
    """The pending that blocks sessions: None when there is none or it is a quarantined denial."""
    pending = capacity.get("pending") if isinstance(capacity, dict) else None
    return None if pending is None or quarantine_of(root, pending) else pending


def quarantine_reserve(root, day):
    """Credits of a quarantined denied attempt that may still be charged (else 0). The reserve
    PERSISTS across UTC days until the pending is resolved: a late charge could land on any
    counter, so every later day's budget subtracts it too."""
    path = Path(root) / "capacity-v2.json"
    try:
        pending = json.loads(path.read_text()).get("pending")
    except (OSError, ValueError, AttributeError):
        return 0
    found = quarantine_of(root, pending)
    return found["reserve"] if found else 0


def _lanes(root):
    path = Path(root) / LANES_FILE
    if not path.exists():
        return {}
    try:
        lanes = json.loads(path.read_text())
    except ValueError:
        lanes = None
    if not isinstance(lanes, dict):
        raise BudgetHold("accounting hold: invalid coverage lanes state")
    return lanes


def hold_lane(root, lane, reason, http_status=None, attempt_seq=None):
    """Mark a coverage lane held (coverage-lanes.json, never capacity-v2). Re-enabling is a
    reviewed human edit of that file; nothing in code clears a hold."""
    lanes = _lanes(root)
    lanes[lane] = dict(
        status="held", reason=reason, http_status=http_status, attempt_seq=attempt_seq, at=_now()
    )
    _atomic_json(Path(root) / LANES_FILE, lanes)


def _body(response, token):
    """-> (sha256 of the full body, first 4 KiB as scrubbed text)."""
    raw = getattr(response, "content", None)
    if not isinstance(raw, bytes):
        text = getattr(response, "text", None)
        try:
            raw = text.encode() if isinstance(text, str) else json.dumps(response.json()).encode()
        except Exception:  # noqa: BLE001 - evidence capture must never mask the attempt
            raw = b""
    head = raw[:_BODY_CAP].decode("utf-8", errors="replace")
    if coverage_contracts.contains_credential(head) or (token and str(token) in head):
        head = "[redacted]"
    return hashlib.sha256(raw).hexdigest(), head


def _count(value):
    if type(value) is int:
        return value
    return int(value) if isinstance(value, str) and re.fullmatch(r"[0-9]+", value) else None


def reconcile_full_cost(run_lock, snapshots):
    """The ONLY code path that clears a coverage pending, and only for a proven FULL charge.

    Needs an owned run lock (duck-typed: assert_owned(), .path inside the state root), a coverage
    pending, its retained attempt record (endpoint + fingerprint + reserved cost + pre counter),
    and free `user` snapshots (dicts with apiRequests/apiRequestsDate[/at]) of the pending's day
    whose latest count - pending.before == possible_cost exactly (a lagged full charge) or more
    (resolved as charged, lane held, coverage_halt). Anything below the reserved cost (zero-charge
    403, partial charge, any number of stable snapshots) and any counter regression is REFUSED:
    no authoritative non-charge evidence exists in code. Counters are never edited; the full pending
    record and its resolution are kept in the append-only coverage-attempts.jsonl (kind=resolution)
    and an observation is appended. capacity-v2.json keeps the legacy key set
    {version, pending, observations}
    so a rollback to the prior immutable runtime still validates it."""

    def refuse(reason):
        return {"resolved": False, "reason": reason}

    try:
        run_lock.assert_owned()
    except Exception:  # noqa: BLE001 - duck-typed lock: any failure means not owned
        return refuse("run_lock_not_owned")
    root = Path(run_lock.path).parent
    path = root / "capacity-v2.json"
    try:
        capacity = json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        capacity = None
    pending = capacity.get("pending") if isinstance(capacity, dict) else None
    if not isinstance(pending, dict):
        return refuse("no_pending")
    endpoint, fingerprint = pending.get("endpoint"), pending.get("fingerprint")
    if not (
        isinstance(endpoint, str)
        and isinstance(fingerprint, str)
        and coverage_lane(endpoint) is not None
    ):
        return refuse("non_coverage_pending")
    day, before, cost = pending.get("day"), pending.get("before"), pending.get("possible_cost")
    if type(before) is not int or type(cost) is not int or not isinstance(day, str):
        return refuse("non_coverage_pending")
    if quarantine_of(root, pending) and any(  # others resumed: a later charge is not proof
        isinstance(o, dict)
        and o.get("endpoint") != "user"
        and type(o.get("possible_cost")) is int
        and o["possible_cost"] > 0
        and isinstance(o.get("day"), str)
        and (o["day"] > day or (o["day"] == day and o.get("account_credits", 0) > before))
        for o in capacity.get("observations", [])
    ):
        return refuse("quarantined_attempt_aggregate_inference_rejected")
    rows = _read_jsonl(root / ATTEMPTS_FILE)
    attempt = next(
        (
            r
            for r in reversed(rows)
            if r.get("kind") == "attempt"
            and r.get("endpoint") == endpoint
            and r.get("fingerprint") == fingerprint
            and r.get("day") == day
            and r.get("reserved_cost") == cost
            and r.get("pre_snapshot", {}).get("count") == before
        ),
        None,
    )
    if attempt is None:
        return refuse("no_matching_attempt_record")
    seen = []
    for item in snapshots if isinstance(snapshots, (list, tuple)) else ():
        count = _count(item.get("apiRequests")) if isinstance(item, dict) else None
        if count is None or not isinstance(item.get("apiRequestsDate"), str):
            return refuse("invalid_snapshot")
        if item["apiRequestsDate"] != day:
            return refuse("wrong_date")
        at = item.get("at")
        seen.append({"count": count, "date": day, "at": at if isinstance(at, str) else None})
    if not seen:
        return refuse("invalid_snapshot")
    series = [before]
    series += [
        r["count"]
        for r in rows
        if r.get("kind") == "snapshot" and r.get("attempt_seq") == attempt["seq"]
    ]
    series += [s["count"] for s in seen]
    if any(b < a for a, b in zip(series, series[1:])):
        return refuse("counter_regression")
    delta = series[-1] - before
    if delta < cost:  # ponytail: no authoritative non-charge evidence exists; a human decides
        return refuse("ambiguous_undercharge_unresolved")
    overspend, lane = delta > cost, coverage_lane(endpoint)
    resolution = dict(
        pending,
        attempt_seq=attempt["seq"],
        resolution="reconciled_full_cost",
        charged=delta,
        snapshots=seen,
        at=_now(),
    )
    _append_jsonl(root / ATTEMPTS_FILE, dict(resolution, kind="resolution"))
    if overspend:
        hold_lane(root, lane, "overspend_after_reconcile", None, attempt["seq"])
    capacity["observations"].append(
        dict(
            day=day,
            account_credits=series[-1],
            account_delta=delta,
            endpoint=endpoint,
            possible_cost=cost,
        )
    )
    # legacy-compatible shape: the full pending record + resolution already live in the append-only
    # attempts sidecar above; capacity-v2.json keeps EXACTLY {version, pending, observations}
    capacity["pending"] = None
    _atomic_json(path, capacity)
    return {
        "resolved": True,
        "charged": delta,
        "coverage_halt": overspend,
        "lane_held": lane if overspend else None,
        "attempt_seq": attempt["seq"],
    }


class AccountProviderSession(StopSession):
    """Every paid transport attempt rechecks free base quota; no extra-credit spending."""

    def __init__(
        self,
        session,
        job_date,
        headroom,
        request,
        receipts,
        *,
        capacity_path=None,
        config=None,
        coverage_lanes=frozenset(),
    ):
        super().__init__(session)
        if not set(coverage_lanes) <= set(coverage_contracts.LANES):
            raise ValueError("unknown coverage lane")
        # default empty: no coverage route is callable
        self.coverage_lanes = frozenset(coverage_lanes)
        if self.coverage_lanes and capacity_path is not None:  # a held lane stays disabled
            held = {k for k, v in _lanes(capacity_path.parent).items() if v.get("status") == "held"}
            self.coverage_lanes -= held
        self.job_date, self.headroom = job_date, headroom
        self.request, self.receipts = request, receipts
        self.last_request = time.monotonic()
        self.deadline = None
        self.in_attempt = False  # a paid attempt past its deadline check must run to resolution
        self.rate_limit = self.rate_remaining = None
        self.server_day = None
        self.reserved_endpoint, self.reserved_cost, self.attempts = None, 0, 0
        self.reserved_binding = None  # coverage reservations bind their exact final params
        self.coverage_halt = False  # an overspending coverage call halts every later coverage call
        self.quarantine = None  # a denied (401/402/403) coverage attempt kept as pending
        self.capacity_path, self.config, self.purpose = capacity_path, config, "routine_refresh"
        self.capacity = {"version": 2, "pending": None, "observations": []}
        if capacity_path is not None and capacity_path.exists():
            try:
                self.capacity = json.loads(capacity_path.read_text())
                if (
                    set(self.capacity) != {"version", "pending", "observations"}
                    or self.capacity["version"] != 2
                    or not isinstance(self.capacity["observations"], list)
                    or any(
                        not isinstance(o, dict)
                        or set(o)
                        != {"day", "account_credits", "account_delta", "endpoint", "possible_cost"}
                        or not isinstance(o["day"], str)
                        or not isinstance(o["endpoint"], str)
                        or any(
                            type(o[k]) is not int or o[k] < 0
                            for k in ("account_credits", "account_delta", "possible_cost")
                        )
                        for o in self.capacity["observations"]
                    )
                ):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise BudgetHold("accounting hold: invalid capacity state") from None
            if self.capacity["pending"] is not None:
                self.quarantine = quarantine_of(self._root(), self.capacity["pending"])
                if self.quarantine is None:
                    raise _pending_hold(self.capacity["pending"])
            self.archive_prior_days()

    def archive_prior_days(self):
        """Move observations older than job_date to capacity-archive/<day>.json, then shrink live.

        Archive first (fsynced), live second, so a crash between leaves duplicates, never loss.
        An existing day file must equal the rows or be an exact prefix of them (re-run after a
        crash); anything else is a conflict and holds. Readers filter by day, so spend is unchanged.
        """
        today = str(self.job_date)
        rows = self.capacity["observations"]
        old = [o for o in rows if o["day"] < today]
        if not old:
            return
        by_day = {}
        for o in old:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", o["day"]):
                raise BudgetHold("accounting hold: malformed day in capacity state")
            by_day.setdefault(o["day"], []).append(o)
        archive = self.capacity_path.parent / "capacity-archive"
        for day, day_rows in by_day.items():
            path = archive / f"{day}.json"
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except ValueError:
                    raise BudgetHold("accounting hold: capacity archive conflict") from None
                if not isinstance(existing, list) or existing != day_rows[: len(existing)]:
                    raise BudgetHold("accounting hold: capacity archive conflict")
                if existing == day_rows:
                    continue
            _atomic_json(path, day_rows)
        self.capacity["observations"] = [o for o in rows if o["day"] >= today]
        self.persist()

    def persist(self):
        _atomic_json(self.capacity_path, self.capacity)

    def _root(self):
        return self.capacity_path.parent

    def _evidence(self, endpoint, kwargs, safe, before, response=None, error=None):
        """Append the attempt record right after send(), before any snapshot logic can fail."""
        path = self._root() / ATTEMPTS_FILE
        params = kwargs["params"]
        record = dict(
            kind="attempt",
            seq=1 + sum(r.get("kind") == "attempt" for r in _read_jsonl(path)),
            day=str(self.job_date),
            endpoint=endpoint,
            fingerprint=coverage_contracts.fingerprint(endpoint, _final(params)),
            reserved_cost=self.reserved_cost,
            pre_snapshot=dict(count=before, date=safe["apiRequestsDate"], at=_now()),
        )
        if error is not None:
            record["error_class"] = type(error).__name__
        else:
            digest, head = _body(response, params.get("api_token"))
            headers = getattr(response, "headers", {})
            record.update(
                http_status=getattr(response, "status_code", None),
                content_type=headers.get("Content-Type"),
                body_sha256=digest,
                body=head,
            )
        _append_jsonl(path, record)
        return record

    def _unresolved(self, endpoint, kwargs, record, delta):
        """Fail closed: pending stays set, lane held with the exact reason, no data returned."""
        status = record.get("http_status")
        if delta < 0:
            reason = "counter_regression"
        elif status in (401, 402, 403):
            reason = f"http_{status}_entitlement_or_auth"
        elif isinstance(status, int) and 500 <= status < 600:
            reason = "http_5xx_provider_error"
        else:
            reason = "undercharge_or_snapshot_lag"
        lane = coverage_lane(endpoint, kwargs["params"])
        hold_lane(self._root(), lane, reason, status, record["seq"])
        self.reserved_endpoint, self.reserved_binding = None, None
        return CoverageUnresolved(reason, record, status)

    def snapshot(self):
        account = self.request(self, "user", {}, expected_type=dict)
        self.day_check()
        safe, effective, remaining = account_usage(
            account, self.job_date, self.headroom, server_day=self.server_day
        )
        return safe, effective, remaining

    def reserve(self, endpoint, cost, purpose="routine_refresh", params=None):
        params = _final(params)
        floor = endpoint_cost(endpoint, params, self.coverage_lanes)
        coverage = coverage_lane(endpoint, params) in self.coverage_lanes
        if floor == 0 or type(cost) is not int or cost < floor or (coverage and cost != floor):
            raise ValueError("paid request has no adequate attempt cost")
        self.reserved_endpoint, self.reserved_cost, self.attempts = endpoint, cost, 0
        self.purpose = purpose
        self.reserved_binding = (  # bind() also requires cost == route_cost(final params)
            coverage_contracts.bind(endpoint, params, cost) if coverage else None
        )

    def day_check(self):
        if datetime.now(UTC).date() != self.job_date:
            raise BudgetHold("UTC day changed during acquisition; restart under the new day budget")

    def rate_check(self):
        if self.rate_remaining is not None and self.rate_remaining < min(
            10, max(2, self.rate_limit // 10)
        ):
            raise BudgetHold("provider minute headroom is low; acquisition stopped")

    def send(self, url, kwargs, *, paid):
        delay = self.last_request + MIN_REQUEST_GAP - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if paid:
            self.day_check()
            self.rate_check()
        self.last_request = time.monotonic()
        if not self.in_attempt and self.deadline is not None and self.last_request >= self.deadline:
            raise BudgetHold("network time budget exhausted")
        response = self.session.get(url, **kwargs)
        headers = getattr(response, "headers", {})
        self.server_day = None
        if headers.get("Date") is not None:
            try:
                server_time = parsedate_to_datetime(headers["Date"])
                if server_time.tzinfo is None:
                    raise ValueError
                self.server_day = server_time.astimezone(UTC).date()
            except (TypeError, ValueError, OverflowError):
                raise BudgetHold("provider UTC response date is unverifiable") from None
        limit, remaining = headers.get("X-RateLimit-Limit"), headers.get("X-RateLimit-Remaining")
        if limit is not None or remaining is not None:
            if (
                not str(limit).isdigit()
                or not str(remaining).isdigit()
                or int(limit) <= 0
                or not 0 <= int(remaining) <= int(limit)
            ):
                raise ValueError("provider returned invalid minute-limit headers")
            self.rate_limit, self.rate_remaining = int(limit), int(remaining)
        elif self.rate_remaining is not None:
            self.rate_remaining -= 1
        return response

    def get(self, url, **kwargs):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "eodhd.com"
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/api/")
        ):
            raise ValueError("provider host/path is outside the read-only data allowlist")
        endpoint = parsed.path.removeprefix("/api/")
        params = copy.deepcopy(kwargs.get("params", {}))
        kwargs["params"] = params  # checked == sent: nothing can mutate the original in between
        floor = endpoint_cost(endpoint, _final(params), self.coverage_lanes)
        coverage = coverage_lane(endpoint, params) in self.coverage_lanes
        allowed = {"api_token", "fmt"}
        if coverage:
            if self.coverage_halt:
                raise BudgetHold("coverage halted: observed spend exceeded its reservation")
            allowed |= set(params)  # route_cost already allowlisted every other param
            # retries re-enter get(): bind final params before any HTTP
            if self.reserved_binding is None:
                raise BudgetHold("coverage reservation missing or already used; reserve again")
            try:
                coverage_contracts.check_bound(self.reserved_binding, endpoint, _final(params))
            except ValueError:
                raise BudgetHold("coverage request differs from reservation") from None
        elif endpoint.startswith("sec-filings/"):
            allowed |= {"page[limit]", "page[offset]"}
        elif endpoint.startswith("eod/"):
            allowed |= {"from", "to", "period", "order"}
        elif endpoint.startswith("splits/"):
            allowed |= {"from", "to"}
        elif endpoint == "calendar/earnings":
            allowed |= {"symbols", "from", "to"}
            eodhd.symbol(kwargs.get("params", {}).get("symbols"))
        elif endpoint == "news":
            allowed |= {"s", "from", "to", "limit", "offset"}
            news = kwargs.get("params", {})
            limit, offset = news.get("limit"), news.get("offset", 0)
            if (
                self.purpose != "routine_refresh"
                or not isinstance(news.get("s"), str)
                or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,15}\.[A-Z]{1,6}", news["s"])
                or any(
                    k in news
                    and not (
                        isinstance(news[k], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", news[k])
                    )
                    for k in ("from", "to")
                )
                or any(type(v) is not int for v in (limit, offset))
                or not 1 <= limit <= 1000
                or not 0 <= offset <= 9000
                or offset + limit > 10000
            ):
                raise ValueError("news request is outside the single-ticker routine allowlist")
        if set(kwargs.get("params", {})) - allowed or set(kwargs) - {
            "params",
            "timeout",
            "allow_redirects",
        }:
            raise ValueError("provider request parameters are outside the data allowlist")
        if kwargs.get("allow_redirects", False):
            raise ValueError("provider redirects are disabled")
        kwargs["allow_redirects"] = False
        if floor:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise BudgetHold("network time budget exhausted")  # before any HTTP or pending
            self.in_attempt = True
            try:
                return self._paid_attempt(url, kwargs, endpoint, coverage)
            finally:
                self.in_attempt = False
        return self.checked(self.send(url, kwargs, paid=False))

    def _paid_attempt(self, url, kwargs, endpoint, coverage=False):
        self.day_check()
        self.rate_check()
        if self.reserved_endpoint != endpoint or self.attempts >= 3:
            raise ValueError("paid request is outside its durable reservation")
        if self.capacity_path is None:
            raise BudgetHold("accounting hold: capacity state unavailable")
        pending = self.capacity["pending"]
        if pending is not None and (self.quarantine is None or "inflight" in pending):
            raise _pending_hold(pending)
        try:
            safe, effective, remaining = self.snapshot()
        except Exception:
            raise BudgetHold("accounting hold: pre-attempt account snapshot unavailable") from None
        self.receipts.append(
            dict(
                dataset="account_preflight",
                safe_account=safe,
                effective_current_usage=effective,
                base_remaining_after_headroom=remaining,
                reserved_credits=0,
                status="checked",
            )
        )
        observations = self.capacity["observations"]
        previous = next((o for o in reversed(observations) if o["day"] == str(self.job_date)), None)
        if previous and effective < previous["account_credits"]:
            raise BudgetHold("accounting hold: account counter regressed")
        observations.append(
            dict(
                day=str(self.job_date),
                account_credits=effective,
                account_delta=0,
                endpoint="user",
                possible_cost=0,
            )
        )
        self.persist()
        hard = self.config["daily_credit_limit"]
        working = self.config.get("daily_working_credit_limit", min(hard, WORKING_CREDITS))
        # a quarantined denial may still be charged: its FULL possible cost is reserved in every
        # cap/headroom check, on this and every later UTC day (never settled as zero)
        reserve = self.quarantine["reserve"] if self.quarantine else 0
        if (
            remaining - reserve < self.reserved_cost
            or effective + reserve + self.reserved_cost > hard
            or (
                effective + reserve + self.reserved_cost > working
                and self.purpose not in _DISCRETIONARY_PURPOSES
            )
        ):
            raise CapacityLimit("account quota or project budget below attempt cost and headroom")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetHold("network time budget exhausted")  # a slow preflight; nothing pending
        inflight = dict(
            day=str(self.job_date),
            endpoint=endpoint,
            possible_cost=self.reserved_cost,
            before=effective,
        )
        if coverage:  # the retained attempt record is matched back to this pending by fingerprint
            inflight["fingerprint"] = coverage_contracts.fingerprint(
                endpoint, _final(kwargs["params"])
            )
        # normal: the in-flight attempt IS the pending. Quarantined: the denied pending is kept
        # verbatim and the new attempt rides along as `inflight` (a crash leaves it -> full hold).
        self.capacity["pending"] = (
            {**self.capacity["pending"], "inflight": inflight} if self.quarantine else inflight
        )
        self.persist()
        self.attempts += 1
        try:
            response = self.send(url, kwargs, paid=True)
        except Exception as exc:
            if coverage:
                self._evidence(endpoint, kwargs, safe, effective, error=exc)
            raise BudgetHold("accounting hold: paid attempt outcome unknown") from None
        record = self._evidence(endpoint, kwargs, safe, effective, response) if coverage else None
        try:
            after_safe, after, after_remaining = self.snapshot()
            if after_safe["apiRequestsDate"] != str(self.job_date) or (
                not coverage and after < effective + self.reserved_cost
            ):
                raise ValueError("counter missing or regressed")
        except Exception:
            raise BudgetHold("accounting hold: post-attempt account snapshot unavailable") from None
        if coverage:
            _append_jsonl(
                self._root() / ATTEMPTS_FILE,
                dict(
                    kind="snapshot",
                    attempt_seq=record["seq"],
                    count=after,
                    date=str(self.job_date),
                    at=_now(),
                ),
            )
            if after - effective < self.reserved_cost:
                raise self._unresolved(endpoint, kwargs, record, after - effective)
        observations.append(
            dict(
                day=str(self.job_date),
                account_credits=after,
                account_delta=after - effective,
                endpoint=endpoint,
                possible_cost=self.reserved_cost,
            )
        )
        self.capacity["pending"] = (
            {k: v for k, v in self.capacity["pending"].items() if k != "inflight"}
            if self.quarantine
            else None
        )
        self.persist()
        self.receipts.append(
            dict(
                dataset="account_usage",
                safe_account=after_safe,
                effective_current_usage=after,
                base_remaining_after_headroom=after_remaining,
                account_delta=after - effective,
                http_attempts=1,
                reserved_credits=0,
                status="checked",
                # the observation row above already pairs possible_cost (route_cost) with the delta
                **({"cost_mismatch": after - effective != self.reserved_cost} if coverage else {}),
            )
        )
        if coverage and after - effective > self.reserved_cost:
            self.coverage_halt = True  # paid data is returned, but nothing more is sent
        if coverage:  # single-use: a pagination loop that forgets to re-reserve cannot pay twice
            self.reserved_endpoint, self.reserved_binding = None, None
        return self.checked(response)


class ProviderSession(StopSession):
    """Every paid transport attempt rechecks free base quota; no extra-credit spending."""

    def __init__(self, session, job_date, headroom, request, receipts):
        super().__init__(session)
        self.job_date, self.headroom = job_date, headroom
        self.request, self.receipts = request, receipts
        self.last_request = time.monotonic()
        self.deadline = None
        self.rate_limit = self.rate_remaining = None
        self.server_day = None
        self.reserved_endpoint, self.reserved_cost, self.attempts = None, 0, 0

    def reserve(self, endpoint, cost):
        if endpoint == "news":
            raise ValueError("news requires an AccountProviderSession")
        floor = endpoint_cost(endpoint)
        if floor == 0 or type(cost) is not int or cost < 3 * floor:
            raise ValueError("paid request has no adequate three-attempt reservation")
        self.reserved_endpoint, self.reserved_cost, self.attempts = endpoint, cost, 0

    def day_check(self):
        if datetime.now(UTC).date() != self.job_date:
            raise BudgetHold("UTC day changed during acquisition; restart under the new day budget")

    def rate_check(self):
        if self.rate_remaining is not None and self.rate_remaining < min(
            10, max(2, self.rate_limit // 10)
        ):
            raise BudgetHold("provider minute headroom is low; acquisition stopped")

    def send(self, url, kwargs, *, paid):
        delay = self.last_request + 0.2 - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if paid:
            self.day_check()
            self.rate_check()
        self.last_request = time.monotonic()
        if self.deadline is not None and self.last_request >= self.deadline:
            raise BudgetHold("network time budget exhausted")
        response = self.session.get(url, **kwargs)
        headers = getattr(response, "headers", {})
        self.server_day = None
        if headers.get("Date") is not None:
            try:
                server_time = parsedate_to_datetime(headers["Date"])
                if server_time.tzinfo is None:
                    raise ValueError
                self.server_day = server_time.astimezone(UTC).date()
            except (TypeError, ValueError, OverflowError):
                raise BudgetHold("provider UTC response date is unverifiable") from None
        limit, remaining = headers.get("X-RateLimit-Limit"), headers.get("X-RateLimit-Remaining")
        if limit is not None or remaining is not None:
            if (
                not str(limit).isdigit()
                or not str(remaining).isdigit()
                or int(limit) <= 0
                or not 0 <= int(remaining) <= int(limit)
            ):
                raise ValueError("provider returned invalid minute-limit headers")
            self.rate_limit, self.rate_remaining = int(limit), int(remaining)
        elif self.rate_remaining is not None:
            self.rate_remaining -= 1
        return self.checked(response)

    def get(self, url, **kwargs):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "eodhd.com"
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/api/")
        ):
            raise ValueError("provider host/path is outside the read-only data allowlist")
        endpoint = parsed.path.removeprefix("/api/")
        if endpoint == "news":
            raise ValueError("news requires an AccountProviderSession")
        floor = endpoint_cost(endpoint)
        allowed = {"api_token", "fmt"}
        if endpoint.startswith("sec-filings/"):
            allowed |= {"page[limit]", "page[offset]"}
        elif endpoint.startswith("eod/"):
            allowed |= {"from", "to", "period", "order"}
        elif endpoint.startswith("splits/"):
            allowed |= {"from", "to"}
        elif endpoint == "calendar/earnings":
            allowed |= {"symbols", "from", "to"}
            eodhd.symbol(kwargs.get("params", {}).get("symbols"))
        if set(kwargs.get("params", {})) - allowed or set(kwargs) - {
            "params",
            "timeout",
            "allow_redirects",
        }:
            raise ValueError("provider request parameters are outside the data allowlist")
        if kwargs.get("allow_redirects", False):
            raise ValueError("provider redirects are disabled")
        kwargs["allow_redirects"] = False
        if floor:
            self.day_check()
            self.rate_check()
            if self.reserved_endpoint != endpoint or self.attempts >= 3:
                raise ValueError("paid request is outside its durable reservation")
            account = self.request(self, "user", {}, expected_type=dict)
            self.day_check()
            safe, effective, remaining = account_usage(
                account, self.job_date, self.headroom, server_day=self.server_day
            )
            self.receipts.append(
                dict(
                    dataset="account_usage",
                    safe_account=safe,
                    effective_current_usage=effective,
                    base_remaining_after_headroom=remaining,
                    reserved_credits=0,
                    status="checked",
                )
            )
            if remaining < self.reserved_cost:
                raise BudgetHold(
                    "account remaining base quota below conservative reservation and headroom"
                )
            self.attempts += 1
        return self.send(url, kwargs, paid=bool(floor))
