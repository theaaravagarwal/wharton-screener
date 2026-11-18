"""Guarded EODHD news page fetch (Phase 1b). Imports no HTTP library.

One call = one durable reservation, one preflight/attempt inside the session; there is no retry
loop here (the session owns retries/accounting). The kill switch defaults to off. Spend is read
from the SHARED capacity-v2.json observations (endpoint "news", account_delta), never from
reservations; ledger.json keeps only halt state (halted, halt_reason, day, day_halted,
retry_until). Every stop that is not a fetched page becomes a receipt with outcome "budget_hold"
or "error", never "empty". A malformed ledger fails closed (reason "ledger_invalid").

Runbook (operator)
- An uncharged failure can leave capacity-v2.json `pending` set (the session post-check raises
  before clearing it when the account counter did not move). That halts news (reason
  "pending_unresolved") AND wedges screening, whose session refuses to start while pending is set.
  An operator must reconcile the pending attempt against the live /api/user counter, then set
  `pending` to null in capacity-v2.json (under the state lock, no run active). News never clears it.
- To clear a news halt (cost anomaly, auth, pending_unresolved) after reconciling: with no run
  active, set `halted` false and `halt_reason` null in <state_root>/news/ledger.json. Fix a
  "ledger_invalid" file the same way; it is never reset automatically.
- 402 halts news for the UTC day; 429 pauses it for 15 minutes.
- Screening owner follow-up: change the post-check so an uncharged failure does not leave pending
  set and wedge screening.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import uuid
from contextlib import contextmanager, nullcontext
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from wharton import news_store

URL = "https://eodhd.com/api/news"
FLOOR = 5  # documented single-ticker/feed cost; the session post-check passes at 5 or 10
DEFAULT_CEILING = 2000
MAX_CEILING = 2000  # user-approved daily news ceiling; raising it needs a new decision
WORKING_CAP = 75000  # user-approved working credits per UTC day (screen_budget.WORKING_CREDITS)


def working_cap(config):
    """Same fallback as screen_budget: min(daily_credit_limit, 50,000) unless set explicitly."""
    hard = config.get("daily_credit_limit", WORKING_CAP)
    return config.get("daily_working_credit_limit", min(hard, WORKING_CAP))


BREADTH_RESERVE = 5000  # credits breadth mode leaves unspent under the working cap
RATE_LIMIT_PAUSE = timedelta(minutes=15)
_LEDGER_KEYS = {"halted", "halt_reason", "day", "day_halted", "retry_until"}
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,15}\.[A-Z]{1,6}")  # same as the guard patch


class NewsSkipped(Exception):
    """News did not start (disabled, or the shared screening lock is held). No call was made."""


class NewsRejected(ValueError):
    """Request shape is outside the news allowlist; rejected before any call."""


def enabled(config) -> bool:
    return (config.get("news") or {}).get("enabled") is True


_HELD = set()  # state roots whose news_lock this process currently holds


def _pending_blocks(session) -> bool:
    """An unresolved pending blocks news, except a quarantined DENIED coverage attempt the guard
    already accepted (its full possible cost stays reserved in every cap check)."""
    pending = session.capacity["pending"]
    quarantine = getattr(session, "quarantine", None)
    return pending is not None and (quarantine is None or "inflight" in pending)


def lock_is_held(state_root) -> bool:
    return str(Path(state_root).resolve()) in _HELD


@contextmanager
def news_lock(state_root):
    """Non-blocking flock on the SAME file screening's run() locks."""
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NewsSkipped("screening or another news run holds the state lock") from None
        key = str(root.resolve())
        _HELD.add(key)
        try:
            yield
        finally:
            _HELD.discard(key)


def _iso(moment):
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate(scope, from_date, to_date, limit, offset):
    if set(scope) - {"s"}:
        raise NewsRejected("only an 's' single-ticker scope is allowed")
    params = {}
    if "s" in scope:
        if not isinstance(scope["s"], str) or not _SYMBOL.fullmatch(scope["s"]):
            raise NewsRejected("scope 's' must be one exchange-qualified symbol (no commas)")
        params["s"] = scope["s"]
    for key, value in (("from", from_date), ("to", to_date)):
        try:
            if value is not None:
                date.fromisoformat(value)
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError
        except (TypeError, ValueError):
            raise NewsRejected(f"{key} must be YYYY-MM-DD") from None
        if value is not None:
            params[key] = value
    for value in (limit, offset):
        if type(value) is not int:
            raise NewsRejected("limit and offset must be integers")
    if not 1 <= limit <= 1000 or not 0 <= offset <= 9000 or offset + limit > 10000:
        raise NewsRejected("limit<=1000, offset<=9000 and offset+limit<=10000 required")
    return {**params, "limit": limit, "offset": offset}


def _ledger(state_root, day):
    """Load ledger.json (halted, halt_reason, day, day_halted, retry_until). A new UTC day resets
    day_halted/retry_until but never the sticky halt. Returns None (fail closed) if malformed."""
    path = Path(state_root) / "news" / "ledger.json"
    if not path.exists():
        return path, dict(
            halted=False, halt_reason=None, day=day, day_halted=False, retry_until=None
        )
    try:
        ledger = json.loads(path.read_text())
        ok = (
            isinstance(ledger, dict)
            and set(ledger) == _LEDGER_KEYS
            and type(ledger["halted"]) is bool
            and type(ledger["day_halted"]) is bool
            and isinstance(ledger["day"], str)
            and all(
                ledger[k] is None or isinstance(ledger[k], str)
                for k in ("halt_reason", "retry_until")
            )
        )
    except (OSError, ValueError):
        ok = False
    if not ok:
        return path, None
    if ledger["day"] != day:
        ledger.update(day=day, day_halted=False, retry_until=None)
    return path, ledger


def _save_ledger(path, ledger):
    news_store.atomic_write(path, news_store.canonical_json(ledger))


def _spend_and_screening(capacity, day):
    """From the shared capacity ledger: news credits observed today, and whether screening spent."""
    obs = [o for o in capacity["observations"] if o["day"] == day]
    news = sum(o["account_delta"] for o in obs if o["endpoint"] == "news")
    screened = any(o["endpoint"] not in {"user", "news"} for o in obs)
    return news, screened


def _receipt(params, page, now, *, status, content, outcome, **extra):
    fields = dict(
        endpoint="news",
        params=params,
        page=page,
        offset=params["offset"],
        limit=params["limit"],
        http_status=status,
        response_sha256=news_store.sha256_hex(content),
        byte_len=len(content),
        bytes_exact=outcome in {"ok", "empty"},
        retrieved_at=_iso(now),
        credits_before=None,
        credits_after=None,
        attempt_id=None,
        parser_version=news_store.PARSER_VERSION,
        outcome=outcome,
    )
    fields.update(extra)
    fields["receipt_id"] = news_store.sha256_hex(news_store.canonical_json(fields))[:32]
    return fields


def _store(root, receipt):
    """Persist a receipt that has no raw page (hold/error); ok pages go through ingest."""
    path = Path(root) / "news" / "receipts" / f"{receipt['receipt_id']}.json"
    news_store.atomic_write(path, news_store.canonical_json(receipt))
    return receipt


def _is(exc, name):
    return any(c.__name__ == name for c in type(exc).__mro__)


def fetch_page(
    session,
    root,
    state_root,
    *,
    scope,
    from_date,
    to_date,
    limit,
    offset,
    config,
    now=None,
    page=0,
    token=None,
    ledger_root=None,
    lock_held=False,
    mode="priority",
):
    """Fetch one news page and return its receipt.

    `session` MUST be a zero-argument factory returning an AccountProviderSession; a pre-built
    session raises NewsRejected (its capacity snapshot would be stale and persist() would lose
    updates). The factory runs inside the shared lock. NewsSkipped (disabled, wrong accounting
    mode, lock held) and NewsRejected (bad shape/config) are raised before any call; every other
    stop returns a "budget_hold" or "error" receipt with a `reason`.
    """
    if lock_held and not lock_is_held(state_root):
        raise NewsRejected("lock_held=True but this process does not hold news_lock(state_root)")
    if not enabled(config):
        raise NewsSkipped("news is disabled (kill switch default off)")
    if config.get("accounting_mode") != "account":
        raise NewsSkipped("news requires accounting_mode: account")
    if not callable(session) or hasattr(session, "get"):
        raise NewsRejected("session must be a zero-argument factory built inside the lock")
    settings = config.get("news") or {}
    if mode not in ("priority", "breadth"):
        raise NewsRejected("mode must be 'priority' or 'breadth'")
    if mode == "breadth":  # the only path that may exceed MAX_CEILING (news_daily --mode breadth)
        working = working_cap(config)
        top = working - settings.get("breadth_reserve", BREADTH_RESERVE)
        ceiling = settings.get("breadth_ceiling")
        if type(ceiling) is not int or not 1 <= ceiling <= top:
            raise NewsRejected(f"breadth_ceiling must be an integer in 1..{top}")
    else:
        ceiling = settings.get("daily_ceiling", DEFAULT_CEILING)
        if type(ceiling) is not int or not 1 <= ceiling <= MAX_CEILING:
            raise NewsRejected(f"daily_ceiling must be an integer in 1..{MAX_CEILING}")
    if not scope:
        raise NewsRejected("a single-ticker 's' scope is required")
    params = _validate(scope, from_date, to_date, limit, offset)
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None or now.utcoffset() is None:
        raise NewsRejected("now must be timezone-aware")
    token = token if token is not None else os.environ.get("EODHD_API_KEY", "").strip()
    if not token:
        raise NewsRejected("no EODHD token available")

    def stop(outcome, reason, *, status=None, content=b"", **extra):
        receipt = _receipt(
            params,
            page,
            now,
            status=status,
            content=content,
            outcome=outcome,
            reason=reason,
            **extra,
        )
        return _store(root, receipt)

    # lock_held: the caller (news_daily) already holds the state lock; flock on a second fd
    # would conflict even within one process.
    with nullcontext() if lock_held else news_lock(state_root):
        try:
            session = session()
        except Exception as exc:  # pending, invalid capacity state, ...
            return stop("budget_hold" if _is(exc, "BudgetHold") else "error", "session_refused")
        day = str(session.job_date)
        if now.astimezone(UTC).date().isoformat() != day:
            return stop("budget_hold", "day_mismatch")
        if _pending_blocks(session):
            return stop("budget_hold", "pending")
        ledger_path, ledger = _ledger(ledger_root or state_root, day)
        if ledger is None:
            return stop("budget_hold", "ledger_invalid")
        if ledger["halted"]:
            return stop("budget_hold", "halted")
        if ledger["day_halted"]:
            return stop("budget_hold", "halted_for_day")
        if ledger["retry_until"] and _iso(now) < ledger["retry_until"]:
            return stop("budget_hold", "rate_limited")
        spent, screened = _spend_and_screening(session.capacity, day)
        if spent + FLOOR > ceiling:
            return stop("budget_hold", "ceiling")
        if mode == "breadth":  # leftover mode: runs after screening, bounded by ALL spend today
            total = sum(
                o["account_delta"] for o in session.capacity["observations"] if o["day"] == day
            )
            if total + FLOOR > working_cap(config):
                return stop("budget_hold", "working_cap")
        elif screened:
            return stop("budget_hold", "screening_started")

        seen = len(session.capacity["observations"])
        attempt_id = f"news-{day}-{uuid.uuid4().hex[:16]}"
        status = content = failure = None
        try:
            session.reserve("news", FLOOR, purpose="routine_refresh")
            response = session.get(
                URL,
                params={**params, "fmt": "json", "api_token": token},
                timeout=(5, 30),
                allow_redirects=False,
            )
            status, content = response.status_code, bytes(response.content)
        except Exception as exc:
            failure = exc
        # What the account actually charged for this attempt, if the session recorded it.
        seen_obs = [o for o in session.capacity["observations"][seen:] if o["endpoint"] == "news"]
        before = after = delta = None
        if seen_obs:
            after, delta = seen_obs[-1]["account_credits"], seen_obs[-1]["account_delta"]
            before = after - delta
        text = str(failure) if failure else ""
        halt = None
        if (seen_obs and delta != FLOOR) or (not seen_obs and failure is None and status == 200):
            halt = f"cost_anomaly observed={delta} floor={FLOOR} attempt={attempt_id}"
        meta = dict(credits_before=before, credits_after=after, attempt_id=attempt_id)
        kind = None  # (outcome, reason) for a failed attempt
        if _pending_blocks(session):  # never cleared here; wedges screening too
            halt, kind = "pending_unresolved", ("error", "pending_unresolved")
        elif status == 402 or "HTTP 402" in text:
            ledger["day_halted"], kind = True, ("budget_hold", "http_402")
        elif status == 429 or "HTTP 429" in text:
            ledger["retry_until"] = _iso(now + RATE_LIMIT_PAUSE)
            kind = ("budget_hold", "rate_limited")
        elif status in {401, 403} or "HTTP 401" in text or "HTTP 403" in text:
            halt, kind = "auth", ("error", "auth")
        elif failure is not None and _is(failure, "CapacityLimit"):
            kind = ("budget_hold", "capacity_limit")
        elif failure is not None and _is(failure, "BudgetHold"):
            ledger["day_halted"], kind = True, ("budget_hold", "session_stop")
        elif failure is not None:
            kind = ("error", "attempt_failed")
        elif status != 200:
            kind = ("error", "http_error")
        if halt:
            ledger["halted"], ledger["halt_reason"] = True, halt
        _save_ledger(ledger_path, ledger)
        if kind:
            return stop(kind[0], kind[1], status=status, content=content or b"", **meta)
        try:
            payload = json.loads(content)
            if not isinstance(payload, list):
                raise ValueError
        except ValueError:
            return stop("error", "bad_payload", status=status, content=content, **meta)
        shape = news_store.page_outcome(len(payload), limit, offset)
        receipt = _receipt(
            params,
            page,
            now,
            status=status,
            content=content,
            outcome="empty" if not payload else "ok",
            n_items=len(payload),
            exhausted=shape != "partial",
            truncated=shape == "partial" and offset + limit >= 10000,
            **meta,
        )
        try:
            news_store.ingest(root, content, receipt, first_seen_at=_iso(now))
        except news_store.NewsPayloadError:  # ingest already stored its error receipt
            return {**receipt, "outcome": "error", "reason": "bad_payload", "bytes_exact": False}
        except (ValueError, OSError):  # credits were spent: one error receipt, never silent
            return stop("error", "ingest_failed", status=status, content=content, **meta)
        return receipt
