"""Bounded daily acquisition and immutable, offline-replayable screening runs."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import html
import io
import json
import os
import platform
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

from wharton import (
    coverage_runtime,
    eodhd,
    http_data,
    screen_budget,
    screen_sources,
    screen_throughput,
    screening,
)
from wharton.screen_budget import (
    COST_FLOORS,
    WORKING_CREDITS,
    AccountProviderSession,
    BudgetHold,
    CapacityLimit,
    ProviderSession,
    SourceUnavailable,
    account_usage,
    reservation_allowed,
    validate_budget,
)
from wharton.screen_budget import (
    StopSession as StopSession,
)
from wharton.screen_budget import (
    endpoint_cost as endpoint_cost,
)

OPERATOR_STOP = (
    "systemctl --user disable --now wharton-screening.timer\n"
    "systemctl --user stop wharton-screening.service"
)


def refresh_context(inputs, as_of):
    now = datetime.now(UTC)
    calendars = {
        screen_sources.CALENDARS.get(r["ticker"].rsplit(".", 1)[-1]) for r in inputs["universe"]
    }
    cutoffs = {}
    for calendar in sorted(calendars - {None}):
        sessions = screen_sources.completed_sessions(
            calendar, inputs.get("market_as_of", as_of), now.isoformat()
        )
        cutoffs[calendar] = str(sessions[-1].date()) if len(sessions) else None
    return {"refresh_epoch": now.strftime("%Y-%m-%dT%H:00Z"), "market_cutoffs": cutoffs}


def event_refresh_due(previous, event, refresh_id):
    observed = screening.day((previous or {}).get("retrieved_at"))
    event_day = screening.day(event)
    return bool(
        previous
        and event_day
        and observed
        and event_day >= observed
        and previous.get("refresh_id") != refresh_id
    )


SourceIdentityConflict = screen_sources.SourceIdentityConflict


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        out.write(encoded(value))
        temporary = out.name
    os.replace(temporary, path)


class Receipts(list):
    """Acquisition receipts plus per-ticker outcome (refreshed/price_excluded/held)."""

    outcomes = None
    verified = ()


# Queue cap only: real spend limits remain the credit budget and the acquire time guard.
MAX_QUEUE = 100000
BOOTSTRAP_FRACTION = 0.25
ACQUISITION_SECONDS = 3300  # acquire() deadline; also the batch estimate's time budget


def bootstrap_fraction(acquisition):
    fraction = acquisition.get("bootstrap_fraction", BOOTSTRAP_FRACTION)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction <= 1
    ):
        raise ValueError("bootstrap_fraction must be in (0, 1]")
    return fraction


def run_seconds(acquisition):
    seconds = acquisition.get("run_seconds", ACQUISITION_SECONDS)
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 600 <= seconds <= 46800:
        raise ValueError("run_seconds must be an integer in 600..46800")
    return seconds


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not config.get("input") or not config.get("state_root"):
        raise ValueError("config requires input and state_root")
    for key in ("input", "state_root", "quarter_supplements"):
        if key not in config:
            continue
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    config.setdefault("batch_size", 1000)
    config.setdefault("evaluation_workers", 1)
    screening.validate_workers(config["evaluation_workers"])
    # An empty YAML `acquisition:` parses as None; every caller expects a mapping.
    config["acquisition"] = config.get("acquisition") or {}
    if not isinstance(config["acquisition"], dict):
        raise ValueError("acquisition must be a mapping")
    if config["batch_size"] != "auto" and (
        isinstance(config["batch_size"], bool)
        or not isinstance(config["batch_size"], int)
        or not 1 <= config["batch_size"] <= MAX_QUEUE
    ):
        raise ValueError(f"batch_size must be 1..{MAX_QUEUE}")
    if config.get("acquisition", {}).get("enabled"):
        validate_budget(config["acquisition"])
    bootstrap_fraction(config.get("acquisition", {}))
    run_seconds(config["acquisition"])
    if "coverage" in config:  # coverage hooks exist only for a validated, enabled block
        config["coverage"] = coverage_runtime.validate_coverage_config(config, base=path.parent)
    return config


def choose_batch(inputs, state, size, *, as_of=None):
    """Holdings and active first, then fair region/cap round robin, oldest seen first.

    Holdings are intentionally uncapped: every actual holding is queued even when that makes
    the batch exceed `size`, so len(batch) <= size + len(holdings).
    """
    records = inputs["universe"]
    cutoff = screening.day(as_of or datetime.now(UTC).date().isoformat())
    learned = {}
    if state.get("last_good"):
        previous = Path(state["last_good"])
        body = (previous / "input.json").read_bytes()
        if digest(body) != read(previous / "manifest.json")["files"].get("input.json"):
            raise ValueError("screen artifact hash mismatch: prior selection input")
        learned = {(r["ticker"], r["issuer_id"]): r for r in json.loads(body)["universe"]}

    def usable_cap(record):
        cap, fresh = screening.evidence(record, "market_cap_usd", cutoff)
        value = screening.number(cap.get("value"))
        period = screening.day(cap.get("period"))
        return (
            value
            if (
                fresh
                and cap.get("currency") == "USD"
                and value is not None
                and value > 0
                and period is not None
                and period <= cutoff
            )
            else None
        )

    holdings = set(inputs.get("portfolio", {}).get("holdings", []))
    priority = holdings | set(inputs.get("active", []))
    seen = state.get("last_seen", {})

    def stalest(rows):
        return sorted(rows, key=lambda r: (seen.get(r["issuer_id"], ""), r["ticker"]))

    # Actual holdings are always queued first; active names rotate by staleness and take at
    # most half of the remaining queue so the universe is never crowded out.
    first = stalest(r for r in records if r["ticker"] in holdings)
    room = max(0, size - len(first))
    first += stalest(r for r in records if r["ticker"] in priority - holdings)[
        : (max(1, room // 2) if room else 0)
    ]
    # Unique issuer is validated by screening before committing a run.
    groups = {}
    for r in records:
        if r["ticker"] in priority:
            continue
        cap = usable_cap(r)
        if cap is None:
            cap = usable_cap(learned.get((r["ticker"], r["issuer_id"]), {}))
        groups.setdefault(
            (str(r.get("region", "unknown")), str(screening.cap_bucket(cap))), []
        ).append(r)
    for group in groups.values():
        group.sort(key=lambda r: (seen.get(r["issuer_id"], ""), r["ticker"]))
    # Rotating the stratum start prevents the first groups starving when batch < groups.
    keys = sorted(groups)
    offset = state.get("rotation", 0) % len(keys) if keys else 0
    keys = keys[offset:] + keys[:offset]
    selected = first
    while len(selected) < size and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop(0))
    return [r["ticker"] for r in selected]


class PublicSECUnavailable(SourceUnavailable):
    def __init__(self, status):
        self.status = status
        super().__init__(f"SEC public HTTP {status}: source quarantined")


class PublicSECSession:
    """At most one public request per second, including retries and restarts."""

    def __init__(self, session, ledger, ledger_path, user_agent, deadline=None):
        self.session, self.ledger, self.ledger_path = session, ledger, ledger_path
        self.user_agent, self.deadline = user_agent, deadline

    def get(self, url, **kwargs):
        delay = self.ledger.get("public_last_request_at", 0) + 1 - time.time()
        if delay > 0:
            time.sleep(delay)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetHold("network time budget exhausted")
        self.ledger["public_last_request_at"] = time.time()
        atomic(self.ledger_path, self.ledger)
        response = self.session.get(url, headers={"User-Agent": self.user_agent}, **kwargs)
        if response.status_code in {401, 403, 404, 429}:
            raise PublicSECUnavailable(response.status_code)
        return response


def acquire(
    config,
    records,
    selected,
    root,
    as_of,
    *,
    request=eodhd.request,
    public_request=http_data.get_json,
    deadline_at=None,
):
    """Durable conservative reservations, event discovery and retained-universe cache reuse.
    `deadline_at` (time.monotonic()) is the run's SHARED deadline: acquisition never outlives it."""
    validate_budget(config)
    if config.get("sec_financials_enabled") and (
        not isinstance(config.get("request_credit_cost"), dict)
        or config["request_credit_cost"].get("sec_financials", 0) < COST_FLOORS["sec_financials"]
    ):
        raise ValueError("SEC financials requires its explicit documented request cost")
    cutoff = screening.day(as_of)
    if cutoff != datetime.now(UTC).date():
        raise ValueError("current-source acquisition requires actual UTC today; use offline replay")
    initial = config.get("mode", "initial") == "initial"
    if config.get("accounting_mode") != "account" and (root / "capacity-v2.json").exists():
        raise BudgetHold("accounting hold: legacy mode cannot bypass account capacity state")
    ledger_path = root / "acquisition-ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else {}
    day_key = str(cutoff)
    market_as_of = config.get("market_as_of", as_of)
    context = refresh_context({"universe": records, "market_as_of": market_as_of}, as_of)
    epoch = config.get("refresh_epoch") or context["refresh_epoch"]
    market_cutoffs = config.get("market_cutoffs", context["market_cutoffs"])
    used = initial_used = 0
    if config.get("accounting_mode") != "account":
        used = ledger.get(day_key, 0)
        # Old ledgers did not distinguish modes: migrate conservatively, never reset bootstrap.
        initial_used = ledger.get(
            "initial_total", sum(v for k, v in ledger.items() if screening.day(k))
        )
        if any(type(v) is not int or v < 0 for v in (used, initial_used)):
            raise ValueError("project credit ledger contains invalid counters")
    receipts, emitted = Receipts(), set()
    # held: price/metadata refresh not completed; done: reached the price stage; halt: first
    # budget/time hold stops further paid attempts for this run.
    held, excluded, done, paid = set(), set(), set(), set()
    identity_blocked, verified, unprovable = set(), set(), set()
    halt = {"all": False, "routine": False}
    deadline = time.monotonic() + run_seconds(config)
    if deadline_at is not None:
        deadline = min(deadline, deadline_at)  # coverage jobs already used part of this budget
    cache = root / "source-cache"
    cache.mkdir(parents=True, exist_ok=True)

    def reserve(cost):
        nonlocal used, initial_used
        used += cost
        if initial:
            initial_used += cost
        ledger[day_key], ledger["initial_total"] = used, initial_used
        atomic(ledger_path, ledger)

    def cache_context(ticker, dataset):
        calendar = screen_sources.CALENDARS.get(ticker.rsplit(".", 1)[-1])
        cutoff = market_cutoffs.get(calendar)
        dependencies = market_cutoffs if dataset == "fx" else cutoff
        return dict(
            refresh_epoch=epoch,
            completed_market_session=cutoff,
            session_key_verified=(
                bool(dependencies) and all(dependencies.values())
                if isinstance(dependencies, dict)
                else bool(dependencies)
            ),
            market_cutoff_signature=digest(encoded(dependencies)),
            refresh_id=digest(
                encoded(
                    {"cutoffs": dependencies}
                    if config.get("accounting_mode") == "account" and dataset != "earnings"
                    else {"epoch": epoch, "cutoffs": dependencies}
                )
            ),
        )

    def pointer_for(ticker, dataset):
        return cache / (digest(f"{ticker}:{dataset}") + ".json")

    def identity_held(record):
        path = cache / (digest(record["ticker"]) + ".identity-quarantine.json")
        if not path.exists():
            return False
        quarantine = read(path)
        if quarantine.get("as_of") != as_of or quarantine.get(
            "expected"
        ) != screen_sources.expected_identity(record):
            return False
        if (
            quarantine.get("file")
            and digest((cache / quarantine["file"]).read_bytes()) != quarantine["sha256"]
        ):
            raise ValueError("identity quarantine raw source hash mismatch")
        record["identity_quarantine"], record["evidence"] = quarantine, {}
        if not any(
            r.get("dataset") == "identity_quarantine" and r.get("ticker") == record["ticker"]
            for r in receipts
        ):
            receipts.append({**quarantine, "cache_hit": True, "reserved_credits": 0})
        return True

    def negative_cached(ticker, dataset):
        path = cache / (digest(f"{ticker}:{dataset}") + ".unavailable.json")
        if path.exists():
            previous = read(path)
            if previous.get("as_of") == as_of:
                return previous
        return None

    def remember_unavailable(receipt):
        receipt.update(as_of=as_of, retrieved_at=datetime.now(UTC).isoformat())
        atomic(
            cache / (digest(f"{receipt['ticker']}:{receipt['dataset']}") + ".unavailable.json"),
            receipt,
        )
        receipts.append(receipt)
        emitted.add((receipt["ticker"], receipt["dataset"]))

    def held_receipt(ticker, dataset, previous):
        if dataset in {"prices", "fundamentals", "sec_annual", "sec_quarterly", "sec_financials"}:
            held.add(ticker)
        if previous is None:
            return None
        receipt = {**previous, "cache_hit": True, "reserved_credits": 0, "refresh_held": True}
        receipts.append(receipt)
        emitted.add((ticker, dataset))
        return receipt

    def cached(ticker, dataset):
        if negative_cached(ticker, dataset):
            return None
        pointer = pointer_for(ticker, dataset)
        if not pointer.exists():
            return None
        previous = read(pointer)
        if previous.get("ticker") != ticker or previous.get("dataset") != dataset:
            raise ValueError("source cache identity mismatch")
        if digest((cache / previous["file"]).read_bytes()) != previous["sha256"]:
            raise ValueError("source cache hash mismatch")
        return previous

    with requests.Session() as raw_session:
        session, remaining = (
            (
                AccountProviderSession(
                    raw_session,
                    cutoff,
                    config.get("account_headroom", 25000),
                    request,
                    receipts,
                    capacity_path=root / "capacity-v2.json",
                    config=config,
                )
                if config.get("accounting_mode") == "account"
                else ProviderSession(
                    raw_session, cutoff, config.get("account_headroom", 25000), request, receipts
                )
            ),
            None,
        )
        session.deadline = deadline

        def fetch(record, dataset, endpoint, params, expected_type, *, daily=False, symbol=None):
            nonlocal remaining
            ticker = record["ticker"]
            if time.monotonic() >= deadline:
                if not halt["all"]:
                    receipts.append(
                        dict(ticker=ticker, dataset=dataset, status="time_hold", reserved_credits=0)
                    )
                halt["all"] = True
                held_receipt(ticker, dataset, None)
                return None
            negative = negative_cached(ticker, dataset)
            if negative:
                receipts.append({**negative, "cache_hit": True, "reserved_credits": 0})
                emitted.add((ticker, dataset))
                return None
            previous = cached(ticker, dataset)
            refresh = cache_context(ticker, dataset)
            max_age = min(config.get("fundamentals_max_age_days", 7), 7)
            event = record.get("filing_event_at")
            event_changed = event and event != (previous or {}).get("filing_event_at")
            if config.get("accounting_mode") == "account":
                fresh = previous and (
                    refresh["session_key_verified"]
                    and previous.get("refresh_id") == refresh["refresh_id"]
                    if daily
                    else 0 <= (cutoff - screening.day(previous["as_of"])).days < max_age
                )
            else:
                fresh = previous and (
                    0 <= (cutoff - screening.day(previous["as_of"])).days
                    and (
                        previous.get("refresh_id") == refresh["refresh_id"]
                        or (
                            not daily and (cutoff - screening.day(previous["as_of"])).days < max_age
                        )
                    )
                )
            if (
                fresh
                and not event_changed
                and not (
                    config.get("accounting_mode") != "account"
                    and previous.get("status") != "coverage_hold"
                    and event_refresh_due(previous, event, refresh["refresh_id"])
                )
            ):
                receipt = {**previous, "cache_hit": True, "reserved_credits": 0}
            else:
                if config.get("accounting_mode") == "account":
                    costs = config["request_credit_cost"]
                    cost = costs[dataset] if isinstance(costs, dict) else costs
                    evidence = record.get("evidence", {})
                    identity, valid_identity = screening.evidence(record, "issuer_identity", cutoff)
                    binding = identity.get("value", {})
                    valid_identity = (
                        valid_identity
                        and isinstance(binding, dict)
                        and binding.get("ticker") == ticker
                        and binding.get("issuer_id") == record["issuer_id"]
                    )
                    purpose, reason = "routine_refresh", "scheduled refresh"
                    if event_changed:
                        purpose, reason = "material_event", "filing event changed"
                    elif previous is None and ticker in config.get("held_instrument_tickers", []):
                        purpose, reason = "held_instrument", "held instrument source missing"
                    elif previous is None and dataset == "fundamentals" and not valid_identity:
                        purpose, reason = "missing_identity", "fundamental identity source missing"
                    elif (
                        previous is None
                        and dataset == "prices"
                        and not evidence.get("daily_total_returns_usd")
                    ):
                        purpose, reason = "missing_price_history", "price history source missing"
                    elif (
                        previous is None
                        and dataset in {"sec_annual", "sec_quarterly", "sec_financials"}
                        and not evidence.get("diluted_eps")
                    ):
                        purpose, reason = "missing_accounting", "reported accounting source missing"
                    if halt["all"] or (halt["routine"] and purpose == "routine_refresh"):
                        return held_receipt(ticker, dataset, previous)
                    session.reserve(endpoint, cost, purpose)
                    try:
                        payload = request(session, endpoint, params, expected_type=expected_type)
                    except CapacityLimit:
                        # Fail safe: one CapacityLimit halts every later paid fetch of its class
                        # ("all" also stops 1-credit holding price fetches); intentional, no retry.
                        halt["routine" if purpose == "routine_refresh" else "all"] = True
                        receipts.append(
                            dict(
                                ticker=ticker,
                                dataset=dataset,
                                status="budget_hold",
                                reserved_credits=0,
                            )
                        )
                        return held_receipt(ticker, dataset, previous)
                    except SourceUnavailable:
                        if not session.attempts:
                            raise BudgetHold("accounting hold: paid transport was bypassed")
                        remember_unavailable(
                            dict(
                                ticker=ticker,
                                dataset=dataset,
                                status="coverage_hold",
                                reserved_credits=0,
                                http_attempts=session.attempts,
                                endpoint=endpoint,
                                purpose=purpose,
                                reason=reason,
                            )
                        )
                        return None
                    if not session.attempts:
                        raise BudgetHold("accounting hold: paid transport was bypassed")
                    body = encoded(payload)
                    sha = digest(body)
                    filename = sha + ".raw.json"
                    if not (cache / filename).exists():
                        (cache / filename).write_text(body)
                    receipt = dict(
                        ticker=ticker,
                        dataset=dataset,
                        as_of=as_of,
                        market_as_of=market_as_of
                        if dataset in {"prices", "fx", "splits"}
                        else None,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        **refresh,
                        file=filename,
                        sha256=sha,
                        endpoint=endpoint,
                        parameters=params,
                        symbol=symbol or ticker,
                        filing_event_at=event,
                        cache_hit=False,
                        reserved_credits=0,
                        http_attempts=session.attempts,
                        purpose=purpose,
                        reason=reason,
                        status="raw_unmapped",
                    )
                    atomic(pointer_for(ticker, dataset), receipt)
                else:
                    costs = config["request_credit_cost"]
                    cost = (costs[dataset] if isinstance(costs, dict) else costs) * 3
                    evidence = record.get("evidence", {})
                    identity, valid_identity = screening.evidence(record, "issuer_identity", cutoff)
                    binding = identity.get("value", {})
                    valid_identity = (
                        valid_identity
                        and isinstance(binding, dict)
                        and binding.get("ticker") == ticker
                        and binding.get("issuer_id") == record["issuer_id"]
                    )
                    purpose, reason = "routine_refresh", "scheduled refresh"
                    if event_changed:
                        purpose, reason = "material_event", "filing event changed"
                    elif previous is None and ticker in config.get("held_instrument_tickers", []):
                        purpose, reason = "held_instrument", "held instrument source missing"
                    elif previous is None and dataset == "fundamentals" and not valid_identity:
                        purpose, reason = "missing_identity", "fundamental identity source missing"
                    elif (
                        previous is None
                        and dataset == "prices"
                        and not evidence.get("daily_total_returns_usd")
                    ):
                        purpose, reason = "missing_price_history", "price history source missing"
                    elif (
                        previous is None
                        and dataset in {"sec_annual", "sec_quarterly", "sec_financials"}
                        and not evidence.get("diluted_eps")
                    ):
                        purpose, reason = "missing_accounting", "reported accounting source missing"
                    if halt["all"] or (halt["routine"] and purpose == "routine_refresh"):
                        return held_receipt(ticker, dataset, previous)
                    allowed = reservation_allowed(
                        config,
                        used_today=used,
                        initial_total=initial_used,
                        cost=cost,
                        purpose=purpose,
                    )
                    if not allowed:
                        held_receipt(ticker, dataset, None)
                        receipts.append(
                            dict(
                                ticker=ticker,
                                dataset=dataset,
                                status="budget_hold",
                                reserved_credits=0,
                            )
                        )
                        # Retain previous evidence with its actual old timestamp.
                        if previous is None:
                            return None
                        receipt = {
                            **previous,
                            "cache_hit": True,
                            "reserved_credits": 0,
                            "refresh_held": True,
                        }
                    else:
                        if remaining is None:
                            account = request(session, "user", {}, expected_type=dict)
                            safe, effective, remaining = account_usage(
                                account,
                                cutoff,
                                config.get("account_headroom", 25000),
                                server_day=session.server_day,
                            )
                            receipts.append(
                                dict(
                                    dataset="account_preflight",
                                    safe_account=safe,
                                    effective_current_usage=effective,
                                    base_remaining_after_headroom=remaining,
                                    reserved_credits=0,
                                    status="checked",
                                )
                            )
                            del account
                        if remaining < cost:
                            raise BudgetHold(
                                "account remaining quota below conservative request "
                                "reservation and headroom"
                            )
                        reserve(cost)
                        remaining -= cost
                        session.reserve(endpoint, cost)
                        try:
                            payload = request(
                                session, endpoint, params, expected_type=expected_type
                            )
                        except SourceUnavailable:
                            remember_unavailable(
                                dict(
                                    ticker=ticker,
                                    dataset=dataset,
                                    status="coverage_hold",
                                    reserved_credits=cost,
                                    endpoint=endpoint,
                                    purpose=purpose,
                                    reason=reason,
                                )
                            )
                            return None
                        body = encoded(payload)
                        sha = digest(body)
                        filename = sha + ".raw.json"
                        if not (cache / filename).exists():
                            (cache / filename).write_text(body)
                        receipt = dict(
                            ticker=ticker,
                            dataset=dataset,
                            as_of=as_of,
                            market_as_of=market_as_of
                            if dataset in {"prices", "fx", "splits"}
                            else None,
                            retrieved_at=datetime.now(UTC).isoformat(),
                            **refresh,
                            file=filename,
                            sha256=sha,
                            endpoint=endpoint,
                            parameters=params,
                            symbol=symbol or ticker,
                            filing_event_at=event,
                            cache_hit=False,
                            reserved_credits=cost,
                            purpose=purpose,
                            reason=reason,
                            status="raw_unmapped",
                        )
                        atomic(pointer_for(ticker, dataset), receipt)
            if dataset in {"prices", "fx", "splits"} and config.get("market_as_of"):
                receipt = {**receipt, "market_as_of": market_as_of}
            if not receipt.get("cache_hit"):
                paid.add(ticker)
            receipts.append(receipt)
            emitted.add((ticker, dataset))
            return receipt

        def fetch_public(record):
            options = config.get("sec_public", {})
            if time.monotonic() >= deadline:
                held.add(record["ticker"])
                receipts.append(
                    dict(
                        ticker=record["ticker"],
                        dataset="sec_public_concept",
                        status="time_hold",
                        reserved_credits=0,
                    )
                )
                return
            if (
                not options.get("enabled")
                or not record["ticker"].endswith(".US")
                or record.get("kind") == "etf"
                or (
                    record.get("evidence", {}).get("pe")
                    and not (
                        record.get("companyfacts_share_basis") or record.get("eps_listing_basis")
                    )
                )
            ):
                return
            cik = str(record.get("identifiers", {}).get("cik", ""))
            if not cik.isdigit() or len(str(int(cik))) > 10:
                return
            companyfacts = bool(
                record.get("companyfacts_share_basis") or record.get("eps_listing_basis")
            )
            ticker = record["ticker"]
            dataset = "sec_companyfacts" if companyfacts else "sec_public_concept"
            negative = negative_cached(ticker, dataset)
            if negative:
                receipts.append(
                    {
                        **negative,
                        "cache_hit": True,
                        "reserved_credits": 0,
                        "reserved_public_requests": 0,
                    }
                )
                emitted.add((ticker, dataset))
                return
            previous = cached(ticker, dataset)
            refresh = cache_context(ticker, dataset)
            event = record.get("filing_event_at")
            if (
                previous
                and previous.get("cik") == str(int(cik))
                and event == previous.get("filing_event_at")
                and not event_refresh_due(previous, event, refresh["refresh_id"])
                and (cutoff - screening.day(previous["as_of"])).days
                < options.get("max_age_days", 45)
            ):
                receipts.append(
                    {
                        **previous,
                        "cache_hit": True,
                        "reserved_credits": 0,
                        "reserved_public_requests": 0,
                    }
                )
                emitted.add((ticker, dataset))
                return
            public_key = "public_requests:" + day_key
            if (
                ledger.get(public_key, 0) + 3 > options["max_requests_per_day"]
                or ledger.get("public_quarantine_day") == day_key
            ):
                held.add(ticker)
                receipts.append(
                    dict(
                        ticker=ticker,
                        dataset=dataset,
                        status="public_budget_or_quarantine_hold",
                        reserved_credits=0,
                    )
                )
                return
            ledger[public_key] = ledger.get(public_key, 0) + 3
            atomic(ledger_path, ledger)
            endpoint = (
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
                if companyfacts
                else f"https://data.sec.gov/api/xbrl/companyconcept/CIK{int(cik):010d}/us-gaap/EarningsPerShareDiluted.json"
            )
            public_session = PublicSECSession(
                raw_session, ledger, ledger_path, options["user_agent"], deadline
            )
            try:
                payload = public_request(
                    public_session, endpoint, {}, "SEC public", expected_type=dict
                )
            except PublicSECUnavailable as exc:
                if exc.status in {401, 403, 429}:
                    ledger["public_quarantine_day"] = day_key
                    atomic(ledger_path, ledger)
                receipts.append(
                    dict(
                        ticker=ticker,
                        dataset=dataset,
                        status="coverage_hold",
                        reserved_credits=0,
                        reserved_public_requests=3,
                        endpoint=endpoint,
                        reason=str(exc),
                    )
                )
                emitted.add((ticker, dataset))
                return
            body = encoded(payload)
            sha = digest(body)
            filename = sha + ".raw.json"
            if not (cache / filename).exists():
                (cache / filename).write_text(body)
            receipt = dict(
                ticker=ticker,
                dataset=dataset,
                as_of=as_of,
                retrieved_at=datetime.now(UTC).isoformat(),
                **refresh,
                file=filename,
                sha256=sha,
                cik=str(int(cik)),
                endpoint=endpoint,
                filing_event_at=event,
                cache_hit=False,
                reserved_credits=0,
                reserved_public_requests=3,
                status="raw_unmapped",
            )
            # Validate source identity and units before publishing the reusable pointer.
            if companyfacts:
                screen_sources.normalize_sec_companyfacts(payload, record, receipt)
            elif not screen_sources.duration_eps_facts(payload, record, receipt):
                receipt.update(status="coverage_hold", reason="no eligible USD quarter facts")
            atomic(pointer_for(ticker, dataset), receipt)
            paid.add(ticker)
            receipts.append(receipt)
            emitted.add((ticker, dataset))

        aux = set(config.get("held_instrument_tickers", []))
        # Held positions keep their deeper fetches even when the new-candidate price gate fails.
        exempt = aux | set(config.get("holding_tickers", []))
        by_ticker = {record["ticker"]: record for record in records}
        # coverage calendar lane: filing events from one retained global window (None = legacy)
        events_seam = coverage_runtime.replace_filing_event_calls(
            config, root, as_of, now=datetime.now(UTC)
        )

        def currency_of(record):
            meta = cached(record["ticker"], "fundamentals")
            if meta:
                return listing_identity(read(cache / meta["file"]), record).get("CurrencyCode")
            value, valid = screening.evidence(record, "listing_metadata", cutoff)
            return (value.get("value") or {}).get("CurrencyCode") if valid else None

        def price_gate(record):
            """Cache-only raw-price admission; never pays. Identity conflicts propagate."""
            ticker = record["ticker"]
            prices = cached(ticker, "prices")
            if not prices:
                # coverage lane: a retained bulk raw close can admit/exclude with no paid call
                bulk = coverage_runtime.enabled(config) and (
                    coverage_runtime.price_state_from_evidence(record, as_of, datetime.now(UTC))
                )
                return bulk or dict(status="unknown", needs="price", reason="price_missing")
            if config.get("market_as_of"):
                prices = {**prices, "market_as_of": market_as_of}
            meta = cached(ticker, "fundamentals")
            kwargs = dict(
                fundamentals=read(cache / meta["file"]) if meta else None,
                fundamentals_receipt=meta,
                cutoff=as_of,
                now=datetime.now(UTC).isoformat(),  # re-validate the native session at consumption
            )
            # ponytail: re-reads cached payloads per call; memoize if profiling shows it.
            payload = read(cache / prices["file"])
            state = screen_sources.price_state(record, payload, prices, **kwargs)
            if state["needs"] == "fx":
                currency = currency_of(record)
                code = screen_sources.MINOR_UNITS.get(currency, (currency,))[0]
                fx = cached(f"{code}USD.FOREX", "fx")
                if fx:
                    state = screen_sources.price_state(
                        record,
                        payload,
                        prices,
                        fx=read(cache / fx["file"]),
                        fx_receipt=fx,
                        **kwargs,
                    )
            return {**state, "cached_as_of": prices.get("as_of", "")}

        def exclude(ticker, state):
            if ticker in excluded:
                return
            excluded.add(ticker)
            receipts.append(
                dict(
                    ticker=ticker,
                    dataset="price_gate",
                    status="price_excluded",
                    price_usd=state["price_usd"],
                    reason=state["reason"] or "price_below_minimum",
                    expected_session=state["expected_session"],
                    reserved_credits=0,
                )
            )

        def fetch_fx(record, code, window):
            ticker, fx_symbol = record["ticker"], f"{code}USD.FOREX"
            if (ticker, "fx") in emitted:
                return
            # Reuse one paid FX history per currency.
            fx = fetch(
                {"ticker": fx_symbol},
                "fx",
                f"eod/{fx_symbol}",
                window,
                list,
                daily=True,
                symbol=fx_symbol,
            )
            if fx:
                receipts.append(
                    {
                        **fx,
                        "ticker": ticker,
                        "dataset": "fx",
                        "reserved_credits": 0,
                        "cache_hit": True,
                    }
                )
                emitted.add((ticker, "fx"))

        costs = config["request_credit_cost"]
        fund_cost = (costs["fundamentals"] if isinstance(costs, dict) else costs) * (
            1 if config.get("accounting_mode") == "account" else 3
        )
        fraction = bootstrap_fraction(config)
        working = config.get(
            "daily_working_credit_limit", min(config["daily_credit_limit"], WORKING_CREDITS)
        )
        # Cold metadata bootstrap may use at most this share of the working budget per run;
        # a zero-credit cap yields zero slots; any positive cap keeps at least one slot.
        cap = int(working * fraction)
        slots = max(1, cap // fund_cost) if cap > 0 else 0

        # Cache-only pre-pass: sub-minimum names are excluded before any paid request.
        admission, queue = {}, []
        for ticker in selected:
            record = by_ticker[ticker]
            if identity_held(record):
                identity_blocked.add(ticker)
                continue
            try:
                state = price_gate(record)
            except SourceIdentityConflict as exc:
                receipts.append(
                    quarantine_identity(root, record, exc, cached(ticker, "fundamentals"))
                )
                identity_blocked.add(ticker)
                continue
            admission[ticker] = state
            if state["status"] == "fail":
                exclude(ticker, state)
            if state["status"] != "fail" or ticker in exempt:
                queue.append(ticker)
        # Actual holdings, then portfolio instruments, then warm names (cheap price refresh),
        # then cold; oldest cache first. Holdings therefore get bootstrap slots first.
        holdings = set(config.get("holding_tickers", []))
        priority = set(config.get("priority_tickers", []))  # coverage jobs: after holdings only
        queue.sort(
            key=lambda t: (
                t not in holdings,
                t not in aux,
                t not in priority,
                not pointer_for(t, "fundamentals").exists()
                and not screening.evidence(by_ticker[t], "listing_metadata", cutoff)[1],
                admission[t].get("cached_as_of", ""),
            )
        )
        network_stopped = False
        for ticker in queue:
            record = by_ticker[ticker]
            fundamental = None
            try:
                eodhd.symbol(ticker)
                first = str((pd.Timestamp(cutoff) - pd.DateOffset(years=6)).date())
                window = {"from": first, "to": market_as_of, "period": "d", "order": "a"}
                if (
                    not pointer_for(ticker, "fundamentals").exists()
                    and not (screening.evidence(record, "listing_metadata", cutoff)[1])
                ):
                    # Only metadata can admit a cold name: capped bootstrap before its price.
                    if slots <= 0:
                        # Without metadata the price cannot admit the name: pay nothing.
                        held.add(ticker)
                        continue
                    slots -= 1
                    fundamental = fetch(record, "fundamentals", f"fundamentals/{ticker}", {}, dict)
                    if fundamental:
                        listing_identity(read(cache / fundamental["file"]), record)
                        screen_sources.identity_metadata(
                            read(cache / fundamental["file"]), record, fundamental
                        )
                # One-credit price before the 10-credit fundamentals refresh and filings.
                fetch(record, "prices", f"eod/{ticker}", window, list, daily=True)
                state = price_gate(record)
                if state["needs"] == "fx" and config.get("fresh_sources"):
                    currency = currency_of(record)
                    fetch_fx(
                        record, screen_sources.MINOR_UNITS.get(currency, (currency,))[0], window
                    )
                    state = price_gate(record)
                if state["status"] == "pass":
                    verified.add(ticker)
                if state["status"] == "fail":
                    exclude(ticker, state)
                    if ticker not in exempt:
                        continue
                if state["status"] == "unknown" and state["needs"] is None:
                    # Fetching cannot make this provable: no deeper paid work.
                    unprovable.add(ticker)
                    if ticker not in exempt:
                        continue
                if ticker in held:
                    continue
                done.add(ticker)
                # Earnings (1 credit) only for names that are neither excluded nor held.
                if config.get("fresh_sources"):
                    covered = events_seam is not None and events_seam.covers(ticker)
                    events = (
                        None
                        if covered
                        else fetch(
                            record,
                            "earnings",
                            "calendar/earnings",
                            {
                                "symbols": ticker,
                                "from": str((pd.Timestamp(cutoff) - pd.Timedelta(days=7)).date()),
                                "to": str((pd.Timestamp(cutoff) + pd.Timedelta(days=7)).date()),
                            },
                            dict,
                            daily=True,
                        )
                    )
                    seam_date = events_seam.date(ticker) if covered else None
                    dates = [seam_date] if seam_date else []
                    if events:
                        data = read(cache / events["file"])
                        if not isinstance(data.get("earnings"), list):
                            raise ValueError("earnings calendar requires documented earnings array")
                        dates = [
                            r.get("report_date")
                            for r in data["earnings"]
                            if r.get("code") == ticker
                            and screening.day(r.get("report_date"))
                            and r["report_date"] <= as_of
                        ]
                    if dates:
                        record["filing_event_at"] = max(dates + [record.get("filing_event_at", "")])
                        # Price was fetched before the event was known: keep its pointer
                        # consistent so the event does not force a refetch every run.
                        price_pointer = pointer_for(ticker, "prices")
                        if price_pointer.exists():
                            atomic(
                                price_pointer,
                                {
                                    **read(price_pointer),
                                    "filing_event_at": record["filing_event_at"],
                                },
                            )
                if fundamental is None:
                    fundamental = fetch(record, "fundamentals", f"fundamentals/{ticker}", {}, dict)
                    if fundamental:
                        listing_identity(read(cache / fundamental["file"]), record)
                        screen_sources.identity_metadata(
                            read(cache / fundamental["file"]), record, fundamental
                        )
                if config.get("fresh_sources"):
                    fetch(
                        record,
                        "splits",
                        f"splits/{ticker}",
                        {"from": first, "to": market_as_of},
                        list,
                        daily=True,
                    )
                    if fundamental:
                        payload = read(cache / fundamental["file"])
                        currency = listing_identity(payload, record).get("CurrencyCode")
                        if (
                            currency
                            and currency not in {"USD", "GBX", "GBp", "ILA", "ZAC"}
                            and re.fullmatch(r"[A-Z]{3}", currency)
                        ):
                            fetch_fx(record, currency, window)
            except SourceIdentityConflict as exc:
                receipts.append(quarantine_identity(root, record, exc, fundamental))
                identity_blocked.add(ticker)
            except BudgetHold as exc:
                if str(exc) != "network time budget exhausted":
                    raise
                receipts.append(
                    dict(ticker=ticker, dataset="transport", status="time_hold", reserved_credits=0)
                )
                held.add(ticker)
                halt["all"] = network_stopped = True
                break
        # Exhaust identity and market coverage before deeper filing requests.
        deep = [t for t in queue if t in done and t not in held]
        if network_stopped:
            held.update(deep)  # price stage finished but filings never reached
            deep = []
        for position, ticker in enumerate(deep):
            record = by_ticker[ticker]
            if identity_held(record):
                continue
            try:
                fundamental = cached(ticker, "fundamentals")
                if fundamental:
                    fetch_public(record)
                if (
                    config.get("fresh_sources")
                    and ticker.endswith(".US")
                    and record.get("kind") != "etf"
                ):
                    if config.get("sec_financials_enabled"):
                        financials = fetch(
                            record, "sec_financials", f"sec-companies/{ticker}/financials", {}, dict
                        )
                        if financials:
                            body = read(cache / financials["file"])
                            data = body.get("data")
                            if (
                                not isinstance(data, dict)
                                or not isinstance(data.get("company"), dict)
                                or not all(
                                    isinstance(data.get(k), list) for k in ("annual", "quarterly")
                                )
                            ):
                                raise ValueError(
                                    "SEC financials requires company, annual and quarterly data"
                                )
                            rows = data["annual"] + data["quarterly"]
                            if any(not isinstance(row, dict) for row in rows):
                                raise ValueError("SEC financials rows must be objects")
                            financial_keys = {
                                "revenue",
                                "net_income",
                                "eps_diluted",
                                "operating_cash_flow",
                                "capital_expenditure",
                            }
                            if not any(
                                row.get(key) is not None for row in rows for key in financial_keys
                            ):
                                financials.update(
                                    status="coverage_hold",
                                    reason="financial statement values unavailable",
                                )
                                atomic(pointer_for(ticker, "sec_financials"), financials)
                    for dataset, form in (("sec_annual", "10k"), ("sec_quarterly", "10q")):
                        filing = fetch(
                            record,
                            dataset,
                            f"sec-filings/{ticker}/{form}",
                            {"page[limit]": 100, "page[offset]": 0},
                            dict,
                            daily=False,
                        )
                        if filing:
                            data = read(cache / filing["file"])
                            if not isinstance(data.get("data"), list):
                                raise ValueError("SEC filings require documented data array")
                            dates = [
                                r.get("filed_at")
                                for r in data["data"]
                                if screening.day(r.get("filed_at")) and r["filed_at"] <= as_of
                            ]
                            if dates:
                                record["filing_event_at"] = max(
                                    dates + [record.get("filing_event_at", "")]
                                )
                            if data.get("links", {}).get("next"):
                                receipts.append(
                                    dict(
                                        ticker=ticker,
                                        dataset=dataset,
                                        status="pagination_hold",
                                        reserved_credits=0,
                                        reason="First 100 filings only; older coverage unknown",
                                    )
                                )

            except SourceIdentityConflict as exc:
                receipts.append(quarantine_identity(root, record, exc, fundamental))
            except BudgetHold as exc:
                if str(exc) != "network time budget exhausted":
                    raise
                receipts.append(
                    dict(ticker=ticker, dataset="transport", status="time_hold", reserved_credits=0)
                )
                halt["all"] = True
                held.update(deep[position:])
                break
        # Rehydrate every retained issuer, not just today's batch. No network is used here.
        datasets = (
            "fundamentals",
            "prices",
            "sec_annual",
            "sec_quarterly",
            "sec_financials",
            "splits",
            "earnings",
            "sec_public_concept",
            "sec_companyfacts",
        )
        for record in records:
            if identity_held(record):
                continue
            fundamental = None
            try:
                for dataset in datasets:
                    if (record["ticker"], dataset) not in emitted:
                        previous = cached(record["ticker"], dataset)
                        if previous:
                            receipts.append(
                                {
                                    **previous,
                                    "cache_hit": True,
                                    "reserved_credits": 0,
                                    **(
                                        {"market_as_of": market_as_of}
                                        if dataset in {"prices", "fx", "splits"}
                                        and config.get("market_as_of")
                                        else {}
                                    ),
                                }
                            )
                fundamental = cached(record["ticker"], "fundamentals")
                if fundamental:
                    currency = listing_identity(read(cache / fundamental["file"]), record).get(
                        "CurrencyCode"
                    )
                    fx = (
                        cached(f"{currency}USD.FOREX", "fx")
                        if currency and currency != "USD"
                        else None
                    )
                    if fx and not any(
                        r.get("ticker") == record["ticker"] and r.get("dataset") == "fx"
                        for r in receipts
                    ):
                        receipts.append(
                            {
                                **fx,
                                "ticker": record["ticker"],
                                "dataset": "fx",
                                "cache_hit": True,
                                "reserved_credits": 0,
                            }
                        )
            except SourceIdentityConflict as exc:
                receipts.append(quarantine_identity(root, record, exc, fundamental))

    def outcome(ticker):
        if ticker in identity_blocked:
            return "identity_held"
        if ticker in excluded:
            return "price_excluded"
        if ticker in held or ticker not in done and ticker not in unprovable:
            return "held"
        if ticker in unprovable:
            return "price_unprovable"
        return "refreshed" if ticker in paid else "cache_current"

    receipts.outcomes = {ticker: outcome(ticker) for ticker in selected}
    receipts.verified = sorted(verified & set(selected))
    return receipts


def listing_identity(payload, record):
    return screen_sources.listing_identity(payload, record)


def quarantine_identity(root, record, conflict, receipt=None):
    source = conflict.receipt or receipt or {}
    quarantine = dict(
        ticker=record["ticker"],
        issuer_id=record["issuer_id"],
        expected=conflict.expected,
        observed=conflict.observed,
        reason=str(conflict),
        as_of=datetime.now(UTC).date().isoformat(),
        dataset="identity_quarantine",
        status="identity_quarantine",
        reserved_credits=0,
        source={
            k: source.get(k)
            for k in ("dataset", "endpoint", "retrieved_at", "as_of", "file", "sha256")
        },
        withheld_evidence_keys=sorted(record.get("evidence", {})),
    )
    if source.get("file"):
        quarantine.update(file=source["file"], sha256=source["sha256"])
    record["identity_quarantine"], record["evidence"] = quarantine, {}
    atomic(
        Path(root) / "source-cache" / (digest(record["ticker"]) + ".identity-quarantine.json"),
        quarantine,
    )
    return quarantine


def invalidate_unproved_ttm(record):
    """Hold old provider multiples and TTM values without applicable listing-class proof."""
    evidence = record.get("evidence", {})
    headline = evidence.get("pe", {})
    if headline.get("valuation_basis") == "provider trailing TTM P/E":
        evidence.setdefault("diagnostic_provider_pe", headline)
        evidence.pop("pe")
    sector = evidence.get("sector_pe", {})
    if "provider trailing TTM P/E" in str(sector.get("peer_definition", "")):
        sector["verified"] = False
        sector["hold_reason"] = "provider headline peer multiples are diagnostic only"
    if sector.get("owner") == "eodhd_screen_service" and sector.get("listing_basis_version") != 2:
        sector["verified"] = False
        sector["hold_reason"] = "peer listing proof predates current admission policy"
    legacy_record = {**record, "eps_listing_basis": {}}
    legacy = bool(screen_sources.ttm_listing_basis(legacy_record))

    def admitted(quarters, date):
        date = screening.day(date)
        if not date or not isinstance(quarters, list) or len(quarters) != 4:
            return False
        spans = []
        for quarter in quarters:
            if not isinstance(quarter, dict) or not quarter.get("source"):
                return False
            start, end, filed = (screening.day(quarter.get(k)) for k in ("start", "end", "filed"))
            if not start or not end or not filed or not start <= end <= filed < date:
                return False
            if not 60 <= (end - start).days + 1 <= 120:
                return False
            spans.append((start, end))
        spans.sort()
        if any((b[0] - a[1]).days != 1 for a, b in zip(spans, spans[1:])):
            return False
        if not 330 <= (spans[-1][1] - spans[0][0]).days + 1 <= 390:
            return False
        return bool(screen_sources.ttm_listing_basis(record, str(spans[0][0]), str(date)))

    for key in ("own_pe_monthly", "pe"):
        item = evidence.get(key, {})
        if item.get("valuation_basis") != "reported US GAAP diluted TTM P/E" or legacy:
            continue
        rows = item.get("value") if key == "own_pe_monthly" else None
        if isinstance(rows, list) and rows:
            valid = all(
                isinstance(row, dict) and admitted(row.get("quarters"), row.get("price_date"))
                for row in rows
            )
        else:
            valid = key == "pe" and admitted(item.get("quarters"), item.get("period"))
        if not valid:
            item["verified"] = False
            item["hold_reason"] = "issuer EPS applicability to listing class is unverified"


def merge_evidence(record, key, value):
    existing = record.setdefault("evidence", {}).get(key)
    value["owner"] = "eodhd_screen_service"
    if existing is None or existing.get("owner") == "eodhd_screen_service":
        record["evidence"][key] = value
        return

    def comparable(item):
        raw = item.get("value")
        if isinstance(raw, list) and all(isinstance(r, dict) for r in raw):
            raw = [
                {
                    k: r.get(k)
                    for k in ("period", "date", "value", "currency", "unit", "units", "basis")
                }
                for r in raw
            ]
        if isinstance(raw, list) and raw and all(isinstance(r, dict) for r in raw):
            raw = sorted(raw, key=lambda r: str(r.get("period") or r.get("date")))
        return {
            "value": raw,
            **{
                k: item.get(k)
                for k in (
                    "currency",
                    "period",
                    "unit",
                    "units",
                    "basis",
                    "statement_source",
                    "frequency",
                    "return_basis",
                )
            },
        }

    if comparable(existing) != comparable(value):
        existing["verified"] = False
        existing["conflict"] = {
            "reason": "curated/provider contradiction; human reconciliation required",
            "provider_evidence": value,
        }


def reconcile_statements(provider, sec, key):
    """Different history lengths are coverage, disagreements on shared years are conflicts."""
    if provider.get("conflict"):
        return provider
    left = {row["period"]: row for row in provider["value"]}
    right = {row["period"]: row for row in sec["value"]}
    conflicting = []
    for period in left.keys() & right.keys():
        a, b = left[period], right[period]
        values = (abs(a["value"]), abs(b["value"])) if key == "capex" else (a["value"], b["value"])
        if a["currency"] != b["currency"] or values[0] != values[1]:
            conflicting.append(period)
    if conflicting:
        return {
            **sec,
            "verified": False,
            "conflict": {
                "reason": (
                    "provider/SEC contradiction on shared fiscal periods; "
                    "human reconciliation required"
                ),
                "periods": sorted(conflicting),
                "provider_evidence": provider,
                "sec_evidence": sec,
            },
        }
    return sec


def normalize_fundamentals(payload, record, receipt):
    """Map only published annual statement fields with explicit filing date/currency.

    Caller retains raw source; calendar EPS, balance-sheet ordinary shares and headline
    historical multiples cannot satisfy diluted-EPS/share or matched historical-P/E gates.
    """
    general = listing_identity(payload, record)
    observed = str(screening.day(receipt["retrieved_at"]))
    fields = {
        "revenue": ("Income_Statement", "totalRevenue"),
        "net_income": ("Income_Statement", "netIncome"),
        "ebitda_annual": ("Income_Statement", "ebitda"),
        "operating_income": ("Income_Statement", "operatingIncome"),
        "debt_annual": ("Balance_Sheet", "shortLongTermDebtTotal"),
        "cash_annual": ("Balance_Sheet", "cashAndShortTermInvestments"),
        "ocf": ("Cash_Flow", "totalCashFromOperatingActivities"),
        "capex": ("Cash_Flow", "capitalExpenditures"),
    }
    normalized = screen_sources.identity_metadata(payload, record, receipt)
    if record.get("identifiers", {}).get("isin") and general.get("ISIN"):
        normalized["issuer_identity"] = {
            "value": {
                "ticker": record["ticker"],
                "issuer_id": record["issuer_id"],
                "isin": general["ISIN"],
            },
            "verified": True,
            "source": f"sources/{receipt['file']}#General.ISIN+Exchange+Code",
            "period": observed,
            "available_at": observed,
            "observed_at": observed,
        }
    for key, (section, field) in fields.items():
        statement = payload.get("Financials", {}).get(section, {})
        rows = []
        for fiscal, row in (statement.get("yearly") or {}).items():
            period = screening.day(row.get("date", fiscal))
            available = screening.day(row.get("filing_date"))
            currency = row.get("currency_symbol") or statement.get("currency_symbol")
            raw = row.get(field)
            try:
                val = float(raw) if not isinstance(raw, bool) else None
            except (ValueError, TypeError):
                val = None
            if (
                period
                and available
                and period <= screening.day(receipt["as_of"])
                and available <= screening.day(receipt["as_of"])
                and currency
                and screening.number(val) is not None
            ):
                rows.append(
                    {
                        "period": str(period),
                        "available_at": str(available),
                        "currency": currency,
                        "value": val,
                        "source": f"EODHD Financials.{section}.yearly.{fiscal}.{field}",
                    }
                )
        if rows:
            normalized[key] = {
                "value": rows,
                "verified": True,
                "source": f"sources/{receipt['file']}#Financials.{section}",
                "period": sorted(r["period"] for r in rows),
                "available_at": observed,
                "observed_at": observed,
            }

    def aligned(left, right, operation):
        lrows = normalized.get(left, {}).get("value", [])
        rrows = {r["period"]: r for r in normalized.get(right, {}).get("value", [])}
        out = []
        for lrow in lrows:
            rrow = rrows.get(lrow["period"])
            if rrow and lrow["currency"] == rrow["currency"]:
                value = operation(lrow["value"], rrow["value"])
                if value is not None:
                    out.append(
                        {
                            **lrow,
                            "value": value,
                            "available_at": max(lrow["available_at"], rrow["available_at"]),
                            "source": lrow["source"] + " ; " + rrow["source"],
                        }
                    )
        return sorted(out, key=lambda r: r["period"])

    margins = aligned("operating_income", "revenue", lambda x, y: x / y if y > 0 else None)
    if margins:
        normalized["operating_margin"] = {**normalized["revenue"], "value": margins}
    netdebt = aligned("debt_annual", "cash_annual", lambda x, y: x - y)
    for key, rows in (
        ("net_debt", netdebt),
        ("ebitda", normalized.get("ebitda_annual", {}).get("value", [])),
    ):
        if rows:
            latest = sorted(rows, key=lambda r: r["period"])[-1]
            normalized[key] = {**latest, "verified": True, "observed_at": observed}
    return normalized


def normalize_prices(payload, fundamentals, record, receipt, sessions):
    """Compatibility entry point; the fresh service supplies explicit FX to the global adapter."""
    listing_identity(fundamentals, record)
    return screen_sources.normalize_global_prices(payload, fundamentals, record, receipt).get(
        "daily_total_returns_usd"
    )


def artifacts(inputs, config, selected, previous):
    results = screening.screen(
        inputs, config["as_of"], selected, workers=config.get("evaluation_workers", 1)
    )
    by_ticker = {r["ticker"]: r for r in previous}
    changes = []
    for result in results:
        before = by_ticker.get(result["ticker"])
        old = {g["gate"]: g["result"] for g in before["gates"]} if before else {}
        changed = [
            {"gate": g["gate"], "before": old.get(g["gate"]), "after": g["result"]}
            for g in result["gates"]
            if old.get(g["gate"]) != g["result"]
        ]
        if changed or before is None or before["stage"] != result["stage"]:
            changes.append(
                {
                    "ticker": result["ticker"],
                    "prior_stage": before["stage"] if before else None,
                    "stage": result["stage"],
                    "gates": changed,
                }
            )
    queue = [
        {
            "ticker": r["ticker"],
            "owner": r["review_owner"],
            "status": "pending_research_review",
            "stage": r["stage"],
            "blockers": r["blockers"],
            "reopening_reason": r["reopening_reason"],
            "identity_quarantine": r.get("identity_quarantine"),
            "event_review": next(
                (
                    record.get("event_review_required")
                    for record in inputs["universe"]
                    if record["ticker"] == r["ticker"]
                ),
                None,
            ),
        }
        for r in results
    ]
    queued = {item["ticker"] for item in queue}
    for record in inputs["universe"]:
        quarantine = record.get("identity_quarantine")
        if quarantine and record["ticker"] not in queued:
            queue.append(
                dict(
                    ticker=record["ticker"],
                    owner=record.get("review_owner", "unassigned"),
                    status="pending_research_review",
                    stage="screen_hold",
                    blockers=["source_identity"],
                    reopening_reason=quarantine["reason"],
                    identity_quarantine=quarantine,
                )
            )
    cutoff = screening.day(config["as_of"])
    records = inputs["universe"]
    fields = sorted({name for record in records for name in record.get("evidence", {})})
    identity_verified = set()
    for record in records:
        identity, valid = screening.evidence(record, "issuer_identity", cutoff)
        binding = identity.get("value", {})
        if (
            valid
            and isinstance(binding, dict)
            and binding.get("ticker") == record["ticker"]
            and binding.get("issuer_id") == record["issuer_id"]
        ):
            identity_verified.add(record["issuer_id"])
    selected_gates = [{g["gate"]: g["result"] for g in r["gates"]} for r in results]
    selected_by_ticker = {r["ticker"]: r for r in results}
    accounting_complete = set()
    for record in records:
        if record.get("kind") == "etf":
            continue
        evaluation = selected_by_ticker.get(record["ticker"]) or screening.evaluate(
            record, config["as_of"]
        )
        gates = {g["gate"]: g["result"] for g in evaluation["gates"]}
        if all(
            gates.get(name) in {"pass", "fail", "not_applicable"}
            for name in ("revenue_3y_cagr", "diluted_eps_3y_cagr", "fcf_3y_cagr")
        ):
            accounting_complete.add(record["issuer_id"])
    records_by_ticker = {r["ticker"]: r for r in records}
    usable_fields = {name: set() for name in fields}
    for result in results:
        record = records_by_ticker[result["ticker"]]
        if record["issuer_id"] not in identity_verified:
            continue
        for name, item in record.get("evidence", {}).items():
            if item.get("value") is None or not screening.evidence(record, name, cutoff)[1]:
                continue
            if any(
                g["result"] in {"pass", "fail"} and item.get("source") in g.get("source", [])
                for g in result["gates"]
            ):
                usable_fields[name].add(record["issuer_id"])
    files = {
        "results.json": encoded(results),
        "identity-quarantines.json": encoded(
            [r["identity_quarantine"] for r in inputs["universe"] if r.get("identity_quarantine")]
        ),
        "daily-changes.json": encoded(changes),
        "coverage.json": encoded(
            {
                "universe_unique_issuers": len({r["issuer_id"] for r in inputs["universe"]}),
                "selected_unique_issuers": len({r["issuer_id"] for r in results}),
                "selected_instruments": len(results),
                "auxiliary_instruments": len(inputs.get("portfolio", {}).get("instruments", [])),
                "screen_eligible_issuers": sum(r["stage"] != "screen_hold" for r in results),
                "screen_excluded_issuers": sum(
                    any(g["category"] == "screen" and g["result"] == "fail" for g in r["gates"])
                    for r in results
                ),
                "screen_unknown_issuers": sum(
                    any(g["category"] == "screen" and g["result"] == "unknown" for g in r["gates"])
                    for r in results
                ),
                "identity_verified_issuers": len(identity_verified),
                "accounting_complete_issuers": len(accounting_complete),
                "data_verified_issuers": len(identity_verified & accounting_complete),
                "raw_verified_flags_by_field": {
                    name: sum(
                        record.get("evidence", {}).get(name, {}).get("verified") is True
                        for record in records
                    )
                    for name in fields
                },
                "selected_usable_fields_by_name": {
                    name: len(issuers) for name, issuers in usable_fields.items()
                },
                "fresh_sourced_fields_by_name": {
                    name: sum(
                        screening.evidence(
                            record,
                            name,
                            cutoff,
                            age=7
                            if name in {"daily_total_returns_usd", "market_cap_usd", "pe", "ptbv"}
                            else 45,
                        )[1]
                        for record in records
                    )
                    for name in fields
                },
                "selected_accounting_complete_issuers": sum(
                    r.get("kind") != "etf"
                    and all(
                        gates.get(name) in {"pass", "fail", "not_applicable"}
                        for name in ("revenue_3y_cagr", "diluted_eps_3y_cagr", "fcf_3y_cagr")
                    )
                    for r, gates in zip(results, selected_gates)
                ),
                "last_price_sessions": {
                    r["ticker"]: r.get("evidence", {})
                    .get("daily_total_returns_usd", {})
                    .get("last_completed_session")
                    for r in inputs["universe"] + inputs.get("portfolio", {}).get("instruments", [])
                },
                "unknown_gate_observations": sum(
                    g["result"] == "unknown" for r in results for g in r["gates"]
                ),
                "verified_gate_observations": sum(
                    g["result"] in {"pass", "fail"} for r in results for g in r["gates"]
                ),
            }
        ),
        "manual-review-queue.json": encoded(queue),
    }
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "ticker",
            "issuer_id",
            "kind",
            "stage",
            "pareto_tier",
            "category",
            "gate",
            "value",
            "threshold",
            "result",
            "source",
            "period",
            "freshness",
            "score",
            "composite_percentile",
            "ranking_cohort",
            "composite_cohort_n",
        ]
    )
    for r in results:
        for g in r["gates"]:
            writer.writerow(
                [
                    r["ticker"],
                    r["issuer_id"],
                    r["kind"],
                    r["stage"],
                    r["pareto_tier"],
                    g["category"],
                    g["gate"],
                    json.dumps(g["value"], sort_keys=True),
                    g["threshold"],
                    g["result"],
                    json.dumps(g["source"], sort_keys=True),
                    json.dumps(g["period"], sort_keys=True),
                    json.dumps(g["freshness"], sort_keys=True),
                    r.get("score"),
                    r.get("composite_percentile"),
                    r.get("ranking_cohort"),
                    r.get("composite_cohort_n"),
                ]
            )
    files["gates.csv"] = buffer.getvalue()
    rows = []
    for r in results:
        values = [
            r.get(k)
            for k in (
                "ticker",
                "kind",
                "cap_bucket",
                "stage",
                "score",
                "composite_percentile",
                "blockers",
            )
        ]
        if r.get("identity_quarantine"):
            values[-1] = str(values[-1]) + "; " + r["identity_quarantine"]["reason"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    synthetic = inputs.get("synthetic", False)
    files["report.html"] = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Daily evidence screen</title><style>body{font:16px system-ui;margin:2rem}"
        "td,th{padding:.6rem;border:1px solid #ccc;text-align:left}"
        "table{border-collapse:collapse}</style>"
        f"<h1>{'SYNTHETIC DEMONSTRATION — ' if synthetic else ''}Daily evidence screen</h1>"
        f"<p>Cutoff {html.escape(config['as_of'])}; {len(results)} issuers in daily batch. "
        "AI research review is distinct from human trade approval. No orders. "
        "Scenarios are assumptions, never forecasts. "
        "Retained-universe percentiles and full evidence are in results.json; "
        "gates.csv includes all blockers.</p>"
        "<table><caption>Strict stage advancement; holds and near misses remain separate</caption>"
        "<thead><tr><th>Ticker<th>Kind<th>Size<th>Stage<th>Weighted score"
        "<th>Cohort percentile<th>Blockers"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></html>\n"
    )
    return files


def code_files(extra=()):
    """Frozen code receipt; `extra` adds coverage modules only when a coverage lane is enabled."""
    return {
        f"code/{module}.py": (Path(__file__).parent / f"{module}.py").read_bytes()
        for module in (
            "screening",
            "screen_ranking",
            "screen_service",
            "screen_sources",
            "screen_budget",
            "eodhd",
            "http_data",
            *extra,
        )
    }


def run_health(config, receipts, ledger, capacity=None):
    budget_holds = sum(
        r.get("status") in {"budget_hold", "public_budget_or_quarantine_hold", "time_hold"}
        for r in receipts
    )
    source_failures = sum(
        r.get("status") in {"coverage_hold", "identity_quarantine", "pagination_hold"}
        for r in receipts
    )
    acquisition = config.get("acquisition", {})
    return dict(
        status="budget_hold"
        if budget_holds
        else "source_failure"
        if source_failures
        else "success",
        acquisition_enabled=bool(acquisition.get("enabled")),
        bootstrap_fraction=bootstrap_fraction(acquisition),
        as_of=config["as_of"],
        refresh_epoch=config.get("refresh_epoch"),
        market_cutoffs=config.get("market_cutoffs", {}),
        budget_holds=budget_holds,
        attempted_instruments=len(
            {
                r.get("ticker")
                for r in receipts
                if not r.get("cache_hit")
                and r.get("dataset") != "fx"
                and r.get("ticker")
                and (
                    r.get("http_attempts", 0)
                    or r.get("reserved_credits", 0)
                    or r.get("reserved_public_requests", 0)
                )
            }
        ),
        datasets_attempted=sorted(
            {
                r.get("dataset")
                for r in receipts
                if not r.get("cache_hit")
                and (
                    r.get("http_attempts", 0)
                    or r.get("reserved_credits", 0)
                    or r.get("reserved_public_requests", 0)
                )
                and r.get("dataset")
            }
        ),
        public_requests_reserved_today=ledger.get("public_requests:" + config["as_of"], 0),
        source_failures=source_failures,
        refresh_holds=sum(bool(r.get("refresh_held")) for r in receipts),
        provider_account_credits=next(
            (
                r["effective_current_usage"]
                for r in reversed(receipts)
                if r.get("dataset") in {"account_preflight", "account_usage"}
            ),
            None,
        ),
        provider_http_attempts=sum(
            r.get("http_attempts", 0) for r in receipts if r.get("dataset") == "account_usage"
        ),
        accounting_unknown=bool(
            capacity and screen_budget.blocking_pending(capacity, config["state_root"])
        ),
        accounting_quarantine=(
            screen_budget.quarantine_of(config["state_root"], capacity["pending"])
            if capacity and capacity.get("pending")
            else None
        ),
        legacy_reserved_credits_today=ledger.get(config["as_of"], 0),
        reserved_credits_today=ledger.get(config["as_of"], 0),
        initial_reserved_total=ledger.get("initial_total", 0),
        limits=dict(
            mode=acquisition.get("mode", "initial"),
            initial_credit_limit=acquisition.get("initial_credit_limit"),
            daily_credit_limit=acquisition.get("daily_credit_limit"),
            daily_working_credit_limit=acquisition.get("daily_working_credit_limit"),
            account_headroom=acquisition.get("account_headroom", 10000),
        ),
        latest_price_refresh_at=max(
            (
                r["retrieved_at"]
                for r in receipts
                if r.get("dataset") == "prices" and r.get("retrieved_at")
            ),
            default=None,
        ),
        code_receipt={
            k: digest(v) for k, v in code_files(coverage_runtime.code_modules(config)).items()
        },
        operator_stop=OPERATOR_STOP,
    )


def settle(inputs, selected, outcomes, verified=()):
    """Separate counts; only budget/time-held and unreached names stay unseen.

    refreshed: at least one paid non-cache fetch this run. cache_current: served from fresh
    cache. verified: raw-price admission passed this run. identity_held is not counted
    refreshed, but, like price_excluded/price_unprovable, it still advances rotation so
    never-provable names cannot monopolise the queue. A missing or unknown outcome is held
    (unseen): callers must record outcomes explicitly.
    """
    known = {
        "refreshed",
        "cache_current",
        "price_excluded",
        "price_unprovable",
        "identity_held",
    }
    final = {t: outcomes.get(t) if outcomes.get(t) in known else "held" for t in selected}
    tally = Counter(final.values())
    counts = dict(
        evaluated=len(inputs["universe"]),
        queued=len(selected),
        refreshed=tally["refreshed"],
        cache_current=tally["cache_current"],
        verified=len(set(verified) & set(selected)),
        price_excluded=tally["price_excluded"],
        price_unprovable=tally["price_unprovable"],
        identity_held=tally["identity_held"],
        held=tally["held"],
    )
    return [t for t, v in final.items() if v != "held"], counts


def publish_state(
    root, state, inputs, selected, output, as_of, refreshed=None, counts=None, coverage_counts=None
):
    """`refreshed` (default: all selected) are the only names whose last_seen/rotation advance."""
    seen = state.get("last_seen", {}).copy()
    advanced = selected if refreshed is None else refreshed
    selected_set = set(advanced)
    for record in inputs["universe"]:
        if record["ticker"] in selected_set:
            seen[record["issuer_id"]] = as_of
    atomic(
        root / "state.json",
        {
            "last_good": str(output),
            "last_success": as_of,
            "last_seen": seen,
            "rotation": state.get("rotation", 0) + len(selected_set),
        },
    )
    health = (
        read(output / "health.json") if (output / "health.json").exists() else {"status": "success"}
    )
    atomic(
        root / "status.json",
        {
            "status": health["status"],
            "published": True,
            "last_good": str(output),
            "batch_count": len(selected),
            "counts": {**(counts or dict(queued=len(selected))), **(coverage_counts or {})},
            "health": health,
            **({"batch_estimate": health["batch_estimate"]} if "batch_estimate" in health else {}),
        },
    )


def attach_news(root, inputs, as_of):
    """Freeze per-ticker news review state into records; no news index means no attachment."""
    from wharton import news_events

    state = news_events.load_review_state(root, as_of)
    if state is None:
        return
    for record in inputs["universe"] + inputs.get("portfolio", {}).get("instruments", []):
        days = state.get("tickers", {}).get(record["ticker"].upper())
        record["news_events"] = (
            {"status": state["status"]}
            if state["status"] != "ok"
            else {"status": "not_covered", "bins": {}}
            if days is None
            else {
                "status": "ok",
                "bins": {k: {"latest_open_event_at": d} for k, d in days.items()},
            }
        )


def run(config_path, *, as_of=None):
    config = load_config(config_path)
    config["as_of"] = as_of or config.get("as_of") or datetime.now(UTC).date().isoformat()
    if screening.day(config["as_of"]) is None:
        raise ValueError("invalid as_of")
    root = Path(config["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    run_started = time.monotonic()  # ONE run deadline, shared by coverage jobs and acquire()
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("screen run already active") from None
        state_file = root / "state.json"
        state = read(state_file) if state_file.exists() else {}
        inputs = read(config["input"])
        original_inputs = encoded(inputs)
        attach_news(root, inputs, config["as_of"])
        for record in inputs["universe"] + inputs.get("portfolio", {}).get("instruments", []):
            invalidate_unproved_ttm(record)
        supplements = (
            read(config["quarter_supplements"]) if config.get("quarter_supplements") else None
        )
        if supplements is not None:
            if not isinstance(supplements, dict):
                raise ValueError("quarter_supplements must map ticker to sourced quarterly facts")
            for record in inputs["universe"]:
                if record["ticker"] in supplements:
                    record["quarter_supplements"] = supplements[record["ticker"]]
        if config.get("acquisition", {}).get("enabled"):
            inputs["market_as_of"] = str(pd.Timestamp(config["as_of"]) - pd.Timedelta(days=1))[:10]
            config["acquisition"]["market_as_of"] = inputs["market_as_of"]
            context = refresh_context(inputs, config["as_of"])
            config.update(context)
            config["acquisition"].update(context)
        # Cached listing classifications must be applied before portfolio and peer evaluation.
        if config.get("acquisition", {}).get("enabled"):
            cache = root / "source-cache"
            for record in inputs["universe"] + inputs.get("portfolio", {}).get("instruments", []):
                pointer = cache / (digest(f"{record['ticker']}:fundamentals") + ".json")
                if not pointer.exists():
                    continue
                receipt = read(pointer)
                if (
                    receipt.get("ticker") != record["ticker"]
                    or receipt.get("dataset") != "fundamentals"
                ):
                    raise ValueError("source cache identity mismatch")
                raw = (cache / receipt["file"]).read_bytes()
                if digest(raw) != receipt["sha256"]:
                    raise ValueError("source cache hash mismatch")
                try:
                    listing_identity(json.loads(raw), record)
                except SourceIdentityConflict as exc:
                    quarantine_identity(root, record, exc, receipt)
        # Coverage lanes (no-op without an enabled lane): retained bulk bytes become verified
        # evidence before any evaluation or paid call; retained events feed the durable queue.
        # A coverage failure is a status note, never an aborted run (an UNRESOLVED accounting
        # state is not masked: soft() re-raises it).
        coverage_report, run_lock, coverage_notes, coverage_jobs = {}, None, [], {}
        coverage_live, coverage_registry, coverage_production = False, None, {}
        if coverage_runtime.enabled(config):
            run_lock = coverage_runtime.RunLock.adopt(lock, root / ".lock")
            config["acquisition"]["coverage"] = config["coverage"]
            cov_records = inputs["universe"] + inputs.get("portfolio", {}).get("instruments", [])
            # the pinned registry must verify before any coverage work or spend
            coverage_registry = coverage_runtime.soft(
                "registry", coverage_notes, coverage_runtime.registry_rows, config, default=None
            )
            coverage_live = coverage_registry is not None
            coverage_prior = None  # the previous decision ledger feeds the fixable-gap producer
            if state.get("last_good"):
                try:
                    coverage_prior = read(Path(state["last_good"]) / "results.json")
                except (OSError, ValueError):
                    coverage_prior = None
            if coverage_live:
                coverage_report = coverage_runtime.soft(
                    "rehydrate", coverage_notes, coverage_runtime.rehydrate,
                    config, root, cov_records, config["as_of"], default={}, now=datetime.now(UTC),
                )  # fmt: skip
                coverage_runtime.soft(
                    "queue", coverage_notes, coverage_runtime.queue_step,
                    config, root, config["as_of"], datetime.now(UTC), run_lock,
                    registry=coverage_registry, report=coverage_report, notes=coverage_notes,
                    universe=inputs["universe"], production=coverage_production,
                    prior=coverage_prior,
                )  # fmt: skip
        # Validate before spending any API credits.
        screening.screen(
            inputs,
            config["as_of"],
            selected=set(),
            workers=config.get("evaluation_workers", 1),
        )
        batch_estimate = None
        size = config["batch_size"]
        if size == "auto":
            portfolio = inputs.get("portfolio", {})
            batch_estimate = screen_throughput.estimate(
                config,
                screen_throughput.load(root),
                universe_size=len(inputs["universe"]),
                credits_remaining=screen_throughput.credits_remaining(
                    config["acquisition"], root, config["as_of"]
                ),
                seconds_available=run_seconds(config["acquisition"]),
                protected=len(portfolio.get("holdings", []))
                + len(portfolio.get("instruments", [])),
            )
            size = batch_estimate["batch"]
        selected = choose_batch(inputs, state, size, as_of=config["as_of"])
        # Coverage production + execution run BEFORE the run identity: pages fetched now are
        # admitted into this run's input, and the due-work signature below makes work that is
        # still due (a new retained event, a budget-unblocked cursor) a NEW run instead of a
        # reuse of a report built before it. Unchanged completed replay keeps the same identity.
        due_work, priority = "", []
        mandatory = len(selected) * coverage_runtime.fundamentals_cost(config)
        if (
            coverage_live
            and coverage_runtime.enabled(config, "queue")
            and config.get("acquisition", {}).get("enabled")
        ):
            coverage_jobs = coverage_runtime.soft(
                "jobs", coverage_notes, coverage_runtime.run_jobs,
                config, root, selected, run_lock,
                now=datetime.now(UTC),
                as_of=config["as_of"],
                remaining_credits=screen_throughput.credits_remaining(
                    config["acquisition"], root, config["as_of"]
                ),
                universe={r["ticker"] for r in inputs["universe"]},
                deadline=run_started + run_seconds(config["acquisition"]),
                registry=coverage_registry,
                default={},
            )  # fmt: skip
            priority = sorted((coverage_jobs or {}).get("priority", {}))
            if priority:  # claimed ticker jobs ride the normal, budgeted acquisition path
                selected = selected + [t for t in priority if t not in selected]
            coverage_report = coverage_runtime.soft(
                "rehydrate", coverage_notes, coverage_runtime.rehydrate,
                config, root, cov_records, config["as_of"], default=coverage_report,
                floors=(coverage_jobs or {}).get("request_floors"), now=datetime.now(UTC),
            )  # fmt: skip
            due_work = coverage_runtime.soft(
                "due_work", coverage_notes, coverage_runtime.due_work_signature,
                config, root, datetime.now(UTC),
                screen_throughput.credits_remaining(config["acquisition"], root, config["as_of"]),
                mandatory, default="",
            )  # fmt: skip
        # Native-session signature: on the as_of day the latest completed session of every listing
        # calendar is part of the run identity, so a report frozen before a later close is never
        # reused after it (the old immutable report stays valid at its own frozen clock).
        sessions = screen_sources.native_session_signature(
            inputs["universe"] + inputs.get("portfolio", {}).get("instruments", []),
            config["as_of"],
            datetime.now(UTC),
        )
        identity = digest(
            encoded(
                {
                    "input": inputs,
                    "config": config,
                    **({"native_sessions": sessions} if sessions else {}),
                    **({"due_work": due_work} if due_work else {}),
                    "code": {
                        k: digest(v)
                        for k, v in code_files(coverage_runtime.code_modules(config)).items()
                    },
                }
            )
        )[:20]
        runs = root / "runs"
        runs.mkdir(exist_ok=True)
        output = runs / f"{config['as_of']}-{identity}"
        # Same configuration/input/day is idempotent even though rotation has advanced.
        if output.exists():
            verify(output)
            if state == read(output / "prior-state.json"):
                queued = read(output / "selection.json")
                frozen_outcomes = output / "acquisition-outcomes.json"
                recorded = read(frozen_outcomes) if frozen_outcomes.exists() else {}
                replayed = recorded.get("outcomes", {})
                if not config.get("acquisition", {}).get("enabled"):
                    replayed = {t: "cache_current" for t in queued}  # served from existing data
                refreshed, counts = settle(inputs, queued, replayed, recorded.get("verified", []))
                publish_state(
                    root,
                    state,
                    inputs,
                    queued,
                    output,
                    config["as_of"],
                    refreshed,
                    counts,
                    coverage_runtime.soft(
                        "counts", coverage_notes, coverage_runtime.status_counts,
                        config, root, coverage_report, default={},
                    ),
                )  # fmt: skip
            return output / "report.html"
        pending = runs / (output.name + ".pending")
        # Incomplete staging is safe to replace: it was never advertised as last good.
        if pending.exists():
            import shutil

            shutil.rmtree(pending)
        pending.mkdir()
        try:
            receipts, outcomes, verified = [], {}, []
            spend_before, started = (
                screen_throughput.day_spend(root, config["as_of"]),
                time.monotonic(),
            )
            if config.get("acquisition", {}).get("enabled"):
                auxiliary = inputs.get("portfolio", {}).get("instruments", [])
                config["acquisition"]["held_instrument_tickers"] = [r["ticker"] for r in auxiliary]
                config["acquisition"]["holding_tickers"] = sorted(
                    inputs.get("portfolio", {}).get("holdings", [])
                )
                records = auxiliary + inputs["universe"]
                if priority:  # set AFTER the run identity: claimed jobs are not part of the config
                    config["acquisition"]["priority_tickers"] = priority
                receipts = acquire(
                    config["acquisition"],
                    records,
                    [r["ticker"] for r in auxiliary] + selected,
                    root,
                    config["as_of"],
                    **(
                        {"deadline_at": run_started + run_seconds(config["acquisition"])}
                        if coverage_live
                        else {}
                    ),
                )
                outcomes = getattr(receipts, "outcomes", None) or {}
                verified = list(getattr(receipts, "verified", ()))
                if (coverage_jobs or {}).get("priority"):  # ack by this run's acquisition outcome
                    coverage_runtime.soft(
                        "priority_ack", coverage_notes, coverage_runtime.finish_priority,
                        config, root, run_lock, datetime.now(UTC), coverage_jobs, outcomes,
                    )  # fmt: skip
                # Freeze original input before event discovery mutates source state.
                by_ticker = {r["ticker"]: r for r in records}
                datasets = {}
                for receipt in receipts:
                    if receipt.get("file") and receipt.get("ticker") in by_ticker:
                        datasets.setdefault(receipt["ticker"], {})[receipt["dataset"]] = receipt
                for ticker, sources in datasets.items():
                    record = by_ticker[ticker]

                    def payload(name):
                        return (
                            read(root / "source-cache" / sources[name]["file"])
                            if name in sources
                            else None
                        )

                    if record.get("identity_quarantine"):
                        continue
                    try:
                        fundamental = payload("fundamentals")
                        if fundamental is None:
                            continue
                        mapped = normalize_fundamentals(
                            fundamental, record, sources["fundamentals"]
                        )
                        if (
                            mapped.get("pe", {}).get("valuation_basis")
                            == "provider trailing TTM P/E"
                        ):
                            mapped["diagnostic_provider_pe"] = mapped.pop("pe")
                        if "sec_financials" in sources:
                            financial_values = screen_sources.normalize_sec_financials(
                                payload("sec_financials"),
                                record,
                                sources["sec_financials"],
                                payload("splits"),
                                sources.get("splits"),
                            )
                            for key, value in financial_values.items():
                                mapped[key] = (
                                    reconcile_statements(mapped[key], value, key)
                                    if key in mapped
                                    else value
                                )
                        if "sec_annual" in sources:
                            sec_values = screen_sources.normalize_sec(
                                payload("sec_annual"),
                                record,
                                sources["sec_annual"],
                                payload("splits"),
                                sources.get("splits"),
                            )
                            for key, value in sec_values.items():
                                if key in mapped:
                                    mapped[key] = reconcile_statements(mapped[key], value, key)
                                else:
                                    mapped[key] = value
                        if "prices" in sources:
                            mapped.update(
                                screen_sources.normalize_global_prices(
                                    payload("prices"),
                                    fundamental,
                                    record,
                                    sources["prices"],
                                    fundamentals_receipt=sources["fundamentals"],
                                    fx=payload("fx"),
                                    fx_receipt=sources.get("fx"),
                                    now=datetime.now(UTC).isoformat(),
                                )
                            )
                            if "sec_annual" in sources:
                                mapped.update(
                                    screen_sources.matched_valuations(
                                        payload("prices"),
                                        payload("sec_annual"),
                                        record,
                                        sources["sec_annual"],
                                        sources["prices"],
                                        payload("splits"),
                                        sources.get("splits"),
                                    )
                                )
                                mapped.update(
                                    screen_sources.matched_book_values(
                                        payload("prices"),
                                        payload("sec_annual"),
                                        payload("sec_quarterly"),
                                        record,
                                        sources["sec_annual"],
                                        sources.get("sec_quarterly"),
                                        sources["prices"],
                                        payload("splits"),
                                        sources.get("splits"),
                                        fundamentals=fundamental,
                                        fundamentals_receipt=sources["fundamentals"],
                                    )
                                )
                            mapped.update(
                                screen_sources.current_capitalization(
                                    payload("prices"),
                                    fundamental,
                                    record,
                                    sources["prices"],
                                    fundamentals_receipt=sources["fundamentals"],
                                    splits=payload("splits"),
                                    split_receipt=sources.get("splits"),
                                    fx=payload("fx"),
                                    fx_receipt=sources.get("fx"),
                                )
                            )
                        if "sec_companyfacts" in sources:
                            companyfacts_record = dict(record)
                            derived = screen_sources.derive_companyfacts_share_basis(
                                payload("sec_companyfacts"),
                                record,
                                fundamental,
                                sources["fundamentals"],
                                payload("splits"),
                                sources.get("splits"),
                                sources["sec_companyfacts"],
                            )
                            companyfacts_record["companyfacts_share_basis"] = {
                                **derived,
                                **record.get("companyfacts_share_basis", {}),
                            }
                            companyfacts = screen_sources.normalize_sec_companyfacts(
                                payload("sec_companyfacts"),
                                companyfacts_record,
                                sources["sec_companyfacts"],
                                payload("splits"),
                                sources.get("splits"),
                            )
                            for key, value in companyfacts.items():
                                mapped[key] = (
                                    reconcile_statements(mapped[key], value, key)
                                    if key in mapped
                                    else value
                                )
                        if (
                            "prices" in sources
                            and not mapped.get("diluted_eps", {}).get("conflict")
                            and ("sec_public_concept" in sources or "sec_companyfacts" in sources)
                        ):
                            concept_name = (
                                "sec_public_concept"
                                if "sec_public_concept" in sources
                                else "sec_companyfacts"
                            )
                            concept = payload(concept_name)
                            if concept_name == "sec_companyfacts":
                                concept = screen_sources.companyfacts_eps_concept(concept)
                            ttm = (
                                screen_sources.normalize_ttm_pe(
                                    payload("prices"),
                                    concept,
                                    record,
                                    sources[concept_name],
                                    sources["prices"],
                                    payload("splits"),
                                    sources.get("splits"),
                                    fundamentals=fundamental,
                                    fundamentals_receipt=sources["fundamentals"],
                                )
                                if concept
                                else {}
                            )
                            if ttm:
                                if "pe" in mapped:
                                    mapped["diagnostic_provider_pe"] = mapped["pe"]
                                mapped.update(ttm)
                        for key, value in mapped.items():
                            merge_evidence(record, key, value)
                        event_dates = []
                        for name in ("sec_annual", "sec_quarterly"):
                            if name in sources:
                                event_dates += [
                                    r["filed_at"]
                                    for r in payload(name).get("data", [])
                                    if screening.day(r.get("filed_at"))
                                    and r["filed_at"] <= config["as_of"]
                                ]
                        if event_dates:
                            latest = max(event_dates + [record.get("filing_event_at", "")])
                            record["filing_event_at"] = latest
                            record["event_review_required"] = dict(
                                event_at=latest,
                                reason="SEC filing discovered; research review required",
                                sources=[
                                    f"sources/{sources[n]['file']}"
                                    for n in ("sec_annual", "sec_quarterly")
                                    if n in sources
                                ],
                            )
                    except SourceIdentityConflict as exc:
                        receipts.append(
                            quarantine_identity(root, record, exc, sources.get("fundamentals"))
                        )
                portfolio = inputs.setdefault("portfolio", {})
                if not portfolio.get("sessions"):
                    dates = screen_sources.completed_sessions(
                        "XNYS",
                        inputs.get("market_as_of", config["as_of"]),
                        datetime.now(UTC).isoformat(),
                    )
                    portfolio.update(
                        sessions=[str(d.date()) for d in dates],
                        sessions_verified=True,
                        session_source="exchange_calendars XNYS; completed market sessions",
                    )
                screen_sources.sector_medians(inputs["universe"], config["as_of"], merge_evidence)
            # Run-wide consumption gate (every record, selected or not, coverage or legacy cache):
            # a stored price whose native session has since advanced is withdrawn BEFORE the
            # engine and the freeze, so the frozen inputs replay exactly (clock recorded below).
            run_clock = datetime.now(UTC)  # ONE clock for every quote, recorded in the manifest
            withdrawn_prices = screen_sources.revalidate_price_sessions(
                inputs["universe"] + inputs.get("portfolio", {}).get("instruments", []),
                config["as_of"],
                run_clock,
            )
            for row in withdrawn_prices:
                receipts.append(
                    dict(
                        ticker=row["ticker"],
                        dataset="price_gate",
                        status="price_stale",
                        reason=row["reason"],
                        expected_session=row["expected_session"],
                        price_revalidated_at=run_clock.isoformat(),
                        reserved_credits=0,
                    )
                )
            previous = []
            if state.get("last_good"):
                previous = read(Path(state["last_good"]) / "results.json")
            frozen = {
                "input.json": encoded(inputs),
                "config.json": encoded(config),
                "original-input.json": original_inputs,
                "prior-state.json": encoded(state),
                "selection.json": encoded(selected),
                "previous.json": encoded(previous),
                "acquisition-receipts.json": encoded(receipts),
                "acquisition-outcomes.json": encoded({"outcomes": outcomes, "verified": verified}),
            }
            ledger_path = root / "acquisition-ledger.json"
            ledger = read(ledger_path) if ledger_path.exists() else {}
            frozen["budget-ledger.json"] = encoded(ledger)
            capacity_path = root / "capacity-v2.json"
            capacity = read(capacity_path) if capacity_path.exists() else None
            if capacity is not None:
                frozen["capacity-v2.json"] = encoded(capacity)
            health = run_health(config, receipts, ledger, capacity)
            if batch_estimate:
                health["batch_estimate"] = batch_estimate
            frozen["health.json"] = encoded(health)
            if supplements is not None:
                frozen["quarter-supplements.json"] = encoded(supplements)
            code_extra = coverage_runtime.code_modules(config)
            frozen.update({k: v.decode() for k, v in code_files(code_extra).items()})
            frozen.update(coverage_runtime.frozen_files(coverage_report, config))
            frozen.update(artifacts(inputs, config, selected, previous))
            for receipt in receipts:
                if receipt.get("file"):
                    frozen["sources/" + receipt["file"]] = (
                        root / "source-cache" / receipt["file"]
                    ).read_text()
            for name, body in frozen.items():
                destination = pending / name
                destination.parent.mkdir(exist_ok=True, parents=True)
                destination.write_text(body)
            manifest = {
                "version": 1,
                "as_of": config["as_of"],
                "synthetic": inputs.get("synthetic", False),
                "files": {name: digest(body) for name, body in frozen.items()},
                **({"code_modules": list(code_extra)} if code_extra else {}),
                **(
                    {
                        "price_revalidation": {
                            "clock": run_clock.isoformat(),
                            "withdrawn": sorted(r["ticker"] for r in withdrawn_prices),
                        }
                    }
                    if str(screening.day(config["as_of"])) == str(run_clock.date())
                    else {}
                ),
                "environment": {"python": platform.python_version()},
                "limitations": [
                    "Normalized source assertions require independent human checking",
                    "Acquisition maps only filing-dated revenue/net income/OCF/capex",
                    "No claim of complete global or historical coverage",
                ],
            }
            (pending / "manifest.json").write_text(encoded(manifest))
            verify(pending)
            pending.rename(output)
            if not config.get("acquisition", {}).get("enabled"):
                outcomes = {t: "cache_current" for t in selected}  # served from existing data
            refreshed, counts = settle(inputs, selected, outcomes, verified)
            publish_state(
                root,
                state,
                inputs,
                selected,
                output,
                config["as_of"],
                refreshed,
                counts,
                coverage_runtime.soft(
                    "counts", coverage_notes, coverage_runtime.status_counts,
                    config, root, coverage_report, default={},
                ),
            )  # fmt: skip
            if run_lock is not None:  # housekeeping under the run lock (a pilot never overlaps)
                coverage_runtime.soft(
                    "prune", coverage_notes, coverage_runtime.prune, root, run_lock
                )
            note = {"coverage_notes": coverage_notes} if coverage_notes else {}
            if coverage_live:  # explicit production / cursor counts (never a silent shortfall)
                note["coverage_production"] = coverage_production
                note["coverage_cursors"] = coverage_runtime.soft(
                    "cursors", coverage_notes, coverage_runtime.cursor_summary,
                    config, root, default=[],
                )  # fmt: skip
            if config.get("acquisition", {}).get("enabled"):
                if config["as_of"] != datetime.now(UTC).date().isoformat():
                    note["throughput_record_skipped"] = "as_of is not the current UTC day"
                else:
                    seen = state.get("last_seen", {})
                    issuer = {r["ticker"]: r["issuer_id"] for r in inputs["universe"]}
                    try:
                        screen_throughput.record(
                            root,
                            dict(
                                counts,
                                day=config["as_of"],
                                bootstrapped=sum(issuer.get(t) not in seen for t in refreshed),
                                credits_spent=screen_throughput.day_spend(root, config["as_of"])
                                - spend_before,
                                acquisition_seconds=time.monotonic() - started,
                            ),
                        )
                    except Exception as exc:  # bounded class only; never fail a published run
                        note["throughput_record_error"] = type(exc).__name__
            if note:
                atomic(root / "status.json", {**read(root / "status.json"), **note})
            return output / "report.html"
        except Exception as exc:
            # Suppress provider payloads/URLs: status persists only a bounded error class.
            atomic(
                root / "status.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "health": {
                        "status": "accounting_hold"
                        if isinstance(exc, BudgetHold) and str(exc).startswith("accounting hold:")
                        else "budget_hold"
                        if isinstance(exc, BudgetHold)
                        else "source_failure",
                        "operator_stop": OPERATOR_STOP,
                    },
                    "last_good": state.get("last_good"),
                    **(
                        coverage_runtime.accounting_status(exc)
                        if getattr(exc, "accounting_unknown", False)
                        else {}
                    ),
                },
            )
            raise


def verify(path):
    path = Path(path)
    manifest = read(path / "manifest.json")
    for name, expected in manifest["files"].items():
        destination = (path / name).resolve()
        if (
            not destination.is_relative_to(path.resolve())
            or digest(destination.read_bytes()) != expected
        ):
            raise ValueError(f"screen artifact hash mismatch: {name}")
    recorded = coverage_runtime.checked_modules(manifest.get("code_modules"))  # legacy: none
    for name, current in code_files(recorded).items():
        if digest(current) != manifest["files"].get(name):
            raise ValueError("screen code differs from frozen run; use matching checkout")
    frozen_config = read(path / "config.json")  # hashed above: authoritative
    if coverage_runtime.ADMISSIONS_FILE in manifest["files"] or coverage_runtime.enabled(
        frozen_config, "bulk_market", "bulk_fundamentals"
    ):  # a coverage run must carry its bundle; it is re-derived offline
        if coverage_runtime.ADMISSIONS_FILE not in manifest["files"]:
            raise ValueError("coverage-enabled run omits its coverage admissions bundle")
        coverage_runtime.verify_bundle(path, frozen_config)
    recorded_clock = (manifest.get("price_revalidation") or {}).get("clock")
    if recorded_clock:  # the frozen quotes must all be current at the ONE recorded run clock
        frozen_inputs = read(path / "input.json")
        for record in frozen_inputs["universe"] + frozen_inputs.get("portfolio", {}).get(
            "instruments", []
        ):
            item = (record.get("evidence") or {}).get("listing_price_usd")
            if screen_sources.price_item_stale(item, manifest["as_of"], recorded_clock):
                raise ValueError(
                    f"frozen price of {record.get('ticker')} is stale at its run clock"
                )
    replay = artifacts(
        read(path / "input.json"),
        read(path / "config.json"),
        read(path / "selection.json"),
        read(path / "previous.json"),
    )
    for name, body in replay.items():
        if digest(body) != manifest["files"].get(name):
            raise ValueError(f"screen replay differs: {name}")
    return {
        "verified": True,
        "offline_replayed": True,
        "files": len(manifest["files"]),
        "as_of": manifest["as_of"],
        "synthetic": manifest["synthetic"],
    }


def status(root):
    root = Path(root)
    result = (
        read(root / "status.json") if (root / "status.json").exists() else {"status": "never_run"}
    )
    result["state"] = read(root / "state.json") if (root / "state.json").exists() else {}
    return result


def benchmark(inputs_path, as_of, output, *, repetitions=3):
    """Actual frozen-input CPU timings; no network and no fabricated speedup."""
    inputs = read(inputs_path)
    records = inputs["universe"]
    if len(records) != 1000:
        raise ValueError("benchmark requires exactly 1000 frozen normalized inputs")
    timings = []
    reference = None
    for workers in (1, 4, 8):
        for repeat in range(repetitions):
            start = time.perf_counter()
            if workers == 1:
                result = [screening.evaluate(r, as_of) for r in records]
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    result = list(pool.map(_benchmark_one, [(r, as_of) for r in records]))
            sha = digest(encoded(result))
            if reference is not None and sha != reference:
                raise ValueError("benchmark worker results differ")
            reference = sha
            timings.append(
                {
                    "workers": workers,
                    "repeat": repeat + 1,
                    "seconds": time.perf_counter() - start,
                    "result_sha256": sha,
                }
            )
    payload = {
        "input_sha256": digest(Path(inputs_path).read_bytes()),
        "issuer_count": 1000,
        "scope": "normalized gate evaluation only; excludes acquisition, portfolio and reporting",
        "synthetic": inputs.get("synthetic", False),
        "timings": timings,
    }
    atomic(Path(output), payload)
    return payload


def _benchmark_one(args):
    return screening.evaluate(*args)
