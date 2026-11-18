"""Cached-first price admission in acquire(); fake requests only, no provider/network calls."""

from datetime import UTC, datetime, timedelta

import pytest

from wharton import screen_service as service
from wharton import screen_sources

_FROZEN = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)  # a Tuesday; keeps sessions deterministic
TODAY = _FROZEN.date().isoformat()
MARKET = str((_FROZEN - timedelta(days=1)).date())
NOW = _FROZEN.isoformat()
QUOTE_DAY = str(screen_sources.completed_sessions("XNYS", MARKET, NOW)[-1].date())


@pytest.fixture(autouse=True)
def frozen_service_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return _FROZEN

    monkeypatch.setattr(service, "datetime", Clock)


def rec(ticker):
    return {"ticker": ticker, "issuer_id": "I-" + ticker, "kind": "stock", "evidence": {}}


def cfg(**extra):
    return {
        "initial_credit_limit": 3000,
        "daily_credit_limit": 3000,
        "request_credit_cost": {"fundamentals": 10, "prices": 1},
        "account_headroom": 25000,
        "market_as_of": MARKET,
        **extra,
    }


def general(ticker):
    return {
        "General": {
            "Code": ticker.split(".")[0],
            "Exchange": "NASDAQ",
            "CurrencyCode": "USD",
            "Type": "Common Stock",
        }
    }


def prices(close):
    return [{"date": QUOTE_DAY, "close": close, "adjusted_close": close}]


def cache_source(root, ticker, dataset, payload, *, days_old=0):
    cache = root / "source-cache"
    cache.mkdir(parents=True, exist_ok=True)
    body = service.encoded(payload)
    sha = service.digest(body)
    (cache / (sha + ".raw.json")).write_text(body)
    when = _FROZEN - timedelta(days=days_old)
    receipt = {
        "ticker": ticker,
        "dataset": dataset,
        "as_of": str(when.date()),
        "retrieved_at": when.isoformat(),
        "market_as_of": MARKET,
        "file": sha + ".raw.json",
        "sha256": sha,
        "symbol": ticker,
        "cache_hit": False,
        "reserved_credits": 0,
        "status": "raw_unmapped",
    }
    service.atomic(cache / (service.digest(f"{ticker}:{dataset}") + ".json"), receipt)


def requester(close=100.0):
    calls = []

    def request(session, endpoint, params, expected_type):
        calls.append(endpoint)
        if endpoint == "user":
            return {"apiRequests": 0, "dailyRateLimit": 100000, "apiRequestsDate": TODAY}
        if endpoint.startswith("fundamentals/"):
            return general(endpoint.split("/")[1])
        return prices(close)

    return request, calls


def test_cached_sub_minimum_name_gets_no_paid_call(tmp_path):
    cache_source(tmp_path, "PENNY.US", "fundamentals", general("PENNY.US"))
    cache_source(tmp_path, "PENNY.US", "prices", prices(3.0))
    request, calls = requester()
    receipts = service.acquire(
        cfg(), [rec("PENNY.US")], ["PENNY.US"], tmp_path, TODAY, request=request
    )
    assert calls == []
    assert any(r.get("status") == "price_excluded" for r in receipts)
    assert receipts.outcomes == {"PENNY.US": "price_excluded"}


def test_fresh_fetch_below_minimum_stops_before_fundamentals_refresh(tmp_path):
    # Warm name (metadata known but week-stale, so a 10-credit refresh would be due).
    cache_source(tmp_path, "WARM.US", "fundamentals", general("WARM.US"), days_old=20)
    request, calls = requester(close=2.5)
    receipts = service.acquire(
        cfg(), [rec("WARM.US")], ["WARM.US"], tmp_path, TODAY, request=request
    )
    assert calls == ["user", "eod/WARM.US"]
    assert receipts.outcomes == {"WARM.US": "price_excluded"}


def test_admitted_name_keeps_deeper_work(tmp_path):
    cache_source(tmp_path, "FINE.US", "fundamentals", general("FINE.US"), days_old=20)
    request, calls = requester(close=50.0)
    receipts = service.acquire(
        cfg(), [rec("FINE.US")], ["FINE.US"], tmp_path, TODAY, request=request
    )
    assert calls == ["user", "eod/FINE.US", "fundamentals/FINE.US"]
    assert receipts.outcomes == {"FINE.US": "refreshed"}


def test_cold_bootstrap_is_capped_and_unbootstrapped_names_stay_held(tmp_path):
    names = ["A.US", "B.US", "C.US"]
    request, calls = requester()
    # 3000 working * 0.01 = 30 credits = one three-attempt (30-credit) fundamentals reservation.
    receipts = service.acquire(
        cfg(bootstrap_fraction=0.01),
        [rec(t) for t in names],
        names,
        tmp_path,
        TODAY,
        request=request,
    )
    assert sum(c.startswith("fundamentals/") for c in calls) == 1
    # Slot-exhausted cold names cannot be admitted without metadata: no price fetch either.
    assert sum(c.startswith("eod/") for c in calls) == 1
    assert sorted(receipts.outcomes.values()) == ["held", "held", "refreshed"]


def test_budget_held_names_stay_unseen_and_counts_stay_separate(tmp_path):
    names = ["A.US", "B.US"]
    records = [rec(t) for t in names]
    # 33 credits: one full cold name (30 + 3) fits, the second is budget-held.
    request, _ = requester()
    receipts = service.acquire(
        cfg(initial_credit_limit=33, daily_credit_limit=33),
        records,
        names,
        tmp_path,
        TODAY,
        request=request,
    )
    assert receipts.outcomes == {"A.US": "refreshed", "B.US": "held"}
    inputs = {"universe": records}
    refreshed, counts = service.settle(inputs, names, receipts.outcomes, receipts.verified)
    assert refreshed == ["A.US"]
    assert counts == dict(
        evaluated=2,
        queued=2,
        refreshed=1,
        cache_current=0,
        verified=1,
        price_excluded=0,
        price_unprovable=0,
        identity_held=0,
        held=1,
    )
    output = tmp_path / "out"
    output.mkdir()
    service.publish_state(tmp_path, {}, inputs, names, output, TODAY, refreshed, counts)
    state = service.read(tmp_path / "state.json")
    assert state["last_seen"] == {"I-A.US": TODAY} and state["rotation"] == 1
    assert service.read(tmp_path / "status.json")["counts"]["held"] == 1


def test_price_excluded_names_advance_rotation_but_unreached_do_not(tmp_path):
    records = [rec(t) for t in ("X.US", "Y.US", "Z.US")]
    inputs = {"universe": records}
    outcomes = {"X.US": "price_excluded", "Y.US": "held"}  # Z.US has no outcome (no acquisition)
    refreshed, counts = service.settle(inputs, ["X.US", "Y.US", "Z.US"], outcomes)
    assert refreshed == ["X.US"]  # unknown outcome is held, never silently seen
    assert counts["price_excluded"] == 1 and counts["held"] == 2 and counts["cache_current"] == 0


def test_settle_unknown_outcome_value_is_held():
    refreshed, counts = service.settle({"universe": [rec("A.US")]}, ["A.US"], {"A.US": "bogus"})
    assert refreshed == [] and counts["held"] == 1


def test_bootstrap_fraction_validated_at_load_config(tmp_path):
    for bad in ("0", "-1", "1.5", "true", "'0.5'"):
        path = tmp_path / "c.yaml"
        path.write_text(
            f"input: i\nstate_root: s\nacquisition: {{enabled: false, bootstrap_fraction: {bad}}}\n"
        )
        with pytest.raises(ValueError, match="bootstrap_fraction"):
            service.load_config(path)


def test_empty_acquisition_section_loads_as_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("input: i\nstate_root: s\nacquisition:\n")
    assert service.load_config(path)["acquisition"] == {}
    path.write_text("input: i\nstate_root: s\nacquisition: [1]\n")
    with pytest.raises(ValueError, match="acquisition must be a mapping"):
        service.load_config(path)


def test_zero_credit_bootstrap_cap_gives_no_slots(tmp_path):
    request, calls = requester()
    receipts = service.acquire(
        cfg(bootstrap_fraction=0.0001), [rec("A.US")], ["A.US"], tmp_path, TODAY, request=request
    )
    assert not any(c.startswith("fundamentals/") for c in calls)
    assert receipts.outcomes == {"A.US": "held"}


def test_run_health_records_effective_bootstrap_fraction():
    health = service.run_health(
        {"as_of": TODAY, "acquisition": {"bootstrap_fraction": 0.1}}, [], {}
    )
    assert health["bootstrap_fraction"] == 0.1
    assert service.run_health({"as_of": TODAY}, [], {})["bootstrap_fraction"] == 0.25


def test_cache_current_and_identity_held_are_not_counted_refreshed(tmp_path):
    request, calls = requester()
    first = service.acquire(cfg(), [rec("A.US")], ["A.US"], tmp_path, TODAY, request=request)
    assert first.outcomes == {"A.US": "refreshed"} and first.verified == ["A.US"]
    paid = len(calls)
    second = service.acquire(cfg(), [rec("A.US")], ["A.US"], tmp_path, TODAY, request=request)
    assert len(calls) == paid  # fresh cache, no paid fetch
    assert second.outcomes == {"A.US": "cache_current"} and second.verified == ["A.US"]

    def renamed(session, endpoint, params, expected_type):
        if endpoint.startswith("fundamentals/"):
            return {"General": {**general("B.US")["General"], "Code": "RENAMED"}}
        return request(session, endpoint, params, expected_type)

    third = service.acquire(cfg(), [rec("B.US")], ["B.US"], tmp_path, TODAY, request=renamed)
    assert third.outcomes == {"B.US": "identity_held"}
    inputs = {"universe": [rec("A.US"), rec("B.US")]}
    advanced, counts = service.settle(
        inputs, ["A.US", "B.US"], {**second.outcomes, **third.outcomes}
    )
    assert advanced == ["A.US", "B.US"]
    assert (
        counts["refreshed"] == 0 and counts["cache_current"] == 1 and counts["identity_held"] == 1
    )


def test_held_position_below_minimum_keeps_deeper_fetches_but_gate_stays_fail(tmp_path):
    for ticker in ("HELD.US", "NEW.US"):
        cache_source(tmp_path, ticker, "fundamentals", general(ticker), days_old=20)
        cache_source(tmp_path, ticker, "prices", prices(3.0))
    request, calls = requester(close=3.0)
    receipts = service.acquire(
        cfg(holding_tickers=["HELD.US"]),
        [rec("HELD.US"), rec("NEW.US")],
        ["HELD.US", "NEW.US"],
        tmp_path,
        TODAY,
        request=request,
    )
    assert "fundamentals/HELD.US" in calls  # stale fundamentals still refreshed for portfolio fit
    assert not any(c.endswith("NEW.US") for c in calls)  # ordinary candidate: nothing paid
    assert receipts.outcomes == {"HELD.US": "price_excluded", "NEW.US": "price_excluded"}
    assert {r["ticker"] for r in receipts if r.get("status") == "price_excluded"} == {
        "HELD.US",
        "NEW.US",
    }


def test_actual_holdings_are_always_queued_and_active_names_are_capped():
    records = [rec(f"H{i}.US") for i in range(5)]
    records += [rec(f"P{i}.US") for i in range(4)] + [rec(f"U{i}.US") for i in range(10)]
    inputs = {
        "universe": records,
        "portfolio": {"holdings": [f"H{i}.US" for i in range(5)]},
        "active": [f"P{i}.US" for i in range(4)],
    }
    small = service.choose_batch(inputs, {}, 4)  # more holdings than the queue: none dropped
    assert {f"H{i}.US" for i in range(5)} <= set(small) and not any(t[0] == "P" for t in small)
    batch = service.choose_batch(inputs, {}, 9)  # 5 holdings, 4 left: at most 2 active names
    assert {f"H{i}.US" for i in range(5)} <= set(batch)
    assert sum(t[0] == "P" for t in batch) == 2 and len(batch) == 9
    assert len(small) <= 4 + 5 and len(batch) <= 9 + 5  # contract: len <= size + holdings


def test_time_hold_stops_paid_work_and_leaves_remaining_names_held(tmp_path, monkeypatch):
    names = ["A.US", "B.US"]
    now = {"t": 0.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: now["t"])
    request, calls = requester()

    def slow(session, endpoint, params, expected_type):
        out = request(session, endpoint, params, expected_type)
        if endpoint.startswith("fundamentals/"):
            now["t"] = 10_000.0  # past the 3300s guard
        return out

    receipts = service.acquire(cfg(), [rec(t) for t in names], names, tmp_path, TODAY, request=slow)
    assert sum(c.startswith("fundamentals/") for c in calls) == 1
    assert not any(c.startswith("eod/") for c in calls)
    assert set(receipts.outcomes.values()) == {"held"}
    assert any(r.get("status") == "time_hold" for r in receipts)


def test_queue_cap_raised_but_bounded(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    path = tmp_path / "c.yaml"
    for size, ok in ((100000, True), (100001, False), (0, False), (True, False)):
        path.write_text(
            f"input: {source}\nstate_root: {tmp_path}/s\nbatch_size: {str(size).lower()}\n"
        )
        if ok:
            assert service.load_config(path)["batch_size"] == size
        else:
            with pytest.raises(ValueError, match="batch_size"):
                service.load_config(path)


def test_priority_names_rotate_and_never_crowd_out_the_universe():
    records = [rec(f"P{i}.US") for i in range(8)] + [rec(f"U{i}.US") for i in range(8)]
    inputs = {"universe": records, "active": [f"P{i}.US" for i in range(8)]}
    batch = service.choose_batch(inputs, {}, 6)
    assert sum(t.startswith("U") for t in batch) >= 3 and len(batch) == 6
    seen = {r["issuer_id"]: TODAY for r in records if r["ticker"] in batch}
    later = service.choose_batch(inputs, {"last_seen": seen}, 6)
    assert {t for t in later if t.startswith("P")}.isdisjoint(t for t in batch if t.startswith("P"))


FRESH = dict(
    fresh_sources=True,
    request_credit_cost={
        "fundamentals": 10,
        "prices": 1,
        "sec_annual": 10,
        "sec_quarterly": 10,
        "splits": 1,
        "earnings": 1,
        "fx": 1,
    },
)


def fresh_requester(close=100.0):
    calls = []

    def request(session, endpoint, params, expected_type):
        calls.append((endpoint, (params or {}).get("symbols")))
        if endpoint == "user":
            return {"apiRequests": 0, "dailyRateLimit": 100000, "apiRequestsDate": TODAY}
        if endpoint.startswith("fundamentals/"):
            return general(endpoint.split("/")[1])
        if endpoint.startswith("eod/"):
            return prices(close)
        return {"earnings": [], "data": []} if expected_type is dict else []

    return request, calls


def test_fresh_sources_cached_sub_minimum_name_pays_nothing_including_earnings(tmp_path):
    cache_source(tmp_path, "PENNY.US", "fundamentals", general("PENNY.US"))
    cache_source(tmp_path, "PENNY.US", "prices", prices(3.0))
    request, calls = fresh_requester()
    service.acquire(cfg(**FRESH), [rec("PENNY.US")], ["PENNY.US"], tmp_path, TODAY, request=request)
    assert calls == []


def test_fresh_sub_minimum_price_stops_before_earnings_and_fundamentals(tmp_path):
    cache_source(tmp_path, "WARM.US", "fundamentals", general("WARM.US"), days_old=20)
    request, calls = fresh_requester(close=2.5)
    service.acquire(cfg(**FRESH), [rec("WARM.US")], ["WARM.US"], tmp_path, TODAY, request=request)
    assert [e for e, _ in calls] == ["user", "eod/WARM.US"]


def test_bootstrap_held_cold_name_pays_no_earnings(tmp_path):
    names = ["A.US", "B.US"]
    request, calls = fresh_requester()
    receipts = service.acquire(
        cfg(bootstrap_fraction=0.01, **FRESH),
        [rec(t) for t in names],
        names,
        tmp_path,
        TODAY,
        request=request,
    )
    assert receipts.outcomes["B.US"] == "held"
    assert [s for e, s in calls if e == "calendar/earnings"] == ["A.US"]


def lse_general(ticker):
    return {
        "General": {
            "Code": ticker.split(".")[0],
            "Exchange": "LSE",
            "CurrencyCode": "GBP",
            "Type": "Common Stock",
        }
    }


@pytest.mark.parametrize("held_position", [False, True])
def test_unprovable_price_pays_no_deeper_work_but_still_rotates(tmp_path, held_position):
    # Unverified unit-risk venue: fetching can never make the price provable.
    cache_source(tmp_path, "U.LSE", "fundamentals", lse_general("U.LSE"), days_old=20)
    base, calls = fresh_requester()

    def request(session, endpoint, params, expected_type):
        if endpoint.startswith("fundamentals/"):
            calls.append((endpoint, None))
            return lse_general("U.LSE")
        return base(session, endpoint, params, expected_type)

    receipts = service.acquire(
        cfg(holding_tickers=["U.LSE"] if held_position else [], **FRESH),
        [rec("U.LSE")],
        ["U.LSE"],
        tmp_path,
        TODAY,
        request=request,
    )
    endpoints = [e for e, _ in calls]
    assert receipts.outcomes == {"U.LSE": "price_unprovable"}
    assert not any(r.get("status") == "price_excluded" for r in receipts)  # gate stays unknown
    if held_position:
        assert "fundamentals/U.LSE" in endpoints  # holdings keep the deeper-fetch exemption
    else:
        assert endpoints == ["user", "eod/U.LSE"]
        advanced, counts = service.settle(
            {"universe": [rec("U.LSE")]}, ["U.LSE"], receipts.outcomes
        )
        assert advanced == ["U.LSE"] and counts["price_unprovable"] == 1


def test_holdings_get_bootstrap_slots_before_other_cold_names(tmp_path):
    names = ["C1.US", "C2.US", "C3.US", "H1.US", "H2.US"]
    request, calls = requester()
    receipts = service.acquire(
        cfg(bootstrap_fraction=0.01, holding_tickers=["H1.US", "H2.US"]),
        [rec(t) for t in names],
        names,
        tmp_path,
        TODAY,
        request=request,
    )
    assert [c for c in calls if c.startswith("fundamentals/")] == ["fundamentals/H1.US"]
    assert receipts.outcomes["H1.US"] == "refreshed"
    assert {receipts.outcomes[t] for t in names if t != "H1.US"} == {"held"}


def test_budget_held_second_pass_marks_name_held_and_unseen(tmp_path):
    # 45 credits: earnings, bootstrap, price and splits fit; the 30-credit SEC filing does not.
    request, _ = fresh_requester()
    receipts = service.acquire(
        cfg(initial_credit_limit=45, daily_credit_limit=45, **FRESH),
        [rec("A.US")],
        ["A.US"],
        tmp_path,
        TODAY,
        request=request,
    )
    assert receipts.outcomes == {"A.US": "held"}
    advanced, counts = service.settle({"universe": [rec("A.US")]}, ["A.US"], receipts.outcomes)
    assert advanced == [] and counts["held"] == 1 and counts["refreshed"] == 0


def test_time_held_second_pass_marks_name_held(tmp_path, monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(service.time, "monotonic", lambda: now["t"])
    request, _ = fresh_requester()

    def slow(session, endpoint, params, expected_type):
        out = request(session, endpoint, params, expected_type)
        if endpoint.startswith("splits/"):
            now["t"] = 10_000.0  # price stage done; filings never reached
        return out

    receipts = service.acquire(cfg(**FRESH), [rec("A.US")], ["A.US"], tmp_path, TODAY, request=slow)
    assert receipts.outcomes == {"A.US": "held"}
