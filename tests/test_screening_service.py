"""Synthetic contract checks; passing software tests establish no investment merit."""

import copy
import fcntl
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from wharton import screen_service as service
from wharton import screening as s
from wharton.cli import app

AS_OF = "2026-10-27"


@pytest.fixture(autouse=True)
def fixed_service_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 27, 23, 30, tzinfo=UTC)

    monkeypatch.setattr(service, "datetime", Clock)


def ev(value, **kwargs):
    return {
        "value": value,
        "verified": True,
        "source": "synthetic:test",
        "period": "2025-12-31",
        "observed_at": AS_OF,
        "available_at": "2026-02-01",
        **kwargs,
    }


def price_ev(value=10.0, **kwargs):
    return ev(
        value,
        **{
            "currency": "USD",
            "period": "2026-10-26",
            "last_completed_session": "2026-10-26",
            "expected_session": "2026-10-26",
            "receipt_as_of": AS_OF,
            "available_at": AS_OF,
            **kwargs,
        },
    )


def annual(vals, *, start=2021, **kwargs):
    return ev(
        [
            {
                "period": f"{start + i}-12-31",
                "value": val,
                "available_at": f"{start + i + 1}-02-01",
                "currency": "USD",
                "source": "synthetic:annual",
            }
            for i, val in enumerate(vals)
        ],
        **kwargs,
    )


def review(**kwargs):
    return {
        "author": "synthetic author",
        "reviewer": "synthetic reviewer",
        "status": "approved",
        "reviewed_at": AS_OF,
        "citations": ["synthetic:test"],
        "rationale": "SYNTHETIC TEST ONLY",
        **kwargs,
    }


def monthly(value=15, n=60):
    months = pd.date_range("2021-11-01", periods=n, freq="MS")
    return ev(
        [
            {
                "period": str(d.date()),
                "price_date": str(d.date()),
                "value": value,
                "earnings_available_at": str((d - pd.Timedelta(days=30)).date()),
                "earnings_period": str((d - pd.Timedelta(days=90)).date()),
                "earnings_currency": "USD",
                "price_currency": "USD",
                "matched": True,
                "source": "synthetic:matched monthly",
            }
            for d in months
        ]
    )


def record(ticker="TEST.US", kind="stock"):
    return {
        "ticker": ticker,
        "issuer_id": ticker,
        "kind": kind,
        "region": "synthetic",
        "sector": "synthetic sector",
        "review_owner": "synthetic analyst",
        "evidence": {
            "market_cap_usd": ev(2e9, currency="USD"),
            "issuer_identity": ev({"ticker": ticker, "issuer_id": ticker}),
            "listing_price_usd": price_ev(),
            "revenue": annual([100 * 1.07**i for i in range(4)], start=2022),
            "diluted_eps": annual(
                [1.05**i for i in range(4)],
                start=2022,
                basis="reported_gaap",
                statement_source="sec_financials",
                accounting_framework="US_GAAP",
                basis_evidence=dict(
                    verified=True,
                    source="synthetic:listing",
                    ticker=ticker,
                    currency="USD",
                    share_basis="diluted",
                    **{"from": "2022-01-01", "through": AS_OF},
                ),
            ),
            "ocf": annual([100 * 1.05**i + 20 for i in range(4)], start=2022),
            "capex": annual([-20] * 4, start=2022),
            "net_income": annual([100], start=2025),
            "diluted_shares": annual(
                [100 * 1.02**i for i in range(4)],
                start=2022,
                basis="weighted_average_diluted_split_adjusted",
            ),
            "pe": ev(15, valuation_basis="reviewed comparable"),
            "sector_pe": ev(15, peer_n=25),
            "own_pe_monthly": {**monthly(), "valuation_basis": "reviewed comparable"},
            "net_debt": ev(200, currency="USD"),
            "ebitda": ev(100, currency="USD"),
            "operating_margin": annual([0.20] * 5),
            "organic_growth": annual([0.04] * 3, start=2023),
            "ptbv": ev(1.1),
            "own_ptbv_monthly": monthly(1.2),
            "roe": ev(0.12),
            "combined_ratio": ev(0.94),
        },
        "reviews": {
            "thesis": review(not_mainly_ai_spending=True),
            "scenarios": review(
                value={"bear": -0.35, "base": 1.1**5 - 1, "bull": 0.7},
                horizon_years=5,
                dividends="reinvested",
                costs=".1% one-way",
                fx="USD no change",
                assumptions="Conditional business cases",
            ),
            "deal": review(ev_fraction=0.2, retained_business_thesis="Core survives"),
            "policy": review(single_policy_earnings_impact=0.2),
            "capital_allocation": review(matches_thesis=True),
            "breakers": review(
                items=[
                    {"metric": "revenue growth", "threshold": 0.03},
                    {"metric": "ROIC", "threshold": 0.1},
                ],
                annual_review=True,
            ),
        },
    }


def gates(rec):
    return {g["gate"]: g for g in s.evaluate(rec, AS_OF)["gates"]}


def test_growth_thresholds_and_negative_values():
    r = record()
    g = gates(r)
    for name in ("revenue_3y_cagr", "diluted_eps_3y_cagr", "fcf_3y_cagr", "annual_dilution"):
        assert g[name]["result"] == "pass"
    r["evidence"]["diluted_eps"]["value"][0]["value"] = -1
    assert gates(r)["diluted_eps_3y_cagr"]["result"] == "fail"
    r["evidence"]["ocf"]["value"][-1]["value"] = 1
    assert gates(r)["positive_fcf_latest3"]["result"] == "fail"
    r["evidence"]["net_income"]["value"][0]["value"] = -1
    assert gates(r)["cash_conversion"]["result"] == "fail"


@pytest.mark.parametrize("change", ["stale", "future", "currency", "duplicate", "gap", "missing"])
def test_unusable_annual_evidence_is_unknown(change):
    r = record()
    item = r["evidence"]["revenue"]
    if change == "stale":
        item["observed_at"] = "2026-01-01"
    elif change == "future":
        item["value"][-1]["available_at"] = "2027-01-01"
    elif change == "currency":
        item["value"][0]["currency"] = "EUR"
    elif change == "duplicate":
        item["value"].append(copy.deepcopy(item["value"][-1]))
    elif change == "gap":
        item["value"][0]["period"] = "2020-12-31"
    else:
        item["value"].pop()
    assert gates(r)["revenue_3y_cagr"]["result"] == "unknown"


def test_eps_calendar_and_ordinary_shares_cannot_pass():
    r = record()
    r["evidence"]["diluted_eps"]["basis"] = "calendar_epsActual"
    r["evidence"]["diluted_shares"]["basis"] = "ordinary_outstanding"
    assert gates(r)["diluted_eps_3y_cagr"]["result"] == "unknown"
    assert gates(r)["annual_dilution"]["result"] == "unknown"


def test_historical_pe_needs_48_matched_months():
    r = record()
    r["evidence"]["own_pe_monthly"] = {**monthly(n=48), "valuation_basis": "reviewed comparable"}
    assert gates(r)["pe_own"]["result"] == "pass"
    r["evidence"]["own_pe_monthly"]["value"].pop()
    assert gates(r)["pe_own"]["result"] == "unknown"
    r["evidence"]["own_pe_monthly"] = {**monthly(), "valuation_basis": "reviewed comparable"}
    for row in r["evidence"]["own_pe_monthly"]["value"]:
        row["earnings_available_at"] = "2027-01-01"
    assert gates(r)["pe_own"]["result"] == "unknown"


def test_financial_replacements_and_limited_exception():
    r = record(kind="insurer")
    g = gates(r)
    for name in ("fcf_3y_cagr", "cash_conversion", "net_debt_ebitda"):
        assert g[name]["result"] == "not_applicable"
    assert g["ptbv_own"]["result"] == "pass"
    assert g["roe"]["result"] == "pass"
    r["evidence"]["combined_ratio"]["value"] = 0.95
    assert gates(r)["combined_ratio"]["result"] == "fail"
    r = record()
    r["evidence"]["net_debt"]["value"] = 300
    assert gates(r)["net_debt_ebitda"]["result"] == "fail"
    r["reviews"]["leverage_exception"] = review(maximum=3)
    assert gates(r)["net_debt_ebitda"]["result"] == "exception"
    r["evidence"]["net_debt"]["value"] = 301
    assert gates(r)["net_debt_ebitda"]["result"] == "fail"
    r["evidence"]["net_debt"]["currency"] = "EUR"
    assert gates(r)["net_debt_ebitda"]["result"] == "unknown"


def test_human_review_never_inferred_and_scenarios_ordered():
    r = record()
    assert gates(r)["authored_scenarios"]["result"] == "pass"
    r["reviews"]["thesis"]["status"] = "draft"
    assert gates(r)["human_thesis"]["result"] == "unknown"
    r["reviews"]["scenarios"]["value"]["bull"] = 0.6
    assert gates(r)["authored_scenarios"]["result"] == "fail"
    r["reviews"]["deal"]["ev_fraction"] = 0.3
    assert gates(r)["deal"]["result"] == "fail"
    r["reviews"]["deal"]["central_thesis"] = True
    assert gates(r)["deal"]["result"] == "pass"


def test_quote_live_fresh_and_limit_only():
    r = record()
    r["evidence"]["quote"] = ev(
        {"bid": 100, "ask": 100.5, "order_type": "limit"}, observed_at=AS_OF + "T23:00:00+00:00"
    )
    r["quote_evaluation_at"] = AS_OF + "T23:00:59+00:00"
    assert gates(r)["limit_order_spread"]["result"] == "pass"
    r["quote_evaluation_at"] = AS_OF + "T23:02:00+00:00"
    assert gates(r)["limit_order_spread"]["result"] == "unknown"


def risk_inputs():
    dates = pd.bdate_range("2021-10-27", AS_OF)
    rng = np.random.default_rng(451)
    records = []
    arrays = {}
    for ticker in ("TEST.US", "HOLD.US", "VT"):
        vals = rng.normal(0.0002, 0.01, len(dates))
        arrays[ticker] = vals
        r = record(ticker)
        r["evidence"]["daily_total_returns_usd"] = ev(
            [{"date": str(d.date()), "value": float(v)} for d, v in zip(dates, vals)],
            currency="USD",
            frequency="daily",
            return_basis="total_return",
        )
        records.append(r)
    portfolio = {
        "holdings": ["HOLD.US"],
        "weights": {"TEST.US": 0.1, "HOLD.US": 0.9},
        "existing_weights": {"HOLD.US": 1.0},
        "sessions": [str(d.date()) for d in dates],
        "sessions_verified": True,
        "session_source": "SYNTHETIC weekdays",
    }
    return {"synthetic": True, "universe": records, "portfolio": portfolio}, arrays


def test_risk_formulas_and_missing_holdings_unknown():
    inputs, arrays = risk_inputs()
    out = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in out["gates"]}
    n = len(arrays["TEST.US"]) - 1  # strict five-year window excludes first day
    expected_es = -np.sort(arrays["TEST.US"][1:])[: int(np.ceil(0.05 * n))].mean()
    assert g["daily_es95"]["value"] == pytest.approx(expected_es)
    cov = np.cov(np.array([arrays["TEST.US"][1:], arrays["HOLD.US"][1:]]))
    weights = np.array([0.1, 0.9])
    expected = 0.1 * (cov @ weights)[0] / (weights @ cov @ weights)
    assert g["euler_variance_fraction"]["value"] == pytest.approx(expected)
    del inputs["portfolio"]["weights"]
    out = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in out["gates"]}
    assert g["euler_variance_fraction"]["result"] == "unknown"
    assert g["volatility_1y"]["result"] == "pass"


def test_no_fill_and_currency_mismatch():
    inputs, _ = risk_inputs()
    rows = inputs["universe"][0]["evidence"]["daily_total_returns_usd"]["value"]
    del rows[:200]
    out = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    assert next(g for g in out["gates"] if g["gate"] == "daily_es95")["result"] == "unknown"
    inputs["universe"][1]["evidence"]["daily_total_returns_usd"]["currency"] = "EUR"
    out = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    assert (
        next(g for g in out["gates"] if g["gate"] == "holding_correlation")["result"] == "unknown"
    )


@pytest.mark.parametrize(
    "cap,bucket",
    [
        (0, "nano"),
        (50e6, "micro"),
        (300e6, "small"),
        (2e9, "mid"),
        (10e9, "large"),
        (200e9, "mega"),
    ],
)
def test_cap_boundaries(cap, bucket):
    assert s.cap_bucket(cap) == bucket


def test_retained_percentiles_midrank_and_peer_minimum():
    universe = [s.evaluate(record(f"T{i}.US"), AS_OF) for i in range(20)]
    chosen = universe[:1]
    s.percentiles(chosen, universe, AS_OF)
    pct = chosen[0]["percentiles"][0]
    assert pct["peer_n"] == 20 and pct["raw_higher_percentile"] == 50
    s.percentiles(chosen, universe[:-1], AS_OF)
    assert chosen[0]["percentiles"][0]["raw_higher_percentile"] is None


def test_weighted_cohort_selection_and_strict_ceilings():
    records = [s.evaluate(record(f"T{i:02}.US"), AS_OF) for i in range(60)]
    for i, r in enumerate(records):
        for g in r["gates"]:
            g["result"] = "pass"
        r["metrics"]["volatility_1y"] = 0.2
        r["metrics"]["revenue_3y_cagr"] = 0.1 + i * 0.001
        next(g for g in r["gates"] if g["gate"] == "revenue_3y_cagr")["value"] = 0.1 + i * 0.001
    lower = []
    for i in range(1140):
        r = copy.deepcopy(records[0])
        r.update(ticker=f"LOW{i}", issuer_id=f"LOW{i}")
        next(g for g in r["gates"] if g["gate"] == "revenue_3y_cagr")["value"] = 0
        lower.append(r)
    ranked = s.stage_and_rank(records, records + lower)
    assert ranked[0]["ticker"] == "T59.US"
    assert sum(r["stage"] == "review_ready" for r in ranked) == 5
    assert sum(r["stage"] == "portfolio_capacity_hold" for r in ranked) == 40
    assert sum(r["stage"] == "diligence_capacity_hold" for r in ranked) == 10
    failed = s.evaluate(record(), AS_OF)
    failed["gates"][0]["result"] = "unknown"
    assert s.stage_and_rank([failed])[0]["stage"] == "screen_hold"


def config(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"synthetic": True, "universe": [record()]}))
    path = tmp_path / "screen.json"
    path.write_text(
        json.dumps(
            {
                "input": str(source),
                "state_root": str(tmp_path / "service"),
                "batch_size": 1000,
                "as_of": AS_OF,
            }
        )
    )
    return path


def test_replay_idempotence_tamper_and_cli(tmp_path):
    cfg = config(tmp_path)
    report = service.run(cfg)
    assert service.verify(report.parent)["offline_replayed"]
    state = (tmp_path / "service/state.json").read_bytes()
    assert service.run(cfg) == report
    assert (tmp_path / "service/state.json").read_bytes() == state
    cli = CliRunner().invoke(app, ["verify-screen", str(report.parent)])
    assert cli.exit_code == 0 and '"verified": true' in cli.stdout
    (report.parent / "results.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        service.verify(report.parent)


def test_failure_preserves_last_good_and_lock(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    service.run(cfg)
    state_file = tmp_path / "service/state.json"
    state = state_file.read_bytes()
    with (tmp_path / "service/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            service.run(cfg)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failed worker")

    monkeypatch.setattr(service, "artifacts", fail)
    with pytest.raises(RuntimeError):
        service.run(cfg, as_of="2026-10-28")
    assert state_file.read_bytes() == state
    assert service.status(tmp_path / "service")["status"] == "failed"


def test_fair_rotation_and_issuer_dedup():
    records = [record(f"T{i}.US") for i in range(6)]
    for i, r in enumerate(records):
        r["region"] = str(i % 3)
    inputs = {"universe": records, "active": ["T5.US"]}
    first = service.choose_batch(inputs, {}, 3)
    assert first[0] == "T5.US" and len(first) == 3
    state = {"last_seen": {t: AS_OF for t in first}, "rotation": 3}
    second = service.choose_batch(inputs, state, 3)
    assert set(second) != set(first)
    inputs["universe"][0]["issuer_id"] = inputs["universe"][1]["issuer_id"]
    with pytest.raises(ValueError, match="deduplicate"):
        s.screen(inputs, AS_OF)


def test_budget_reservation_stop_and_cache(tmp_path):
    cfg = {
        "initial_credit_limit": 33,
        "daily_credit_limit": 33,
        "request_credit_cost": {"fundamentals": 10, "prices": 1},
        "account_headroom": 25000,
    }
    calls = []

    def request(session, endpoint, params, expected_type):
        calls.append(endpoint)
        if endpoint == "user":
            return {
                "apiRequests": 0,
                "dailyRateLimit": 100000,
                "apiRequestsDate": AS_OF,
                "secret_unwanted_field": "must never be retained",
            }
        return {"General": {"Code": "TEST", "Exchange": "NASDAQ"}} if expected_type is dict else []

    rec = [record()]
    receipts = service.acquire(cfg, rec, ["TEST.US"], tmp_path, AS_OF, request=request)
    assert len(calls) == 3 and sum(r["reserved_credits"] for r in receipts) == 33
    assert "secret_unwanted_field" not in json.dumps(receipts)
    receipts = service.acquire(cfg, rec, ["TEST.US"], tmp_path, AS_OF, request=request)
    assert len(calls) == 3 and all(r["cache_hit"] for r in receipts)
    cfg["initial_credit_limit"] = 40001
    with pytest.raises(ValueError, match="exceed 40000"):
        service.validate_budget(cfg)

    class Session:
        def get(self, *args, **kwargs):
            return type("Response", (), {"status_code": 429})()

    with pytest.raises(ValueError, match="429"):
        service.StopSession(Session()).get("irrelevant")


def test_source_mapping_changes_gate_and_does_not_invent_eps():
    r = record()
    rows = {
        f"{year}-12-31": {
            "date": f"{year}-12-31",
            "filing_date": f"{year + 1}-02-01",
            "currency_symbol": "USD",
            "totalRevenue": 100 * 1.1 ** (year - 2022),
        }
        for year in range(2022, 2026)
    }
    payload = {
        "General": {"Code": "TEST", "Exchange": "NASDAQ"},
        "Financials": {"Income_Statement": {"yearly": rows}},
        "Earnings": {"History": {"epsActual": 999}},
    }
    receipt = {
        "file": "synthetic.raw.json",
        "as_of": AS_OF,
        "retrieved_at": AS_OF + "T23:30:00+00:00",
    }
    mapped = service.normalize_fundamentals(payload, r, receipt)
    r["evidence"].update(mapped)
    assert gates(r)["revenue_3y_cagr"]["value"] == pytest.approx(0.1)
    assert "diluted_eps" not in mapped and "diluted_shares" not in mapped
    rows["2025-12-31"]["totalRevenue"] = 50
    r["evidence"].update(service.normalize_fundamentals(payload, r, receipt))
    assert gates(r)["revenue_3y_cagr"]["result"] == "fail"


def test_incomplete_staging_recovered(tmp_path):
    cfg = config(tmp_path)
    report = service.run(cfg)
    pending = report.parent.with_name(report.parent.name + ".pending")
    report.parent.rename(pending)
    (tmp_path / "service/state.json").unlink()
    (pending / "results.json").write_text("incomplete")
    assert service.run(cfg) == report
    assert service.verify(report.parent)["verified"]


def test_crash_after_publish_before_state_recovers(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    real_atomic = service.atomic

    def crash(path, value):
        if path.name == "state.json":
            raise RuntimeError("crash after run directory publication")
        return real_atomic(path, value)

    monkeypatch.setattr(service, "atomic", crash)
    with pytest.raises(RuntimeError):
        service.run(cfg)
    monkeypatch.setattr(service, "atomic", real_atomic)
    report = service.run(cfg)
    assert service.status(tmp_path / "service")["state"]["last_good"] == str(report.parent)


def test_price_mapping_no_fill_and_usd_only():
    receipt = {"file": "prices.raw.json", "as_of": AS_OF, "retrieved_at": AS_OF + "T23:30:00+00:00"}
    metadata = {
        "General": {
            "Code": "TEST",
            "Exchange": "NASDAQ",
            "CurrencyCode": "USD",
            "Type": "Common Stock",
        }
    }
    rows = [
        {"date": "2026-10-23", "adjusted_close": 100},
        {"date": "2026-10-27", "adjusted_close": 110},
    ]
    sessions = ["2026-10-23", "2026-10-26", "2026-10-27"]
    mapped = service.normalize_prices(rows, metadata, record(), receipt, sessions)
    assert mapped["value"] == []  # Monday gap also invalidates Tuesday's daily return
    metadata["General"]["CurrencyCode"] = "EUR"
    assert service.normalize_prices(rows, metadata, record(), receipt, sessions) is None


def test_etf_separate_review_and_stock_risk_exemptions():
    r = record(kind="etf")
    g = gates(r)
    assert g["revenue_3y_cagr"]["result"] == "not_applicable"
    assert g["fund_review"]["result"] == "unknown"
    r["reviews"]["fund"] = review(
        strategy="managed futures", costs=".8%", liquidity="adequate", structure="ETF"
    )
    assert gates(r)["fund_review"]["result"] == "pass"


def test_provider_refresh_replaces_owned_and_quarantines_curated():
    r = record()
    fresh = annual([100, 90, 80, 70], start=2022)
    service.merge_evidence(r, "revenue", fresh)
    assert r["evidence"]["revenue"]["verified"] is False
    assert gates(r)["revenue_3y_cagr"]["result"] == "unknown"
    r["evidence"]["revenue"] = {
        **annual([100, 110, 120, 130], start=2022),
        "owner": "eodhd_screen_service",
    }
    service.merge_evidence(r, "revenue", fresh)
    assert gates(r)["revenue_3y_cagr"]["result"] == "fail"


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("currency", "EUR"),
        ("period", "2024-12-31"),
        ("basis", "incomparable"),
        ("units", "millions"),
    ],
)
def test_equal_scalar_value_different_currency_period_or_basis_conflicts(field, new_value):
    r = record()
    fresh = copy.deepcopy(r["evidence"]["net_debt"])
    fresh[field] = new_value
    service.merge_evidence(r, "net_debt", fresh)
    assert r["evidence"]["net_debt"]["verified"] is False
    assert gates(r)["net_debt_ebitda"]["result"] == "unknown"


def test_equal_annual_values_different_basis_conflict():
    r = record()
    fresh = copy.deepcopy(r["evidence"]["revenue"])
    fresh["basis"] = "restated_incomparable_scope"
    service.merge_evidence(r, "revenue", fresh)
    assert gates(r)["revenue_3y_cagr"]["result"] == "unknown"


def test_historical_acquisition_rejected_and_retrieval_not_backdated(tmp_path):
    cfg = {
        "initial_credit_limit": 40,
        "daily_credit_limit": 40,
        "request_credit_cost": {"fundamentals": 10, "prices": 1},
    }
    with pytest.raises(ValueError, match="actual UTC today"):
        service.acquire(cfg, [], [], tmp_path, "2026-10-26")
    r = record()
    payload = {
        "General": {"Code": "TEST", "Exchange": "NASDAQ"},
        "Financials": {
            "Income_Statement": {
                "yearly": {
                    row["period"]: {
                        "date": row["period"],
                        "filing_date": row["available_at"],
                        "currency_symbol": "USD",
                        "totalRevenue": row["value"],
                    }
                    for row in r["evidence"]["revenue"]["value"]
                }
            }
        },
    }
    receipt = {
        "file": "synthetic.raw.json",
        "as_of": AS_OF,
        "retrieved_at": "2026-10-28T00:00:01+00:00",
    }
    mapped = service.normalize_fundamentals(payload, r, receipt)
    assert mapped["revenue"]["observed_at"] == "2026-10-28"
    r["evidence"].update(mapped)
    assert gates(r)["revenue_3y_cagr"]["result"] == "unknown"


def test_quota_preflight_stops_before_dataset_calls(tmp_path):
    cfg = {
        "initial_credit_limit": 40,
        "daily_credit_limit": 40,
        "request_credit_cost": {"fundamentals": 10, "prices": 1},
    }
    calls = []

    def request(session, endpoint, params, expected_type):
        calls.append(endpoint)
        return {"apiRequests": 99998, "dailyRateLimit": 100000, "apiRequestsDate": AS_OF}

    with pytest.raises(ValueError, match="remaining quota"):
        service.acquire(cfg, [record()], ["TEST.US"], tmp_path, AS_OF, request=request)
    assert calls == ["user"]
    assert not (tmp_path / "acquisition-ledger.json").exists()


def test_zero_weight_is_no_concentration_clearance():
    inputs, _ = risk_inputs()
    inputs["portfolio"]["weights"] = {"TEST.US": 0.0, "HOLD.US": 1.0}
    out = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    assert (
        next(g for g in out["gates"] if g["gate"] == "euler_variance_fraction")["result"] == "fail"
    )


def test_vt_identity_cannot_be_reconfigured():
    inputs, _ = risk_inputs()
    inputs["portfolio"]["benchmark"] = "SPY.US"
    with pytest.raises(ValueError, match="must be VT"):
        s.screen(inputs, AS_OF)


def test_qualified_listing_and_issuer_identity_validation():
    r = record()
    with pytest.raises(ValueError, match="qualified listing"):
        service.listing_identity({"General": {"Code": "TEST", "Exchange": "LSE"}}, r)
    r["identifiers"] = {"isin": "US-EXPECTED"}
    with pytest.raises(service.SourceIdentityConflict, match="isin mismatch"):
        service.listing_identity(
            {"General": {"Code": "TEST", "Exchange": "NYSE", "ISIN": "US-OTHER"}}, r
        )
    del r["evidence"]["issuer_identity"]
    assert gates(r)["issuer_identity"]["result"] == "unknown"


def test_organic_growth_compounds_three_years_and_rejects_invalid_alignment():
    r = record()
    r["evidence"]["organic_growth"] = annual([0, 0.08, 0.08], start=2023)
    gate = gates(r)["organic_growth"]
    assert gate["result"] == "pass"
    assert gate["value"] == pytest.approx(1.08 ** (2 / 3) - 1)
    r["evidence"]["organic_growth"]["value"][0]["value"] = -1
    assert gates(r)["organic_growth"]["result"] == "unknown"
    r["evidence"]["organic_growth"] = annual([0, 0.08, 0.08], start=2023)
    for row in r["evidence"]["organic_growth"]["value"]:
        row["currency"] = "EUR"
    assert gates(r)["organic_growth"]["result"] == "pass"  # Independently exceeds 4%.


def test_one_year_volatility_does_not_require_five_year_history():
    inputs, _ = risk_inputs()
    item = inputs["universe"][0]["evidence"]["daily_total_returns_usd"]
    item["value"] = [r for r in item["value"] if r["date"] > "2025-10-27"]
    result = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in result["gates"]}
    assert g["volatility_1y"]["result"] == "pass"
    for key in ("max_drawdown_5y", "daily_es95", "holding_correlation", "downside_beta"):
        assert g[key]["result"] == "unknown"


def test_old_existing_and_new_proposed_portfolios_independent():
    inputs, _ = risk_inputs()
    inputs["portfolio"]["weights"] = {"TEST.US": 1.0}
    result = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in result["gates"]}
    for key in ("holding_correlation", "portfolio_correlation", "downside_beta"):
        assert g[key]["value"] is not None
    assert g["euler_variance_fraction"]["value"] == pytest.approx(1.0)
    inputs["portfolio"].pop("existing_weights")
    result = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in result["gates"]}
    assert g["portfolio_correlation"]["result"] == "unknown"
    assert g["downside_beta"]["value"] is not None


def test_risk_gate_coverage_uses_its_actual_panel():
    inputs, _ = risk_inputs()
    inputs["portfolio"]["weights"] = {"MISSING.US": 1.0}
    result = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in result["gates"]}
    for key in ("holding_correlation", "downside_beta"):
        assert g[key]["value"] is not None
        assert g[key]["freshness"]["aligned_sessions"] > 1000
        assert g[key]["freshness"]["coverage"] == 1
        assert "MISSING.US" not in g[key]["freshness"]["instruments"]
        assert g[key]["period"][0] > "2021-10-27"
    assert g["volatility_1y"]["freshness"]["expected_sessions"] < 270
    assert g["euler_variance_fraction"]["result"] == "unknown"


def test_one_year_calendar_suffices_only_for_one_year_volatility():
    inputs, _ = risk_inputs()
    inputs["portfolio"]["sessions"] = [
        d for d in inputs["portfolio"]["sessions"] if d > "2025-10-27"
    ]
    result = s.screen(inputs, AS_OF, selected={"TEST.US"})[0]
    g = {g["gate"]: g for g in result["gates"]}
    assert g["volatility_1y"]["result"] == "pass"
    assert g["max_drawdown_5y"]["result"] == "unknown"
    assert g["downside_beta"]["result"] == "unknown"


def test_two_runs_rotate_using_learned_verified_caps_without_mutating_seed(tmp_path, monkeypatch):
    records = [record(f"{name}.US") for name in "ABCD"]
    for r in records:
        r["evidence"].pop("market_cap_usd")
        r["capitalization_basis"] = dict(
            verified=True,
            source="synthetic cap currency",
            field="Highlights.MarketCapitalization",
            currency="USD",
        )
    source = tmp_path / "input.json"
    original = json.dumps(dict(synthetic=True, universe=records))
    source.write_text(original)
    path = tmp_path / "config.json"
    cfg = dict(
        input=str(source),
        state_root=str(tmp_path / "service"),
        batch_size=4,
        acquisition=dict(
            enabled=True,
            initial_credit_limit=40,
            daily_credit_limit=40,
            request_credit_cost=dict(fundamentals=10, prices=1),
        ),
    )
    path.write_text(json.dumps(cfg))
    caps = dict(zip([r["ticker"] for r in records], [1e12, 1e8, 1e9, 1e11]))

    def acquire(config, universe, selected, root, as_of):
        cache = root / "source-cache"
        cache.mkdir(exist_ok=True)
        receipts = []
        for r in universe:
            body = service.encoded(
                dict(
                    General=dict(
                        Code=r["ticker"].split(".")[0],
                        Exchange="NASDAQ",
                        CurrencyCode="USD",
                        Type="Common Stock",
                    ),
                    Highlights=dict(MarketCapitalization=caps[r["ticker"]]),
                )
            )
            name = service.digest(body) + ".raw.json"
            (cache / name).write_text(body)
            receipts.append(
                dict(
                    ticker=r["ticker"],
                    dataset="fundamentals",
                    file=name,
                    sha256=service.digest(body),
                    as_of=AS_OF,
                    retrieved_at=AS_OF + "T23:30:00+00:00",
                    reserved_credits=0,
                )
            )
        return receipts

    monkeypatch.setattr(service, "acquire", acquire)
    first = service.run(path, as_of=AS_OF)
    assert all(
        r["evidence"]["market_cap_usd"]["verified"]
        for r in service.read(first.parent / "input.json")["universe"]
    )
    cfg["batch_size"] = 2
    path.write_text(json.dumps(cfg))
    second = service.run(path, as_of="2026-10-28")
    assert service.read(second.parent / "selection.json") == ["D.US", "A.US"]
    assert source.read_text() == original
    assert all(
        "market_cap_usd" not in r["evidence"]
        for r in service.read(second.parent / "original-input.json")["universe"]
    )
    assert service.verify(second.parent)["offline_replayed"]
    state = (tmp_path / "service/state.json").read_bytes()
    assert service.run(path, as_of="2026-10-28") == second
    assert (tmp_path / "service/state.json").read_bytes() == state


@pytest.mark.parametrize(
    "changes",
    [
        dict(verified=False),
        dict(currency="EUR"),
        dict(source=""),
        dict(observed_at="2026-01-01", available_at="2026-01-01"),
        dict(observed_at="2026-10-29"),
        dict(oldest_source_observed_at="2026-01-01"),
    ],
)
def test_rotation_learned_caps_reject_unverified_stale_or_incompatible_evidence(tmp_path, changes):
    seed = [record("A.US"), record("Z.US")]
    for r in seed:
        r["evidence"].pop("market_cap_usd")
    learned = copy.deepcopy(seed)
    learned[1]["evidence"]["market_cap_usd"] = (
        ev(1e12, currency="USD", **changes) if "currency" not in changes else ev(1e12, **changes)
    )
    body = service.encoded(dict(universe=learned))
    (tmp_path / "input.json").write_text(body)
    service.atomic(tmp_path / "manifest.json", dict(files={"input.json": service.digest(body)}))
    state = dict(last_good=str(tmp_path), rotation=1)
    assert service.choose_batch(dict(universe=seed), state, 1, as_of=AS_OF) == ["A.US"]
    learned[1]["evidence"]["market_cap_usd"] = ev(1e12, currency="USD")
    body = service.encoded(dict(universe=learned))
    (tmp_path / "input.json").write_text(body)
    service.atomic(tmp_path / "manifest.json", dict(files={"input.json": service.digest(body)}))
    assert service.choose_batch(dict(universe=seed), state, 1, as_of=AS_OF) == ["Z.US"]
    learned[1]["issuer_id"] = "different-issuer"
    body = service.encoded(dict(universe=learned))
    (tmp_path / "input.json").write_text(body)
    service.atomic(tmp_path / "manifest.json", dict(files={"input.json": service.digest(body)}))
    assert service.choose_batch(dict(universe=seed), state, 1, as_of=AS_OF) == ["A.US"]


def test_rotation_usable_external_cap_wins_and_prior_input_must_match_hash(tmp_path):
    seed = [record("A.US"), record("B.US"), record("Z.US")]
    seed[0]["evidence"]["market_cap_usd"] = ev(1e8, currency="USD")
    seed[1]["evidence"].pop("market_cap_usd")
    seed[2]["evidence"]["market_cap_usd"] = ev(1e12, currency="USD")
    learned = copy.deepcopy(seed)
    learned[2]["evidence"]["market_cap_usd"] = ev(1e6, currency="USD")
    body = service.encoded(dict(universe=learned))
    (tmp_path / "input.json").write_text(body)
    service.atomic(tmp_path / "manifest.json", dict(files={"input.json": service.digest(body)}))
    state = dict(last_good=str(tmp_path), rotation=1)
    assert service.choose_batch(dict(universe=seed), state, 1, as_of=AS_OF) == ["Z.US"]
    (tmp_path / "input.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        service.choose_batch(dict(universe=seed), state, 1, as_of=AS_OF)


def test_identity_quarantine_continues_batch_freezes_original_and_blocks_all_metrics(tmp_path):
    bad, good = record("BAD.US"), record("GOOD.US")
    bad["evidence"]["daily_total_returns_usd"] = ev(
        [dict(date=AS_OF, value=0.01)],
        currency="USD",
        frequency="daily",
        return_basis="total_return",
    )
    source, cfg_path = tmp_path / "input.json", tmp_path / "config.json"
    original = json.dumps(dict(synthetic=True, universe=[bad, good]))
    source.write_text(original)
    cfg = dict(
        input=str(source),
        state_root=str(tmp_path / "service"),
        batch_size=2,
        acquisition=dict(
            enabled=True,
            initial_credit_limit=200,
            daily_credit_limit=200,
            bootstrap_fraction=0.5,  # two cold-name metadata slots: BAD and GOOD
            request_credit_cost=dict(fundamentals=10, prices=1),
        ),
    )
    cfg_path.write_text(json.dumps(cfg))
    calls = []

    def request(session, endpoint, params, expected_type):
        calls.append(endpoint)
        if endpoint == "user":
            return dict(apiRequests=0, dailyRateLimit=100000, apiRequestsDate=AS_OF)
        if endpoint.startswith("fundamentals/"):
            ticker = endpoint.split("/")[1]
            return dict(
                General=dict(
                    Code="RENAMED" if ticker == "BAD.US" else "GOOD",
                    Exchange="NASDAQ",
                    CurrencyCode="USD",
                    Type="Common Stock",
                )
            )
        days = pd.bdate_range("2021-10-01", AS_OF)
        return [dict(date=str(d.date()), adjusted_close=100, close=100) for d in days]

    original_acquire = service.acquire
    # This local monkeypatch is scoped explicitly so the service still exercises real acquisition.
    from unittest.mock import patch

    with patch.object(service, "acquire", lambda *args: original_acquire(*args, request=request)):
        report = service.run(cfg_path, as_of=AS_OF)
    assert "eod/BAD.US" not in calls and "eod/GOOD.US" in calls
    results = {r["ticker"]: r for r in service.read(report.parent / "results.json")}
    assert set(results) == {"BAD.US", "GOOD.US"}
    held = results["BAD.US"]
    assert held["stage"] == "screen_hold" and held["cap_bucket"] is None
    assert all(g["result"] == "unknown" for g in held["gates"])
    assert all(v is None for v in held["metrics"].values())
    assert results["GOOD.US"]["metrics"]["revenue_3y_cagr"] == pytest.approx(0.07)
    quarantine = held["identity_quarantine"]
    assert quarantine["expected"]["ticker"] == "BAD.US"
    assert quarantine["observed"]["Code"] == "RENAMED"
    assert (
        service.digest((report.parent / "sources" / quarantine["file"]).read_bytes())
        == quarantine["sha256"]
    )
    assert service.read(report.parent / "identity-quarantines.json")[0]["reason"]
    queue = {r["ticker"]: r for r in service.read(report.parent / "manual-review-queue.json")}
    assert queue["BAD.US"]["identity_quarantine"]["reason"]
    assert source.read_text() == original
    frozen_original = service.read(report.parent / "original-input.json")["universe"][0]
    assert frozen_original["evidence"] == bad["evidence"]
    assert "identity_quarantine" not in frozen_original
    assert service.verify(report.parent)["offline_replayed"]
    # Same-day quarantine suppresses all repeat spending for the bad issuer, including prices.
    retry_records = [record("BAD.US")]
    previous_calls = list(calls)
    receipts = service.acquire(
        cfg["acquisition"], retry_records, ["BAD.US"], tmp_path / "service", AS_OF, request=request
    )
    assert calls == previous_calls and receipts[0]["dataset"] == "identity_quarantine"
    assert retry_records[0]["evidence"] == {}


def test_identity_quarantine_never_leaks_curated_peer_or_risk_metrics():
    records = [record(f"T{i}.US") for i in range(21)]
    records[-1]["identity_quarantine"] = dict(
        reason="synthetic issuer contradiction", source="synthetic raw receipt", as_of=AS_OF
    )
    results = s.screen(dict(universe=records), AS_OF)
    held = next(r for r in results if r["ticker"] == "T20.US")
    assert all(value is None for value in held["metrics"].values())
    assert held["stage"] == "screen_hold"
    # Twenty usable peers remain; the quarantined issuer cannot supply the twenty-first value.
    for result in results:
        if result["ticker"] != held["ticker"]:
            assert all(row["peer_n"] <= 20 for row in result["percentiles"])
    for key in ("issuer_identity", "pe", "market_cap_usd"):
        assert s.evidence(records[-1], key, s.day(AS_OF))[1] is False


@pytest.mark.parametrize("mode", ["hydrate", "normalize"])
def test_cached_identity_conflict_is_quarantined_per_issuer(tmp_path, monkeypatch, mode):
    records = [record("BAD.US"), record("GOOD.US")]
    cache = tmp_path / "service/source-cache"
    cache.mkdir(parents=True)
    receipts = []
    for r in records:
        payload = dict(
            General=dict(
                Code="OTHER" if r["ticker"] == "BAD.US" else "GOOD",
                Exchange="NASDAQ",
                CurrencyCode="USD",
                Type="Common Stock",
            )
        )
        body = service.encoded(payload)
        filename = service.digest(body) + ".raw.json"
        (cache / filename).write_text(body)
        receipt = dict(
            ticker=r["ticker"],
            dataset="fundamentals",
            file=filename,
            sha256=service.digest(body),
            as_of=AS_OF,
            retrieved_at=AS_OF + "T23:30:00+00:00",
            reserved_credits=0,
        )
        service.atomic(cache / (service.digest(r["ticker"] + ":fundamentals") + ".json"), receipt)
        receipts.append(receipt)
    cfg = dict(
        initial_credit_limit=40,
        daily_credit_limit=40,
        request_credit_cost=dict(fundamentals=10, prices=1),
    )
    if mode == "hydrate":
        service.acquire(
            cfg,
            records,
            [],
            tmp_path / "service",
            AS_OF,
            request=lambda *a, **k: pytest.fail("hydration must not request"),
        )
        assert records[0]["identity_quarantine"] and records[0]["evidence"] == {}
        assert records[1]["evidence"]["pe"]["verified"]
    else:
        source, path = tmp_path / "input.json", tmp_path / "config.json"
        source.write_text(json.dumps(dict(universe=records, synthetic=True)))
        path.write_text(
            json.dumps(
                dict(
                    input=str(source),
                    state_root=str(tmp_path / "service"),
                    batch_size=2,
                    acquisition=dict(enabled=True, **cfg),
                )
            )
        )
        monkeypatch.setattr(service, "acquire", lambda *a, **k: receipts)
        report = service.run(path, as_of=AS_OF)
        results = {r["ticker"]: r for r in service.read(report.parent / "results.json")}
        assert results["BAD.US"]["identity_quarantine"]
        assert results["GOOD.US"]["metrics"]["revenue_3y_cagr"] == pytest.approx(0.07)


@pytest.mark.parametrize("failure", ["auth", "schema"])
def test_auth_and_malformed_general_remain_globally_fatal_with_last_good_preserved(
    tmp_path, monkeypatch, failure
):
    path = config(tmp_path)
    previous = service.run(path, as_of=AS_OF)
    state_path = tmp_path / "service/state.json"
    state = state_path.read_bytes()
    cfg = json.loads(path.read_text())
    cfg["acquisition"] = dict(
        enabled=True,
        initial_credit_limit=40,
        daily_credit_limit=40,
        request_credit_cost=dict(fundamentals=10, prices=1),
    )
    path.write_text(json.dumps(cfg))

    def request(session, endpoint, params, expected_type):
        if endpoint == "user":
            return dict(apiRequests=0, dailyRateLimit=100000, apiRequestsDate=AS_OF)
        if failure == "auth":
            raise ValueError("EODHD HTTP 403: acquisition stopped")
        return dict(error="provider error object is not company metadata")

    original = service.acquire
    monkeypatch.setattr(service, "acquire", lambda *args: original(*args, request=request))
    with pytest.raises(ValueError, match="403|well-formed General"):
        service.run(path, as_of=AS_OF)
    assert state_path.read_bytes() == state
    assert service.status(tmp_path / "service")["last_good"] == str(previous.parent)


@pytest.mark.parametrize("derived", [False, True])
def test_companyfacts_conflict_aux_cash_and_restart(tmp_path, monkeypatch, derived):
    """Real source normalizers feed a frozen run; auxiliary holdings stay out of selection."""
    issuer = record("A.US")
    issuer["identifiers"] = {"cik": "1"}
    issuer["sec_accounting"] = dict(
        verified=True,
        source="filing:accounting",
        framework="US_GAAP",
        currency="USD",
        per_share_basis="period_end",
    )
    issuer["evidence"].pop("diluted_eps", None)
    facts, bases, filings = [], {}, []
    for year in range(2021, 2025):
        accession = f"0000000001-{year % 100:02d}-000001"
        facts.append(
            dict(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                filed=f"{year + 1}-02-01",
                val=float(year - 2019),
                form="10-K",
                accn=accession,
            )
        )
        bases[accession] = dict(
            verified=True,
            source=f"filing/{accession}",
            currency="USD",
            share_basis="diluted",
            ticker="A.US",
            share_basis_at=f"{year}-12-31",
        )
        filings.append(
            dict(
                period_of_report=f"{year}-12-31",
                filed_at=f"{year + 1}-02-01",
                accession_number=accession,
                cik="1",
                eps_diluted=float(year - 2020),
            )
        )
    issuer["eps_listing_basis"] = dict(
        verified=True,
        source="synthetic issuer:share-class",
        ticker="A.US",
        currency="USD",
        share_basis="diluted",
        **{"from": "2020-01-01", "through": "2026-10-26"},
    )
    if derived:
        issuer["eps_listing_basis"] = dict(
            verified=True,
            source="issuer:share-class",
            ticker="A.US",
            currency="USD",
            share_basis="diluted",
            **{"from": "2020-01-01", "through": "2026-10-26"},
        )
    else:
        issuer["companyfacts_share_basis"] = bases
    held = record("ADR.US")
    held["issuer_id"] = issuer["issuer_id"]
    inputs = dict(
        synthetic=True,
        universe=[issuer],
        portfolio=dict(
            instruments=[held],
            holdings=["ADR.US"],
            existing_weights={"ADR.US": 0.25},
            cash_weight=0.75,
            cash_currency="USD",
        ),
    )
    input_path, config_path = tmp_path / "input.json", tmp_path / "config.json"
    input_path.write_text(json.dumps(inputs))
    config_path.write_text(
        json.dumps(
            dict(
                input=str(input_path),
                state_root=str(tmp_path / "state"),
                batch_size=1,
                acquisition=dict(
                    enabled=True,
                    mode="daily",
                    initial_credit_limit=40000,
                    daily_credit_limit=100,
                    request_credit_cost=dict(fundamentals=10, prices=1),
                ),
            )
        )
    )
    calls = []

    def acquire(config, records, selected, root, as_of):
        calls.append((records, selected))
        assert selected == ["ADR.US", "A.US"]
        cache = root / "source-cache"
        cache.mkdir(exist_ok=True)
        documents = {
            "fundamentals": {
                "General": {
                    "Code": "A",
                    "Exchange": "NASDAQ",
                    "CurrencyCode": "USD",
                    "Type": "Common Stock",
                    "CIK": "1",
                    "HomeCategory": "Domestic",
                    "PrimaryTicker": "A.US",
                }
            },
            "sec_annual": {"data": filings},
            "sec_companyfacts": {
                "cik": 1,
                "facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": facts}}}},
            },
            "splits": [],
        }
        receipts = []
        for dataset, payload in documents.items():
            body = service.encoded(payload)
            name = service.digest(body) + ".raw.json"
            (cache / name).write_text(body)
            receipts.append(
                dict(
                    ticker="A.US",
                    dataset=dataset,
                    file=name,
                    sha256=service.digest(body),
                    as_of=as_of,
                    market_as_of="2026-10-26",
                    retrieved_at=as_of + "T23:30:00Z",
                    endpoint={"fundamentals": "fundamentals/A.US", "splits": "splits/A.US"}.get(
                        dataset
                    ),
                    parameters={"from": "2020-01-01", "to": "2026-10-26"},
                    reserved_credits=0,
                )
            )
        return receipts

    monkeypatch.setattr(service, "acquire", acquire)
    report = service.run(config_path, as_of=AS_OF)
    frozen = service.read(report.parent / "input.json")
    eps = frozen["universe"][0]["evidence"]["diluted_eps"]
    assert eps["verified"] is False and eps["conflict"]["periods"]
    if derived:
        assert "companyfacts_share_basis" not in frozen["universe"][0]
    assert frozen["portfolio"]["cash_weight"] == 0.75
    assert service.read(report.parent / "selection.json") == ["A.US"]
    assert service.read(report.parent / "coverage.json")["auxiliary_instruments"] == 1
    assert service.verify(report.parent)["offline_replayed"]
    assert service.run(config_path, as_of=AS_OF) == report and len(calls) == 1


def test_coverage_identity_counts_require_binding_and_freshness():
    valid = record("A.US")
    no_identity = record("B.US")
    no_identity["evidence"].pop("issuer_identity")
    stale = record("C.US")
    stale["evidence"]["issuer_identity"]["observed_at"] = "2025-01-01"
    files = service.artifacts(
        dict(universe=[valid, no_identity, stale]), {"as_of": AS_OF}, ["A.US", "B.US", "C.US"], []
    )
    coverage = json.loads(files["coverage.json"])
    assert coverage["identity_verified_issuers"] == 1
    assert coverage["raw_verified_flags_by_field"]["market_cap_usd"] == 3
    assert coverage["fresh_sourced_fields_by_name"]["issuer_identity"] == 1


def test_run_health_counts_only_fresh_reserved_instruments_and_datasets():
    receipts = [
        {"ticker": "A.US", "dataset": "prices", "reserved_credits": 3, "cache_hit": False},
        {"ticker": "EURUSD.FOREX", "dataset": "fx", "reserved_credits": 3, "cache_hit": False},
        {
            "ticker": "B.US",
            "dataset": "sec_public_concept",
            "reserved_public_requests": 1,
            "cache_hit": True,
        },
        {"ticker": "C.US", "dataset": "fundamentals", "reserved_credits": 0, "cache_hit": False},
    ]
    health = service.run_health({"as_of": "2026-10-28"}, receipts, {})
    assert health["attempted_instruments"] == 1
    assert health["datasets_attempted"] == ["fx", "prices"]


def test_weighted_admission_holds_ties_small_and_unknown_cohorts():
    records = [s.evaluate(record(f"T{i:02}.US"), AS_OF) for i in range(20)]
    for r in records:
        for g in r["gates"]:
            g["result"] = "pass"
    ranked = s.stage_and_rank(records)
    assert all(r["stage"] == "screen_percentile_hold" for r in ranked)
    assert all(r["composite_percentile"] == 50 for r in ranked)
    records[0]["cap_bucket"] = None
    ranked = s.stage_and_rank(records)
    assert all(r["composite_percentile"] is None for r in ranked)
    assert all(r["stage"] == "screen_percentile_hold" for r in ranked)


def test_run_seconds_default_and_validation(tmp_path):
    assert service.ACQUISITION_SECONDS == 3300 == service.run_seconds({})
    path = tmp_path / "c.yaml"
    path.write_text("input: i\nstate_root: s\nacquisition: {enabled: false, run_seconds: 45000}\n")
    assert service.load_config(path)["acquisition"]["run_seconds"] == 45000
    for bad in ("true", "'600'", "599", "46801", "3300.5"):
        path.write_text(f"input: i\nstate_root: s\nacquisition: {{run_seconds: {bad}}}\n")
        with pytest.raises(ValueError, match="run_seconds"):
            service.load_config(path)


def test_run_seconds_sets_acquire_deadline(tmp_path, monkeypatch):
    def request(session, endpoint, params, expected_type):
        if endpoint == "user":
            return {"apiRequests": 0, "dailyRateLimit": 100000, "apiRequestsDate": AS_OF}
        return {"General": {"Code": "TEST", "Exchange": "NASDAQ"}} if expected_type is dict else []

    def statuses(extra, root):
        ticks = iter([0.0])  # clock jumps to 4000s after the deadline is set
        monkeypatch.setattr(service.time, "monotonic", lambda: next(ticks, 4000.0))
        cfg = {
            "initial_credit_limit": 33,
            "daily_credit_limit": 33,
            "request_credit_cost": {"fundamentals": 10, "prices": 1},
            "account_headroom": 25000,
            **extra,
        }
        receipts = service.acquire(cfg, [record()], ["TEST.US"], root, AS_OF, request=request)
        return {r["status"] for r in receipts if "status" in r}

    assert "time_hold" in statuses({}, tmp_path / "a")
    assert "time_hold" not in statuses({"run_seconds": 45000}, tmp_path / "b")
