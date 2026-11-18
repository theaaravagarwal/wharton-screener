"""Wholly synthetic source contracts and adverse variants; no API calls."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from wharton import screen_service as service
from wharton import screen_sources as sources
from wharton import screening

FIXTURES = Path(__file__).parent / "fixtures/screen_sources"
AS_OF = "2026-10-28"
NOW = AS_OF + "T02:50:00+00:00"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def receipt(name="synthetic-industrial-10k.json", **extra):
    return dict(file=name, as_of=AS_OF, retrieved_at=NOW, **extra)


def record(ticker="SYNTHIND.US", cik="9900000001", kind="stock"):
    return dict(
        ticker=ticker,
        issuer_id=ticker,
        kind=kind,
        sector="Technology",
        region="US",
        identifiers={"cik": cik},
        sec_accounting=dict(
            verified=True,
            source="test-only reviewed assertion; not an approval",
            framework="US_GAAP",
            currency="USD",
            per_share_basis="period_end",
        ),
        evidence={},
        reviews={},
    )


def synthetic_proofs(rec, *payloads):
    """Test-only issuer/listing assertions; synthetic facts establish no real-world claim."""
    rec["eps_listing_basis"] = dict(
        verified=True,
        source="synthetic listing assertion",
        ticker=rec["ticker"],
        currency="USD",
        share_basis="diluted",
        **{"from": "2000-01-01", "through": AS_OF},
    )
    rec["sec_issuer_bindings"] = {
        row["accession_number"]: dict(
            verified=True, cik=rec["identifiers"]["cik"], source="synthetic accession assertion"
        )
        for payload in payloads
        for row in payload["data"]
    }
    return rec


def split_receipt():
    return receipt("splits.json", parameters={"from": "2000-01-01", "to": AS_OF})


def metadata(
    ticker="SYNTHIND", exchange="NASDAQ", currency="USD", industry="Software - Infrastructure"
):
    return dict(
        General=dict(
            Code=ticker,
            Exchange=exchange,
            CurrencyCode=currency,
            Type="Common Stock",
            CIK="9900000001",
            Industry=industry,
            Sector="Technology",
        ),
        SharesStats=dict(SharesOutstanding=400e6),
        Valuation=dict(TrailingPE=20),
        Financials={},
    )


def price_rows(
    dates=("2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28"), values=(100, 110, 121, 999)
):
    return [dict(date=d, close=v, adjusted_close=v) for d, v in zip(dates, values)]


def test_synthetic_fixture_manifest_and_field_contracts():
    provenance = load("provenance.json")
    assert provenance["synthetic"] is True
    assert set(provenance["files"]) == {
        path.name for path in FIXTURES.glob("*.json") if path.name != "provenance.json"
    }
    for name, source in provenance["files"].items():
        assert source["synthetic"] is True and source["purpose"]
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == source["sha256"]
        payload = load(name)
        assert payload["synthetic"] is True
        if "data" not in payload:
            continue
        assert set(payload) == {"synthetic", "data", "meta", "links"}
        assert payload["meta"]["total"] == len(payload["data"])
        for row in payload["data"]:
            assert {
                "accession_number",
                "filed_at",
                "period_of_report",
                "eps_diluted",
                "weighted_avg_shares_diluted",
            } <= row.keys()
            assert row["accession_number"].startswith("990000000")
    assert len(load("synthetic-industrial-10k.json")["data"]) == 4
    assert len(load("synthetic-bank-10k.json")["data"]) == 1


@pytest.mark.parametrize(
    "name,ticker,cik,kind",
    [
        ("synthetic-industrial-10k.json", "SYNTHIND.US", "9900000001", "stock"),
        ("synthetic-insurer-10k.json", "SYNTHINS.US", "9900000002", "insurer"),
        ("synthetic-bank-10k.json", "SYNTHBNK.US", "9900000003", "bank"),
    ],
)
def test_synthetic_industrial_insurer_bank_sec_mapping(name, ticker, cik, kind):
    payload, rec = load(name), record(ticker, cik, kind)
    assert sources.normalize_sec(payload, rec, receipt(name), [], split_receipt()) == {}
    synthetic_proofs(rec, payload)
    mapped = sources.normalize_sec(payload, rec, receipt(name), [], split_receipt())
    assert mapped["diluted_eps"]["basis"] == "reported_diluted"
    assert mapped["diluted_eps"]["statement_source"] == "sec_financials"
    assert mapped["diluted_shares"]["basis"] == "weighted_average_diluted_split_adjusted"
    assert (
        mapped["diluted_shares"]["value"][-1]["value"]
        == payload["data"][0]["weighted_avg_shares_diluted"]
    )
    rec["evidence"] = mapped
    if kind == "bank":
        assert screening.annual(rec, "diluted_eps", screening.day(AS_OF), 4) is None
    else:
        assert len(screening.annual(rec, "diluted_eps", screening.day(AS_OF), 4)) == 4
    rec.pop("sec_accounting")
    assert sources.normalize_sec(payload, rec, receipt(name), [], split_receipt()) == {}


def test_sec_vintage_selection_identity_split_and_incomplete_coverage():
    raw, rec = load("synthetic-industrial-10k.json"), record()
    original = raw["data"][0]
    amendment = dict(
        original, accession_number="9900000001-26-999999", eps_diluted=99, filed_at="2026-11-01"
    )
    raw["data"].append(amendment)
    synthetic_proofs(rec, raw)
    mapped = sources.normalize_sec(raw, rec, receipt(), [], split_receipt())
    assert mapped["diluted_eps"]["value"][-1]["value"] == original["eps_diluted"]
    amendment["filed_at"] = original["filed_at"]
    mapped = sources.normalize_sec(raw, rec, receipt(), [], split_receipt())
    assert original["period_of_report"] not in mapped["diluted_eps"]["period"]
    rec["sec_vintages"] = {original["period_of_report"]: original["accession_number"]}
    splits = [{"date": "2026-06-01", "split": "2.000000/1.000000"}]
    mapped = sources.normalize_sec(raw, rec, receipt(), splits, split_receipt())
    assert mapped["diluted_eps"]["value"][-1]["value"] == original["eps_diluted"] / 2
    assert (
        mapped["diluted_shares"]["value"][-1]["value"]
        == original["weighted_avg_shares_diluted"] * 2
    )
    short = split_receipt()
    short["parameters"]["from"] = "2026-01-01"
    assert "diluted_eps" not in sources.normalize_sec(raw, rec, receipt(), [], short)
    with pytest.raises(ValueError, match="endpoint listing identity mismatch"):
        sources.normalize_sec(
            raw, rec, receipt(endpoint="sec-filings/WRONG.US/10k"), [], split_receipt()
        )


def test_current_identity_financial_kind_pe_and_no_generic_share_cap():
    rec = record()
    meta = metadata(industry="Insurance - Property & Casualty")
    mapped = service.normalize_fundamentals(meta, rec, receipt("fund.json"))
    assert rec["kind"] == "insurer"
    assert mapped["issuer_identity"]["value"]["cik"] == "9900000001"
    assert mapped["pe"]["value"] == 20
    assert mapped["pe"]["valuation_basis"] == "provider trailing TTM P/E"
    assert "market_cap_usd" not in mapped
    meta["General"]["CIK"] = "2"
    with pytest.raises(ValueError, match="cik mismatch"):
        service.normalize_fundamentals(meta, rec, receipt("fund.json"))


def test_completed_sessions_excludes_open_bar_and_foreign_fx_is_date_matched():
    mapped = sources.normalize_global_prices(
        price_rows(), metadata(), record(), receipt("price.json")
    )
    value = mapped["daily_total_returns_usd"]
    assert value["last_completed_session"] == "2026-10-27"
    assert max(r["date"] for r in value["value"]) == "2026-10-27"
    assert value["value"][-1]["value"] == pytest.approx(0.1)
    foreign = record("ALFA.ST")
    native = metadata("ALFA", "ST", "SEK")
    fx = price_rows(values=(0.1, 0.11, 0.121, 1))
    assert sources.normalize_global_prices(price_rows(), native, foreign, receipt()) == {}
    out = sources.normalize_global_prices(
        price_rows(),
        native,
        foreign,
        receipt(),
        fx=fx,
        fx_receipt=receipt("fx.json", symbol="SEKUSD.FOREX"),
    )
    assert out["daily_total_returns_usd"]["value"][-1]["value"] == pytest.approx(0.21)
    gap = [r for r in fx if r["date"] != "2026-10-26"]
    out = sources.normalize_global_prices(
        price_rows(),
        native,
        foreign,
        receipt(),
        fx=gap,
        fx_receipt=receipt("fx.json", symbol="SEKUSD.FOREX"),
    )
    assert out["daily_total_returns_usd"]["value"] == []
    native["General"]["Exchange"] = "UNSUPPORTED"
    foreign["ticker"] = "ALFA.UNSUPPORTED"
    assert sources.normalize_global_prices(price_rows(), native, foreign, receipt()) == {}
    assert sources.normalize_global_prices(price_rows()[:-2], metadata(), record(), receipt()) == {}


def test_ordinary_cap_requires_explicit_applicability_and_exact_close_fx():
    rec = record("ALFA.ST")
    rec["capitalization_basis"] = dict(
        verified=True,
        source="test-only ordinary class review",
        field="SharesStats.SharesOutstanding",
        share_class="ordinary_common",
        price_applies_to_all_shares=True,
        shares_as_of="2026-10-01",
    )
    mapped = sources.current_capitalization(
        price_rows(),
        metadata("ALFA", "ST", "SEK"),
        rec,
        receipt(),
        fundamentals_receipt=receipt("fund.json"),
        splits=[],
        split_receipt=split_receipt(),
        fx=price_rows(values=(0.1, 0.1, 0.1, 0.1)),
        fx_receipt=receipt("fx.json", symbol="SEKUSD.FOREX"),
    )
    assert mapped["market_cap_usd"]["value"] == pytest.approx(121 * 400e6 * 0.1)
    assert "fund.json" in mapped["market_cap_usd"]["source"]
    rec["capitalization_basis"]["share_class"] = "weighted_average_diluted"
    assert sources.current_capitalization(price_rows(), metadata(), rec, receipt()) == {}


def test_annual_valuations_are_diagnostics_and_no_adjusted_close_multiple():
    rows = price_rows()
    for r in rows:
        r["adjusted_close"] = 1
    annual = load("synthetic-industrial-10k.json")
    rec = synthetic_proofs(record(), annual)
    mapped = sources.matched_valuations(
        rows,
        annual,
        rec,
        receipt(),
        receipt("prices.json"),
        [],
        split_receipt(),
    )
    assert "pe" not in mapped and "own_pe_monthly" not in mapped
    assert mapped["diagnostic_annual_pe"]["value"] == pytest.approx(121 / 8)
    assert mapped["diagnostic_annual_pe_monthly"]["value"][-1]["price_date"] == "2026-10-27"


def config(**extra):
    return dict(
        initial_credit_limit=40000,
        daily_credit_limit=7500,
        account_headroom=25000,
        fresh_sources=True,
        request_credit_cost=dict(
            fundamentals=10, prices=1, sec_annual=10, sec_quarterly=10, splits=1, earnings=1, fx=1
        ),
        **extra,
    )


@pytest.fixture
def fixed_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 28, 2, 50, tzinfo=UTC)

    monkeypatch.setattr(service, "datetime", Clock)
    try:
        from wharton import screen_budget
    except ImportError:
        pass
    else:
        monkeypatch.setattr(screen_budget, "datetime", Clock)


def fake_request(calls):
    def request(session, endpoint, params, expected_type):
        calls.append((endpoint, params))
        if endpoint == "user":
            return dict(apiRequests=200, dailyRateLimit=100000, apiRequestsDate=AS_OF)
        if endpoint == "calendar/earnings":
            return dict(
                earnings=[dict(code="SYNTHIND.US", report_date="2026-10-27", date="2026-08-28")]
            )
        if endpoint.endswith("/10k"):
            return load("synthetic-industrial-10k.json")
        if endpoint.endswith("/10q"):
            return load("synthetic-industrial-10q.json")
        if endpoint.startswith("fundamentals/"):
            return metadata()
        if endpoint.startswith("splits/"):
            return []
        return price_rows()

    return request


def test_fresh_endpoints_events_retained_cache_and_total_bootstrap(tmp_path, fixed_clock):
    calls, rec = [], record()
    cfg = config()
    first = service.acquire(
        cfg, [rec], [rec["ticker"]], tmp_path, AS_OF, request=fake_request(calls)
    )
    assert rec["filing_event_at"] == "2026-10-27"
    assert {r["dataset"] for r in first} >= {
        "sec_annual",
        "sec_quarterly",
        "splits",
        "earnings",
        "prices",
        "fundamentals",
    }
    assert all(p.get("page[limit]") == 100 for e, p in calls if e.startswith("sec-filings"))
    retained = service.acquire(
        cfg,
        [rec],
        [],
        tmp_path,
        AS_OF,
        request=lambda *a, **k: pytest.fail("retained cache must not request"),
    )
    assert len(retained) == 6
    ledger = service.read(tmp_path / "acquisition-ledger.json")
    assert ledger["initial_total"] == sum(r["reserved_credits"] for r in first)
    ledger["initial_total"] = 40000
    ledger["2026-10-27"] = 40000
    service.atomic(tmp_path / "acquisition-ledger.json", ledger)
    other = record("NEW.US")
    held = service.acquire(
        cfg,
        [other],
        [other["ticker"]],
        tmp_path,
        AS_OF,
        request=lambda *a, **k: pytest.fail("bootstrap must not reset across dates"),
    )
    assert all(r["status"] == "budget_hold" for r in held)
    with pytest.raises(ValueError, match="75000"):
        service.validate_budget({**cfg, "daily_credit_limit": 75001})


def test_fresh_run_freezes_sources_replays_and_keeps_human_holds(
    tmp_path, fixed_clock, monkeypatch
):
    rec = record()
    input_path, config_path = tmp_path / "input.json", tmp_path / "config.json"
    input_path.write_text(json.dumps(dict(universe=[rec], synthetic=True)))
    cfg = dict(
        input=str(input_path),
        state_root=str(tmp_path / "state"),
        batch_size=1,
        acquisition={"enabled": True, **config()},
    )
    config_path.write_text(json.dumps(cfg))
    original_acquire = service.acquire
    monkeypatch.setattr(
        service, "acquire", lambda *args: original_acquire(*args, request=fake_request([]))
    )
    report = service.run(config_path, as_of=AS_OF)
    assert service.verify(report.parent)["offline_replayed"] is True
    assert (
        "filing_event_at" not in service.read(report.parent / "original-input.json")["universe"][0]
    )
    acquired = service.read(report.parent / "input.json")["universe"][0]
    assert acquired["reviews"] == {}
    assert acquired["event_review_required"]["event_at"]
    result = service.read(report.parent / "results.json")[0]
    gates = {g["gate"]: g for g in result["gates"]}
    assert gates["pe_own"]["result"] == "unknown"
    assert gates["human_thesis"]["result"] == "unknown"
    assert service.run(config_path, as_of=AS_OF) == report


def test_current_sector_median_minimum_and_comparability_with_retained_records():
    records = []
    for i in range(21):
        rec = record(f"T{i}.US")
        meta = metadata(f"T{i}")
        meta["Valuation"]["TrailingPE"] = i + 1
        meta["Highlights"] = {"MarketCapitalization": 2e9}
        rec["capitalization_basis"] = dict(
            verified=True,
            source="test-only provider cap currency review",
            field="Highlights.MarketCapitalization",
            currency="USD",
        )
        rec["evidence"] = service.normalize_fundamentals(meta, rec, receipt("fund.json"))
        records.append(rec)
    sources.sector_medians(records, AS_OF, service.merge_evidence)
    assert records[0]["evidence"]["sector_pe"]["peer_n"] == 20
    assert records[0]["evidence"]["sector_pe"]["value"] == 11.5
    records[-1]["kind"] = "bank"
    sources.sector_medians(records, AS_OF, service.merge_evidence)
    assert "sector_pe" not in records[0]["evidence"]


def test_synthetic_book_point_values_build_monthlies_only_with_explicit_basis():
    rec = record("SYNTHINS.US", "9900000002", "insurer")
    synthetic_proofs(rec, load("synthetic-insurer-10k.json"), load("synthetic-insurer-10q.json"))
    rec["sec_accounting"].update(
        book_equity_basis="common_equity_excluding_preferred", intangibles_exclude_goodwill=True
    )
    dates = sources.completed_sessions("XNYS", AS_OF, NOW)
    prices = [dict(date=str(d.date()), close=200, adjusted_close=1) for d in dates]
    mapped = sources.matched_book_values(
        prices,
        load("synthetic-insurer-10k.json"),
        load("synthetic-insurer-10q.json"),
        rec,
        receipt("synthetic-insurer-10k.json"),
        receipt("synthetic-insurer-10q.json"),
        receipt("price.json"),
        [],
        split_receipt(),
        fundamentals={
            "General": {
                "Code": "SYNTHINS",
                "Exchange": "NYSE",
                "CurrencyCode": "USD",
                "Type": "Common Stock",
                "CIK": "9900000002",
            }
        },
        fundamentals_receipt=receipt("fund.json"),
    )
    assert len(mapped["own_ptbv_monthly"]["value"]) == 60
    rec["evidence"] = mapped
    assert screening.historical_median(rec, "own_ptbv_monthly", screening.day(AS_OF)) > 0
    assert mapped["ptbv"]["value"] == pytest.approx(200 / 114)
    assert mapped["own_ptbv_monthly"]["value"][-1]["earnings_period"] == "2026-06-30"
    assert mapped["own_ptbv_monthly"]["value"][-1]["earnings_available_at"] < "2026-10-27"
    rec["sec_accounting"].pop("intangibles_exclude_goodwill")
    assert (
        sources.matched_book_values(
            prices,
            load("synthetic-insurer-10k.json"),
            load("synthetic-insurer-10q.json"),
            rec,
            receipt(),
            receipt(),
            receipt(),
            [],
            split_receipt(),
        )
        == {}
    )


def test_event_invalidates_older_review_without_manufacturing_approval():
    rec = record()
    rec["filing_event_at"] = "2026-10-27"
    rec["reviews"]["thesis"] = dict(
        author="test",
        reviewer="test",
        status="approved",
        reviewed_at="2026-10-26",
        citations=["test"],
        rationale="test",
        not_mainly_ai_spending=True,
    )
    assert screening.human(rec, "thesis", screening.day(AS_OF))[1] is False
    assert rec["reviews"]["thesis"]["status"] == "approved"


def test_global_native_sessions_preserved_and_joint_intervals_match():
    rec, benchmark = record("ALFA.ST"), record("VT.US")
    native = sources.normalize_global_prices(
        [
            dict(date=str(d.date()), close=100 + i, adjusted_close=100 + i)
            for i, d in enumerate(sources.completed_sessions("XSTO", AS_OF, NOW))
        ],
        metadata("ALFA", "ST", "USD"),
        rec,
        receipt("price.json"),
        fundamentals_receipt=receipt("fund.json"),
    )
    rec["evidence"].update(native)
    reference = sources.completed_sessions("XNYS", AS_OF, NOW)
    market = sources.normalize_global_prices(
        [
            dict(date=str(d.date()), close=100 + i + (i % 2), adjusted_close=100 + i + (i % 2))
            for i, d in enumerate(reference)
        ],
        metadata("VT"),
        benchmark,
        receipt("vt.json"),
        fundamentals_receipt=receipt("fund.json"),
    )
    benchmark["evidence"].update(market)
    results = [dict(ticker=rec["ticker"], kind="stock", gates=[], metrics={})]
    portfolio = dict(
        sessions=[str(d.date()) for d in reference],
        sessions_verified=True,
        session_source="exchange_calendars XNYS",
        benchmark="VT.US",
    )
    screening.portfolio_gates([rec, benchmark], results, portfolio, AS_OF)
    gates = {g["gate"]: g for g in results[0]["gates"]}
    native_index = pd.to_datetime(rec["evidence"]["daily_total_returns_usd"]["native_sessions"])
    expected = native_index[native_index > pd.Timestamp(AS_OF) - pd.DateOffset(years=1)]
    assert gates["volatility_1y"]["freshness"]["expected_sessions"] == len(expected)
    own_count = len(
        [r for r in native["daily_total_returns_usd"]["value"] if r["date"] > "2025-10-28"]
    )
    assert gates["volatility_1y"]["freshness"]["aligned_sessions"] == own_count
    assert set(native_index) - set(reference)  # Foreign sessions on US holidays exist.
    # Joint observations are strictly fewer because unequal actual intervals are withheld.
    assert gates["downside_beta"]["freshness"]["aligned_sessions"] < len(reference)


def test_cap_inherits_oldest_dated_source_and_applies_intervening_split():
    rec = record()
    rec["capitalization_basis"] = dict(
        verified=True,
        source="test-only ordinary share count",
        field="SharesStats.SharesOutstanding",
        share_class="ordinary_common",
        price_applies_to_all_shares=True,
        shares_as_of="2026-10-01",
    )
    split = [{"date": "2026-10-15", "split": "2/1"}]
    mapped = sources.current_capitalization(
        price_rows(),
        metadata(),
        rec,
        receipt("prices.json"),
        fundamentals_receipt=receipt("fund.json"),
        splits=split,
        split_receipt=split_receipt(),
    )
    cap = mapped["market_cap_usd"]
    assert cap["value"] == 121 * 800e6
    assert cap["oldest_source_observed_at"] == "2026-10-01"
    rec["capitalization_basis"]["shares_as_of"] = "2026-08-01"
    assert (
        sources.current_capitalization(
            price_rows(),
            metadata(),
            rec,
            receipt(),
            fundamentals_receipt=receipt(),
            splits=[],
            split_receipt=split_receipt(),
        )
        == {}
    )
    rec["evidence"] = mapped
    cap["oldest_source_observed_at"] = "2026-08-01"
    assert screening.evidence(rec, "market_cap_usd", screening.day(AS_OF))[1] is False


def test_earnings_event_later_than_latest_sec_is_not_reset(tmp_path, fixed_clock, monkeypatch):
    rec = record()
    input_path, config_path = tmp_path / "input.json", tmp_path / "config.json"
    input_path.write_text(json.dumps(dict(universe=[rec], synthetic=True)))
    config_path.write_text(
        json.dumps(
            dict(
                input=str(input_path),
                state_root=str(tmp_path / "state"),
                batch_size=1,
                acquisition={"enabled": True, **config()},
            )
        )
    )
    original_acquire = service.acquire
    monkeypatch.setattr(
        service, "acquire", lambda *args: original_acquire(*args, request=fake_request([]))
    )
    report = service.run(config_path, as_of=AS_OF)
    acquired = service.read(report.parent / "input.json")["universe"][0]
    assert acquired["filing_event_at"] == "2026-10-27"
    assert acquired["event_review_required"]["event_at"] == "2026-10-27"


def test_sec_provider_overlap_conflicts_do_not_confuse_history_coverage():
    raw = load("synthetic-industrial-10k.json")
    sec = sources.normalize_sec(
        raw, synthetic_proofs(record(), raw), receipt(), [], split_receipt()
    )["revenue"]
    provider = copy.deepcopy(sec)
    provider["value"] = provider["value"][-4:]
    assert service.reconcile_statements(provider, sec, "revenue")["verified"] is True
    provider["value"][-1]["value"] += 1
    conflict = service.reconcile_statements(provider, sec, "revenue")
    assert conflict["verified"] is False
    assert conflict["conflict"]["provider_evidence"] == provider
    assert conflict["conflict"]["sec_evidence"] == sec


@pytest.mark.parametrize("status", [401, 403, 429])
def test_auth_and_rate_errors_stop_immediately(status):
    class Session:
        def get(self, *args, **kwargs):
            return type("Response", (), {"status_code": status})()

    with pytest.raises(ValueError, match=str(status)):
        service.StopSession(Session()).get("redacted")


def test_unsupported_source_is_a_budgeted_coverage_hold(tmp_path, fixed_clock):
    calls = []
    real = fake_request(calls)

    def request(session, endpoint, params, expected_type):
        if endpoint.startswith("sec-filings"):
            raise service.SourceUnavailable("missing coverage")
        return real(session, endpoint, params, expected_type)

    receipts = service.acquire(
        config(), [record()], ["SYNTHIND.US"], tmp_path, AS_OF, request=request
    )
    holds = [r for r in receipts if r.get("status") == "coverage_hold"]
    assert len(holds) == 2 and sum(r["reserved_credits"] for r in holds) == 60
    assert any(r.get("dataset") == "prices" and r.get("file") for r in receipts)


def test_valid_us_otc_identity_with_unsupported_calendar_is_unknown():
    rec = record("AAIGF.US")
    meta = metadata("AAIGF", "PINK", "USD", "Insurance - Life")
    assert service.listing_identity(meta, rec)["Exchange"] == "PINK"
    assert sources.normalize_global_prices(price_rows(), meta, rec, receipt()) == {}
    meta["General"]["Exchange"] = "ST"
    with pytest.raises(ValueError, match="identity mismatch"):
        sources.normalize_global_prices(price_rows(), meta, rec, receipt())


def test_same_day_negative_paid_cache_does_not_repeat_missing_coverage(tmp_path, fixed_clock):
    calls = []
    real = fake_request(calls)
    missing = []

    def request(session, endpoint, params, expected_type):
        if endpoint.startswith("sec-filings"):
            missing.append(endpoint)
            raise service.SourceUnavailable("missing coverage")
        return real(session, endpoint, params, expected_type)

    rec = record()
    service.acquire(config(), [rec], ["SYNTHIND.US"], tmp_path, AS_OF, request=request)
    assert len(missing) == 2
    service.acquire(config(), [rec], ["SYNTHIND.US"], tmp_path, AS_OF, request=request)
    assert len(missing) == 2


@pytest.mark.parametrize(
    "alias,calendar",
    [
        ("AS", "XAMS"),
        ("SW", "XSWX"),
        ("MC", "XMAD"),
        ("MI", "XMIL"),
        ("CO", "XCSE"),
        ("HE", "XHEL"),
        ("OL", "XOSL"),
        ("TSE", "XTKS"),
        ("SG", "XSES"),
        ("KO", "XKRX"),
        ("TW", "XTAI"),
        ("MX", "XMEX"),
        ("SN", "XSGO"),
        ("JSE", "XJSE"),
        ("TA", "XTAE"),
        ("SR", "XSAU"),
    ],
)
def test_installed_explicit_global_calendars_cover_current_five_years(alias, calendar):
    assert sources.CALENDARS[alias] == calendar
    sessions = sources.completed_sessions(calendar, AS_OF, NOW)
    assert len(sessions) > 1100
    assert sessions.min() <= pd.Timestamp("2021-10-28")
    assert sessions.max() <= pd.Timestamp("2026-10-28")


def test_unsupported_or_out_of_bounds_calendars_are_unknown_without_substitution():
    assert "NSE" not in sources.CALENDARS
    assert sources.completed_sessions("XNSE", AS_OF, NOW).empty
    # Installed Saudi calendar has no pre-2021 sessions; an older five-year window is unknown.
    assert sources.completed_sessions("XSAU", "2025-01-15", "2025-01-15T23:00:00+00:00").empty


@pytest.mark.parametrize(
    "field,key,value",
    [("CIK", "cik", "2"), ("OpenFigi", "figi", "WRONG-FIGI"), ("ISIN", "isin", "WRONG-ISIN")],
)
def test_well_formed_identifier_contradictions_use_typed_conflict(field, key, value):
    rec = record()
    rec["identifiers"][key] = "9900000001" if key == "cik" else "EXPECTED-ID"
    meta = metadata()
    meta["General"][field] = value
    with pytest.raises(sources.SourceIdentityConflict) as caught:
        service.normalize_fundamentals(meta, rec, receipt())
    assert caught.value.expected["identifiers"][key] == rec["identifiers"][key]
    assert caught.value.observed[field] == value


def test_malformed_general_and_identifier_are_global_schema_errors():
    for payload in (
        {},
        {"General": []},
        {"General": {"Code": "SYNTHIND"}},
        {"General": {"Code": [], "Exchange": "NASDAQ"}},
    ):
        with pytest.raises(ValueError) as caught:
            service.listing_identity(payload, record())
        assert type(caught.value) is ValueError
    meta = metadata()
    meta["General"]["CIK"] = "not-a-CIK"
    with pytest.raises(ValueError) as caught:
        service.listing_identity(meta, record())
    assert type(caught.value) is ValueError


def test_market_cutoff_excludes_later_completed_bar():
    result = sources.normalize_global_prices(
        price_rows(), metadata(), record(), receipt("price.json", market_as_of="2026-10-26")
    )["daily_total_returns_usd"]
    assert result["last_completed_session"] == "2026-10-26"
    assert all(row["date"] <= "2026-10-26" for row in result["value"])


def test_business_description_is_provider_evidence_only():
    payload, rec = metadata(), record()
    payload["General"]["Description"] = "  Makes software.  "
    result = sources.identity_metadata(payload, rec, receipt())
    assert result["business_description"]["value"] == "Makes software."
    assert result["business_description"]["source"].endswith("#General.Description")
    payload["General"]["Description"] = " "
    assert "business_description" not in sources.identity_metadata(payload, rec, receipt())


def test_consolidated_financials_requires_identity_and_filing_vintage():
    raw, rec = load("synthetic-industrial-10k.json"), record()
    payload = {
        "data": {
            "company": {"cik": "9900000001", "ticker": "SYNTHIND", "exchange": "US"},
            "annual": raw["data"],
            "quarterly": [],
        }
    }
    source = receipt("financials.json", endpoint="sec-companies/SYNTHIND.US/financials")
    assert "diluted_eps" not in sources.normalize_sec_financials(
        payload, rec, source, [], split_receipt()
    )
    synthetic_proofs(rec, raw)
    result = sources.normalize_sec_financials(payload, rec, source, [], split_receipt())
    assert result["diluted_eps"]["value"]
    assert "#data.annual/" in result["diluted_eps"]["value"][0]["source"]
    payload["data"]["annual"] = [{"period": "2025-12-31", "eps_diluted": 2}]
    assert sources.normalize_sec_financials(payload, rec, source, [], split_receipt()) == {}
    payload["data"]["company"]["cik"] = "7"
    with pytest.raises(sources.SourceIdentityConflict):
        sources.normalize_sec_financials(payload, rec, source, [], split_receipt())


def test_sector_medians_skips_records_without_a_supported_pe_basis_before_the_peer_loop(
    monkeypatch,
):
    """Root latency fix: 13,000 records with no PE / unsupported basis do ZERO peer evidence work
    (it was O(N^2) = 169M peer checks), and the supported cohort output is unchanged."""
    from wharton import screening

    calls = []
    real = screening.evidence
    monkeypatch.setattr(screening, "evidence", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    records = [
        {"ticker": f"N{i}.US", "sector": "Tech", "kind": "stock", "evidence": {}}
        for i in range(13000)
    ]
    records[0]["evidence"]["pe"] = {"value": 20.0, "valuation_basis": "reviewed comparable"}
    sources.sector_medians(records, AS_OF, service.merge_evidence)
    assert calls == []  # no PE call at all
    assert all("sector_pe" not in r["evidence"] for r in records)
    stale = {"owner": "eodhd_screen_service", "value": 1}
    records[1]["evidence"]["sector_pe"] = stale
    sources.sector_medians(records, AS_OF, service.merge_evidence)
    assert "sector_pe" not in records[1]["evidence"]  # the owned withdrawal still happens first
