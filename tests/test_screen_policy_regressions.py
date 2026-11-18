"""Independent end-to-end checks of the automatic $5 penny-stock policy.

Synthetic data only; no network, provider or paid calls.
"""

import copy

import pytest
from test_screen_price_pipeline import (  # noqa: F401  (frozen_service_clock is autouse)
    TODAY,
    cache_source,
    cfg,
    frozen_service_clock,
    general,
    prices,
    rec,
    requester,
)
from test_screen_sources import metadata, receipt
from test_screening_service import AS_OF, record, review

from wharton import screen_budget, screen_ranking, screen_sources, screening
from wharton import screen_service as service

# ---- helpers -------------------------------------------------------------

CLOSE_STAMP = AS_OF + "T23:30:00+00:00"  # after the 20:00 UTC NYSE/Nasdaq close on AS_OF


def rcpt(name, **extra):
    return {**receipt(name), "as_of": AS_OF, "retrieved_at": CLOSE_STAMP, **extra}


def quote(close, day=AS_OF):
    # adjusted_close is deliberately wild: it must never decide anything.
    return [{"date": day, "close": close, "adjusted_close": 1000.0}]


def normalized(close, ticker="TEST.US", rows=None, **kwargs):
    rec_ = record(ticker)
    meta = kwargs.pop("meta", metadata(ticker.split(".")[0]))
    return screen_sources.normalize_listing_price(
        quote(close) if rows is None else rows,
        rec_,
        rcpt("prices.json", symbol=ticker),
        fundamentals=meta,
        fundamentals_receipt=rcpt("fundamentals.json"),
        **kwargs,
    )


def with_price(close, **kwargs):
    """Strong-metrics synthetic record whose price evidence came from the real normalizer."""
    r = record()
    out = normalized(close, **kwargs)
    if out:
        r["evidence"].update(out)
    else:
        r["evidence"].pop("listing_price_usd", None)
    return r


def price_gate(r, **kw):
    gates = screening.evaluate(r, AS_OF, **kw)["gates"]
    return next(g for g in gates if g["gate"] == "listing_price_usd")


def ranked_row(ticker, price_result="pass", stage="review_ready", kind="stock", cap="mid", v=1):
    names = list(screen_ranking.SCREEN_GATES) + list(screen_ranking.LATER_GATES)
    gates = [dict(gate=n, result="pass", value=v) for n in names]
    next(g for g in gates if g["gate"] == "listing_price_usd")["result"] = price_result
    return dict(
        ticker=ticker, issuer_id=ticker, kind=kind, cap_bucket=cap, stage=stage, gates=gates
    )


def run_of(rows):
    return {"as_of": AS_OF, "run_id": "r", "results_sha256": "h", "rows": rows}


# ---- 1/3. price boundary end to end: normalizer -> evaluate -> qualification ----


@pytest.mark.parametrize(("close", "result"), [(4.99, "fail"), (5.0, "pass"), (5.01, "pass")])
def test_boundary_end_to_end_through_normalizer_evaluate_and_qualification(close, result):
    r = with_price(close)
    gate = price_gate(r)
    assert gate["result"] == result and gate["value"] == close
    row = screening.screen({"universe": [r]}, AS_OF)[0]
    eligible, _, screen, _ = screen_ranking.qualification(row)
    assert screen["listing_price_usd"] == result
    if result == "fail":
        assert not eligible and row["stage"] == "screen_hold"
        assert "listing_price_usd" in row["blockers"]
    else:
        assert "listing_price_usd" not in row["blockers"]


def test_adjusted_close_is_never_substituted_for_raw_close():
    out = normalized(4.99)["listing_price_usd"]  # adjusted_close is 1000
    assert out["value"] == 4.99
    only_adjusted = [{"date": AS_OF, "adjusted_close": 1000.0}]
    assert normalized(None, rows=only_adjusted) == {}
    state = screen_sources.price_state(
        record(),
        only_adjusted,
        rcpt("prices.json", symbol="TEST.US"),
        fundamentals=metadata("TEST"),
        fundamentals_receipt=rcpt("fundamentals.json"),
        cutoff=AS_OF,
    )
    assert state["status"] == "unknown" and state["price_usd"] is None


def test_foreign_price_boundary_uses_exact_day_raw_fx_close():
    # FX must be retrieved on a later day than the session, so the session is AS_OF - 1 day.
    ticker, day, stamp = "ALFA.ST", "2026-10-26", AS_OF + "T08:00:00+00:00"
    meta = metadata("ALFA", "ST", "SEK")
    px = {**receipt("prices.json"), "as_of": AS_OF, "retrieved_at": stamp, "symbol": ticker}
    fxr = {**receipt("fx.json"), "as_of": AS_OF, "retrieved_at": stamp, "symbol": "SEKUSD.FOREX"}

    def usd(local, rate=0.125, fx_date=day, fx_receipt=fxr):
        return screen_sources.normalize_listing_price(
            quote(local, day),
            record(ticker),
            px,
            fundamentals=meta,
            fundamentals_receipt={**receipt("f.json"), "as_of": AS_OF, "retrieved_at": stamp},
            fx=[{"date": fx_date, "close": rate, "adjusted_close": 99}],
            fx_receipt=fx_receipt,
        )

    assert usd(40)["listing_price_usd"]["value"] == 5
    assert usd(39.92)["listing_price_usd"]["value"] < 5
    assert usd(40, rate=0.1249)["listing_price_usd"]["value"] < 5
    # Wrong-day FX, wrong-symbol FX, zero rate, missing FX: unknown, never a guessed rate.
    assert usd(40, fx_date="2026-10-23") == {}
    assert usd(40, fx_receipt=dict(fxr, symbol="EURUSD.FOREX")) == {}
    assert usd(40, rate=0) == {}
    assert usd(40, fx_receipt=None) == {}


@pytest.mark.parametrize("currency", ["GBX", "GBp", "ILA", "ZAC", "", "US", "usd"])
def test_minor_or_unknown_currency_units_never_reach_a_pass(currency):
    assert normalized(100, meta=metadata("TEST", currency=currency)) == {}


# ---- 2. stale / unknown never pass ---------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        dict(period="2026-10-23"),
        dict(last_completed_session="2026-10-23"),
        dict(expected_session="2026-10-28"),
        dict(receipt_as_of="2026-10-26"),
        dict(currency="EUR"),
        dict(verified=False),
        dict(observed_at="2026-08-01"),
        dict(value=None),
        dict(value=float("nan")),
    ],
)
def test_stale_or_unproved_price_evidence_is_unknown_not_pass_and_not_fail(change):
    r = with_price(100)
    r["evidence"]["listing_price_usd"].update(change)
    gate = price_gate(r)
    assert gate["result"] == "unknown" and gate["value"] is None
    row = screening.screen({"universe": [r]}, AS_OF)[0]
    assert row["stage"] == "screen_hold" and "listing_price_usd" in row["blockers"]
    assert not screen_ranking.price_failed(row)  # unknown is a hold, not an exclusion


def test_missing_stale_future_or_unmatched_source_never_normalizes_to_pass():
    assert normalized(100, rows=[]) == {}
    assert normalized(100, rows=quote(100, "2026-10-23")) == {}  # stale session
    assert normalized(100, rows=quote(100, "2026-10-28")) == {}  # future session
    assert normalized(100, rows=quote(100) * 2) == {}  # duplicate rows
    assert normalized(0) == {} and normalized(-1) == {}
    stale_meta = screen_sources.normalize_listing_price(
        quote(100),
        record(),
        rcpt("prices.json", symbol="TEST.US"),
        fundamentals=metadata("TEST"),
        fundamentals_receipt=rcpt("f.json", retrieved_at="2026-08-01T00:00:00+00:00"),
    )
    assert stale_meta == {}
    wrong_symbol = screen_sources.normalize_listing_price(
        quote(100),
        record(),
        rcpt("prices.json", symbol="OTHER.US"),
        fundamentals=metadata("TEST"),
        fundamentals_receipt=rcpt("f.json"),
    )
    assert wrong_symbol == {}


def test_future_market_session_and_missing_evidence_are_holds():
    r = with_price(100)
    assert price_gate(r)["result"] == "pass"
    assert price_gate(r, market_as_of="2026-10-20")["result"] == "unknown"
    r["evidence"].pop("listing_price_usd")
    assert price_gate(r)["result"] == "unknown"


def test_etf_price_gate_is_not_applicable_and_does_not_block():
    r = record(kind="etf")
    r["evidence"].pop("listing_price_usd", None)
    assert price_gate(r)["result"] == "not_applicable"


# ---- 4. gate noncompensation by rank --------------------------------------


def test_strong_metrics_penny_stock_cannot_qualify_or_advance():
    r = with_price(4.99)  # every other metric is the strong synthetic fixture
    row = screening.screen({"universe": [r]}, AS_OF)[0]
    assert row["stage"] == "screen_hold" and "listing_price_usd" in row["blockers"]
    good = screening.screen({"universe": [with_price(5.0)]}, AS_OF)[0]
    assert good["stage"] != "screen_hold" or "listing_price_usd" not in good["blockers"]


def test_rank_and_review_ready_stage_cannot_offset_price_fail_or_unknown():
    rows = [ranked_row(f"P{i}") for i in range(25)]
    rows += [ranked_row("PENNY", "fail", v=1e9), ranked_row("UNK", "unknown", v=1e9)]
    out = {r["ticker"]: r for r in screen_ranking.build([run_of(rows)])[0]}
    for ticker, status in [("PENNY", "fail"), ("UNK", "unknown")]:
        row = out[ticker]
        assert not row["screen_qualified"] and not row["guidelines_qualified"]
        assert row["screen_gate_status"]["listing_price_usd"] == status
    assert out["PENNY"]["price_ineligible"] is True
    assert out["P0"]["guidelines_qualified"] and out["P0"]["screen_qualified"]


def test_price_failed_names_are_excluded_from_peer_cohort_and_percentile():
    peers = [ranked_row(f"P{i}", v=i + 1) for i in range(20)]
    penny = ranked_row("PENNY", "fail", v=10**6)
    with_penny = {r["ticker"]: r for r in screen_ranking.build([run_of(peers + [penny])])[0]}
    without = {r["ticker"]: r for r in screen_ranking.build([run_of(peers)])[0]}
    assert with_penny["PENNY"]["ranking_cohort"] is None
    assert with_penny["PENNY"]["composite_percentile"] is None
    for p in peers:
        t = p["ticker"]
        assert with_penny[t]["composite_cohort_n"] == without[t]["composite_cohort_n"] == 20
        assert with_penny[t]["composite_percentile"] == without[t]["composite_percentile"]
    # An unknown price is a hold but the name stays a peer until proven below $5.
    unk = {
        r["ticker"]: r
        for r in screen_ranking.build([run_of(peers + [ranked_row("U", "unknown")])])[0]
    }
    assert unk["P0"]["composite_cohort_n"] == 21


def test_stage_and_rank_keeps_penny_out_of_screening_top_ordering():
    results = []
    for i, close in enumerate([4.99, 5.0, 5.01, 50]):
        r = with_price(close)
        r["ticker"] = r["issuer_id"] = f"T{i}.US"
        r["evidence"]["issuer_identity"]["value"] = dict(ticker=r["ticker"], issuer_id=r["ticker"])
        results.append(r)
    out = screening.screen({"universe": results}, AS_OF)
    by = {x["ticker"]: x for x in out}
    assert by["T0.US"]["stage"] == "screen_hold"
    assert [x["ticker"] for x in out][-1] == "T0.US"  # price-failed sorts last whatever the score


# ---- 5. every cohort selected correctly ------------------------------------

KINDS = ["stock", "bank", "insurer", "reit"]
BUCKETS = ["nano", "micro", "small", "mid", "large", "mega"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_and_cap_bucket_cohort_is_formed_separately(kind):
    rows = []
    for bucket in BUCKETS:
        rows += [
            ranked_row(f"{kind}-{bucket}-{i}", kind=kind, cap=bucket, v=i + 1) for i in range(20)
        ]
    out = screen_ranking.build([run_of(rows)])[0]
    assert len(out) == len(rows)
    for row in out:
        bucket = row["cap_bucket"]
        assert screen_ranking.cohort_key(row) == (kind, bucket)
        assert row["composite_cohort_n"] == 20, (row["ticker"], row["composite_cohort_n"])
        assert row["ranking_cohort"] is not None and bucket in str(row["ranking_cohort"])


def test_cohorts_with_fewer_than_twenty_peers_or_no_cap_are_not_pooled():
    rows = [ranked_row(f"S{i}", cap="small", v=i + 1) for i in range(19)]
    rows += [ranked_row(f"M{i}", cap="mid", v=i + 1) for i in range(20)]
    rows += [ranked_row("NOCAP", cap=None), ranked_row("BADCAP", cap="giant")]
    out = {r["ticker"]: r for r in screen_ranking.build([run_of(rows)])[0]}
    assert all(out[f"S{i}"]["composite_percentile"] is None for i in range(19))
    assert all(out[f"M{i}"]["composite_cohort_n"] == 20 for i in range(20))
    assert out["NOCAP"]["ranking_cohort"] is None and out["BADCAP"]["ranking_cohort"] is None


@pytest.mark.parametrize(
    ("cap", "bucket"),
    [
        (-1, None),
        (None, None),
        (0, "nano"),
        (49_999_999, "nano"),
        (50e6, "micro"),
        (300e6, "small"),
        (2e9, "mid"),
        (10e9, "large"),
        (200e9, "mega"),
    ],
)
def test_cap_bucket_boundaries(cap, bucket):
    assert screening.cap_bucket(cap) == bucket


# ---- 6/7/8. large queue keeps budget and original inputs --------------------


def big_inputs(n=6000):
    regions = ["US", "EU", "JP", "EM"]
    caps = [10e6, 100e6, 1e9, 5e9, 50e9, 500e9]
    universe = []
    for i in range(n):
        r = record(f"T{i:05d}.US")
        r["region"] = regions[i % 4]
        r["evidence"]["market_cap_usd"] = dict(
            r["evidence"]["market_cap_usd"], value=caps[(i // 4) % 6]
        )
        universe.append(r)
    return dict(
        universe=universe,
        portfolio=dict(holdings=[f"T{i:05d}.US" for i in range(0, 4000, 4)]),
        active=[f"T{i:05d}.US" for i in range(1, 400, 4)],
    )


def test_large_queue_is_exact_unique_fair_and_leaves_inputs_untouched():
    inputs, state = big_inputs(), {"last_seen": {}, "rotation": 7}
    before_i, before_s = copy.deepcopy(inputs), copy.deepcopy(state)
    size = 3000
    chosen = service.choose_batch(inputs, state, size, as_of=AS_OF)
    assert inputs == before_i and state == before_s
    assert len(chosen) == size == len(set(chosen))
    priority = set(inputs["portfolio"]["holdings"]) | set(inputs["active"])
    assert len(priority & set(chosen)) <= size // 2  # priority never crowds out the universe
    assert len(set(chosen) - priority) >= size // 2
    assert set(chosen) <= {r["ticker"] for r in inputs["universe"]}


def test_queue_above_old_1000_ceiling_rotates_without_starvation():
    inputs = big_inputs(2500)
    inputs["portfolio"] = {}
    inputs["active"] = []
    state, seen_all = {"last_seen": {}, "rotation": 0}, set()
    for day in range(1, 4):
        chosen = service.choose_batch(inputs, state, 1000, as_of=AS_OF)
        assert len(chosen) == 1000
        if day < 3:
            assert not seen_all & set(chosen)  # nothing repeats before all are seen
        seen_all |= set(chosen)
        for r in inputs["universe"]:
            if r["ticker"] in chosen:
                state["last_seen"][r["issuer_id"]] = f"2026-10-{day:02d}"
        state["rotation"] += len(chosen)
    assert len(seen_all) == 2500  # 3 x 1000 > 2500: every name refreshed


def test_queue_limit_and_budget_constants_are_not_loosened():
    assert service.MAX_QUEUE >= 5000
    base = dict(
        initial_credit_limit=1000,
        request_credit_cost={"fundamentals": 10, "prices": 1},
        account_headroom=25000,
    )
    screen_budget.validate_budget(dict(base, daily_credit_limit=75000))
    with pytest.raises(ValueError):
        screen_budget.validate_budget(dict(base, daily_credit_limit=75001))
    with pytest.raises(ValueError):
        screen_budget.validate_budget(dict(base, daily_credit_limit=75000, account_headroom=24999))
    with pytest.raises(ValueError):
        screen_budget.validate_budget(
            dict(base, daily_credit_limit=75000, daily_working_credit_limit=75001)
        )
    screen_budget.validate_budget(
        dict(base, daily_credit_limit=75000, daily_working_credit_limit=50000)
    )


def test_batch_size_bounds_are_validated(tmp_path):
    for bad in (0, True, service.MAX_QUEUE + 1, "5"):
        path = tmp_path / "c.yaml"
        path.write_text(f"input: i.json\nstate_root: s\nbatch_size: {bad!r}\n")
        with pytest.raises(ValueError):
            service.load_config(path)


# ---- 7. excluded names trigger zero paid calls -------------------------------


@pytest.mark.parametrize("close", [4.99, 0.01])
def test_cached_sub_five_name_triggers_zero_paid_calls_and_is_not_refreshed(tmp_path, close):
    today = TODAY
    cache_source(tmp_path, "PENNY.US", "fundamentals", general("PENNY.US"))
    cache_source(tmp_path, "PENNY.US", "prices", prices(close))
    request, calls = requester()
    inputs = [rec("PENNY.US")]
    before = copy.deepcopy(inputs)
    receipts = service.acquire(cfg(), inputs, ["PENNY.US"], tmp_path, today, request=request)
    assert calls == []
    assert receipts.outcomes == {"PENNY.US": "price_excluded"}
    assert inputs == before
    assert all(r.get("reserved_credits", 0) == 0 for r in receipts)


@pytest.mark.parametrize(("close", "outcome"), [(5.0, "refreshed"), (4.99, "price_excluded")])
def test_five_dollar_edge_outcome_is_exact(tmp_path, close, outcome):
    today = TODAY
    cache_source(tmp_path, "EDGE.US", "fundamentals", general("EDGE.US"), days_old=1)
    cache_source(tmp_path, "EDGE.US", "prices", prices(close))
    request, _ = requester(close=close)
    receipts = service.acquire(
        cfg(), [rec("EDGE.US")], ["EDGE.US"], tmp_path, today, request=request
    )
    assert receipts.outcomes == {"EDGE.US": outcome}


def test_held_names_do_not_advance_last_seen(tmp_path):
    state = {"last_seen": {"A": "2026-10-01"}, "rotation": 0}
    inputs = {
        "universe": [{"ticker": "A.US", "issuer_id": "A"}, {"ticker": "B.US", "issuer_id": "B"}]
    }
    out = tmp_path / "out"
    out.mkdir()
    service.publish_state(
        tmp_path,
        state,
        inputs,
        ["A.US", "B.US"],
        out,
        "2026-10-27",
        refreshed=["B.US"],
        counts=dict(queued=2, refreshed=1, held=1),
    )
    saved = service.read(tmp_path / "state.json")
    assert saved["last_seen"] == {"A": "2026-10-01", "B": "2026-10-27"}
    assert service.read(tmp_path / "status.json")["counts"]["held"] == 1


# ---- real-flow consistency ------------------------------------------------------


def _morning(stamp):
    return screen_sources.normalize_listing_price(
        [{"date": "2026-10-26", "close": 9.0, "adjusted_close": 9.0}],
        record(),
        {**receipt("p.json"), "as_of": AS_OF, "retrieved_at": stamp, "symbol": "TEST.US"},
        fundamentals=metadata("TEST"),
        fundamentals_receipt={**receipt("f.json"), "as_of": AS_OF, "retrieved_at": stamp},
    )


def test_morning_run_prior_session_quote_reaches_pass_and_pre_as_of_receipt_is_unknown():
    out = _morning(AS_OF + "T12:30:00+00:00")  # as_of=today, prior-session quote
    assert out["listing_price_usd"]["period"] == "2026-10-26"
    r = record()
    r["evidence"].update(out)
    assert price_gate(r)["result"] == "pass"
    assert _morning("2026-10-26T23:30:00+00:00") == {}  # retrieved before the as_of date


# ---- stage 4 review fixes -------------------------------------------------


def _thesis_gate(key):
    r = record()
    r["reviews"]["thesis"] = review(**{key: True})
    gates = screening.evaluate(r, AS_OF)["gates"]
    return next(x for x in gates if x["gate"] == "human_thesis")


def test_thesis_requires_the_ai_spending_answer():
    # The renamed key asks a stricter question; a legacy not_mainly_ai answer stays unknown.
    assert _thesis_gate("not_mainly_ai_spending")["result"] == "pass"
    assert _thesis_gate("not_mainly_ai")["result"] == "unknown"


def _metric_row(kind="stock", **gate_over):
    row = ranked_row("X.US", kind=kind)
    for g in row["gates"]:
        if g["gate"] in gate_over:
            g.update(gate_over[g["gate"]])
    return row


def test_provider_not_applicable_cannot_exempt_a_stock_term():
    row = _metric_row(pe_own={"result": "not_applicable", "value": 10})
    assert screen_ranking.component(row, "pe_own")[0] == "unknown"
    bank = _metric_row(kind="bank", net_debt_ebitda={"result": "not_applicable"})
    assert screen_ranking.component(bank, "net_debt_ebitda")[0] == "not_applicable"


def test_usable_value_in_small_cohort_scores_zero_and_is_reported():
    out, _ = screen_ranking.build([run_of([ranked_row(f"T{i}.US") for i in range(3)])])
    row = out[0]
    assert row["pe_own_score"] == 0.0
    assert row["coverage"] == row["usable"] / row["applicable"]  # coverage != scoring
    assert 0 < row["small_cohort_terms"] <= row["usable"]


def test_stage_and_rank_orders_unscored_rows_last_explicitly():
    scored = ranked_row("A.US")
    for g in scored["gates"]:
        g["category"] = "screen"
    etf = dict(ticker="E.US", issuer_id="E.US", kind="etf", cap_bucket=None, gates=[])
    out = screening.stage_and_rank([etf, scored])
    assert [r["ticker"] for r in out] == ["A.US", "E.US"]
