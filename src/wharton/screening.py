"""Conservative screening of normalized, attributed evidence; never orders or forecasts.

The adapter contract deliberately distinguishes reported diluted EPS/shares and matched
historical P/E from superficially similar provider fields. Missing evidence stays unknown.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from itertools import repeat
from statistics import median, stdev

import numpy as np
import pandas as pd

from wharton import screen_ranking
from wharton.resources import limit_threads, worker_limit

RESULTS = {"pass", "fail", "unknown", "exception", "not_applicable"}
CAPS = (50e6, 300e6, 2e9, 10e9, 200e9)
PRICE_FLOOR_USD = 5.0
PRICE_TOLERANCE = 1e-9  # close*FX float noise only
CAP_NAMES = ("nano", "micro", "small", "mid", "large", "mega")


def number(value):
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def day(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def cap_bucket(value):
    value = number(value)
    return None if value is None or value < 0 else CAP_NAMES[sum(value >= x for x in CAPS)]


def evidence(record, key, cutoff, *, age=45):
    item = record.get("evidence", {}).get(key, {})
    observed, available = day(item.get("observed_at")), day(item.get("available_at"))
    oldest_source = day(item.get("oldest_source_observed_at", item.get("observed_at")))
    valid = (
        not record.get("identity_quarantine")
        and item.get("verified") is True
        and bool(item.get("source"))
        and observed is not None
        and available is not None
        and available <= cutoff
        and available <= observed <= cutoff
        and 0 <= (cutoff - observed).days <= age
        and oldest_source is not None
        and 0 <= (cutoff - oldest_source).days <= age
    )
    return item, valid


def annual(record, key, cutoff, n, *, basis=None):
    item, valid = evidence(record, key, cutoff)
    rows = item.get("value")
    if not valid or not isinstance(rows, list) or (basis and item.get("basis") != basis):
        return None
    eligible = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        period, available = day(row.get("period")), day(row.get("available_at"))
        if period is None or available is None or available < period:
            return None
        if period <= cutoff and available <= cutoff:
            if (
                not row.get("source")
                or number(row.get("value")) is None
                or not row.get("currency")
                or available > day(item["observed_at"])
            ):
                return None
            eligible.append(row)
    # Duplicate fiscal periods require a selected restatement vintage, never last-row wins.
    eligible.sort(key=lambda r: r["period"])
    if len({r["period"] for r in eligible}) != len(eligible) or len(eligible) < n:
        return None
    chosen = eligible[-n:]
    periods = [day(r["period"]) for r in chosen]
    if (
        (cutoff - periods[-1]).days > 550
        or len({r["currency"] for r in chosen}) != 1
        or any(not 350 <= (b - a).days <= 380 for a, b in zip(periods, periods[1:]))
    ):
        return None
    return chosen


def values(rows):
    return [float(r["value"]) for r in rows] if rows else None


def cagr(rows):
    vals = values(rows)
    if not vals or any(x <= 0 for x in vals):
        return None
    return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1


def review_evidence_digest(item):
    """Stable digest of substantive evidence, excluding retrieval-only timestamps."""

    def substantive(value):
        if isinstance(value, dict):
            return {
                k: substantive(v)
                for k, v in value.items()
                if k
                not in {"observed_at", "oldest_source_observed_at", "retrieved_at", "retrieval_at"}
            }
        if isinstance(value, list):
            return [substantive(v) for v in value]
        return value

    payload = json.dumps(substantive(item), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def human(record, key, cutoff):
    item = record.get("reviews", {}).get(key, {})
    reviewed = day(item.get("reviewed_at"))
    event_date = day(record.get("filing_event_at"))
    news = record.get("news_events")
    bins = news.get("bins") if isinstance(news, dict) else None
    entry = bins.get(key) if isinstance(bins, dict) else None
    news_day = day(entry.get("latest_open_event_at")) if isinstance(entry, dict) else None
    valid = (
        bool(item.get("author"))
        and bool(item.get("reviewer"))
        and item.get("author") != item.get("reviewer")
        and item.get("review_type") in (None, "human", "ai_research")
        and item.get("status") == "approved"
        and bool(item.get("citations"))
        and bool(item.get("rationale"))
        and reviewed is not None
        and 0 <= (cutoff - reviewed).days <= 366
        and (event_date is None or reviewed >= event_date)
        and (news_day is None or reviewed >= news_day)
        and not (isinstance(news, dict) and news.get("status") in ("unavailable", "stale"))
    )
    if item.get("review_type") == "ai_research":
        reviewed_evidence = item.get("reviewed_evidence")
        current = record.get("evidence", {})
        excluded = {
            "pe",
            "sector_pe",
            "own_pe_monthly",
            "own_ptbv_monthly",
            "ptbv",
            "quote",
            "market_cap_usd",
            "daily_total_returns_usd",
            "listing_metadata",
        }
        if key == "scenarios":
            excluded -= {"pe", "ptbv", "quote"}
        required = {k for k in current if k not in excluded and not k.startswith("diagnostic_")}
        anchor, anchor_valid = evidence(record, f"{key}_facts", cutoff)
        valid = (
            valid
            and anchor_valid
            and bool(anchor.get("value"))
            and item.get("reviewed_event_at") == record.get("filing_event_at")
            and isinstance(reviewed_evidence, dict)
            and bool(reviewed_evidence)
        )
        if valid:
            try:
                valid = required <= set(reviewed_evidence) and all(
                    k in current and digest == review_evidence_digest(current[k])
                    for k, digest in reviewed_evidence.items()
                )
            except (TypeError, ValueError):
                valid = False
    return item, valid


def proved_ttm_listing(record, item):
    if item.get("valuation_basis") == "provider trailing TTM P/E":
        return False
    if item.get("valuation_basis") in {"reviewed comparable", "independently reviewed matched P/E"}:
        return True
    if item.get("valuation_basis") != "reported US GAAP diluted TTM P/E":
        return False
    proof = record.get("eps_listing_basis", {})
    if not isinstance(proof, dict):
        return False
    first, last = day(proof.get("from")), day(proof.get("through"))
    if not (
        proof.get("verified") is True
        and isinstance(proof.get("source"), str)
        and proof["source"].strip()
        and proof.get("ticker") == record.get("ticker")
        and proof.get("currency") == "USD"
        and proof.get("share_basis") == "diluted"
        and first
        and last
        and first <= last
    ):
        return False
    rows = item.get("value")
    if not isinstance(rows, list):
        rows = [{"quarters": item.get("quarters"), "price_date": item.get("period")}]
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        price = day(row.get("price_date"))
        quarters = row.get("quarters")
        if not price or not isinstance(quarters, list) or len(quarters) != 4:
            return False
        spans = []
        for quarter in quarters:
            if not isinstance(quarter, dict) or not quarter.get("source"):
                return False
            start, end, filed = (day(quarter.get(k)) for k in ("start", "end", "filed"))
            if not start or not end or not filed or not start <= end <= filed < price:
                return False
            if not 60 <= (end - start).days + 1 <= 120:
                return False
            spans.append((start, end))
        spans.sort()
        if (
            first > spans[0][0]
            or last < price
            or any((b[0] - a[1]).days != 1 for a, b in zip(spans, spans[1:]))
            or not 330 <= (spans[-1][1] - spans[0][0]).days + 1 <= 390
            or not 0 <= (price - spans[-1][1]).days <= 180
        ):
            return False
    return True


def historical_median(record, key, cutoff):
    item, valid = evidence(record, key, cutoff)
    if (
        not valid
        or not isinstance(item.get("value"), list)
        or (key == "own_pe_monthly" and not proved_ttm_listing(record, item))
    ):
        return None
    valid_rows = []
    seen = set()
    for row in item["value"]:
        if not isinstance(row, dict):
            continue
        period, priced, available = (
            day(row.get(k)) for k in ("period", "price_date", "earnings_available_at")
        )
        v = number(row.get("value"))
        if (
            period is None
            or priced is None
            or available is None
            or v is None
            or v <= 0
            or not row.get("source")
            or row.get("matched") is not True
            or not row.get("earnings_period")
            or row.get("earnings_currency") != row.get("price_currency")
            or not row.get("earnings_currency")
            or available > priced
            or priced > cutoff
            or period.year != priced.year
            or period.month != priced.month
        ):
            continue
        month = period.year * 12 + period.month
        if 1 <= cutoff.year * 12 + cutoff.month - month <= 60:
            if month in seen:
                return None
            seen.add(month)
            valid_rows.append(v)
    return median(valid_rows) if len(valid_rows) >= 48 else None


def evaluate(record, as_of, market_as_of=None):
    """Return auditable categories without converting unknown or review into a pass."""
    cutoff = day(as_of)
    if cutoff is None:
        raise ValueError("as_of must be an ISO date")
    gates = []
    metrics = {}

    def add(name, category, value, threshold, ok, keys=(), *, status=None, note=""):
        items = [record.get("evidence", {}).get(k, {}) for k in keys]
        freshness = [
            {
                "observed_at": x.get("observed_at"),
                "available_at": x.get("available_at"),
                "fresh": evidence(record, k, cutoff)[1],
            }
            for k, x in zip(keys, items)
        ]
        review_keys = {
            "human_thesis": "thesis",
            "authored_scenarios": "scenarios",
            "deal": "deal",
            "policy": "policy",
            "capital_allocation": "capital_allocation",
            "thesis_breakers": "breakers",
            "fund_review": "fund",
        }
        review = record.get("reviews", {}).get(review_keys.get(name), {})
        gates.append(
            {
                "gate": name,
                "category": category,
                "value": value,
                "threshold": threshold,
                "result": status or ("unknown" if ok is None else "pass" if ok else "fail"),
                "source": [x.get("source") for x in items] or review.get("citations", []),
                "period": [x.get("period") for x in items] or [review.get("reviewed_at")],
                "freshness": freshness
                or {
                    "reviewed_at": review.get("reviewed_at"),
                    "valid": human(record, review_keys.get(name), cutoff)[1],
                },
                "note": note,
            }
        )
        metrics[name] = value

    def scalar(key):
        item, valid = evidence(record, key, cutoff)
        if key == "market_cap_usd" and item.get("currency") != "USD":
            valid = False
        return number(item.get("value")) if valid else None

    def comparison(name, category, value, threshold, predicate, keys=(), **kw):
        add(
            name,
            category,
            value,
            threshold,
            predicate(value) if value is not None else None,
            keys,
            **kw,
        )

    kind = record.get("kind", "stock")
    identity, bound = evidence(record, "issuer_identity", cutoff)
    binding = identity.get("value", {})
    add(
        "issuer_identity",
        "screen",
        binding or None,
        "verified qualified ticker to issuer binding",
        (
            binding.get("ticker") == record["ticker"]
            and binding.get("issuer_id") == record["issuer_id"]
        )
        if bound
        else None,
        ["issuer_identity"],
    )
    financial = kind in {"bank", "insurer"}
    etf = kind == "etf"
    price, price_ok = evidence(record, "listing_price_usd", cutoff)
    session = price.get("period")
    market = day(market_as_of) if market_as_of is not None else None
    # Current only if the quote is the expected completed session of this run's receipt.
    current = bool(
        price_ok
        and price.get("currency") == "USD"
        and session
        and session == price.get("last_completed_session") == price.get("expected_session")
        and price.get("receipt_as_of") == str(cutoff)
        and (market is None or (day(session) is not None and day(session) <= market))
        and number(price.get("value")) is not None
    )
    price_value = number(price.get("value")) if current else None
    add(
        "listing_price_usd",
        "screen",
        price_value,
        f">= {PRICE_FLOOR_USD:g} USD raw completed-session close (current session)",
        None if price_value is None else price_value >= PRICE_FLOOR_USD - PRICE_TOLERANCE,
        ["listing_price_usd"],
        status="not_applicable" if etf else None,
    )
    for key, threshold, basis in [("revenue", 0.07, None), ("diluted_eps", 0.05, "reported_gaap")]:
        rows = annual(record, key, cutoff, 4, basis=None if key == "diluted_eps" else basis)
        if key == "diluted_eps":
            item = record.get("evidence", {}).get(key, {})
            proof = item.get("basis_evidence", {})
            start = day(proof.get("from")) if isinstance(proof, dict) else None
            end = day(proof.get("through")) if isinstance(proof, dict) else None
            adjusted = day(item.get("share_basis_date")) if item.get("share_basis_date") else None
            typed = (
                item.get("basis") in {"reported_diluted", "reported_gaap"}
                and item.get("accounting_framework") in {"US_GAAP", "IFRS"}
                and item.get("statement_source")
                in {"sec_financials", "sec_companyfacts", "issuer_filing"}
                and isinstance(proof, dict)
                and proof.get("verified") is True
                and (
                    isinstance(proof.get("source"), str)
                    and bool(proof["source"].strip())
                    or isinstance(proof.get("source"), list)
                    and bool(proof["source"])
                    and all(isinstance(v, str) and v.strip() for v in proof["source"])
                )
                and proof.get("ticker") == record.get("ticker")
                and isinstance(proof.get("currency"), str)
                and bool(proof["currency"].strip())
                and proof.get("share_basis") == "diluted"
                and start is not None
                and end is not None
                and start <= end
                and (not item.get("share_basis_date") or adjusted is not None and adjusted <= end)
                and (not rows or all(row["currency"] == proof["currency"] for row in rows))
                and (
                    not rows
                    or start
                    <= (
                        pd.Timestamp(rows[0]["period"])
                        - pd.DateOffset(years=1)
                        + pd.Timedelta(days=1)
                    ).date()
                )
                and (not rows or day(rows[-1]["period"]) <= end)
            )
            if not typed:
                rows = None
        growth = cagr(rows)
        # Known nonpositive EPS/revenue fails the positive-base growth screen.
        ok = None if rows is None else growth is not None and growth >= threshold - 1e-12
        add(
            f"{key}_3y_cagr",
            "screen",
            growth,
            f">= {threshold}",
            ok,
            [key],
            status="not_applicable" if etf else None,
        )

    def free_cash_flow(n):
        ocf, capex = (annual(record, k, cutoff, n) for k in ("ocf", "capex"))
        if (
            ocf
            and capex
            and [(r["period"], r["currency"]) for r in ocf]
            == [(r["period"], r["currency"]) for r in capex]
        ):
            return [{**a, "value": a["value"] - abs(b["value"])} for a, b in zip(ocf, capex)]
        return None

    fcf = free_cash_flow(4)
    latest_fcf = free_cash_flow(3)
    na = "not_applicable" if financial or etf else None
    comparison(
        "fcf_3y_cagr",
        "screen",
        cagr(fcf),
        ">= .05; OCF - abs(capex)",
        lambda x: x >= 0.05 - 1e-12,
        ["ocf", "capex"],
        status=na,
    )
    add(
        "positive_fcf_latest3",
        "screen",
        values(latest_fcf),
        "each > 0",
        all(r["value"] > 0 for r in latest_fcf) if latest_fcf else None,
        ["ocf", "capex"],
        status=na,
    )
    pe = (
        scalar("pe")
        if proved_ttm_listing(record, record.get("evidence", {}).get("pe", {}))
        else None
    )
    own_pe = historical_median(record, "own_pe_monthly", cutoff)
    sector_pe = scalar("sector_pe")
    pe_ratio = pe / own_pe if pe is not None and own_pe and pe > 0 else None
    comparison(
        "pe_own",
        "screen",
        pe_ratio,
        "positive P/E <= 1.10 own 5y median; >=48 months",
        lambda x: x <= 1.10,
        ["pe", "own_pe_monthly"],
        status="not_applicable" if etf else ("fail" if pe is not None and pe <= 0 else None),
    )
    sector_ratio = (
        pe / sector_pe if pe is not None and pe > 0 and sector_pe and sector_pe > 0 else None
    )
    comparison(
        "pe_sector",
        "screen",
        sector_ratio,
        "positive P/E <= 1.25 sector median",
        lambda x: x <= 1.25,
        ["pe", "sector_pe"],
        status="not_applicable" if etf else ("fail" if pe is not None and pe <= 0 else None),
    )
    debt, ebitda = scalar("net_debt"), scalar("ebitda")
    debt_item, _ = evidence(record, "net_debt", cutoff)
    ebitda_item, _ = evidence(record, "ebitda", cutoff)
    same_basis = (
        debt_item.get("currency")
        and debt_item.get("currency") == ebitda_item.get("currency")
        and debt_item.get("period") == ebitda_item.get("period")
        and day(debt_item.get("period")) is not None
        and 0 <= (cutoff - day(debt_item["period"])).days <= 550
    )
    leverage = debt / ebitda if debt is not None and ebitda and ebitda > 0 and same_basis else None
    case, approved = human(record, "leverage_exception", cutoff)
    exception = approved and case.get("maximum") == 3
    lev_status = na or (
        "exception" if leverage is not None and 2 < leverage <= 3 and exception else None
    )
    comparison(
        "net_debt_ebitda",
        "screen",
        leverage,
        "<=2; <=3 only cited written case",
        lambda x: x <= 2,
        ["net_debt", "ebitda"],
        status=lev_status or ("fail" if ebitda is not None and ebitda <= 0 else None),
        note=case.get("rationale", "") if exception else "",
    )
    gates[-1]["exception_verified"] = lev_status == "exception"
    ni = annual(record, "net_income", cutoff, 1)
    conversion_fcf = free_cash_flow(1)
    conversion = None
    if (
        conversion_fcf
        and ni
        and ni[-1]["period"] == conversion_fcf[-1]["period"]
        and (ni[-1]["currency"] == conversion_fcf[-1]["currency"])
        and ni[-1]["value"] > 0
    ):
        conversion = conversion_fcf[-1]["value"] / ni[-1]["value"]
    comparison(
        "cash_conversion",
        "screen",
        conversion,
        "FCF / positive latest FY NI >= .8; FCF = OCF - abs(capex)",
        lambda x: x >= 0.8,
        ["ocf", "capex", "net_income"],
        status=na or ("fail" if ni and ni[-1]["value"] <= 0 else None),
    )
    shares = annual(
        record, "diluted_shares", cutoff, 4, basis="weighted_average_diluted_split_adjusted"
    )
    growths = (
        [b["value"] / a["value"] - 1 for a, b in zip(shares, shares[1:])]
        if shares and all(r["value"] > 0 for r in shares)
        else None
    )
    add(
        "annual_dilution",
        "screen",
        growths,
        "each latest3 <= .02; split verified",
        max(growths) <= 0.02 + 1e-12 if growths is not None else None,
        ["diluted_shares"],
        status="not_applicable" if etf else None,
    )
    if financial:
        ptbv, own = scalar("ptbv"), historical_median(record, "own_ptbv_monthly", cutoff)
        ratio = ptbv / own if ptbv and ptbv > 0 and own else None
        comparison(
            "ptbv_own",
            "screen",
            ratio,
            "positive PTBV <= own5y median (48 months)",
            lambda x: x <= 1,
            ["ptbv", "own_ptbv_monthly"],
        )
        comparison("roe", "screen", scalar("roe"), ">= .12", lambda x: x >= 0.12, ["roe"])
        comparison(
            "combined_ratio",
            "screen",
            scalar("combined_ratio"),
            "< .95",
            lambda x: x < 0.95,
            ["combined_ratio"],
            status="not_applicable" if kind == "bank" else None,
        )
    margins = annual(record, "operating_margin", cutoff, 5)
    margins_3y = annual(record, "operating_margin", cutoff, 4)
    comparison(
        "margin_change_3y",
        "diligence",
        margins_3y[-1]["value"] - margins_3y[-4]["value"] if margins_3y else None,
        ">= -.02",
        lambda x: x >= -0.02 - 1e-12,
        ["operating_margin"],
        status="not_applicable" if etf else None,
    )
    comparison(
        "margin_sample_sd_5y",
        "diligence",
        stdev(values(margins)) if margins else None,
        "<= .03; sample ddof=1",
        lambda x: x <= 0.03,
        ["operating_margin"],
        status="not_applicable" if etf else None,
    )
    organic = annual(record, "organic_growth", cutoff, 3)
    reported = annual(record, "revenue", cutoff, 4)
    organic_ok = organic_cagr = None
    if organic and all(o["value"] > -1 for o in organic):
        organic_cagr = math.prod(1 + o["value"] for o in organic) ** (1 / 3) - 1
        if organic_cagr >= 0.04 - 1e-12:
            organic_ok = True
        elif reported:
            aligned = [r["period"] for r in organic] == [r["period"] for r in reported[-3:]] and {
                r["currency"] for r in organic
            } == {r["currency"] for r in reported}
            reported_cagr = cagr(reported)
            if aligned and reported_cagr is not None:
                organic_ok = organic_cagr >= 0.5 * reported_cagr - 1e-12
    add(
        "organic_growth",
        "diligence",
        organic_cagr,
        "3y compounded organic CAGR >=.04 or >= half 3y reported CAGR",
        organic_ok,
        ["organic_growth", "revenue"],
        status="not_applicable" if etf else None,
    )
    thesis, reviewed = human(record, "thesis", cutoff)
    add(
        "human_thesis",
        "screen",
        thesis.get("not_mainly_ai_spending"),
        "cited independent research review; thesis not mainly reliant on AI spending",
        thesis["not_mainly_ai_spending"]
        if reviewed and isinstance(thesis.get("not_mainly_ai_spending"), bool)
        else None,
        status="not_applicable" if etf else None,
        note=thesis.get("rationale", ""),
    )
    if etf:
        fund, reviewed = human(record, "fund", cutoff)
        add(
            "fund_review",
            "screen",
            {k: fund.get(k) for k in ("strategy", "costs", "liquidity", "structure")},
            "separate cited ETF strategy/cost/liquidity/structure review",
            all(fund.get(k) for k in ("strategy", "costs", "liquidity", "structure"))
            if reviewed
            else None,
        )
    scenarios, reviewed = human(record, "scenarios", cutoff)
    scenario = scenarios.get("value", {})
    bear, base, bull = (number(scenario.get(k)) for k in ("bear", "base", "bull"))
    assumptions = all(scenarios.get(k) for k in ("dividends", "costs", "fx", "assumptions"))
    scenario_ok = None
    if (
        reviewed
        and assumptions
        and scenarios.get("horizon_years") == 5
        and all(x is not None for x in (bear, base, bull))
    ):
        scenario_ok = (
            bear <= base <= bull
            and bear >= -0.35
            and base > -1
            and (1 + base) ** 0.2 - 1 >= 0.10 - 1e-12
            and bull >= 2 * abs(min(bear, 0))
        )
    add(
        "authored_scenarios",
        "diligence",
        scenario or None,
        "5y: base CAGR>=.10; bear>=-.35; bull>=2*abs(bear loss); ordered",
        scenario_ok,
        note="Authored conditional scenarios, never forecasts",
    )
    deal, reviewed = human(record, "deal", cutoff)
    fraction = number(deal.get("ev_fraction"))
    add(
        "deal",
        "pretrade",
        fraction,
        "<=.20 or cited central thesis; retained business thesis",
        (
            fraction is not None
            and fraction >= 0
            and (fraction <= 0.2 or deal.get("central_thesis") is True)
            and bool(deal.get("retained_business_thesis"))
        )
        if reviewed
        else None,
    )
    policy, reviewed = human(record, "policy", cutoff)
    impact = number(policy.get("single_policy_earnings_impact"))
    comparison(
        "policy",
        "pretrade",
        impact if reviewed else None,
        "absolute impact <=.20",
        lambda x: abs(x) <= 0.2,
    )
    capital, reviewed = human(record, "capital_allocation", cutoff)
    add(
        "capital_allocation",
        "pretrade",
        capital.get("matches_thesis"),
        "matches thesis",
        capital.get("matches_thesis") is True if reviewed else None,
    )
    breakers, reviewed = human(record, "breakers", cutoff)
    items = breakers.get("items", [])
    add(
        "thesis_breakers",
        "pretrade",
        items or None,
        "2-3 measurable breakers; annual review",
        (
            2 <= len(items) <= 3
            and all(
                isinstance(x, dict) and x.get("metric") and number(x.get("threshold")) is not None
                for x in items
            )
            and breakers.get("annual_review") is True
        )
        if reviewed
        else None,
    )
    quote, valid = evidence(record, "quote", cutoff, age=0)
    quote_value = quote.get("value", {})
    bid, ask = (number(quote_value.get(k)) for k in ("bid", "ask"))
    stamp = quote.get("observed_at")
    # Daily reports cannot bless a quote: require an explicit evaluation instant <=60s later.
    now = record.get("quote_evaluation_at")
    live = False
    try:
        delta = datetime.fromisoformat(now) - datetime.fromisoformat(stamp)
        live = 0 <= delta.total_seconds() <= 60 and "+" in now and "+" in stamp
    except (TypeError, ValueError):
        pass
    spread = (
        (ask - bid) / ((ask + bid) / 2)
        if bid and ask and 0 < bid <= ask and valid and live
        else None
    )
    comparison(
        "limit_order_spread",
        "pretrade",
        spread,
        "<=.005 fresh live bid/ask; limit order",
        lambda x: x <= 0.005 and quote_value.get("order_type") == "limit",
        ["quote"],
    )
    return {
        "ticker": record["ticker"],
        "issuer_id": record["issuer_id"],
        "kind": kind,
        "sector": record.get("sector"),
        "region": record.get("region"),
        "cap_bucket": cap_bucket(scalar("market_cap_usd")),
        "gates": gates,
        "metrics": metrics,
        "review_owner": record.get("review_owner", "unassigned"),
    }


def portfolio_gates(records, results, portfolio, as_of, market_as_of=None):
    """Aligned explicit USD daily total returns; no fill, supplied holdings/weights only."""
    backend = portfolio.get("numeric_backend", "cpu")
    if backend not in {"cpu", "cuda", "mps"}:
        raise ValueError("numeric_backend must be cpu, cuda, or mps")
    if backend != "cpu":
        from wharton.gpu import require_device

        require_device(backend)
    cutoff = pd.Timestamp(market_as_of or as_of)
    sessions = portfolio.get("sessions", [])
    weights = portfolio.get("weights", {})
    holdings = portfolio.get("holdings", [])
    benchmark = portfolio.get("benchmark", "VT")
    if benchmark not in {"VT", "VT.US"}:
        raise ValueError("downside benchmark must be VT or qualified VT.US")
    expected = pd.DatetimeIndex(pd.to_datetime(sessions))
    calendar_valid = (
        bool(sessions)
        and len(expected) >= 200
        and not expected.has_duplicates
        and expected.is_monotonic_increasing
        and bool(portfolio.get("session_source"))
        and portfolio.get("sessions_verified") is True
        and expected.max() <= cutoff
        and expected.min() <= cutoff - pd.DateOffset(years=1) + pd.Timedelta(days=7)
        and expected.max() >= cutoff - pd.Timedelta(days=7)
        and all(expected.weekday < 5)
    )

    def valid_weights(value):
        return (
            isinstance(value, dict)
            and bool(value)
            and all(number(v) is not None and v >= 0 for v in value.values())
            and abs(sum(value.values()) - 1) < 1e-8
        )

    proposed_valid = valid_weights(weights)
    existing = portfolio.get("existing_weights", {})
    cash = number(portfolio.get("cash_weight", 0))
    existing_valid = (
        isinstance(existing, dict)
        and set(existing) == set(holdings)
        and cash is not None
        and cash >= 0
        and (cash == 0 or portfolio.get("cash_currency") == "USD")
        and all(number(v) is not None and v >= 0 for v in existing.values())
        and abs(sum(existing.values()) + cash - 1) < 1e-8
    )
    five_year_calendar_valid = (
        calendar_valid
        and len(expected) >= 1100
        and expected.min() <= cutoff - pd.DateOffset(years=5) + pd.Timedelta(days=7)
    )
    expected = expected[expected > cutoff - pd.DateOffset(years=5)]
    sources = {r["ticker"]: r for r in records}

    def native_calendar(ticker):
        item = sources.get(ticker, {}).get("evidence", {}).get("daily_total_returns_usd", {})
        if "native_sessions" not in item:
            return expected if calendar_valid else None
        try:
            dates = pd.DatetimeIndex(pd.to_datetime(item["native_sessions"]))
            if (
                item.get("native_sessions_verified") is not True
                or not item.get("session_source")
                or len(dates) < 200
                or dates.hasnans
                or dates.has_duplicates
                or not dates.is_monotonic_increasing
                or dates.tz is not None
                or dates.max() > cutoff
                or dates.min() > cutoff - pd.DateOffset(years=1) + pd.Timedelta(days=7)
                or dates.max() < cutoff - pd.Timedelta(days=7)
            ):
                return None
            return dates[dates > cutoff - pd.DateOffset(years=5)]
        except (ValueError, TypeError):
            return None

    def series(ticker, calendar=None):
        record = sources.get(ticker, {})
        item, fresh = evidence(record, "daily_total_returns_usd", day(as_of), age=7)
        rows = item.get("value", [])
        if (
            not fresh
            or item.get("currency") != "USD"
            or item.get("frequency") != "daily"
            or item.get("return_basis") != "total_return"
            or not isinstance(rows, list)
        ):
            return None
        try:
            dates = pd.DatetimeIndex(pd.to_datetime([r["date"] for r in rows]))
            vals = [number(r["value"]) for r in rows]
            if dates.has_duplicates or any(v is None or v <= -1 for v in vals):
                return None
            target = expected if calendar is None else calendar
            out = pd.Series(vals, index=dates).reindex(target)
            target_days = [str(d.date()) for d in target]
            previous = dict(zip(target_days[1:], target_days))
            # Only equal actual close-to-close intervals enter a joint reference panel.
            for row in rows:
                if (
                    "interval_start" in row
                    and row["date"] in previous
                    and row["interval_start"] != previous[row["date"]]
                ):
                    out.loc[row["date"]] = np.nan
            return out
        except (ValueError, TypeError, KeyError):
            return None

    def aligned(tickers):
        calendars = {ticker: native_calendar(ticker) for ticker in dict.fromkeys(tickers)}
        if (
            not five_year_calendar_valid
            or not calendars
            or any(v is None for v in calendars.values())
        ):
            return pd.DataFrame()
        common = expected
        for calendar in calendars.values():
            common = common.intersection(calendar)
        # Require a representative five-year sample before comparing actual intervals.
        if (
            len(common) < 1100
            or len(common) < 0.9 * len(expected)
            or common.min() > cutoff - pd.DateOffset(years=5) + pd.Timedelta(days=7)
            or common.max() < cutoff - pd.Timedelta(days=7)
        ):
            return pd.DataFrame()
        # Compare only returns covering the same actual close-to-close interval.
        predecessors = [
            pd.Series(calendar[:-1], index=calendar[1:]).reindex(common)
            for calendar in calendars.values()
        ]
        comparable = pd.Series(True, index=common)
        for previous in predecessors[1:]:
            comparable &= (previous == predecessors[0]) | (previous.isna() & predecessors[0].isna())
        common = common[comparable.to_numpy()]
        if (
            len(common) < 1100
            or len(common) < 0.9 * len(expected)
            or common.min() > cutoff - pd.DateOffset(years=5) + pd.Timedelta(days=7)
            or common.max() < cutoff - pd.Timedelta(days=7)
        ):
            return pd.DataFrame()
        columns = {ticker: series(ticker, calendar) for ticker, calendar in calendars.items()}
        if any(v is None for v in columns.values()):
            return pd.DataFrame()
        panel = pd.DataFrame({ticker: values.reindex(common) for ticker, values in columns.items()})
        return panel if len(panel) > 2 and not panel.isna().any().any() else pd.DataFrame()

    for result in results:
        ticker = result["ticker"]
        measures = {}
        metric_panels = {}
        own_expected = native_calendar(ticker)
        standalone = series(ticker, own_expected) if own_expected is not None else None
        if standalone is not None:
            standalone = standalone.dropna()
            last_year = standalone.loc[standalone.index > cutoff - pd.DateOffset(years=1)]
            year_expected = own_expected[own_expected > cutoff - pd.DateOffset(years=1)]
            if len(last_year) == len(year_expected) and len(last_year) > 2:
                measures["volatility_1y"] = float(last_year.std(ddof=1) * math.sqrt(252))
                metric_panels["volatility_1y"] = (last_year, year_expected, [ticker])
            if (
                len(own_expected) >= 1100
                and own_expected.min() <= cutoff - pd.DateOffset(years=5) + pd.Timedelta(days=7)
                and len(standalone) == len(own_expected)
                and len(standalone) > 2
            ):
                wealth = np.r_[1.0, (1 + standalone.to_numpy()).cumprod()]
                measures["max_drawdown_5y"] = float(
                    (wealth / np.maximum.accumulate(wealth) - 1).min()
                )
                n = math.ceil(0.05 * len(standalone))
                measures["daily_es95"] = max(0.0, -float(np.sort(standalone.to_numpy())[:n].mean()))
                for name in ("max_drawdown_5y", "daily_es95"):
                    metric_panels[name] = (standalone, own_expected, [ticker])
        holding_panel = aligned([ticker, *holdings]) if holdings else pd.DataFrame()
        if not holding_panel.empty:
            cors = [holding_panel[ticker].corr(holding_panel[s]) for s in holdings if s != ticker]
            measures["holding_correlation"] = max(cors) if cors else None
            metric_panels["holding_correlation"] = (
                holding_panel,
                holding_panel.index,
                list(holding_panel),
            )
        existing_panel = aligned([ticker, *existing]) if existing_valid else pd.DataFrame()
        if not existing_panel.empty:
            measures["portfolio_correlation"] = existing_panel[ticker].corr(
                sum(existing_panel[s] * w for s, w in existing.items())
            )
            metric_panels["portfolio_correlation"] = (
                existing_panel,
                existing_panel.index,
                list(existing_panel),
            )
        market_panel = aligned([ticker, benchmark])
        if not market_panel.empty:
            market = market_panel[benchmark]
            mask = market < 0
            variance = market[mask].var(ddof=1)
            measures["downside_beta"] = (
                market_panel[ticker][mask].cov(market[mask]) / variance
                if mask.sum() > 2 and variance > 0
                else None
            )
            metric_panels["downside_beta"] = (market_panel, market_panel.index, list(market_panel))
        proposed_panel = aligned(weights) if proposed_valid else pd.DataFrame()
        if not proposed_panel.empty:
            if backend == "cpu":
                covariance = proposed_panel[list(weights)].cov().to_numpy()
            else:
                from wharton.gpu import portfolio_covariance

                covariance = portfolio_covariance(proposed_panel[list(weights)].to_numpy(), backend)
            w = np.array(list(weights.values()))
            total = w @ covariance @ w
            measures["euler_variance_fraction"] = (
                float(
                    w[list(weights).index(ticker)]
                    * (covariance @ w)[list(weights).index(ticker)]
                    / total
                )
                if ticker in weights and weights[ticker] > 0 and total > 0
                else None
            )
            metric_panels["euler_variance_fraction"] = (
                proposed_panel,
                proposed_panel.index,
                list(proposed_panel),
            )
        specs = [
            ("volatility_1y", 0.45, False),
            ("holding_correlation", 0.6, False),
            ("portfolio_correlation", 0.7, False),
            ("downside_beta", 1.2, False),
            ("euler_variance_fraction", 0.2, False),
            ("max_drawdown_5y", -0.5, True),
            ("daily_es95", 0.04, False),
        ]
        for name, threshold, greater in specs:
            value = number(measures.get(name))
            category = "screen" if name == "volatility_1y" else "portfolio"
            measured, denominator, instruments = metric_panels.get(
                name,
                (
                    pd.DataFrame(),
                    expected[expected > cutoff - pd.DateOffset(years=1)]
                    if name == "volatility_1y"
                    else expected,
                    [],
                ),
            )
            na = result["kind"] == "etf" and name in {"euler_variance_fraction", "max_drawdown_5y"}
            result["gates"].append(
                {
                    "gate": name,
                    "category": category,
                    "value": value,
                    "threshold": f"{'>=' if greater else '<='} {threshold}",
                    "result": "fail"
                    if name == "euler_variance_fraction" and weights.get(ticker) == 0 and not na
                    else "not_applicable"
                    if na
                    else "unknown"
                    if value is None
                    else "pass"
                    if (value >= threshold if greater else value <= threshold)
                    else "fail",
                    "source": [
                        sources.get(s, {})
                        .get("evidence", {})
                        .get("daily_total_returns_usd", {})
                        .get("source")
                        for s in instruments
                    ],
                    "period": [str(measured.index.min().date()), str(measured.index.max().date())]
                    if len(measured)
                    else None,
                    "freshness": {
                        "aligned_sessions": len(measured),
                        "expected_sessions": len(denominator),
                        "coverage": len(measured) / len(denominator) if len(denominator) else 0,
                        "instruments": instruments,
                    },
                    "note": (
                        "Supplied USD total returns; standalone native sessions; "
                        "joint reference calendar with matching actual intervals; no fill"
                    ),
                }
            )
            result["metrics"][name] = value


def percentiles(results, universe_results, cutoff):
    """Midrank ties over retained universe; never silently use just today's batch."""
    directions = {
        "revenue_3y_cagr": 1,
        "diluted_eps_3y_cagr": 1,
        "fcf_3y_cagr": 1,
        "pe_own": -1,
        "net_debt_ebitda": -1,
        "volatility_1y": -1,
        "max_drawdown_5y": 1,
    }
    # Price-eligible, non-ETF peers are grouped once per run, with sorted values per metric.
    groups = {}
    for p in universe_results:
        if p["kind"] == "etf" or screen_ranking.price_failed(p) or p.get("cap_bucket") is None:
            continue
        for key in (
            ("size", p["cap_bucket"], p["kind"]),
            ("sector_size", p["cap_bucket"], p["kind"], p.get("sector")),
        ):
            group = groups.setdefault(key, {"n": 0, "values": {m: [] for m in directions}})
            group["n"] += 1
            for metric in directions:
                x = number(p["metrics"].get(metric))
                if x is not None:
                    group["values"][metric].append(x)
    for group in groups.values():
        for values in group["values"].values():
            values.sort()
    empty = {"n": 0, "values": {m: [] for m in directions}}
    for result in results:
        rows = []
        for scope in ("size", "sector_size"):
            # Industrial/financial definitions stay separate even inside a size cohort.
            key = (scope, result.get("cap_bucket"), result["kind"])
            if scope == "sector_size":
                key += (result.get("sector"),)
            group = groups.get(key, empty) if result.get("cap_bucket") is not None else empty
            for metric, direction in directions.items():
                valid = group["values"][metric]
                value = number(result["metrics"].get(metric))
                raw = None
                if len(valid) >= 20 and value is not None:
                    raw = screen_ranking.midrank_percentile(valid, value)
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "value": value,
                        "raw_higher_percentile": raw,
                        "favorable_percentile": raw if direction == 1 or raw is None else 100 - raw,
                        "peer_n": len(valid),
                        "cohort_n": group["n"],
                        "coverage": len(valid) / group["n"] if group["n"] else 0,
                        "source_cutoff": cutoff,
                        "result": "pass" if raw is not None else "unknown",
                    }
                )
        result["percentiles"] = rows


def stage_and_rank(results, universe_results=None):
    allowed = {"pass", "not_applicable", "exception"}
    for result in results:
        gates = result["gates"]
        result["stage"] = "screen_hold"
        result["pareto_tier"] = None
        for category, stage in [
            ("screen", "diligence"),
            ("diligence", "portfolio"),
            ("portfolio", "pretrade"),
            ("pretrade", "review_ready"),
        ]:
            if any(g["result"] not in allowed for g in gates if g["category"] == category):
                break
            result["stage"] = stage
        result["blockers"] = [g["gate"] for g in gates if g["result"] not in allowed]
        result["reopening_reason"] = "Provide fresh verified evidence or remedy: " + ", ".join(
            result["blockers"]
        )
    # Score the retained universe so today's acquisition batch cannot redefine peers.
    ranked, _ = screen_ranking.build(
        [
            {
                "as_of": "",
                "run_id": "current",
                "results_sha256": "",
                "rows": universe_results if universe_results is not None else results,
            }
        ]
    )
    by_ticker = {row["ticker"]: row for row in ranked}
    for result in results:
        rank = by_ticker.get(result["ticker"])
        if rank is None:  # ETFs retain their separately reviewed fund path.
            continue
        for key in (
            "score",
            "coverage",
            "composite_percentile",
            "ranking_cohort",
            "composite_cohort_n",
        ):
            result[key] = rank[key]
        percentile = rank["composite_percentile"]
        if result["stage"] != "screen_hold" and (
            percentile is None or percentile < screen_ranking.TOP_PERCENTILE
        ):
            result["stage"] = "screen_percentile_hold"
            result["blockers"].append("cohort_percentile")
            result["reopening_reason"] = (
                "Requires at least 20 retained kind/size peers and a composite percentile >=95; "
                "all applicable hard gates remain required."
            )
    results.sort(
        key=lambda r: (
            screen_ranking.price_failed(r),
            r.get("score") is None,  # unscored rows (ETFs) sort after scored, explicitly
            -(r.get("score") or 0),
            -(r.get("coverage") or 0),
            r["ticker"],
        )
    )
    # Ceilings restrict advancement; no floor can manufacture a qualifying company.
    for stage, limit in [("diligence", 50), ("portfolio", 10), ("pretrade", 5)]:
        beyond = {
            "diligence": {"diligence", "portfolio", "pretrade", "review_ready"},
            "portfolio": {"portfolio", "pretrade", "review_ready"},
            "pretrade": {"pretrade", "review_ready"},
        }[stage]
        admitted = [r for r in results if r["stage"] in beyond]
        for r in admitted[limit:]:
            r["stage"] = stage + "_capacity_hold"
    return results


def validate_workers(workers):
    if workers != "auto" and (
        isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
    ):
        raise ValueError("evaluation workers must be 'auto' or a positive integer")
    return workers


def screen(inputs, as_of, selected=None, *, workers=1):
    validate_workers(workers)
    workers = worker_limit(workers)
    records = inputs["universe"]
    if (
        not isinstance(records, list)
        or any(not r.get("ticker") or not r.get("issuer_id") for r in records)
        or len({r["ticker"] for r in records}) != len(records)
        or len({r["issuer_id"] for r in records}) != len(records)
    ):
        raise ValueError(
            "universe requires unique ticker and issuer_id; deduplicate issuers upstream"
        )
    for r in records:
        if r.get("kind", "stock") not in {"stock", "bank", "insurer", "etf"}:
            raise ValueError("kind must explicitly identify stock, bank, insurer, or etf")
        for section in ("evidence", "reviews"):
            if not isinstance(r.get(section, {}), dict) or any(
                not isinstance(v, dict) for v in r.get(section, {}).values()
            ):
                raise ValueError(f"{section} must contain named evidence objects")
    portfolio = inputs.get("portfolio", {})
    instruments = portfolio.get("instruments", [])
    if (
        not isinstance(instruments, list)
        or any(
            not isinstance(r, dict) or not r.get("ticker") or not r.get("issuer_id")
            for r in instruments
        )
        or len({r["ticker"] for r in instruments}) != len(instruments)
        or {r["ticker"] for r in instruments} & {r["ticker"] for r in records}
    ):
        raise ValueError("auxiliary instruments require unique tickers disjoint from universe")
    for r in instruments:
        for section in ("evidence", "reviews"):
            if not isinstance(r.get(section, {}), dict) or any(
                not isinstance(v, dict) for v in r.get(section, {}).values()
            ):
                raise ValueError(f"{section} must contain named evidence objects")
    market_as_of = inputs.get("market_as_of")
    if market_as_of is not None and (day(market_as_of) is None or day(market_as_of) > day(as_of)):
        raise ValueError("market_as_of must be an ISO date no later than as_of")
    if workers == 1 or not records:
        results = [evaluate(record, as_of, market_as_of) for record in records]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers, initializer=limit_threads) as pool:
            # Bound queued evidence payloads; map preserves the universe's input order.
            for start in range(0, len(records), workers * 32):
                results.extend(
                    pool.map(
                        evaluate,
                        records[start : start + workers * 32],
                        repeat(as_of),
                        repeat(market_as_of),
                        chunksize=8,
                    )
                )
    portfolio_gates([*records, *instruments], results, portfolio, as_of, market_as_of)
    for record, result in zip(records, results):
        quarantine = record.get("identity_quarantine")
        if quarantine:
            result["identity_quarantine"] = quarantine
            result["cap_bucket"] = None
            result["metrics"] = {key: None for key in result["metrics"]}
            for gate in result["gates"]:
                gate.update(value=None, result="unknown", note=quarantine["reason"])
            result["gates"].append(
                dict(
                    gate="source_identity",
                    category="screen",
                    value=None,
                    threshold="matching reviewed issuer and listing identity",
                    result="unknown",
                    source=quarantine.get("source"),
                    period=quarantine.get("as_of"),
                    freshness={"identity_quarantine": quarantine},
                    note=quarantine["reason"],
                )
            )
    chosen = [r for r in results if selected is None or r["ticker"] in selected]
    percentiles(chosen, results, as_of)
    return stage_and_rank(chosen, results)
