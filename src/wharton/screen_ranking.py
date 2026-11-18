"""Shared descriptive screening score; hard gates remain independent of ranking."""

import bisect
import collections
import json
import math

BASE = (
    "revenue_3y_cagr",
    "diluted_eps_3y_cagr",
    "fcf_3y_cagr",
    "positive_fcf_latest3",
    "pe_own",
    "pe_sector",
    "net_debt_ebitda",
    "cash_conversion",
    "annual_dilution",
    "volatility_1y",
)
FINANCIAL = ("ptbv_own", "roe", "combined_ratio")
METRICS = BASE + FINANCIAL
# Equal unit weights preserve the reviewed baseline; no return-fitted preferences.
METRIC_WEIGHTS = {name: 1.0 for name in METRICS}
HIGH = {"revenue_3y_cagr", "diluted_eps_3y_cagr", "fcf_3y_cagr", "cash_conversion", "roe"}
SCREEN_GATES = ("issuer_identity", "listing_price_usd", *BASE, "human_thesis")
LATER_GATES = (
    "margin_change_3y",
    "margin_sample_sd_5y",
    "organic_growth",
    "authored_scenarios",
    "holding_correlation",
    "portfolio_correlation",
    "downside_beta",
    "euler_variance_fraction",
    "max_drawdown_5y",
    "daily_es95",
    "deal",
    "policy",
    "capital_allocation",
    "thesis_breakers",
    "limit_order_spread",
)
MIN_COHORT = 20
TOP_PERCENTILE = 95
CAP_BUCKETS = {"nano", "micro", "small", "mid", "large", "mega"}


def price_failed(row):
    """Verified sub-$5 names are ineligible: kept as audit rows, never ranked or used as peers."""
    return bool(row.get("price_ineligible")) or any(
        g.get("gate") == "listing_price_usd" and g.get("result") == "fail"
        for g in row.get("gates", [])
    )


def midrank_percentile(sorted_values, value):
    """Midrank percentile (0-100) of value within an ascending list that contains ties."""
    lo = bisect.bisect_left(sorted_values, value)
    hi = bisect.bisect_right(sorted_values, value)
    return 100.0 * (lo + (hi - lo) / 2) / len(sorted_values)


def cohort_key(row):
    cap = row.get("cap_bucket")
    if price_failed(row):
        return None
    return (row["kind"], cap) if cap in CAP_BUCKETS else None


FINANCIAL_NA = {"fcf_3y_cagr", "positive_fcf_latest3", "net_debt_ebitda", "cash_conversion"}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def na_allowed(kind, name):
    """The only (kind, gate) pairs a provider may mark not_applicable."""
    return (kind in ("bank", "insurer") and name in FINANCIAL_NA) or (
        kind == "bank" and name == "combined_ratio"
    )


def component(row, name):
    gate = next((g for g in row["gates"] if g["gate"] == name), None)
    default_na = (
        (row["kind"] in ("bank", "insurer") and name in FINANCIAL_NA)
        or (row["kind"] == "bank" and name == "combined_ratio")
        or (row["kind"] == "stock" and name in FINANCIAL)
    )
    status = gate["result"] if gate else ("not_applicable" if default_na else "unknown")
    if gate and status == "not_applicable" and not na_allowed(row["kind"], name):
        status = "unknown"  # same restriction as qualification(): no score inflation
    raw = gate.get("value") if gate else None
    value = raw
    if name == "positive_fcf_latest3":
        value = (
            int(all(x > 0 for x in raw))
            if isinstance(raw, list) and len(raw) == 3 and all(finite(x) for x in raw)
            else None
        )
    elif name == "annual_dilution":
        value = (
            max(raw)
            if isinstance(raw, list) and len(raw) == 3 and all(finite(x) for x in raw)
            else None
        )
    elif not finite(value) or (name in ("pe_own", "pe_sector", "ptbv_own") and value <= 0):
        value = None
    usable = status in ("pass", "fail", "exception") and value is not None
    return status, raw, value if usable else None


def qualification(row):
    required = list(SCREEN_GATES)
    if row["kind"] in ("bank", "insurer"):
        required.extend(FINANCIAL)
    gates = collections.defaultdict(list)
    for gate in row["gates"]:
        gates[gate["gate"]].append(gate)
    if "source_identity" in gates:
        required.append("source_identity")

    def status(name):
        matches = gates[name]
        if len(matches) != 1:
            return "unknown"
        gate = matches[0]
        result = gate.get("result")
        if result in ("pass", "fail"):
            return result
        if (
            result == "exception"
            and name == "net_debt_ebitda"
            and gate.get("exception_verified") is True
        ):
            return result
        if result == "not_applicable" and na_allowed(row["kind"], name):
            return result
        return "unknown"

    screen = {name: status(name) for name in required}
    later = {name: status(name) for name in LATER_GATES}
    allowed = {"pass", "not_applicable", "exception"}
    eligible = row.get("stage") in {
        "diligence",
        "portfolio",
        "pretrade",
        "review_ready",
        "screen_percentile_hold",
        "diligence_capacity_hold",
        "portfolio_capacity_hold",
        "pretrade_capacity_hold",
    } and all(value in allowed for value in screen.values())
    qualified = (
        eligible
        and row.get("stage") == "review_ready"
        and all(value in allowed for value in later.values())
    )
    return eligible, qualified, screen, later


def build(runs):
    latest = {}
    historical = []
    for index, run in enumerate(runs):
        for row in run["rows"]:
            item = dict(
                row,
                as_of=run["as_of"],
                run_id=run["run_id"],
                results_sha256=run["results_sha256"],
                run_health=json.dumps(run["health"], sort_keys=True, separators=(",", ":"))
                if run.get("health") is not None
                else None,
                input_run_index=index,
            )
            historical.append(item)
            key = (run["as_of"], index)
            if row["ticker"] not in latest or key >= latest[row["ticker"]][0]:
                latest[row["ticker"]] = (key, item)
    rows = [v[1] for v in latest.values() if v[1]["kind"] != "etf"]
    cohorts = collections.defaultdict(list)
    for row in rows:
        for name in METRICS:
            _, _, value = component(row, name)
            if value is not None and name != "positive_fcf_latest3" and cohort_key(row):
                cohorts[cohort_key(row), name].append(value)
    for values in cohorts.values():
        values.sort()
    issuers = collections.Counter(r["issuer_id"] for r in rows if r.get("issuer_id"))
    output = []
    for row in rows:
        record = {
            k: row.get(k)
            for k in (
                "ticker",
                "company_name",
                "issuer_id",
                "kind",
                "region",
                "sector",
                "cap_bucket",
                "stage",
                "as_of",
                "run_id",
                "results_sha256",
            )
        }
        eligible, qualified, screen_status, later_status = qualification(row)
        identity = screen_status["issuer_identity"]
        record.update(
            price_ineligible=price_failed(row),
            screen_qualified=eligible,
            guidelines_qualified=qualified,
            screen_fail_count=sum(v == "fail" for v in screen_status.values()),
            screen_unknown_count=sum(v == "unknown" for v in screen_status.values()),
            failed_criteria=sorted(
                {
                    name
                    for name, status in {**screen_status, **later_status}.items()
                    if status == "fail"
                }
                | {g["gate"] for g in row["gates"] if g.get("result") == "fail"}
            ),
            missing_criteria=[
                name
                for name, status in {**screen_status, **later_status}.items()
                if status == "unknown"
            ],
            screen_gate_status=screen_status,
            later_gate_status=later_status,
            identity_status=identity,
            identity_unverified=identity != "pass",
            potential_duplicate_issuer=bool(row.get("issuer_id") and issuers[row["issuer_id"]] > 1),
            gate_results={g["gate"]: g["result"] for g in row["gates"]},
        )
        total = total_weight = count = usable = small = 0
        for name in METRICS:
            status, raw, value = component(row, name)
            score = None if status == "not_applicable" else 0.0
            cohort = cohorts.get((cohort_key(row), name), [])
            if score is not None:
                count += 1
                total_weight += METRIC_WEIGHTS[name]
                if value is not None:
                    usable += 1
                    if name == "positive_fcf_latest3":
                        score = 100.0 * value
                    elif len(cohort) >= MIN_COHORT:
                        score = midrank_percentile(cohort, value)
                        if name not in HIGH:
                            score = 100.0 - score
                    else:
                        small += 1  # usable but cohort < MIN_COHORT: scores 0.0 (as always)
                total += score * METRIC_WEIGHTS[name]
            record.update(
                {
                    name + "_status": status,
                    name + "_value": raw,
                    name + "_score": score,
                    name + "_cohort_n": len(cohort) if name != "positive_fcf_latest3" else None,
                }
            )
        record.update(
            score=total / total_weight if total_weight else 0.0,
            coverage=usable / count if count else 0.0,
            usable=usable,
            small_cohort_terms=small,
            applicable=count,
        )
        output.append(record)
    output.sort(key=lambda r: (price_failed(r), -r["score"], -r["coverage"], r["ticker"]))
    composite_cohorts = collections.defaultdict(list)
    for row in output:
        if cohort_key(row):
            composite_cohorts[cohort_key(row)].append(row["score"])
    for scores in composite_cohorts.values():
        scores.sort()
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
        key = cohort_key(row)
        scores = composite_cohorts.get(key, [])
        row["ranking_cohort"] = ":".join(key) if key else None
        row["composite_cohort_n"] = len(scores)
        row["composite_percentile"] = None
        if len(scores) >= MIN_COHORT:
            row["composite_percentile"] = midrank_percentile(scores, row["score"])
    return output, historical
