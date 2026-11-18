"""Documented EODHD source adapters; unsupported accounting and calendars stay unknown.

Current retrievals are descriptive. Filing dates do not prove historical snapshot vintages.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from statistics import median

import exchange_calendars as xcals
import pandas as pd
from exchange_calendars.errors import CalendarError, DateOutOfBounds

from wharton import eodhd, screening

# Exchange aliases are explicit; no guessed weekday calendar for unsupported exchanges.
CALENDARS = {
    "US": "XNYS",
    "NASDAQ": "XNYS",
    "NYSE": "XNYS",
    "NYSE ARCA": "XNYS",
    "NYSE MKT": "XNYS",
    "AMEX": "XNYS",
    "NYSE AMERICAN": "XNYS",
    "BATS": "XNYS",
    "ST": "XSTO",
    "LSE": "XLON",
    "PA": "XPAR",
    "XETRA": "XETR",
    "HK": "XHKG",
    "TO": "XTSE",
    "SA": "BVMF",
    "AU": "XASX",
    "AS": "XAMS",
    "SW": "XSWX",
    "MC": "XMAD",
    "MI": "XMIL",
    "CO": "XCSE",
    "HE": "XHEL",
    "OL": "XOSL",
    "TSE": "XTKS",
    "SG": "XSES",
    "KO": "XKRX",
    "TW": "XTAI",
    "MX": "XMEX",
    "SN": "XSGO",
    "JSE": "XJSE",
    "TA": "XTAE",
    "SR": "XSAU",
}


US_EXCHANGES = {
    "US",
    "NASDAQ",
    "NYSE",
    "NYSE ARCA",
    "NYSE MKT",
    "AMEX",
    "NYSE AMERICAN",
    "BATS",
    "OTC",
    "OTCQX",
    "OTCQB",
    "PINK",
}


def expected_identity(record):
    return {
        "ticker": record["ticker"],
        "issuer_id": record["issuer_id"],
        "identifiers": {k: record.get("identifiers", {}).get(k) for k in ("cik", "figi", "isin")},
    }


class SourceIdentityConflict(ValueError):
    """Well-formed source identification contradicts the explicit expected security."""

    def __init__(self, reason, record, observed, receipt=None):
        super().__init__(reason)
        self.expected = expected_identity(record)
        self.observed, self.receipt = observed, receipt


def listing_identity(payload, record):
    general = payload.get("General")
    if not isinstance(general, dict) or any(
        not isinstance(general.get(k), str) or not general[k].strip() for k in ("Code", "Exchange")
    ):
        raise ValueError("fundamentals requires well-formed General.Code and Exchange")
    expected = record.get("identifiers", {})
    observed = {k: general.get(k) for k in ("Code", "Exchange", "CIK", "OpenFigi", "ISIN")}
    if str(general.get("HomeCategory", "")).upper() == "ETD":
        observed["HomeCategory"] = general["HomeCategory"]
        raise SourceIdentityConflict("ETD is not an equity listing", record, observed)
    mismatches = []
    for key, field in (("isin", "ISIN"), ("cik", "CIK"), ("figi", "OpenFigi")):
        left, right = expected.get(key), general.get(field)
        if left in (None, "") or right in (None, ""):
            continue
        if key == "cik":
            if not str(left).isdigit() or not str(right).isdigit():
                raise ValueError("invalid CIK identifier")
            left, right = str(int(left)), str(int(right))
        elif not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(f"malformed {key} identifier")
        if left != right:
            mismatches.append(key)
    code, exchange = record["ticker"].rsplit(".", 1)
    actual = general["Exchange"].upper()
    if general["Code"] != code or not (
        actual in US_EXCHANGES if exchange == "US" else actual == exchange
    ):
        raise SourceIdentityConflict(
            "fundamentals qualified listing identity mismatch", record, observed
        )
    if mismatches:
        raise SourceIdentityConflict(
            "fundamentals issuer " + ",".join(mismatches) + " mismatch", record, observed
        )
    return general


def numeric(value):
    try:
        return None if isinstance(value, bool) else screening.number(float(value))
    except (TypeError, ValueError):
        return None


def stamp(receipt):
    return str(screening.day(receipt["retrieved_at"]))


def item(value, receipt, field, **extra):
    observed = stamp(receipt)
    return dict(
        value=value,
        verified=True,
        source=f"sources/{receipt['file']}#{field}",
        period=observed,
        available_at=observed,
        observed_at=observed,
        **extra,
    )


@lru_cache(maxsize=256)
def _session_closes(calendar_name, cutoff):
    """Sessions and closes per (calendar name, cutoff); independent of retrieval time."""
    calendar = xcals.get_calendar(calendar_name)
    first = cutoff - pd.DateOffset(years=5) - pd.Timedelta(days=7)
    dates = calendar.sessions_in_range(first, cutoff)
    return dates, pd.DatetimeIndex([calendar.session_close(d) for d in dates])


def completed_sessions(calendar_name, as_of, retrieved_at):
    """Exclude today's scheduled bar until the exchange close has actually completed."""
    try:
        now = pd.Timestamp(retrieved_at)
        dates, closes = _session_closes(calendar_name, pd.Timestamp(as_of))
        if now.tz is None:
            raise ValueError("retrieval timestamp must include timezone")
        return dates[closes <= now]
    except (CalendarError, DateOutOfBounds):
        return pd.DatetimeIndex([])


def latest_completed_session(calendar_name, as_of, now):
    """ISO date of the latest native session completed at min(now, end of the as_of UTC day), or
    None when the calendar cannot say. The one definition shared by price_state, the coverage
    replay check and the pilot window planner."""
    try:
        cut = pd.Timestamp(str(screening.day(as_of)) + "T23:59:59+00:00")
        clock = pd.Timestamp(now)
        if clock is pd.NaT or clock.tz is None:
            return None  # an unusable consumption clock is unproved, never a pass
        end = min(clock, cut)
        done = completed_sessions(calendar_name, str(screening.day(as_of)), end.isoformat())
    except (ValueError, TypeError):
        return None
    return str(done[-1].date()) if len(done) else None


def price_item_stale(item, as_of, now):
    """None when a stored listing_price_usd item is still the latest completed native session at
    `now` (clipped to the end of the as_of day), else a reason: price_stale (a later session has
    completed) or session_calendar_unavailable (unproved). Items that name no exchange calendar
    (curated/legacy) are not judged here. Only meaningful on the as_of day itself: a historical
    as_of is never re-judged against a later clock, so replays stay deterministic."""
    if not isinstance(item, dict) or not str(item.get("session_source", "")).startswith(
        "exchange_calendars "
    ):
        return None
    try:
        clock = pd.Timestamp(now) if now is not None else pd.NaT
    except (ValueError, TypeError):
        clock = pd.NaT  # unparseable clock: unproved, never a raise or a pass
    if clock is pd.NaT or clock.tz is None:
        return "session_calendar_unavailable"
    if str(screening.day(as_of)) != str(clock.tz_convert("UTC").date()):
        return None
    calendar = item["session_source"].removeprefix("exchange_calendars ").strip()
    latest = latest_completed_session(calendar, as_of, clock)
    if latest is None:
        return "session_calendar_unavailable"
    return None if latest == item.get("expected_session") else "price_stale"


def native_session_signature(records, as_of, now):
    """{calendar: latest completed native session at `now` (clipped to the end of the as_of
    day)} for every listing calendar the records use (ticker suffix or a stored price item's
    session_source). Empty unless as_of is the clock's UTC day, so historical identities stay
    stable. Part of the run identity: a report frozen before a later close is never reused."""
    try:
        clock = pd.Timestamp(now)
    except (ValueError, TypeError):
        return {}
    if (
        clock is pd.NaT
        or clock.tz is None
        or str(screening.day(as_of)) != str(clock.tz_convert("UTC").date())
    ):
        return {}
    calendars = set()
    for record in records:
        calendars.add(CALENDARS.get(str(record.get("ticker", "")).rsplit(".", 1)[-1]))
        evidence = record.get("evidence")
        item = evidence.get("listing_price_usd") if isinstance(evidence, dict) else None
        source = item.get("session_source") if isinstance(item, dict) else None
        if isinstance(source, str) and source.startswith("exchange_calendars "):
            calendars.add(source.removeprefix("exchange_calendars ").strip())
    return {
        name: latest_completed_session(name, as_of, clock) for name in sorted(calendars - {None})
    }


def revalidate_price_sessions(records, as_of, now):
    """Run-wide consumption gate: withdraw every record's listing_price_usd whose native session
    has advanced (or cannot be proved) so the engine sees NO price (unknown, refetch needed), not
    a stale pass. Covers cached/unselected/replayed records alike. Returns audit rows
    [{ticker, reason, expected_session}]; the caller records them with the run clock."""
    out = []
    for record in records:
        evidence = record.get("evidence")
        item = evidence.get("listing_price_usd") if isinstance(evidence, dict) else None
        reason = price_item_stale(item, as_of, now)
        if reason:
            out.append(
                {
                    "ticker": record.get("ticker"),
                    "reason": reason,
                    "expected_session": item.get("expected_session"),
                }
            )
            evidence.pop("listing_price_usd")
    return out


def split_events(payload, receipt):
    """Documented date/split 'new/old' values; reject malformed/duplicate actions."""
    if not isinstance(payload, list):
        raise ValueError("split history must be an array")
    out = {}
    for row in payload:
        day = screening.day(row.get("date"))
        parts = str(row.get("split", "")).split("/")
        if day is None or len(parts) != 2 or str(day) in out:
            raise ValueError("invalid or duplicate split action")
        new, old = map(numeric, parts)
        if new is None or old is None or new <= 0 or old <= 0:
            raise ValueError("invalid split ratio")
        if str(day) <= receipt["as_of"]:
            out[str(day)] = new / old
    return out


def split_covered(receipt, after, through):
    params = (receipt or {}).get("parameters", {})
    return bool(
        params.get("from")
        and params.get("to")
        and params["from"] <= after
        and params["to"] >= through
    )


def split_factor(events, after, through):
    return math.prod(value for day, value in events.items() if after < day <= through)


def identity_metadata(payload, record, receipt):
    """Listing validation is performed by service; bind only matching supplied identifiers."""
    general = listing_identity(payload, record)
    expected = record.get("identifiers", {})
    matched = {}
    for key, field in (("isin", "ISIN"), ("cik", "CIK"), ("figi", "OpenFigi")):
        left, right = expected.get(key), general.get(field)
        if left is None or right in (None, ""):
            continue
        if key == "cik":
            if not str(left).isdigit() or not str(right).isdigit():
                raise ValueError("invalid CIK identifier")
            left, right = str(int(left)), str(int(right))
        if str(left) != str(right):
            raise SourceIdentityConflict(
                f"fundamentals issuer {key} mismatch", record, {field: right}, receipt
            )
        matched[key] = str(left)
    out = {}
    if matched:
        out["issuer_identity"] = item(
            dict(ticker=record["ticker"], issuer_id=record["issuer_id"], **matched),
            receipt,
            "General identifiers+Code+Exchange",
        )
    out["listing_metadata"] = item(
        {
            k: general.get(k)
            for k in ("Code", "Exchange", "CurrencyCode", "Type", "Industry", "Sector")
        },
        receipt,
        "General",
    )
    industry = str(general.get("Industry") or "").lower()
    if "insurance" in industry or "insurer" in industry:
        record["kind"] = "insurer"
    elif "bank" in industry or "capital markets" in industry:
        record["kind"] = "bank"
    if general.get("Sector"):
        record["sector"] = general["Sector"]
    description = general.get("Description")
    if isinstance(description, str) and description.strip():
        out["business_description"] = item(description.strip(), receipt, "General.Description")
    trailing = numeric(payload.get("Valuation", {}).get("TrailingPE"))
    if trailing is not None and trailing > 0:
        out["pe"] = item(
            trailing, receipt, "Valuation.TrailingPE", valuation_basis="provider trailing TTM P/E"
        )
    roe = numeric(payload.get("Highlights", {}).get("ReturnOnEquityTTM"))
    if roe is not None:
        out["roe"] = item(roe, receipt, "Highlights.ReturnOnEquityTTM")
    # The provider docs do not specify market-cap currency: require an attributed basis.
    cap_basis = record.get("capitalization_basis", {})
    cap = numeric(payload.get("Highlights", {}).get("MarketCapitalization"))
    if (
        cap is not None
        and cap > 0
        and cap_basis.get("verified") is True
        and cap_basis.get("source")
        and cap_basis.get("field") == "Highlights.MarketCapitalization"
        and cap_basis.get("currency") == "USD"
    ):
        out["market_cap_usd"] = item(
            cap, receipt, "Highlights.MarketCapitalization; " + cap_basis["source"], currency="USD"
        )
    return out


def sec_rows(payload, record, receipt, *, form, currency, issuer_bound=False):
    """Flat official data/meta/links contract; duplicate periods need explicit vintage choice.

    SEC API does not declare reporting currency/accounting framework per row. The supplied
    accounting assertion must establish domestic US GAAP and original period-end share basis.
    Issuer binding requires a matching row CIK, a validated consolidated company CIK,
    or sec_issuer_bindings[accession] with verified, matching cik, and cited source.
    Accessions alone identify submissions, not necessarily the issuer.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("SEC filings require documented data array envelope")
    basis = record.get("sec_accounting", {})
    if not (
        record["ticker"].endswith(".US")
        and basis.get("verified") is True
        and basis.get("source")
        and basis.get("framework") == "US_GAAP"
        and basis.get("currency") == currency
        and currency == "USD"
        and basis.get("per_share_basis") == "period_end"
    ):
        return []
    cik = str(record.get("identifiers", {}).get("cik", ""))
    if not cik.isdigit():
        return []
    endpoint = receipt.get("endpoint")
    if endpoint and endpoint != f"sec-filings/{record['ticker']}/{form}":
        raise SourceIdentityConflict(
            "SEC endpoint listing identity mismatch", record, {"endpoint": endpoint}, receipt
        )
    choices = record.get("sec_vintages", {})
    periods = {}
    rows = []
    for raw in payload["data"]:
        if not isinstance(raw, dict):
            raise ValueError("SEC filing row must be an object")
        period, available = (
            screening.day(raw.get("period_of_report")),
            screening.day(raw.get("filed_at")),
        )
        accession = raw.get("accession_number")
        if (
            not period
            or not available
            or available < period
            or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", str(accession))
        ):
            continue
        row_cik = raw.get("cik")
        if row_cik is not None and (not str(row_cik).isdigit() or int(row_cik) != int(cik)):
            raise SourceIdentityConflict("SEC filing row issuer mismatch", record, raw, receipt)
        binding = record.get("sec_issuer_bindings", {}).get(accession, {})
        reviewed = (
            isinstance(binding, dict)
            and binding.get("verified") is True
            and str(binding.get("cik", "")).isdigit()
            and int(binding["cik"]) == int(cik)
            and isinstance(binding.get("source"), str)
            and bool(binding["source"].strip())
        )
        if row_cik is None and not issuer_bound and not reviewed:
            continue
        if str(available) > receipt["as_of"] or str(period) > receipt["as_of"]:
            continue
        selected = choices.get(str(period))
        if selected and selected != accession:
            continue
        row = dict(
            raw,
            period=str(period),
            available_at=str(available),
            currency=currency,
            source=f"sources/{receipt['file']}#data/{accession}"
            + (f"; {binding['source']}" if row_cik is None and not issuer_bound else ""),
            form=form,
        )
        periods.setdefault(str(period), []).append(row)
    for versions in periods.values():
        if len(versions) == 1:
            rows.extend(versions)
    return sorted(rows, key=lambda r: r["period"])


def normalize_sec(payload, record, receipt, splits=None, split_receipt=None, *, issuer_bound=False):
    rows = sec_rows(payload, record, receipt, form="10k", currency="USD", issuer_bound=issuer_bound)
    mapped = {}
    fields = {
        "diluted_eps": "eps_diluted",
        "diluted_shares": "weighted_avg_shares_diluted",
        "revenue": "revenue",
        "net_income": "net_income",
        "ocf": "operating_cash_flow",
        "capex": "capital_expenditure",
        "ebitda_annual": "ebitda",
        "operating_income": "operating_income",
    }
    events = split_events(splits, split_receipt) if splits is not None and split_receipt else None
    target = (split_receipt or {}).get(
        "market_as_of", receipt.get("market_as_of", receipt["as_of"])
    )
    target_day = screening.day(target)
    for key, field in fields.items():
        values = []
        for row in rows:
            value = numeric(row.get(field))
            if value is None:
                continue
            if key == "diluted_eps" and not ttm_listing_basis(
                record, str(screening.day(row["period"]) - pd.Timedelta(days=380)), target
            ):
                continue
            if key in {"diluted_eps", "diluted_shares"}:
                if (
                    events is None
                    or target_day is None
                    or target > receipt["as_of"]
                    or row["period"] > target
                    or not split_covered(split_receipt, row["period"], target)
                    or any(row["period"] < d <= row["available_at"] for d in events)
                ):
                    continue
                factor = split_factor(events, row["period"], target)
                value = value / factor if key == "diluted_eps" else value * factor
            values.append(
                {
                    k: row[k]
                    for k in (
                        "period",
                        "available_at",
                        "currency",
                        "source",
                        "accession_number",
                        "form",
                    )
                }
                | {"value": value}
            )
        if values:
            mapped[key] = item(values, receipt, f"data.{field}")
            mapped[key]["period"] = [row["period"] for row in values]
            if key == "diluted_eps":
                proof = record["eps_listing_basis"]
                mapped[key].update(
                    basis="reported_diluted",
                    statement_source="sec_financials",
                    accounting_framework="US_GAAP",
                    basis_evidence={
                        "verified": True,
                        "source": proof["source"],
                        "ticker": proof["ticker"],
                        "currency": "USD",
                        "share_basis": "diluted",
                        "from": proof["from"],
                        "through": proof["through"],
                    },
                )
            if key == "diluted_shares":
                mapped[key].update(basis="weighted_average_diluted_split_adjusted")
            if key in {"diluted_eps", "diluted_shares"}:
                mapped[key]["split_source"] = f"sources/{split_receipt['file']}#date+split"
                mapped[key]["share_basis_date"] = target
    return mapped


def normalize_sec_financials(payload, record, receipt, splits=None, split_receipt=None):
    """Normalize documented annual rows only when filing provenance is actually present."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("company"), dict):
        raise ValueError("SEC financials requires company and annual/quarterly envelope")
    if not isinstance(data.get("annual"), list) or not isinstance(data.get("quarterly"), list):
        raise ValueError("SEC financials requires annual and quarterly arrays")
    company = data["company"]
    cik = str(company.get("cik", ""))
    expected = str(record.get("identifiers", {}).get("cik", ""))
    ticker, exchange = record["ticker"].rsplit(".", 1)
    if not cik.isdigit() or not expected.isdigit():
        raise ValueError("SEC financials requires issuer CIK")
    if (
        int(cik) != int(expected)
        or company.get("ticker") != ticker
        or company.get("exchange") != exchange
    ):
        raise SourceIdentityConflict(
            "SEC financials issuer or listing mismatch", record, company, receipt
        )
    endpoint = receipt.get("endpoint")
    if endpoint and endpoint != f"sec-companies/{record['ticker']}/financials":
        raise SourceIdentityConflict(
            "SEC financials endpoint mismatch", record, {"endpoint": endpoint}, receipt
        )
    annual = []
    for row in data["annual"]:
        if not isinstance(row, dict):
            raise ValueError("SEC annual financials row must be an object")
        if (
            row.get("period_of_report")
            and row.get("period")
            and row["period_of_report"] != row["period"]
        ):
            raise ValueError("SEC financials conflicting period")
        annual.append({**row, "period_of_report": row.get("period_of_report") or row.get("period")})
    adjusted = {**receipt, "endpoint": f"sec-filings/{record['ticker']}/10k"}
    result = normalize_sec(
        {"data": annual}, record, adjusted, splits, split_receipt, issuer_bound=True
    )
    prefix = f"sources/{receipt['file']}#data"
    for evidence in result.values():
        evidence["source"] = evidence["source"].replace(prefix, prefix + ".annual", 1)
        for row in evidence["value"]:
            row["source"] = row["source"].replace(prefix, prefix + ".annual", 1)
    return result


# Minor-unit codes: (major currency, divisor). Admitted only with record["price_unit_verified"].
MINOR_UNITS = {"GBX": ("GBP", 100), "GBp": ("GBP", 100), "ILA": ("ILS", 100), "ZAC": ("ZAR", 100)}
UNIT_RISK_EXCHANGES = {"LSE", "TA", "JSE"}  # majors can be quoted in pence/agorot/cents


def price_state(
    record,
    payload,
    receipt,
    *,
    fundamentals=None,
    fundamentals_receipt=None,
    fx=None,
    fx_receipt=None,
    cutoff=None,
    now=None,
):
    """Pure, offline USD raw-close state: pass (>= $5), fail (< $5) or unknown.

    Only the nominal `close` of the latest completed native session is used, never
    `adjusted_close`. `needs` is the only signal that a fetch could help:
    "metadata" (metadata_receipt_missing, metadata_missing_or_stale, metadata_stale),
    "fx" (fx_missing, fx_stale, fx_not_after_session, fx_rate_missing) or "price"
    (price_symbol_mismatch, price_payload_invalid, price_stale, price_row_missing,
    price_row_duplicate, price_close_invalid, price_receipt_invalid when the receipt is
    malformed). It is None for non-fetchable unknowns: price_receipt_invalid when the
    receipt is future-dated vs the cutoff, identity_quarantine, unsupported_listing,
    currency_unproved, unit_unproved, session_calendar_unavailable; callers must skip
    paid work for those.
    Listing identity conflicts still raise SourceIdentityConflict for quarantine.

    `now` (consumption time, tz-aware) re-validates the latest completed native session when the
    receipt is consumed after it was retrieved: if a later session has completed (clipped to the
    end of the cutoff day) the receipt is price_stale (refetch needed). The receipt's own
    retrieved_at is never altered. Without `now` the state depends on the receipt alone.
    """
    state = dict(status="unknown", price_usd=None, needs=None, reason="", expected_session=None)

    def hold(reason, needs=None):
        state.update(reason=reason, needs=needs)
        return state

    receipt = receipt or {}
    cut = screening.day(cutoff if cutoff is not None else receipt.get("as_of"))
    observed = screening.day(receipt.get("retrieved_at"))
    market_cutoff = screening.day(receipt.get("market_as_of", receipt.get("as_of")))
    if (
        cut is None
        or (observed is not None and observed > cut)
        or (market_cutoff is not None and market_cutoff > cut)
    ):
        # future-dated vs the frozen cutoff: a refetch is later still
        return hold("price_receipt_invalid")
    if observed is None or market_cutoff is None or not receipt.get("file"):
        return hold(
            "price_receipt_invalid", "price"
        )  # malformed receipt: a fresh fetch replaces it
    if receipt.get("symbol") != record.get("ticker"):
        return hold("price_symbol_mismatch", "price")
    if not isinstance(payload, list):
        return hold("price_payload_invalid", "price")
    if record.get("identity_quarantine"):
        return hold("identity_quarantine")
    if fundamentals is not None:
        general = listing_identity(fundamentals, record)
        if not fundamentals_receipt or not fundamentals_receipt.get("file"):
            return hold("metadata_receipt_missing", "metadata")
        metadata_observed = screening.day(fundamentals_receipt.get("retrieved_at"))
        metadata_source = f"sources/{fundamentals_receipt['file']}#General"
    else:
        cached, valid = screening.evidence(record, "listing_metadata", cut)
        if not valid or not isinstance(cached.get("value"), dict):
            return hold("metadata_missing_or_stale", "metadata")
        general = listing_identity({"General": cached["value"]}, record)
        metadata_observed = screening.day(
            cached.get("oldest_source_observed_at", cached.get("observed_at"))
        )
        metadata_source = cached["source"]
    if metadata_observed is None or not 0 <= (cut - metadata_observed).days <= 45:
        return hold("metadata_stale", "metadata")
    exchange = str(general.get("Exchange", "")).upper()
    calendar = CALENDARS.get(exchange)
    currency = general.get("CurrencyCode")
    if not calendar or general.get("Type") not in {"Common Stock", "ETF"}:
        return hold("unsupported_listing")
    divisor, fx_code = 1, currency
    if not isinstance(currency, str):
        return hold("currency_unproved")
    if currency in MINOR_UNITS:
        fx_code, divisor = MINOR_UNITS[currency]
    elif not re.fullmatch(r"[A-Z]{3}", currency):
        return hold("currency_unproved")
    if (currency in MINOR_UNITS or exchange in UNIT_RISK_EXCHANGES) and (
        record.get("price_unit_verified") is not True
    ):
        return hold("unit_unproved")
    if observed < cut:
        return hold("price_stale", "price")  # receipt retrieved before the as_of day
    try:
        sessions = completed_sessions(calendar, str(market_cutoff), receipt["retrieved_at"])
        # Latest session completed at min(retrieved_at, end of as_of day).
        seen = pd.Timestamp(receipt["retrieved_at"])
        if now is not None:
            clock = pd.Timestamp(now)
            if clock is pd.NaT or clock.tz is None:
                return hold("consumption_clock_invalid")  # unusable clock: never a silent pass
            seen = max(seen, clock)  # consumption time may be later than retrieval
        end = min(seen, pd.Timestamp(f"{cut}T23:59:59+00:00"))
        due = completed_sessions(calendar, str(cut), end.isoformat())
    except ValueError:
        return hold("price_receipt_invalid", "price")  # unparseable/naive retrieved_at
    if sessions.empty or due.empty:
        return hold("session_calendar_unavailable")
    quote_day, state["expected_session"] = str(sessions[-1].date()), str(due[-1].date())
    if quote_day != state["expected_session"]:
        return hold("price_stale", "price")
    matches = [r for r in payload if isinstance(r, dict) and r.get("date") == quote_day]
    if len(matches) != 1:
        return hold("price_row_missing" if not matches else "price_row_duplicate", "price")
    close = numeric(matches[0].get("close"))
    if close is None or close <= 0:
        return hold("price_close_invalid", "price")
    rate, constituents = 1.0, [str(observed), str(metadata_observed)]
    if fx_code != "USD":
        if (
            not fx_receipt
            or fx_receipt.get("symbol") != f"{fx_code}USD.FOREX"
            or not fx_receipt.get("file")
            or not isinstance(fx, list)
        ):
            return hold("fx_missing", "fx")
        fx_observed = screening.day(fx_receipt.get("retrieved_at"))
        if fx_observed is None or not 0 <= (cut - fx_observed).days <= 45:
            return hold("fx_stale", "fx")
        if str(fx_observed) <= quote_day:
            return hold("fx_not_after_session", "fx")
        rows = [r for r in fx if isinstance(r, dict) and r.get("date") == quote_day]
        rate = numeric(rows[0].get("close")) if len(rows) == 1 else None
        if rate is None or rate <= 0:
            return hold("fx_rate_missing", "fx")
        constituents.append(str(fx_observed))
    value = close / divisor * rate
    if screening.number(value) is None:
        return hold("price_close_invalid", "price")
    out = item(value, receipt, f"{quote_day}.close", currency="USD")
    out.update(
        period=quote_day,
        local_currency=currency,
        local_close=close,
        metadata_source=metadata_source,
        last_completed_session=quote_day,
        expected_session=state["expected_session"],
        receipt_as_of=str(screening.day(receipt.get("as_of"))),
        session_source=f"exchange_calendars {calendar}",
        observed_at=max(constituents),
        available_at=max(constituents),
        oldest_source_observed_at=min(constituents),
    )
    if fx_code != "USD":
        out["fx_source"] = f"sources/{fx_receipt['file']}#{quote_day}.close"
    ok = value >= screening.PRICE_FLOOR_USD - screening.PRICE_TOLERANCE
    state.update(
        status="pass" if ok else "fail",
        price_usd=value,
        reason="ok" if ok else "below_floor",
        evidence=out,
    )
    return state


def normalize_listing_price(
    payload,
    record,
    receipt,
    *,
    fundamentals=None,
    fundamentals_receipt=None,
    fx=None,
    fx_receipt=None,
    now=None,
):
    """Evidence item for verified pass/fail prices; {} when unknown (see price_state). `now` is
    the consumption clock: a price whose native session has since advanced yields no evidence."""
    state = price_state(
        record,
        payload,
        receipt,
        fundamentals=fundamentals,
        fundamentals_receipt=fundamentals_receipt,
        fx=fx,
        fx_receipt=fx_receipt,
        cutoff=receipt.get("as_of"),
        now=now,
    )
    return {"listing_price_usd": state["evidence"]} if "evidence" in state else {}


def normalize_global_prices(
    payload,
    fundamentals,
    record,
    receipt,
    *,
    fundamentals_receipt=None,
    fx=None,
    fx_receipt=None,
    now=None,
):
    general = listing_identity(fundamentals, record)
    exchange = str(general.get("Exchange", "")).upper()
    calendar_name = CALENDARS.get(exchange)
    currency = general.get("CurrencyCode")
    code, suffix = record["ticker"].rsplit(".", 1)
    if general.get("Code") != code or not (
        exchange == suffix or suffix == "US" and exchange in US_EXCHANGES
    ):
        raise SourceIdentityConflict(
            "price listing metadata identity mismatch", record, general, fundamentals_receipt
        )
    if not calendar_name or not currency or general.get("Type") not in {"Common Stock", "ETF"}:
        return {}
    # GBX/GBp quoted pence and other unit multipliers require an explicit verified mapping.
    if currency in {"GBX", "GBp", "ILA", "ZAC"}:
        return {}
    sessions = completed_sessions(
        calendar_name, receipt.get("market_as_of", receipt["as_of"]), receipt["retrieved_at"]
    )
    if sessions.empty:
        return {}
    start, end = str(sessions[0].date()), str((sessions[-1] + pd.Timedelta(days=1)).date())
    prices = eodhd.adjusted_series(payload, record["ticker"], start, end).reindex(sessions)
    if pd.isna(prices.iloc[-1]):
        return {}
    local = prices.pct_change(fill_method=None)
    if currency == "USD":
        returns = local
    else:
        if fx is None or not fx_receipt or fx_receipt.get("symbol") != f"{currency}USD.FOREX":
            return {}
        fx_series = eodhd.adjusted_series(fx, fx_receipt["symbol"], start, end).reindex(sessions)
        if pd.isna(fx_series.iloc[-1]):
            return {}
        returns = (1 + local) * (1 + fx_series.pct_change(fill_method=None)) - 1
    previous = {d: sessions[i - 1] for i, d in enumerate(sessions) if i}
    out = item(
        [
            dict(date=str(d.date()), value=float(v), interval_start=str(previous[d].date()))
            for d, v in returns.dropna().items()
        ],
        receipt,
        "adjusted_close (split+dividend); verified listing calendar/currency",
        currency="USD",
        frequency="daily",
        return_basis="total_return",
    )
    out.update(
        period=[start, str(sessions[-1].date())],
        calendar=calendar_name,
        local_currency=currency,
        last_completed_session=str(sessions[-1].date()),
        session_source=f"exchange_calendars {calendar_name}",
        native_sessions=[str(d.date()) for d in sessions],
        native_sessions_verified=True,
        local_returns=[dict(date=str(d.date()), value=float(v)) for d, v in local.dropna().items()],
    )
    constituents = [receipt] + ([fx_receipt] if fx_receipt else [])
    out["observed_at"] = out["available_at"] = max(stamp(r) for r in constituents)
    out["oldest_source_observed_at"] = min(stamp(r) for r in constituents)
    if fundamentals_receipt:
        out["metadata_source"] = (
            f"sources/{fundamentals_receipt['file']}#General.Code+Exchange+CurrencyCode+Type"
        )
        out["verified"] = (
            0
            <= (screening.day(out["observed_at"]) - screening.day(stamp(fundamentals_receipt))).days
            <= 45
        )
    else:
        out["verified"] = False
        out["metadata_hold"] = "listing metadata receipt missing"
    if fx_receipt:
        out["fx_source"] = f"sources/{fx_receipt['file']}#date-matched {currency}USD"
    return {
        "daily_total_returns_usd": out,
        **normalize_listing_price(
            payload,
            record,
            receipt,
            fundamentals=fundamentals,
            fundamentals_receipt=fundamentals_receipt,
            fx=fx,
            fx_receipt=fx_receipt,
            now=now,
        ),
    }


def matched_valuations(
    prices, annual_payload, record, annual_receipt, price_receipt, splits, split_receipt
):
    """Annual earnings/book multiples on actual unadjusted closes; never calendar EPS.

    Annual denominator is consistent for current and history. Snapshot availability is
    descriptive, not a PIT backtest; amendments/unknown split basis are withheld.
    """
    rows = sec_rows(annual_payload, record, annual_receipt, form="10k", currency="USD")
    if not rows or splits is None or not split_receipt:
        return {}
    events = split_events(splits, split_receipt)
    price_map = {}
    for raw in prices:
        date, close = screening.day(raw.get("date")), numeric(raw.get("close"))
        if (
            not date
            or close is None
            or close <= 0
            or str(date) > price_receipt.get("market_as_of", price_receipt["as_of"])
        ):
            continue
        if str(date) in price_map:
            raise ValueError("duplicate valuation price date")
        price_map[str(date)] = close
    if not price_map:
        return {}
    # Do not value an open session or a provider bar on a non-session date.
    allowed = {
        str(d.date())
        for d in completed_sessions(
            "XNYS",
            price_receipt.get("market_as_of", price_receipt["as_of"]),
            price_receipt["retrieved_at"],
        )
    }
    price_map = {d: v for d, v in price_map.items() if d in allowed}
    if not price_map:
        return {}
    monthly = {}
    for date in sorted(price_map):
        monthly[date[:7]] = date
    histories = {"pe": [], "ptbv": []}
    last_date = max(price_map)
    for date in sorted(set(monthly.values()) | {last_date}):
        eligible = [
            r
            for r in rows
            if r["available_at"] < date
            and (pd.Timestamp(date) - pd.Timestamp(r["period"])).days <= 550
        ]
        if not eligible:
            continue
        row = eligible[-1]
        if not split_covered(
            split_receipt, row["period"], price_receipt.get("market_as_of", price_receipt["as_of"])
        ) or any(row["period"] < d <= row["available_at"] for d in events):
            continue
        factor = split_factor(events, row["period"], date)
        eps = numeric(row.get("eps_diluted"))
        equity, goodwill, intangibles, shares = (
            numeric(row.get(k))
            for k in ("stockholders_equity", "goodwill", "intangible_assets", "shares_outstanding")
        )
        tangible = (
            (equity - goodwill - intangibles) / (shares * factor)
            if all(v is not None for v in (equity, goodwill, intangibles, shares)) and shares > 0
            else None
        )
        denominators = {"pe": eps / factor if eps is not None else None, "ptbv": tangible}
        for key, denominator in denominators.items():
            if denominator is None or denominator <= 0:
                continue
            histories[key].append(
                dict(
                    period=date[:7] + "-01",
                    price_date=date,
                    value=price_map[date] / denominator,
                    earnings_period=row["period"],
                    earnings_available_at=row["available_at"],
                    earnings_currency="USD",
                    price_currency="USD",
                    matched=True,
                    source=row["source"]
                    + f"; sources/{price_receipt['file']}#close"
                    + f"; sources/{split_receipt['file']}#split",
                    basis="latest filed annual US GAAP; original close; no PIT vintage claim",
                )
            )
    mapped = {}
    for key, history in histories.items():
        if history:
            mapped[f"diagnostic_annual_{key}_monthly"] = item(
                history,
                price_receipt,
                "close + filing-dated annual denominator + splits",
                valuation_basis="annual; current retrieval; not historical PIT",
            )
            latest = history[-1]
            if latest["price_date"] == last_date:
                mapped[f"diagnostic_annual_{key}"] = item(
                    latest["value"],
                    price_receipt,
                    latest["source"],
                    currency="USD",
                    valuation_basis="latest filed annual; not TTM",
                )
                mapped[f"diagnostic_annual_{key}"]["period"] = last_date
    return mapped


def sector_medians(records, as_of, merge):
    """Separate positive current TTM P/E cohort by verified sector and accounting kind."""
    cutoff = screening.day(as_of)
    for record in records:
        previous = record.setdefault("evidence", {}).get("sector_pe", {})
        if previous.get("owner") == "eodhd_screen_service":
            record["evidence"].pop("sector_pe")
        valid = []
        own_basis = record.get("evidence", {}).get("pe", {}).get("valuation_basis")
        if not record.get("sector") or own_basis not in {
            "provider trailing TTM P/E",
            "reported US GAAP diluted TTM P/E",
        }:
            continue  # the inner loop only matches an own_basis in this set (hoisted, same output)
        for peer in records:
            if (
                peer is record
                or peer.get("sector") != record.get("sector")
                or peer.get("kind") != record.get("kind")
            ):
                continue
            value, fresh = screening.evidence(peer, "pe", cutoff)
            cap, cap_fresh = screening.evidence(peer, "market_cap_usd", cutoff)
            metadata, metadata_fresh = screening.evidence(peer, "listing_metadata", cutoff)
            if (
                fresh
                and (
                    own_basis != "reported US GAAP diluted TTM P/E"
                    or screening.proved_ttm_listing(peer, value)
                )
                and cap_fresh
                and metadata_fresh
                and metadata.get("value", {}).get("Sector") == record.get("sector")
                and cap.get("currency") == "USD"
                and numeric(cap.get("value")) is not None
                and cap["value"] > 0
                and value.get("valuation_basis") == own_basis
                and own_basis in {"provider trailing TTM P/E", "reported US GAAP diluted TTM P/E"}
                and numeric(value.get("value")) is not None
                and value["value"] > 0
            ):
                valid.append((peer["ticker"], value))
        if len(valid) >= 20:
            value = dict(
                value=median(v["value"] for _, v in valid),
                verified=True,
                source=[v["source"] for _, v in valid],
                period=as_of,
                available_at=as_of,
                observed_at=as_of,
                listing_basis_version=2,
                peer_n=len(valid),
                oldest_source_observed_at=min(v["observed_at"] for _, v in valid),
                peer_definition=(
                    f"retained positive {own_basis}; "
                    f"{record.get('sector')}; {record.get('kind')}; excluding subject"
                ),
                peer_tickers=[t for t, _ in valid],
                owner="eodhd_screen_service",
            )
            merge(record, "sector_pe", value)


def current_capitalization(
    prices,
    fundamentals,
    record,
    receipt,
    *,
    fundamentals_receipt=None,
    splits=None,
    split_receipt=None,
    fx=None,
    fx_receipt=None,
):
    """Ordinary-share cap only with attributed class and unit applicability; never diluted WA."""
    basis = record.get("capitalization_basis", {})
    if not (
        basis.get("verified") is True
        and basis.get("source")
        and basis.get("field") == "SharesStats.SharesOutstanding"
        and basis.get("share_class") == "ordinary_common"
        and basis.get("price_applies_to_all_shares") is True
    ):
        return {}
    shares_date = screening.day(basis.get("shares_as_of"))
    if not fundamentals_receipt or shares_date is None:
        return {}
    general = fundamentals["General"]
    calendar = CALENDARS.get(str(general.get("Exchange", "")).upper())
    if not calendar:
        return {}
    sessions = completed_sessions(
        calendar, receipt.get("market_as_of", receipt["as_of"]), receipt["retrieved_at"]
    )
    if sessions.empty:
        return {}
    date = str(sessions[-1].date())
    if (
        not 0 <= (screening.day(date) - shares_date).days <= 45
        or splits is None
        or not split_covered(split_receipt, str(shares_date), date)
    ):
        return {}
    actions = split_events(splits, split_receipt)
    matches = [r for r in prices if r.get("date") == date]
    shares = numeric(fundamentals.get("SharesStats", {}).get("SharesOutstanding"))
    close = numeric(matches[0].get("close")) if len(matches) == 1 else None
    if shares is None or shares <= 0 or close is None or close <= 0:
        return {}
    shares *= split_factor(actions, str(shares_date), date)
    currency, rate = general.get("CurrencyCode"), 1
    if currency != "USD":
        if not fx_receipt or fx_receipt.get("symbol") != f"{currency}USD.FOREX":
            return {}
        matches = [r for r in (fx or []) if r.get("date") == date]
        rate = numeric(matches[0].get("close")) if len(matches) == 1 else None
        if rate is None or rate <= 0:
            return {}
    if not fundamentals_receipt:
        return {}
    out = item(
        close * shares * rate,
        receipt,
        "close; "
        + f"sources/{fundamentals_receipt['file']}#SharesStats.SharesOutstanding; "
        + basis["source"],
        currency="USD",
    )
    out["period"] = date
    out["shares_as_of"] = str(shares_date)
    out["split_source"] = f"sources/{split_receipt['file']}#date+split"
    out["oldest_source_observed_at"] = min(
        str(shares_date),
        stamp(fundamentals_receipt),
        stamp(receipt),
        stamp(split_receipt),
        stamp(fx_receipt) if fx_receipt else stamp(receipt),
    )
    if fx_receipt:
        out["fx_source"] = f"sources/{fx_receipt['file']}#{date}.close"
    return {"market_cap_usd": out}


def us_usd_quote_identity(record, fundamentals):
    general = (fundamentals or {}).get("General", {})
    code, suffix = record["ticker"].rsplit(".", 1)
    exchange = str(general.get("Exchange", "")).upper()
    cik, expected = str(general.get("CIK", "")), str(record.get("identifiers", {}).get("cik", ""))
    return (
        suffix == "US"
        and general.get("Code") == code
        and exchange in US_EXCHANGES
        and CALENDARS.get(exchange) == "XNYS"
        and general.get("CurrencyCode") == "USD"
        and general.get("Type") == "Common Stock"
        and cik.isdigit()
        and expected.isdigit()
        and int(cik) == int(expected)
    )


def matched_book_values(
    prices,
    annual,
    quarterly,
    record,
    annual_receipt,
    quarterly_receipt,
    price_receipt,
    splits,
    split_receipt,
    *,
    fundamentals=None,
    fundamentals_receipt=None,
):
    """Point-date book statements avoid the quarterly income duration ambiguity."""
    if not fundamentals_receipt or not us_usd_quote_identity(record, fundamentals):
        return {}
    basis = record.get("sec_accounting", {})
    if not (
        basis.get("book_equity_basis") == "common_equity_excluding_preferred"
        and basis.get("intangibles_exclude_goodwill") is True
    ):
        return {}
    if splits is None or not split_receipt:
        return {}
    annual_rows = sec_rows(annual, record, annual_receipt, form="10k", currency="USD")
    quarter_rows = (
        sec_rows(quarterly, record, quarterly_receipt, form="10q", currency="USD")
        if quarterly and quarterly_receipt
        else []
    )
    rows = sorted(annual_rows + quarter_rows, key=lambda r: r["period"])
    if len({r["period"] for r in rows}) != len(rows):
        return {}
    events = split_events(splits, split_receipt)
    sessions = completed_sessions(
        "XNYS",
        price_receipt.get("market_as_of", price_receipt["as_of"]),
        price_receipt["retrieved_at"],
    )
    if sessions.empty:
        return {}
    cutoff_month = pd.Period(price_receipt["as_of"], freq="M")
    month_ends = {}
    for date in sessions:
        month = date.to_period("M")
        if 1 <= cutoff_month.ordinal - month.ordinal <= 60:
            month_ends[month] = str(date.date())
    current_date = str(sessions[-1].date())
    allowed = set(month_ends.values()) | {current_date}
    monthly = {}
    for raw in sorted(prices, key=lambda r: r.get("date", "")):
        date, close = raw.get("date"), numeric(raw.get("close"))
        if date in allowed and close is not None and close > 0:
            monthly[date[:7]] = (date, close)
    history = []
    for date, close in monthly.values():
        eligible = [
            r
            for r in rows
            if r["available_at"] < date
            and (pd.Timestamp(date) - pd.Timestamp(r["period"])).days <= 180
        ]
        if not eligible:
            continue
        row = eligible[-1]
        equity, goodwill, intangibles, shares = (
            numeric(row.get(k))
            for k in ("stockholders_equity", "goodwill", "intangible_assets", "shares_outstanding")
        )
        if (
            any(v is None for v in (equity, goodwill, intangibles, shares))
            or shares <= 0
            or equity - goodwill - intangibles <= 0
        ):
            continue
        if not split_covered(
            split_receipt, row["period"], price_receipt.get("market_as_of", price_receipt["as_of"])
        ):
            continue
        denominator = (equity - goodwill - intangibles) / (
            shares * split_factor(events, row["period"], date)
        )
        history.append(
            dict(
                period=date[:7] + "-01",
                price_date=date,
                value=close / denominator,
                earnings_period=row["period"],
                earnings_available_at=row["available_at"],
                earnings_currency="USD",
                price_currency="USD",
                matched=True,
                source=row["source"]
                + f"; sources/{price_receipt['file']}#close"
                + f"; sources/{split_receipt['file']}#split"
                + f"; sources/{fundamentals_receipt['file']}#General",
                basis=(
                    "quarter-end common tangible book; actual close; "
                    "current retrieval, no PIT vintage claim"
                ),
            )
        )
    if not history:
        return {}
    mapped = {
        "own_ptbv_monthly": item(
            [row for row in history if row["price_date"] in month_ends.values()],
            price_receipt,
            "close / filing-dated tangible book per ordinary share",
        )
    }
    last_date = max((r.get("date", "") for r in prices if r.get("date") in allowed), default="")
    if history[-1]["price_date"] == last_date:
        mapped["ptbv"] = item(
            history[-1]["value"],
            price_receipt,
            history[-1]["source"],
            valuation_basis="latest filed quarter-end common tangible book",
        )
        mapped["ptbv"]["period"] = last_date
    return mapped


def duration_eps_facts(payload, record, receipt):
    """SEC duration-tagged entity facts, plus individually attributed issuer releases."""
    expected_cik = str(record.get("identifiers", {}).get("cik", ""))
    actual_cik = str(payload.get("cik", ""))
    if not expected_cik.isdigit() or not actual_cik.isdigit():
        raise ValueError("SEC companyconcept malformed CIK")
    if int(expected_cik) != int(actual_cik):
        raise SourceIdentityConflict(
            "SEC companyconcept issuer identity mismatch", record, {"CIK": actual_cik}, receipt
        )
    if payload.get("taxonomy") != "us-gaap" or payload.get("tag") != "EarningsPerShareDiluted":
        raise ValueError("SEC companyconcept taxonomy/tag mismatch")
    unit_map = payload.get("units")
    if not isinstance(unit_map, dict):
        raise ValueError("SEC companyconcept requires valid per-share units")
    for unit, facts in unit_map.items():
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("SEC companyconcept requires valid per-share units")
        if type(facts) is dict and not facts:
            continue
        if not isinstance(facts, list) or any(not isinstance(fact, dict) for fact in facts):
            raise ValueError("SEC companyconcept requires valid USD/shares facts")
    if "USD/shares" not in unit_map:
        return []
    units = unit_map["USD/shares"]
    if type(units) is dict and not units:
        units = []
    rows = []
    raw_facts = [(raw, False) for raw in units]
    raw_facts += [(raw, True) for raw in record.get("quarter_supplements", [])]
    for raw, supplement in raw_facts:
        start, end, filed = (screening.day(raw.get(k)) for k in ("start", "end", "filed"))
        value = numeric(raw.get("val"))
        if not start or not end or not filed or not 60 <= (end - start).days + 1 <= 120:
            continue
        if end > filed or str(filed) > receipt["as_of"] or value is None:
            continue
        if supplement:
            if (
                raw.get("verified") is not True
                or not raw.get("source")
                or raw.get("currency") != "USD"
                or raw.get("form") != "issuer_release"
                or raw.get("basis") != "reported_gaap_diluted"
            ):
                continue
            source = raw["source"] + "; frozen quarter_supplements"
        else:
            if raw.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                continue
            if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", str(raw.get("accn"))):
                continue
            source = f"sources/{receipt['file']}#units.USD/shares/{raw['accn']}/{start}/{end}"
        basis = screening.day(raw.get("share_basis_at"))
        if basis is not None and (
            basis > filed or basis < end or not raw.get("share_basis_source")
        ):
            continue
        rows.append(
            dict(
                start=str(start),
                end=str(end),
                filed=str(filed),
                value=value,
                source=source,
                accn=raw.get("accn"),
                form=raw.get("form"),
                share_basis_at=str(basis) if basis else None,
                share_basis_source=raw.get("share_basis_source"),
            )
        )
    return rows


def ttm_eps_at(facts, date, splits, split_receipt):
    """Latest eligible vintage per quarter; no missing-quarter or annual-minus-YTD inference."""
    by_end = {}
    for fact in facts:
        if fact["filed"] < date:
            by_end.setdefault(fact["end"], []).append(fact)
    quarters = []
    for end in sorted(by_end)[-4:]:
        versions = by_end[end]
        latest = max(r["filed"] for r in versions)
        choices = [r for r in versions if r["filed"] == latest]
        if len({(r["start"], r["value"], r["share_basis_at"]) for r in choices}) != 1:
            return None
        quarters.append(sorted(choices, key=lambda r: r["source"])[0])
    if len(quarters) != 4:
        return None
    if any(
        screening.day(b["start"]) != screening.day(a["end"]) + pd.Timedelta(days=1)
        for a, b in zip(quarters, quarters[1:])
    ):
        return None
    span = (screening.day(quarters[-1]["end"]) - screening.day(quarters[0]["start"])).days + 1
    age = (screening.day(date) - screening.day(quarters[-1]["end"])).days
    if not 330 <= span <= 390 or not 0 <= age <= 180:
        return None
    values = []
    for row in quarters:
        basis = row["share_basis_at"] or row["end"]
        if not split_covered(split_receipt, basis, date):
            return None
        if not row["share_basis_at"] and any(row["end"] < d <= row["filed"] for d in splits):
            return None
        values.append(row["value"] / split_factor(splits, basis, date))
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        return None
    return dict(
        value=total,
        quarters=quarters,
        period=quarters[-1]["end"],
        available_at=max(r["filed"] for r in quarters),
    )


def ttm_listing_basis(record, start=None, through=None):
    """Require sourced evidence that issuer diluted EPS applies to this exact listing."""
    proof = record.get("eps_listing_basis", {})
    source = proof.get("source")
    first, last = screening.day(proof.get("from")), screening.day(proof.get("through"))
    if not (
        proof.get("verified") is True
        and isinstance(source, str)
        and source.strip()
        and proof.get("ticker") == record.get("ticker")
        and proof.get("currency") == "USD"
        and proof.get("share_basis") == "diluted"
        and first
        and last
        and first <= last
    ):
        return None
    if start is not None and (not screening.day(start) or screening.day(start) < first):
        return None
    if through is not None and (not screening.day(through) or screening.day(through) > last):
        return None
    return source


def normalize_ttm_pe(
    prices,
    concept,
    record,
    concept_receipt,
    price_receipt,
    splits,
    split_receipt,
    *,
    fundamentals=None,
    fundamentals_receipt=None,
):
    """Four-quarter GAAP diluted EPS on verified US USD month ends and original closes."""
    if splits is None or not split_receipt or not fundamentals or not fundamentals_receipt:
        return {}
    if str(fundamentals.get("General", {}).get("HomeCategory", "")).upper() == "ETD":
        listing_identity(fundamentals, record)
    if not us_usd_quote_identity(record, fundamentals) or not ttm_listing_basis(record):
        return {}
    observed = max(stamp(price_receipt), stamp(concept_receipt))
    if not 0 <= (screening.day(observed) - screening.day(stamp(fundamentals_receipt))).days <= 45:
        return {}
    facts = duration_eps_facts(concept, record, concept_receipt)
    actions = split_events(splits, split_receipt)
    sessions = completed_sessions(
        "XNYS",
        price_receipt.get("market_as_of", price_receipt["as_of"]),
        price_receipt["retrieved_at"],
    )
    allowed = {str(d.date()) for d in sessions}
    price_map = {}
    for raw in prices:
        date, close = raw.get("date"), numeric(raw.get("close"))
        if date in price_map:
            raise ValueError("duplicate TTM valuation price date")
        if date in allowed and close is not None and close > 0:
            price_map[date] = close
    if sessions.empty:
        return {}
    cutoff_month = pd.Period(price_receipt["as_of"], freq="M")
    month_ends = {}
    for date in sessions:
        month = date.to_period("M")
        if 1 <= cutoff_month.ordinal - month.ordinal <= 60:
            month_ends[month] = str(date.date())
    current_date = str(sessions[-1].date())
    targets = set(month_ends.values()) | {current_date}
    history, current_row = [], None
    for date in sorted(targets):
        if date not in price_map:
            continue
        eps = ttm_eps_at(facts, date, actions, split_receipt)
        if eps is None:
            continue
        listing_source = ttm_listing_basis(record, eps["quarters"][0]["start"], date)
        if not listing_source:
            continue
        row = dict(
            period=date[:7] + "-01",
            price_date=date,
            value=price_map[date] / eps["value"],
            earnings_period=eps["period"],
            earnings_available_at=eps["available_at"],
            earnings_currency="USD",
            price_currency="USD",
            matched=True,
            source=[r["source"] for r in eps["quarters"]]
            + [
                f"sources/{price_receipt['file']}#close",
                f"sources/{split_receipt['file']}#split",
                f"sources/{fundamentals_receipt['file']}#General.Code+Exchange+CurrencyCode+CIK",
                listing_source,
            ],
            quarters=eps["quarters"],
            ttm_diluted_eps=eps["value"],
            basis=(
                "reported US GAAP diluted TTM EPS; dated vintages; "
                "no audited PIT reconstruction claim"
            ),
        )
        if date in month_ends.values():
            history.append(row)
        if date == current_date:
            current_row = row
    if not history and current_row is None:
        return {}
    mapped = {
        "own_pe_monthly": item(
            history,
            concept_receipt,
            "duration-tagged four-quarter EPS + completed-month closes + covered splits",
            valuation_basis="reported US GAAP diluted TTM P/E",
            calendar="XNYS",
            monthly_sampling="last_completed_session_of_completed_month",
        )
    }
    pe = item(
        current_row["value"] if current_row else None,
        concept_receipt,
        "duration-tagged EPS/current original close",
        valuation_basis="reported US GAAP diluted TTM P/E",
    )
    pe["source"] = current_row["source"] if current_row else []
    pe["verified"] = current_row is not None
    pe["period"] = current_date
    if current_row:
        pe["quarters"] = current_row["quarters"]
    observations = [
        stamp(concept_receipt),
        stamp(price_receipt),
        stamp(split_receipt),
        stamp(fundamentals_receipt),
    ]
    for evidence in (mapped["own_pe_monthly"], pe):
        evidence["observed_at"] = evidence["available_at"] = max(observations)
        evidence["oldest_source_observed_at"] = min(observations)
    mapped["pe"] = pe
    return mapped


def companyfacts_eps_concept(payload):
    """Expose the US-GAAP diluted EPS concept to the existing quarterly TTM adapter."""
    concept = payload.get("facts", {}).get("us-gaap", {}).get("EarningsPerShareDiluted")
    if not isinstance(concept, dict):
        return None
    return {
        **concept,
        "cik": payload.get("cik"),
        "taxonomy": "us-gaap",
        "tag": "EarningsPerShareDiluted",
    }


def derive_companyfacts_share_basis(
    payload, record, fundamentals, fundamentals_receipt, splits, split_receipt, receipt
):
    """Derive no-split basis only after independently proved EPS/listing class applicability."""
    if not isinstance(fundamentals_receipt, dict) or not isinstance(split_receipt, dict):
        return {}
    if (
        fundamentals_receipt.get("endpoint") != f"fundamentals/{record['ticker']}"
        or split_receipt.get("endpoint") != f"splits/{record['ticker']}"
    ):
        return {}
    assertion = record.get("eps_listing_basis", {})
    target = split_receipt.get("market_as_of", receipt.get("market_as_of", receipt["as_of"]))
    if not (screening.day(target) and target <= receipt["as_of"]):
        return {}
    if not (
        assertion.get("verified") is True
        and assertion.get("source")
        and assertion.get("ticker") == record["ticker"]
        and assertion.get("currency") == "USD"
        and assertion.get("share_basis") == "diluted"
        and screening.day(assertion.get("from"))
        and screening.day(assertion.get("through"))
        and assertion["through"] >= target
    ):
        return {}
    general = listing_identity(fundamentals, record)
    cik = str(record.get("identifiers", {}).get("cik", ""))
    actual = str(payload.get("cik", ""))
    observed = str(general.get("CIK", ""))
    if not (
        cik.isdigit()
        and actual.isdigit()
        and observed.isdigit()
        and int(cik) == int(actual) == int(observed)
    ):
        return {}
    if not (
        general.get("Type") == "Common Stock"
        and general.get("HomeCategory") == "Domestic"
        and general.get("PrimaryTicker") == record["ticker"]
        and general.get("CurrencyCode") == "USD"
    ):
        return {}
    if not isinstance(splits, list):
        return {}
    actions = split_events(splits, split_receipt)
    if any(date <= target for date in actions):
        return {}
    facts = (
        payload.get("facts", {})
        .get("us-gaap", {})
        .get("EarningsPerShareDiluted", {})
        .get("units", {})
        .get("USD/shares", [])
    )
    if not isinstance(facts, list):
        return {}
    periods = {}
    for fact in facts:
        if not isinstance(fact, dict):
            return {}
        start, end, filed = (screening.day(fact.get(k)) for k in ("start", "end", "filed"))
        value, accession = numeric(fact.get("val")), str(fact.get("accn", ""))
        if not (
            start
            and end
            and filed
            and value is not None
            and 330 <= (end - start).days + 1 <= 380
            and end <= filed
            and str(filed) <= receipt["as_of"]
            and str(filed) <= target
            and fact.get("form") in {"10-K", "10-K/A"}
            and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
        ):
            continue
        key = (accession, str(end))
        periods.setdefault(key, set()).add((str(start), value, str(filed)))
    if not periods or any(len(values) != 1 for values in periods.values()):
        return {}
    latest_periods = set(sorted({end for _, end in periods})[-4:])
    if len(latest_periods) != 4:
        return {}
    result = {}
    for (accession, end), values in periods.items():
        if end not in latest_periods:
            continue
        start, _, filed = next(iter(values))
        if assertion["from"] > start or not split_covered(split_receipt, end, target):
            return {}
        proof = dict(
            verified=True,
            ticker=record["ticker"],
            currency="USD",
            share_basis="diluted",
            share_basis_at=filed,
            source=(
                f"sources/{receipt['file']}#facts/us-gaap/"
                f"EarningsPerShareDiluted/USD/shares/{accession}; "
                f"sources/{fundamentals_receipt['file']}#General; "
                f"sources/{split_receipt['file']}#date+split; "
                f"{assertion['source']}"
            ),
        )
        if accession in result and result[accession]["share_basis_at"] != filed:
            return {}
        result[accession] = proof
    return result


def normalize_sec_companyfacts(payload, record, receipt, splits=None, split_receipt=None):
    """Annual reported diluted EPS only with issuer, unit and share-basis proof."""
    expected = str(record.get("identifiers", {}).get("cik", ""))
    actual = str(payload.get("cik", ""))
    if not expected.isdigit() or not actual.isdigit():
        raise ValueError("SEC companyfacts malformed CIK")
    if int(expected) != int(actual):
        raise SourceIdentityConflict(
            "SEC companyfacts issuer identity mismatch", record, {"CIK": actual}, receipt
        )
    if splits is None or not split_receipt:
        return {}
    target = split_receipt.get("market_as_of", receipt.get("market_as_of", receipt["as_of"]))
    if screening.day(target) is None or target > receipt["as_of"]:
        return {}
    events = split_events(splits, split_receipt)
    candidates, latest_filing, filing_counts = [], {}, {}
    for taxonomy, tag, framework in (
        ("us-gaap", "EarningsPerShareDiluted", "US_GAAP"),
        ("ifrs-full", "DilutedEarningsLossPerShare", "IFRS"),
    ):
        concept = payload.get("facts", {}).get(taxonomy, {}).get(tag, {})
        for unit, facts in concept.get("units", {}).items():
            if not re.fullmatch(r"[A-Z]{3}/shares", unit) or not isinstance(facts, list):
                continue
            currency = unit.split("/")[0]
            for fact in facts:
                start, end, filed = (screening.day(fact.get(k)) for k in ("start", "end", "filed"))
                accession = str(fact.get("accn", ""))
                value = numeric(fact.get("val"))
                if (
                    start
                    and end
                    and filed
                    and value is not None
                    and 330 <= (end - start).days + 1 <= 380
                    and end <= filed
                    and str(filed) <= receipt["as_of"]
                    and fact.get("form") in {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
                    and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                ):
                    latest_filing[str(end)] = max(latest_filing.get(str(end), ""), str(filed))
                    key = (str(end), str(filed))
                    filing_counts[key] = filing_counts.get(key, 0) + 1
                basis = record.get("companyfacts_share_basis", {}).get(accession, {})
                basis_date = screening.day(basis.get("share_basis_at"))
                if not (
                    start
                    and end
                    and filed
                    and value is not None
                    and 330 <= (end - start).days + 1 <= 380
                    and end <= filed
                    and str(filed) <= receipt["as_of"]
                    and fact.get("form") in {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
                    and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                    and basis.get("verified") is True
                    and basis.get("source")
                    and basis.get("currency") == currency
                    and basis.get("share_basis") == "diluted"
                    and basis.get("ticker") == record["ticker"]
                    and basis_date
                    and end <= basis_date <= filed
                    and str(basis_date) <= target
                    and split_covered(split_receipt, str(basis_date), target)
                    and not any(str(basis_date) < d <= str(filed) for d in events)
                ):
                    continue
                candidates.append(
                    dict(
                        period=str(end),
                        start=str(start),
                        available_at=str(filed),
                        value=value / split_factor(events, str(basis_date), target),
                        currency=currency,
                        accession_number=accession,
                        form=fact["form"],
                        source=f"sources/{receipt['file']}#facts/{taxonomy}/{tag}/{unit}/{accession}/{start}/{end}",
                        framework=framework,
                        basis_source=basis["source"],
                        share_basis_at=str(basis_date),
                    )
                )
    by_period = {}
    for row in candidates:
        by_period.setdefault(row["period"], []).append(row)
    rows = []
    for period in sorted(latest_filing)[-4:]:
        versions = by_period.get(period, [])
        latest = latest_filing[period]
        choices = [row for row in versions if row["available_at"] == latest]
        if (
            not choices
            or len(choices) != filing_counts[(period, latest)]
            or len(
                {
                    (r["start"], r["value"], r["currency"], r["framework"], r["share_basis_at"])
                    for r in choices
                }
            )
            != 1
        ):
            return {}
        rows.append(sorted(choices, key=lambda r: r["source"])[0])
    rows = rows[-4:]
    if len(rows) != 4 or len({(r["currency"], r["framework"]) for r in rows}) != 1:
        return {}
    if any(
        not 330 <= (screening.day(b["period"]) - screening.day(a["period"])).days <= 400
        for a, b in zip(rows, rows[1:])
    ):
        return {}
    currency, framework = rows[0]["currency"], rows[0]["framework"]
    evidence = item(
        rows,
        receipt,
        "facts annual diluted EPS",
        basis="reported_diluted",
        statement_source="sec_companyfacts",
        accounting_framework=framework,
        basis_evidence=dict(
            verified=True,
            source=[r["basis_source"] for r in rows],
            currency=currency,
            share_basis="diluted",
        ),
        split_source=f"sources/{split_receipt['file']}#date+split",
        share_basis_date=target,
    )
    assertion = record.get("eps_listing_basis", {})
    first, last = screening.day(assertion.get("from")), screening.day(assertion.get("through"))
    if (
        assertion.get("verified") is True
        and isinstance(assertion.get("source"), str)
        and assertion["source"].strip()
        and assertion.get("ticker") == record.get("ticker")
        and assertion.get("currency") == currency
        and assertion.get("share_basis") == "diluted"
        and first
        and last
        and first <= min(screening.day(r["start"]) for r in rows)
        and last >= screening.day(target)
    ):
        evidence["basis_evidence"].update(
            ticker=record["ticker"], **{"from": str(first), "through": str(last)}
        )
    evidence["period"] = [r["period"] for r in rows]
    return {"diluted_eps": evidence}
