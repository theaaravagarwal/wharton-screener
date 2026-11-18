"""News entity resolution, watchlist and coverage states. Pure stdlib, no network.

Provider `symbols` are CLAIMS. An article attaches to a listing only on T1/T2 (exact CODE.EXCH
registry symbol with a proven issuer). Nothing is ever propagated to another listing of the same
issuer (ADR vs ordinary share, GOOG vs GOOGL, .US vs .LSE): each listing needs its own exact claim.
"""

import csv
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wharton import news_store

NOT_QUERIED = "not_queried"
COMPLETE_NONE = "queried_complete_none"
COMPLETE_ITEMS = "queried_complete_items"
PARTIAL = "partial"
UNAVAILABLE = "unavailable"
BUDGET_DEFERRED = "budget_deferred"
STALE = "stale"
IDENTITY_HOLD = "identity_hold"
STATES = (
    NOT_QUERIED,
    COMPLETE_NONE,
    COMPLETE_ITEMS,
    PARTIAL,
    UNAVAILABLE,
    BUDGET_DEFERRED,
    STALE,
    IDENTITY_HOLD,
)
TIERS = ("T1", "T2", "T3", "T4")
# The only thing COMPLETE_NONE may assert. Never "safe", never "no event", never a gate pass.
NO_NEWS_CLAIM = "no items from source under filter in window at time"

# Registry semantics (scripts/build_screen_universe.py): issuer_id is "cik:"/"isin:" (proven),
# "primary:" or "symbol:" (unproven); issuer_proof is "proven"/"unproven" derived from the prefix.
# duplicate_flag is none | not_applicable | proven | unverified; only none/proven clear a listing.
PROVEN = "proven"
DUPLICATE_CLEAR = {"none", "proven"}
COMMON = "common stock"
# An article tagging more symbols than this is a roundup: a ticker among many is weak evidence.
ROUNDUP_MAX_SYMBOLS = 10
REGISTRY_COLUMNS = (
    "ticker name exchange listing_exchange currency type isin issuer_id "
    "issuer_proof dup_group duplicate_flag"
).split()


def load_registry(csv_path):
    """ticker (upper) -> row dict with the registry CSV columns.

    Raises ValueError if any required column is missing (never a silent all-hold registry).
    """
    registry = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REGISTRY_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("registry missing required columns: " + ", ".join(missing))
        for raw in reader:
            row = {c: (raw.get(c) or "").strip() for c in REGISTRY_COLUMNS}
            if row["ticker"]:
                row["ticker"] = row["ticker"].upper()
                registry[row["ticker"]] = row
    return registry


def _norm_cik(value):
    return str(value).lower().removeprefix("cik:").strip().lstrip("0")


def _identifier_check(ids, row):
    """-> (matched:bool, conflict:str|None) comparing caller-supplied article identifiers."""
    matched = False
    if not isinstance(ids, dict):
        return False, None
    isin = str(ids.get("isin") or "").strip().upper()
    if isin and row["isin"]:
        if isin != row["isin"].upper():
            return False, "isin_conflict"
        matched = True
    cik = _norm_cik(ids.get("cik") or "")
    iid = row["issuer_id"].lower()
    if cik and iid.startswith("cik:"):
        if cik != _norm_cik(iid):
            return False, "cik_conflict"
        matched = True
    return matched, None


def _exact(claim, row, article):
    base = dict(ticker=row["ticker"], issuer_id=row["issuer_id"] or None)
    # Order matters: ETFs/preferreds carry duplicate_flag=not_applicable and an unproven
    # symbol: issuer; their hold reason is the listing/issuer, never a duplicate verdict.
    if row["type"].lower() != COMMON:
        return dict(base, tier="T4", attach=False, reason="hold:non_common_listing")
    if row["issuer_proof"].lower() != PROVEN:
        return dict(base, tier="T4", attach=False, reason="hold:unproven_issuer")
    if row["duplicate_flag"].lower() not in DUPLICATE_CLEAR:
        return dict(base, tier="T4", attach=False, reason="hold:duplicate_flag_unverified")
    matched, conflict = _identifier_check(article.get("identifiers"), row)
    if conflict:
        return dict(base, tier="T4", attach=False, reason="hold:" + conflict)
    if matched:
        return dict(base, tier="T1", attach=True, reason="exact_symbol_and_identifier")
    return dict(base, tier="T2", attach=True, reason="exact_symbol_proven_issuer")


def attachable_tickers(registry):
    """Registry tickers an exact-symbol claim can attach to (same rules as `_exact`)."""
    return frozenset(t for t, row in (registry or {}).items() if _exact(t, row, {})["attach"])


NAME_SUFFIXES = {
    "inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "sa", "ag", "nv",
    "se", "holdings", "holding", "group",
}  # fmt: skip


def _word_in(needle, text):
    return re.search(r"(?<!\w)" + re.escape(needle.casefold()) + r"(?!\w)", text) is not None


def _title_names(row, title):
    """True if the title names this listing: bare code (>=3 chars) or name stem (>=4 chars)."""
    text = title.casefold()
    code = row["ticker"].rsplit(".", 1)[0]
    words = re.sub(r"[,()]", " ", row["name"]).split()
    while words:
        last = words[-1].strip(".").casefold()
        if last in NAME_SUFFIXES:
            words.pop()
        elif len(words) > 1 and len(last) == 1 and words[-2].casefold() == "class":
            del words[-2:]
        else:
            break
    stem = " ".join(words).strip(" .,&-")
    return (len(code) >= 3 and _word_in(code, text)) or (len(stem) >= 4 and _word_in(stem, text))


def resolve_article(article, registry, *, aliases=None):
    """-> list of dict(ticker, issuer_id, tier, attach, reason), one per ticker, sorted by ticker.

    article: dict with `symbols` (provider claims), optional `title`/`excerpt`, optional
    `identifiers` {isin, cik, figi} supplied independently of the provider symbols.
    aliases: optional {alias text: registry ticker}, supplied by the caller from filings/registry.
    Never generated here, and an alias match is T3 and NEVER attaches.
    Roundup (> ROUNDUP_MAX_SYMBOLS distinct claims): an otherwise attaching exact claim keeps
    its tier only if the stored (sanitized) title names it as a whole word (bare code >= 3
    chars, or registry name minus corporate suffixes >= 4 chars); else hold:roundup_many_symbols.
    """
    out = {}

    def put(entry):
        old = out.get(entry["ticker"])
        if old is None or TIERS.index(entry["tier"]) < TIERS.index(old["tier"]):
            out[entry["ticker"]] = entry

    by_code = {}
    for t, row in registry.items():
        by_code.setdefault(t.rsplit(".", 1)[0], []).append(row)
    claims = sorted({str(c).strip().upper() for c in article.get("symbols") or []} - {""})
    roundup = len(claims) > ROUNDUP_MAX_SYMBOLS
    for claim in claims:
        if "." in claim:
            row = registry.get(claim)
            if row is None:
                put(
                    dict(
                        ticker=claim,
                        issuer_id=None,
                        tier="T4",
                        attach=False,
                        reason="hold:symbol_not_in_registry",
                    )
                )
            else:
                put(_exact(claim, row, article))
            continue
        rows = by_code.get(claim, [])
        reason = (
            "hold:bare_code_multiple_exchanges"
            if len(rows) > 1
            else ("hold:bare_code_no_exchange" if rows else "hold:bare_code_unknown")
        )
        for row in rows or [dict(ticker=claim, issuer_id="")]:
            put(
                dict(
                    ticker=row["ticker"],
                    issuer_id=row["issuer_id"] or None,
                    tier="T4",
                    attach=False,
                    reason=reason,
                )
            )
    if roundup:
        title = str(article.get("title") or "")
        for e in out.values():
            if e["attach"] and _title_names(registry[e["ticker"]], title):
                e["reason"] = "exact_symbol_roundup_title_match"
            elif e["attach"]:  # identity holds keep their more specific reason
                e.update(tier="T4", attach=False, reason="hold:roundup_many_symbols")
    text = " ".join(str(article.get(k) or "") for k in ("title", "excerpt")).casefold()
    for alias, ticker in sorted((aliases or {}).items()):
        row = registry.get(str(ticker).upper())
        if not (
            row
            and alias.strip()
            and re.search(r"(?<!\w)" + re.escape(alias.strip().casefold()) + r"(?!\w)", text)
        ):
            continue
        old = out.get(row["ticker"])
        if old is None:
            out[row["ticker"]] = dict(
                ticker=row["ticker"],
                issuer_id=row["issuer_id"] or None,
                tier="T3",
                attach=False,
                reason="hold:name_alias_only",
            )
        elif not old["attach"] and "+name_alias_hit" not in old["reason"]:
            old["reason"] += "+name_alias_hit"  # keep the existing hold (e.g. isin_conflict)
    return [out[t] for t in sorted(out)]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _up(ticker):
    return str(ticker).strip().upper()


SKIP_OUTCOMES = {"price_excluded", "price_unprovable", "identity_held"}


def watchlist(input_path, *, results_path=None, outcomes_path=None):
    """-> list of dict(ticker, tier A|B|C, reason), sorted by ticker, upper-cased. Read-only.

    input_path: prefer the run's frozen output input.json (it carries the identity_quarantine
    annotations screening added during the run); the configured input works but lacks them.
    Input shape (screen_service run()): {universe:[{ticker, identity_quarantine?, issuer_proof?}],
    portfolio:{holdings:[ticker], instruments:[{ticker, identity_quarantine?}]}, active:[ticker]}.
    outcomes_path: optional acquisition-outcomes.json ({"outcomes": {ticker: outcome}} or a bare
    {ticker: outcome} map). Precedence:
    C "identity_quarantine" (input or results; wins even over a holding) >
    A (holdings/instruments, `active` theses; with results_path also results.json finalists,
    stage portfolio/pretrade/review_ready) >
    C "screening_outcome:<outcome>" for price_excluded/price_unprovable/identity_held (names
    screening will not evaluate, so news does not spend on them) >
    C "unproven_issuer" (non-holding universe row that EXPOSES issuer_proof != "proven";
    screening input rows normally carry only issuer_id, so without that field the row stays B) >
    B "universe_listing".
    """
    inputs = _read(input_path)
    portfolio = inputs.get("portfolio") or {}
    instruments = portfolio.get("instruments") or []
    tier_a = {_up(t): "holding" for t in portfolio.get("holdings") or []}
    for r in instruments:
        tier_a.setdefault(_up(r["ticker"]), "holding")
    for t in inputs.get("active") or []:
        tier_a.setdefault(_up(t), "active_thesis")
    quarantined = {
        _up(r["ticker"])
        for r in (inputs.get("universe") or []) + instruments
        if r.get("identity_quarantine")
    }
    outcomes = _read(outcomes_path) if outcomes_path is not None else {}
    if isinstance(outcomes.get("outcomes"), dict):
        outcomes = outcomes["outcomes"]
    skipped = {_up(t): o for t, o in outcomes.items() if o in SKIP_OUTCOMES}
    if results_path is not None:
        for r in _read(results_path):
            if r.get("identity_quarantine"):
                quarantined.add(_up(r["ticker"]))
            elif r.get("stage") in {"portfolio", "pretrade", "review_ready"}:
                tier_a.setdefault(_up(r["ticker"]), "finalist:" + r["stage"])
    rows = {}
    for r in inputs.get("universe") or []:
        proof = r.get("issuer_proof")
        unproven = proof is not None and str(proof).lower() != PROVEN
        rows[_up(r["ticker"])] = ("C", "unproven_issuer") if unproven else ("B", "universe_listing")
    rows.update({t: ("C", "screening_outcome:" + o) for t, o in skipped.items() if t in rows})
    rows.update({t: ("A", why) for t, why in tier_a.items()})
    rows.update({t: ("C", "identity_quarantine") for t in quarantined})
    return [dict(ticker=t, tier=tier, reason=why) for t, (tier, why) in sorted(rows.items())]


_store_utc = news_store.utc


def _utc(value):
    """Strict tz-aware UTC datetime (news_store semantics: naive or garbage -> ValueError)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return value.astimezone(timezone.utc)
    return _store_utc(value)


def _usable(receipt, window, when):
    """Parsed receipt dict, or None if malformed, naive/future-dated (lookahead) or, when a
    window is given, its params lack either from/to bound or do not cover `window`."""
    try:
        at = _utc(receipt["retrieved_at"])
        if at > when:
            return None
        if window:
            p = receipt["params"]
            if not (
                str(p["from"])[:10] <= str(window[0])[:10]
                and str(p["to"])[:10] >= str(window[1])[:10]
            ):
                return None
        outcome = receipt["outcome"]
        n = receipt.get("n_items", 0 if outcome == "empty" else None)
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 0):
            return None
        limit = receipt.get("limit")
        return dict(
            at=at,
            outcome=outcome,
            n=n,
            limit=limit if isinstance(limit, int) and not isinstance(limit, bool) else None,
            page=int(receipt.get("page") or 0),
            exhausted=receipt.get("exhausted") is True,
            truncated=bool(receipt.get("truncated")),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def coverage_state(receipts, window, now, *, fresh_hours=36, identity_held=False):
    """Coverage state string for one listing over `window` (from_date, to_date) at `now`.

    Receipt keys used: outcome ("ok"|"empty"|"error"|"budget_hold"), retrieved_at, page
    (default 0), limit, and n_items:int (RAW provider items on that page, before normalization),
    exhausted:bool, truncated:bool. Unusable receipts are ignored, never raised on: malformed,
    retrieved_at missing/invalid/naive or later than `now` (lookahead), or (when `window` is
    given) params missing either from or to, or not covering `window`. Anything not proven
    complete is never COMPLETE_*: errors/timeouts/auth/budget give
    UNAVAILABLE/BUDGET_DEFERRED/PARTIAL.
    Exhaustion proof (a missing `exhausted` key means False): an ok page with n_items == 0, or
    a page with n_items < limit AND an explicit exhausted=True. A short page alone proves
    nothing; n_items >= limit (100-of-100) is never exhausted whatever the flag says, and any
    truncated receipt gives PARTIAL. A scan is complete only if a final exhausted page exists
    and every earlier page (0..page-1) has an ok receipt. ponytail: pages are matched by number
    only, not by scan run; a mixed old/new scan could be over-credited. Key by scan id if so.
    """
    if identity_held:
        return IDENTITY_HOLD
    when = _utc(now)  # ValueError on a naive/invalid `now`: a caller bug, not a receipt issue
    rs = [u for u in (_usable(r, window, when) for r in receipts) if u]
    if not rs:
        return NOT_QUERIED
    ok = [r for r in rs if r["outcome"] in ("ok", "empty")]
    if not ok:
        return BUDGET_DEFERRED if any(r["outcome"] == "budget_hold" for r in rs) else UNAVAILABLE
    if any(r["truncated"] for r in ok):
        return PARTIAL

    def exhausted(r):
        if r["n"] is None:
            return False
        if r["n"] == 0:
            return True
        return r["exhausted"] and r["limit"] is not None and r["n"] < r["limit"]

    pages = {r["page"] for r in ok}
    final = [r for r in ok if exhausted(r) and all(p in pages for p in range(r["page"]))]
    if not final:
        return PARTIAL
    if max(r["at"] for r in final) < when - timedelta(hours=fresh_hours):
        return STALE
    return COMPLETE_ITEMS if any((r["n"] or 0) > 0 for r in ok) else COMPLETE_NONE


def coverage_report(rows):
    """rows: iterable of (ticker, tier, exchange, state) -> counts plus the allowed claim text."""
    rows = list(rows)
    bad = [r for r in rows if r[3] not in STATES]
    if bad:
        raise ValueError("unknown coverage state: " + repr(bad[0][3]))

    def nest(i):
        out = {}
        for r in rows:
            out.setdefault(r[i], Counter())[r[3]] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    by_state = Counter(r[3] for r in rows)
    return dict(
        total=len(rows),
        by_state={s: by_state.get(s, 0) for s in STATES},
        by_tier_state=nest(1),
        by_exchange_state=nest(2),
        claims={COMPLETE_NONE: NO_NEWS_CLAIM},
    )
