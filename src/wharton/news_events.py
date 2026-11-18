"""Offline filing events for the news layer (stdlib only, no I/O except write_outputs).

SEC filings are CONFIRMED events; attached news articles are only LEADS (a co-occurrence
heuristic, never proof) and never reopen a review bin. All input text is untrusted data.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

from . import news_store
from .news_store import NewsPayloadError, atomic_write, canonical_json, sha256_hex

# ponytail: news_store.utc is being made public; fall back to the private name until then
_utc = news_store.utc

# Review types screening.py passes to human() (checked against its source by a test).
NEWS_BINS = (
    "thesis",
    "scenarios",
    "deal",
    "policy",
    "capital_allocation",
    "breakers",
    "leverage_exception",
    "fund",
)
CURRENT_FORMS = {"8-K", "8-K/A", "6-K", "6-K/A"}
VERSION_INDEX = "news-review-1"
VERSION_EVENTS = "news-events-1"

# Form 8-K item -> review bins it reopens (bins = screening.py human() review types).
# Chosen by severity of what the item discloses for the thesis. `fund` and `policy` receive
# nothing from any 8-K item: they are slow, analyst-owned judgements no single current report
# should reopen. An 8-K's bins are the union over its items.
_ALL4 = ("thesis", "scenarios", "breakers", "leverage_exception")
ITEM_BINS = {
    "4.02": _ALL4,  # non-reliance on financials: everything is suspect
    "1.03": _ALL4,  # bankruptcy / receivership
    "1.01": ("deal", "thesis"),  # material agreement entered
    "1.02": ("deal", "thesis"),  # material agreement terminated
    "2.01": ("deal", "capital_allocation", "thesis"),  # acquisition / disposition completed
    "5.01": ("thesis", "deal", "breakers"),  # change in control
    "5.06": ("thesis", "deal", "breakers"),  # change in shell company status
    "2.05": ("scenarios", "capital_allocation"),  # exit / disposal costs
    "2.06": ("scenarios", "capital_allocation"),  # material impairments
    "2.02": ("scenarios",),  # results of operations
    "2.03": ("leverage_exception", "capital_allocation"),  # new direct financial obligation
    "2.04": ("leverage_exception", "breakers"),  # obligation accelerated
    "3.01": ("breakers",),  # delisting notice
    "4.01": ("breakers",),  # auditor change
    "1.05": ("breakers",),  # cybersecurity incident
    "5.05": ("breakers",),  # code-of-ethics waiver
    "3.02": ("capital_allocation",),  # unregistered equity sales
    "3.03": ("capital_allocation",),  # modification of security holder rights
    "5.02": ("capital_allocation",),  # officer / director change
    # Queue-visible, reopen nothing:
    **{k: () for k in ("1.04", "5.03", "5.04", "5.07", "5.08", "7.01", "8.01", "9.01")},
    **{f"6.0{i}": () for i in range(1, 7)},
}
_UNKNOWN_ITEMS = ("thesis",)  # unknown or empty items on an 8-K are never dropped

_ACC = re.compile(r"\d{10}-\d{2}-\d{6}")
_DOC = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
_COLUMNS = (
    "accessionNumber filingDate reportDate acceptanceDateTime form items primaryDocument"
).split()
WINDOW_BEFORE, WINDOW_AFTER = timedelta(days=1), timedelta(days=3)


def _bad(msg):
    raise NewsPayloadError(f"submissions: {msg}")


def parse_submissions(raw: bytes, *, cik: str) -> list[dict]:
    """SEC submissions JSON -> current-report filings. Anything malformed raises (fail closed)."""
    try:
        data = json.loads(raw)
        recent = data["filings"]["recent"]
    except (ValueError, TypeError, KeyError):
        _bad("not a submissions document")
    if not isinstance(recent, dict) or not re.fullmatch(r"[0-9]+", str(cik)):
        _bad("bad structure")
    if str(data.get("cik", "")).lstrip("0") != str(cik).lstrip("0") or "cik" not in data:
        _bad("cik missing or mismatch")
    cols = [recent.get(c) for c in _COLUMNS]
    if not all(isinstance(c, list) for c in cols) or len({len(c) for c in cols}) != 1:
        _bad("missing or unequal-length columns")
    if not all(isinstance(v, str) for c in cols for v in c):
        _bad("non-string column entry")
    out, seen = [], set()
    for acc, filed, rep, acc_at, form, items, doc in zip(*cols):
        if form not in CURRENT_FORMS or acc in seen:
            continue  # duplicate accession rows: keep the first
        seen.add(acc)
        if not (isinstance(acc, str) and _ACC.fullmatch(acc) and isinstance(doc, str)):
            _bad("bad accession or document")
        if doc and not _DOC.fullmatch(doc):  # '' = paper-era row: kept, linked by its folder
            _bad("bad primary document")
        try:
            accepted = _utc(acc_at).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            accepted = None  # unparsable: never confirmable
        out.append(
            {
                "cik": str(cik).lstrip("0"),
                "accession": acc,
                "form": form,
                "filing_date": filed,
                "report_date": rep,
                "acceptance_at": accepted,
                "items": [i.strip() for i in str(items or "").split(",") if i.strip()],
                "primary_document": doc,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{acc.replace('-', '')}/{doc}",
                "amends": None,
            }
        )
    return out


def submissions_coverage(raw: bytes) -> dict:
    """recent_only is True when older filings exist in files[] that this parse did not read."""
    try:
        files = json.loads(raw)["filings"]["files"]
    except (ValueError, TypeError, KeyError):
        _bad("not a submissions document")
    return {"recent_only": bool(files)}


def submissions_identity(raw: bytes, ticker: str) -> dict:
    """Does SEC's own `tickers` list name our ticker code? Exact, case-insensitive, never raises.

    Any SEC document without a non-empty `tickers` array cannot confirm identity, so all of
    that issuer's filing events become unconfirmed. Entries are stripped; returned strings are
    cut to 16 characters (matching uses the full string).
    """
    try:
        listed = json.loads(raw).get("tickers")
    except (ValueError, TypeError, AttributeError, RecursionError):
        listed = None
    sec = (
        [u for t in listed if isinstance(t, str) and (u := t.strip().upper())]
        if isinstance(listed, list)
        else []
    )
    reason = (
        "no_sec_tickers"
        if not sec
        else "match"
        if ticker.rsplit(".", 1)[0].upper() in sec
        else "not_listed"
    )
    return {"match": reason == "match", "reason": reason, "sec_tickers": [t[:16] for t in sec[:5]]}


def _bins(filing) -> list[str]:
    if filing["form"].startswith("6-K"):
        return []  # foreign private issuer reports have no item codes: queue-visible only
    items = filing["items"] or [None]
    return sorted({b for i in items for b in ITEM_BINS.get(i, _UNKNOWN_ITEMS)})


def build_events(
    filings, attached_articles, *, issuer_id, ticker, as_of, identity_ok=True
) -> list[dict]:
    """identity_ok=False (SEC `tickers` lack our ticker): every filing event is unconfirmed.

    Any SEC document without a non-empty `tickers` array makes all of that issuer's filing
    events unconfirmed (identity cannot be confirmed); identity_ok defaults to True only for
    callers that have no SEC document.
    """
    cutoff = _utc(as_of)
    events = []
    for f in filings:
        if f["acceptance_at"] is not None and _utc(f["acceptance_at"]) > cutoff:
            continue  # lookahead guard
        # unknown time or wrong SEC entity: visible in queue, reopens nothing
        unconfirmed = f["acceptance_at"] is None or not identity_ok
        kind = "foreign_current_report" if f["form"].startswith("6-K") else "current_report"
        fields = {
            k: f[k] for k in ("accession", "form", "items", "acceptance_at", "primary_document")
        }
        events.append(
            {
                "event_id": sha256_hex(f"{issuer_id}|{kind}|{f['accession']}")[:32],
                "kind": kind,
                "status": "unconfirmed" if unconfirmed else "confirmed",
                "issuer_id": issuer_id,
                "ticker": ticker,
                "accession": f["accession"],
                "form": f["form"],
                "items": f["items"],
                "bins": [] if unconfirmed else _bins(f),
                "acceptance_at": f["acceptance_at"],
                "url": f["url"],
                "revision_digest": sha256_hex(canonical_json(fields)),
                "leads": [],
                "link_rule": "co_occurrence_window",
                **({} if identity_ok else {"identity": "sec_ticker_mismatch"}),
            }
        )
    events.sort(key=lambda e: (e["acceptance_at"] or "", e["accession"]))
    for a in sorted(attached_articles, key=lambda a: a["article_id"]):
        if not a.get("attach") or a.get("issuer_id") != issuer_id:
            continue
        seen = _utc(a["first_seen_at"])
        if seen > cutoff:
            continue
        near = [
            (abs(seen - _utc(e["acceptance_at"])), e["accession"], e)
            for e in events
            if e["status"] == "confirmed"
            and _utc(e["acceptance_at"]) - WINDOW_BEFORE
            <= seen
            <= _utc(e["acceptance_at"]) + WINDOW_AFTER
        ]
        if near:
            min(near, key=lambda t: t[:2])[2]["leads"].append(a["article_id"])
        else:
            events.append(
                {
                    "event_id": sha256_hex(f"{issuer_id}|lead|{a['article_id']}")[:32],
                    "kind": "news_lead",
                    "status": "lead",
                    "issuer_id": issuer_id,
                    "ticker": ticker,
                    "accession": None,
                    "form": None,
                    "items": [],
                    "bins": [],
                    "acceptance_at": None,
                    "url": a.get("canonical_url"),
                    "revision_digest": None,
                    "leads": [a["article_id"]],
                    "link_rule": None,
                }
            )
    return sorted(events, key=lambda e: e["event_id"])


def _valid_dispositions(dispositions, events, cutoff):
    """-> (valid {event_id: disposition}, sorted invalid event_ids).

    A disposition must name the event's CURRENT revision_digest and be dated <= as_of.
    """
    digest = {e["event_id"]: e["revision_digest"] for e in events}
    valid, invalid = {}, []
    for eid, d in (dispositions or {}).items():
        try:
            ok = (
                isinstance(d, dict)
                and d.get("status") in ("reviewed", "dismissed")
                and all(
                    isinstance(d.get(k), str) and d[k].strip()
                    for k in ("author", "reviewer", "rationale", "reviewed_at")
                )
                and d["reviewer"].strip() != d["author"].strip()
                and digest.get(eid) is not None
                and d.get("revision_digest") == digest[eid]
                and _utc(
                    d["reviewed_at"] if "T" in d["reviewed_at"] else d["reviewed_at"] + "T00:00:00Z"
                )
                <= cutoff
            )
        except ValueError:
            ok = False
        if ok:
            valid[eid] = d
        else:
            invalid.append(eid)
    return valid, sorted(invalid)


def review_index(events, dispositions, *, as_of) -> dict:
    """Per ticker/bin: latest acceptance DAY over confirmed, non-dismissed events <= as_of.

    Dismissal can only lower it. screening.py compares it with the review date.
    """
    cutoff = _utc(as_of)
    valid, invalid = _valid_dispositions(dispositions, events, cutoff)
    tickers: dict = {}
    for e in events:
        if e["status"] != "confirmed" or _utc(e["acceptance_at"]) > cutoff:
            continue
        if valid.get(e["event_id"], {}).get("status") == "dismissed":
            continue
        for b in e["bins"]:
            s = tickers.setdefault(e["ticker"].upper(), {"bins": {}})["bins"].setdefault(
                b,
                {
                    "latest_open_event_at": "",
                    "open_event_days": [],
                    "event_ids": [],
                    "open_count": 0,
                },
            )
            s["open_event_days"].append(e["acceptance_at"][:10])
            s["event_ids"].append(e["event_id"])
            s["open_count"] += 1
    for t in tickers.values():
        for s in t["bins"].values():
            s["event_ids"].sort()
            s["open_event_days"] = sorted(set(s["open_event_days"]))
            s["latest_open_event_at"] = s["open_event_days"][-1]
    return {
        "version": VERSION_INDEX,
        "as_of_ts": as_of,
        "tickers": tickers,
        "invalid_dispositions": invalid,
    }


def _filings_cov(c) -> str:
    if isinstance(c, dict) and c.get("none") is True:
        return "none"  # SEC has no submissions for this CIK
    if not isinstance(c, dict) or not isinstance(c.get("recent_only"), bool):
        return "unknown"
    return "recent_only" if c["recent_only"] else "complete"


def review_queue(
    events, coverage_by_ticker, watchlist_rows, dispositions, *, as_of, filings_coverage=None
) -> list[dict]:
    """Human worklist of open items per ticker. Informational: no gate, rank or score."""
    cutoff = _utc(as_of)
    valid, _ = _valid_dispositions(dispositions, events, cutoff)
    tier = {r["ticker"]: r.get("tier") for r in watchlist_rows}
    rows: dict = {}
    for e in events:
        if e["event_id"] in valid:
            continue  # reviewed or dismissed
        if e["status"] == "confirmed" and _utc(e["acceptance_at"]) > cutoff:
            continue
        r = rows.setdefault(
            e["ticker"],
            {
                "ticker": e["ticker"],
                "tier": tier.get(e["ticker"]),
                "coverage_state": (coverage_by_ticker or {}).get(e["ticker"]),
                "filings_coverage": _filings_cov((filings_coverage or {}).get(e["ticker"])),
                "open_confirmed": [],
                "unconfirmed": [],
                "leads": [],
                "bins_reopened": set(),
                "note": "Filings are confirmed; leads are co-occurrence heuristics, not proof.",
            },
        )
        if e["status"] == "confirmed":
            r["open_confirmed"].append(e["event_id"])
            r["bins_reopened"].update(e["bins"])
        if e["status"] == "unconfirmed":
            r["unconfirmed"].append(e["event_id"])
        r["leads"].extend(e["leads"])
    out = []
    for t in sorted(rows):
        r = rows[t]
        r["bins_reopened"] = sorted(r["bins_reopened"])
        r["open_confirmed"].sort()
        r["unconfirmed"].sort()
        r["leads"].sort()
        out.append(r)
    return out


def write_outputs(root, *, as_of, events, index, queue) -> None:
    d = Path(root) / "news" / "events"
    doc = {
        "events.json": {"version": VERSION_EVENTS, "as_of_ts": as_of, "events": events},
        "review-queue.json": {"version": VERSION_EVENTS, "as_of_ts": as_of, "rows": queue},
        "review-index.json": index,  # last: consumers gate on it, so it must never lead the rest
    }
    for name, obj in doc.items():
        atomic_write(d / name, canonical_json(obj))


_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_review_state(root, as_of):
    """review-index.json -> None (absent) | {status ok|unavailable|stale, tickers {T: {bin: day}}}.

    Per ticker and bin: the latest open event day <= as_of's UTC day (bins with none omitted).
    """
    path = Path(root) / "news" / "events" / "review-index.json"
    unavailable = {"status": "unavailable", "tickers": {}}
    if not path.exists():
        return None
    # run() passes a bare YYYY-MM-DD; a bad as_of is a caller bug, so it raises here.
    if isinstance(as_of, str) and _DAY.fullmatch(as_of):
        want = date.fromisoformat(as_of)
    else:
        want = _utc(as_of).date()
    try:
        idx = json.loads(path.read_bytes())
        if idx["version"] != VERSION_INDEX:
            return unavailable
        built = _utc(idx["as_of_ts"]).date()
        tickers = {}
        for t, v in idx["tickers"].items():
            for b, s in v["bins"].items():
                days = s["open_event_days"]
                if b not in NEWS_BINS or not all(
                    isinstance(d, str) and _DAY.fullmatch(d) and date.fromisoformat(d) for d in days
                ):
                    return unavailable
                ok = [d for d in days if d <= want.isoformat()]
                if ok:
                    tickers.setdefault(t.upper(), {})[b] = max(ok)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return unavailable
    return {"status": "stale" if built < want else "ok", "tickers": tickers}
