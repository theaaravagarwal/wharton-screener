"""Frozen interfaces and central cost rules for the coverage lanes (bulk, events, queue). v2.0

Plain JSON-compatible dicts and stdlib only. This module never performs I/O and never calls the
provider; the guarded transport owns every paid attempt. Costs here are the single source of
truth that the budget guard validates a request spec against before any paid attempt, and the
guard binds the FINAL http params to the reserved cost (fingerprint/check_bound) on every send,
retries included.

v2.0 changes: job identity gains an `op` namespace (kind+op+scope+window) and an optional
`revision` that reopens a finished job (reopen); route rules follow the provider docs
(qualified fundamentals symbols, version 1.2, no invented EOD pagination, flat-100 splits and
dividends); observations carry row provenance verifiable against the retained raw bytes.
"""
# ruff: noqa: E501 - dense contract docstrings/comments; wrapping them hurts readability

import hashlib
import json
import re
from datetime import date

CONTRACT_VERSION = "2.6"
JOB_KINDS = ("market", "fundamentals", "calendar", "evidence")
JOB_OPS = {
    "market": ("eod", "splits", "dividends"),
    "fundamentals": ("bulk", "single"),
    "calendar": ("earnings", "splits", "dividends"),
    "evidence": ("completion",),
}
JOB_STATUSES = ("queued", "claimed", "completed", "hold", "failed")
LANES = ("bulk_market", "bulk_fundamentals", "calendar", "queue")

BULK_PAGE_MAX = 500  # symbols per request or rows per fundamentals page
BULK_PAGE_CREDITS = 100  # no-symbol request; with N symbols (eod/fundamentals) it costs 100 + N
BULK_SNAPSHOT_V2_CREDITS = 5000  # whole-exchange v2 snapshot: documented, DISABLED here
CALENDAR_CREDITS = 1  # per request; symbol-filtered multi-ticker pricing is unverified (pilot)
PILOT_CAP_CREDITS = 500  # additional charged credits for the single-exchange pilot, inside quota

BULK_ROUTES = ("bulk-fundamentals", "eod-bulk-last-day")
CALENDAR_ROUTES = ("calendar/earnings", "calendar/splits", "calendar/dividends")
BULK_TYPES = ("eod", "splits", "dividends")
FUNDAMENTALS_VERSION = "1.2"
_V2_VERSIONS = {"2", "2.0", "v2"}
_EXCHANGE = re.compile(r"[A-Z0-9]{1,8}")
# exact source code, plain ("AAPL") or exchange-qualified ("AAPL.US", "BRK-B.US")
_SYMBOL = re.compile(r"[A-Za-z0-9_-]{1,20}(?:\.[A-Za-z0-9_-]{1,10})?")
_FUNDAMENTALS_PARAMS = {"offset", "limit", "symbols", "version", "fmt"}
_EOD_PARAMS = {"date", "type", "symbols", "fmt"}  # the route has NO pagination
_CALENDAR_PARAMS = {"from", "to", "symbols", "fmt"}
# the upcoming-dividends route also takes JSON:API-style filters and paging (unverified cost: pilot)
_DIVIDEND_FILTERS = {"filter[date_eq]", "filter[date_from]", "filter[date_to]", "filter[symbol]"}
_DIVIDEND_PAGING = {"page[limit]", "page[offset]"}
DIVIDEND_PAGE_MAX = 1000
# Credential / holding detection works on whole key tokens, not substrings, so ordinary keys such
# as cap_key, sort_key, event_key or tokens_est are fine while api_token / api_key / secret_* are not.
_FORBIDDEN_TOKENS = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "apikey",
        "authorization",
        "bearer",
        "holding",
        "holdings",
        "payload",
    }
)
_FORBIDDEN_PAIRS = ("api_key", "api-key", "access_key", "secret_key", "private_key", "auth_token")
_CREDENTIAL_TEXT = re.compile(
    r"(?i)(?:api[_-]?token|api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|bearer)\s*[=:]\s*\S+"
    r"|[?&](?:api_token|apikey|token)="
)


class DisabledRoute(ValueError):
    """The route exists in the provider docs but is intentionally unavailable here."""


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def _day(value, name):
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise ValueError(f"{name} must be an ISO date") from None


def credential_key(key):
    """True for keys that name a credential or a holding payload (whole-token match)."""
    low = str(key).lower()
    tokens = set(re.split(r"[^a-z0-9]+", low))
    return bool(tokens & _FORBIDDEN_TOKENS) or any(pair in low for pair in _FORBIDDEN_PAIRS)


def contains_credential(text):
    """True when free text looks like it carries a credential (api_token=..., Bearer ..., ?token=)."""
    return isinstance(text, str) and bool(_CREDENTIAL_TEXT.search(text))


def _forbid_credentials(mapping, what):
    if any(credential_key(k) for k in mapping):
        raise ValueError(f"{what} must not carry credentials or holding payloads")


# ---- jobs -------------------------------------------------------------------------------


def job_id(kind, op, scope, window):
    """sha256 over canonical kind + op + scope + window. Revision is NOT part of identity: a
    revised event is the same job, reopened (see reopen)."""
    return _sha({"kind": kind, "op": op, "scope": scope, "window": window})


def make_job(kind, op, scope, window, reason, estimated_credits, status="queued", revision=None):
    """Job: job_id, kind, op (namespace within kind), scope (exact ticker or exchange),
    window{from,to}, reason, estimated_credits, status, optional revision. No API key and no
    holding payload may appear in a job."""
    job = {
        "kind": kind,
        "op": op,
        "scope": scope,
        "window": {
            "from": _day(window["from"], "window.from"),
            "to": _day(window["to"], "window.to"),
        },
        "reason": reason,
        "estimated_credits": estimated_credits,
        "status": status,
    }
    if revision is not None:
        job["revision"] = revision
    job["job_id"] = job_id(kind, op, scope, job["window"])
    return validate_job(job)


def validate_job(job):
    if not isinstance(job, dict):
        raise ValueError("job must be a dict")
    required = {"job_id", "kind", "op", "scope", "window", "reason", "estimated_credits", "status"}
    optional = {
        "revision",
        "pending_revision",
        "attempts",
        "updated_at",
        "detail",
        "hold_reason",
        "meta",
    }
    extra = set(job) - required - optional
    if required - set(job) or extra:
        raise ValueError(
            "job fields: missing %s extra %s" % (sorted(required - set(job)), sorted(extra))
        )
    _forbid_credentials(job, "job")
    if job["kind"] not in JOB_KINDS or job["status"] not in JOB_STATUSES:
        raise ValueError("job kind/status outside the frozen vocabulary")
    for field in ("scope", "reason", "hold_reason", "detail"):
        if contains_credential(job.get(field)) or (
            isinstance(job.get(field), dict) and contains_credential(canonical(job[field]))
        ):
            raise ValueError("job %s must not carry credential text" % field)
    if job["op"] not in JOB_OPS[job["kind"]]:
        raise ValueError("op %r is not valid for kind %r" % (job["op"], job["kind"]))
    if not isinstance(job["scope"], str) or not job["scope"] or not isinstance(job["reason"], str):
        raise ValueError("job scope/reason must be non-empty strings")
    credits = job["estimated_credits"]
    if isinstance(credits, bool) or not isinstance(credits, int) or credits < 0:
        raise ValueError("estimated_credits must be a nonnegative integer")
    if "revision" in job and not (isinstance(job["revision"], str) and job["revision"]):
        raise ValueError("job revision must be a non-empty string")
    if "meta" in job and not isinstance(job["meta"], dict):
        raise ValueError("job meta must be a dict")
    if "pending_revision" in job and not (
        job["status"] == "claimed"
        and isinstance(job["pending_revision"], str)
        and job["pending_revision"]
    ):
        raise ValueError("pending_revision is a non-empty string and only valid on a claimed job")
    window = job["window"]
    start, end = _day(window.get("from"), "window.from"), _day(window.get("to"), "window.to")
    if start > end:
        raise ValueError("window.from must not be after window.to")
    if job["job_id"] != job_id(job["kind"], job["op"], job["scope"], {"from": start, "to": end}):
        raise ValueError("job_id does not match kind+op+scope+window")
    return job


def reopen(job, revision):
    """Revision arrival: NEVER drops a revision. Returns a new dict (job is not mutated).

    queued    -> the queued job takes the new revision immediately (one pending unit of work).
    claimed   -> the running work is not disturbed; the arrival is stored durably as
                 `pending_revision` (latest arrival wins; arrival equal to the claimed revision
                 clears it) and ack() requeues it when the claim is acknowledged.
    finished  -> (completed/hold/failed) requeued with the new revision and attempts reset.
    The same revision as the job's current one changes nothing."""
    validate_job(job)
    if not (isinstance(revision, str) and revision):
        raise ValueError("revision must be a non-empty string")
    current = job.get("revision")
    if job["status"] == "claimed":
        updated = dict(job)
        if revision == current:
            updated.pop("pending_revision", None)
        else:
            updated["pending_revision"] = revision
        return validate_job(updated)
    if revision == current:
        return dict(job)
    if job["status"] == "queued":
        return validate_job({**job, "revision": revision})
    reopened = {k: v for k, v in job.items() if k not in ("hold_reason", "detail")}
    reopened.update(status="queued", revision=revision, attempts=0)
    return validate_job(reopened)


def ack(job, status, hold_reason=None):
    """Acknowledge a claimed job as completed/hold/failed. If a revision arrived while it was
    claimed (`pending_revision`) the job is requeued with that revision instead of finishing, so
    a seen revision always has durable work behind it. Returns a new dict."""
    validate_job(job)
    if job["status"] != "claimed" or status not in ("completed", "hold", "failed"):
        raise ValueError("ack needs a claimed job and a completed/hold/failed status")
    pending = job.get("pending_revision")
    if pending is not None and pending != job.get("revision"):
        requeued = {
            k: v for k, v in job.items() if k not in ("pending_revision", "hold_reason", "detail")
        }
        requeued.update(status="queued", revision=pending, attempts=0)
        return validate_job(requeued)
    done = {k: v for k, v in job.items() if k != "pending_revision"}
    done["status"] = status
    if hold_reason is not None:
        done["hold_reason"] = hold_reason
    return validate_job(done)


# ---- route costs and request binding ------------------------------------------------------


def _symbols(params):
    raw = params.get("symbols")
    if raw in (None, ""):
        return None
    if not isinstance(raw, str) or raw != raw.strip() or any(ch.isspace() for ch in raw):
        raise ValueError("symbols must be one comma-joined string without whitespace")
    items = raw.split(",")
    if not items or len(items) > BULK_PAGE_MAX or not all(_SYMBOL.fullmatch(s) for s in items):
        raise ValueError(
            "symbols must be 1..%d exact source codes (plain or exchange-qualified)" % BULK_PAGE_MAX
        )
    if len({s.upper() for s in items}) != len(items):
        raise ValueError("duplicate symbols")
    return items


def _int(params, name, default, low, high=None):
    value = params.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < low
        or (high and value > high)
    ):
        raise ValueError("%s must be an integer in %d..%s" % (name, low, high if high else "inf"))
    return value


def route_cost(endpoint, params=None):
    """Exact credits a coverage request is expected to charge, or raise.

    bulk-fundamentals/{EXCH}: params offset, limit (1..500), symbols (plain or qualified, <=500),
      version "1.2" (default); no symbols -> 100, N symbols -> 100 + N; version 2 snapshot (5000)
      raises DisabledRoute; any other version is rejected.
    eod-bulk-last-day/{EXCH}: params date, type (eod|splits|dividends), symbols, fmt. NO
      pagination params. type eod: no symbols -> 100, N symbols -> 100 + N. type splits/dividends:
      symbols are ignored by the provider and the cost is a flat 100.
    calendar/{earnings,splits,dividends}: 1 per request (multi-symbol pricing unverified).
    """
    params = dict(params or {})
    _forbid_credentials(params, "request params")
    if endpoint in CALENDAR_ROUTES:
        allowed = _CALENDAR_PARAMS | (
            _DIVIDEND_FILTERS | _DIVIDEND_PAGING if endpoint == "calendar/dividends" else set()
        )
        if set(params) - allowed:
            raise ValueError("unsupported calendar params: %s" % sorted(set(params) - allowed))
        for name in _DIVIDEND_FILTERS & set(params):
            if name == "filter[symbol]":
                _symbols({"symbols": params[name]})
            else:
                _day(params[name], name)
        if "filter[date_from]" in params and "filter[date_to]" in params:
            if _day(params["filter[date_from]"], "filter[date_from]") > _day(
                params["filter[date_to]"], "filter[date_to]"
            ):
                raise ValueError("filter[date_from] must not be after filter[date_to]")
        if "filter[date_eq]" in params and ({"filter[date_from]", "filter[date_to]"} & set(params)):
            raise ValueError("filter[date_eq] cannot be combined with a date range")
        _int(params, "page[limit]", DIVIDEND_PAGE_MAX, 1, DIVIDEND_PAGE_MAX)
        _int(params, "page[offset]", 0, 0)
        if (
            "from" in params
            and "to" in params
            and _day(params["from"], "from") > _day(params["to"], "to")
        ):
            raise ValueError("calendar from must not be after to")
        _symbols(params)
        return CALENDAR_CREDITS
    route, _, exchange = endpoint.partition("/")
    if route not in BULK_ROUTES or not _EXCHANGE.fullmatch(exchange):
        raise ValueError("route is outside the coverage allowlist")
    if route == "bulk-fundamentals":
        if set(params) - _FUNDAMENTALS_PARAMS:
            raise ValueError(
                "unsupported fundamentals params: %s" % sorted(set(params) - _FUNDAMENTALS_PARAMS)
            )
        version = str(params.get("version", FUNDAMENTALS_VERSION))
        if version in _V2_VERSIONS:
            raise DisabledRoute(
                "v2 whole-exchange snapshot (%d credits) is disabled" % BULK_SNAPSHOT_V2_CREDITS
            )
        if version != FUNDAMENTALS_VERSION:
            raise ValueError("fundamentals version must be %s" % FUNDAMENTALS_VERSION)
        _int(params, "limit", BULK_PAGE_MAX, 1, BULK_PAGE_MAX)
        _int(params, "offset", 0, 0)
        symbols = _symbols(params)
        return BULK_PAGE_CREDITS + (len(symbols) if symbols else 0)
    if set(params) - _EOD_PARAMS:
        raise ValueError(
            "eod-bulk-last-day has no pagination/version params: %s"
            % sorted(set(params) - _EOD_PARAMS)
        )
    if "date" in params:
        _day(params["date"], "date")
    bulk_type = params.get("type", "eod")
    if bulk_type not in BULK_TYPES:
        raise ValueError("type must be one of %s" % (BULK_TYPES,))
    if bulk_type != "eod":
        return BULK_PAGE_CREDITS  # splits/dividends ignore symbols: flat 100
    symbols = _symbols(params)
    return BULK_PAGE_CREDITS + (len(symbols) if symbols else 0)


def request_spec(endpoint, params, estimated_credits):
    """Request spec = endpoint + params + estimated_credits; the estimate must equal route_cost."""
    expected = route_cost(endpoint, params)
    if estimated_credits != expected:
        raise ValueError("estimated_credits %r != contract cost %r" % (estimated_credits, expected))
    return {"endpoint": endpoint, "params": dict(params or {}), "estimated_credits": expected}


def fingerprint(endpoint, params=None):
    """Order-insensitive digest of the exact request: endpoint + params (None dropped, values
    stringified, symbols sorted case-sensitively)."""
    norm = {}
    for key, value in dict(params or {}).items():
        if value is None:
            continue
        if key == "symbols":
            symbols = _symbols({"symbols": value})
            norm[key] = sorted(symbols) if symbols else []
        else:
            norm[key] = str(value)
    return _sha({"endpoint": endpoint, "params": norm})


def bind(endpoint, params, reserved_credits):
    """Binding recorded at reservation time: the reserved cost must equal route_cost now."""
    cost = route_cost(endpoint, params)
    if cost != reserved_credits:
        raise ValueError("reserved %r != contract cost %r" % (reserved_credits, cost))
    return {"endpoint": endpoint, "fingerprint": fingerprint(endpoint, params), "cost": cost}


def check_bound(binding, endpoint, params):
    """Call immediately before EVERY send, retries included. Any change to endpoint or params
    (symbols grown, offset/date/type changed) or a cost above the reservation raises ValueError
    so the guard can refuse before any HTTP call."""
    if endpoint != binding["endpoint"]:
        raise ValueError("endpoint changed after reservation")
    cost = route_cost(endpoint, params)
    if cost > binding["cost"]:
        raise ValueError("final request costs %d > reserved %d" % (cost, binding["cost"]))
    if fingerprint(endpoint, params) != binding["fingerprint"]:
        raise ValueError("final request params differ from the reserved request")
    return cost


# ---- observations, holds, row provenance ---------------------------------------------------


def row_digest(row):
    return _sha(row)


def provenance(payload_sha256, row_index, row):
    """Where an observation came from: retained payload digest + row index + canonical row digest."""
    return {"payload_sha256": payload_sha256, "row_index": row_index, "row_sha256": row_digest(row)}


def _no_duplicate_keys(pairs):
    """json object hook: duplicate keys would silently collapse to the last value, so a retained
    payload with them is ambiguous evidence and is refused."""
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("retained payload has duplicate JSON keys")
    return dict(pairs)


def _rows(parsed):
    """Row list of a retained payload: a bare list, {data: [...]}, or an index-keyed object
    {"0": row, "1": row, ...} (bulk fundamentals); the row index is the integer key order."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        if len(parsed) != 1:
            raise ValueError("ambiguous payload: {data: [...]} must be the only key")
        return parsed["data"]
    if (
        isinstance(parsed, dict)
        and parsed
        and all(isinstance(k, str) and k.isdigit() for k in parsed)
    ):
        keys = sorted(parsed, key=int)
        if len({int(k) for k in keys}) != len(keys):
            raise ValueError("retained payload has duplicate integer keys")
        if [int(k) for k in keys] != list(range(len(keys))):
            raise ValueError("index-keyed payload keys must be 0..N-1 without gaps")
        return [parsed[k] for k in keys]
    raise ValueError(
        "retained payload is neither a row list, {data: [...]} nor an index-keyed object"
    )


class RetainedPayload:
    """Retained raw bytes parsed and hashed ONCE; many observations are then verified against it
    in O(1) each (verify_provenance on every row re-hashed and re-parsed the payload: O(rows^2))."""

    def __init__(self, payload_bytes):
        if not isinstance(payload_bytes, (bytes, bytearray)):
            raise ValueError("retained payload must be bytes")
        self.sha256 = hashlib.sha256(payload_bytes).hexdigest()
        self.rows = _rows(json.loads(payload_bytes, object_pairs_hook=_no_duplicate_keys))

    def verify(self, prov):
        if not isinstance(prov, dict) or set(prov) != {"payload_sha256", "row_index", "row_sha256"}:
            raise ValueError("provenance needs exactly payload_sha256, row_index, row_sha256")
        if self.sha256 != prov["payload_sha256"]:
            raise ValueError("retained bytes do not match payload_sha256")
        index = prov["row_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.rows):
            raise ValueError("row_index outside the retained payload")
        if row_digest(self.rows[index]) != prov["row_sha256"]:
            raise ValueError("retained row does not match row_sha256")
        return self.rows[index]


def verify_provenance(prov, payload_bytes):
    """Check ONE provenance record against the RETAINED raw bytes (never against in-memory rows).
    For many records of the same payload build a RetainedPayload once and call .verify(prov)."""
    return RetainedPayload(payload_bytes).verify(prov)


def observation(ticker, evidence, receipt_ref, prov):
    """One usable observation: {ticker, evidence, receipt_ref, provenance}. Evidence follows the
    existing normalizers' field contract (value + period + sources + observed/available times)."""
    if not (
        isinstance(ticker, str)
        and ticker
        and isinstance(evidence, dict)
        and isinstance(receipt_ref, str)
    ):
        raise ValueError("observation needs ticker, evidence dict and receipt_ref")
    if not isinstance(prov, dict) or set(prov) != {"payload_sha256", "row_index", "row_sha256"}:
        raise ValueError("observation needs provenance {payload_sha256,row_index,row_sha256}")
    return {"ticker": ticker, "evidence": evidence, "receipt_ref": receipt_ref, "provenance": prov}


def row_hold(ticker, reason, detail=None):
    """Explicit per-row hold; unknown/ambiguous rows are held, never passed."""
    if not (isinstance(ticker, str) and isinstance(reason, str) and reason):
        raise ValueError("row hold needs ticker and reason")
    hold = {"ticker": ticker, "hold": reason}
    if detail is not None:
        hold["detail"] = detail
    return hold


def lane_enabled(config, lane):
    """Lane flags live in config['coverage'] and default to DISABLED (explicit opt-in)."""
    if lane not in LANES:
        raise ValueError("unknown lane %r" % (lane,))
    return bool(((config or {}).get("coverage") or {}).get(lane))
