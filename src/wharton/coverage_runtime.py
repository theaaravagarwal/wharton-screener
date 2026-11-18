"""Coverage runtime: config, adapters, retained bytes, evidence merge, qualified financial
contracts, events seam, queue wiring, job execution, per-lane pilot, frozen audit bundle and the
code-receipt module list. stdlib only, plain dicts.

screen_service.py calls into this module through thin hooks that do NOTHING unless the config has
a `coverage` block with an enabled lane (cc.lane_enabled). Every provider call goes through a
guarded session built or passed in by the caller; this module never reads a credential itself.

Config block (all lanes default False; unknown keys are rejected; no opt-in/permission flags):
  coverage: {bulk_market, bulk_fundamentals, calendar, queue: bool,
             registry: {path, sha256},               # pinned registry CSV (bulk/calendar)
             registry_overrides: {path, sha256},     # pinned issuer-proof downgrades (optional)
             pilot: {enabled, exchange, cap_credits <= 500, kind: plan, symbols: [<=3 tickers]},
             bulk_exchanges: [EXCH, ...], windows: {calendar_days: 1..31},
             max_priority_tickers: 1..500}

Adapters receive the pinned REGISTRY rows (news_resolve.load_registry), never screening records.

Retained bytes: state_root/coverage-raw/<sha256>.json plus one receipt sidecar per retrieval
(<sha256>.<fingerprint16>.<retrieved8>.receipt.json) carrying `retrieved_at` and a monotonic `seq`
assigned under the run lock. Supersession order is (retrieved_at, seq), never hash or value.
Evidence is only ever re-derived from those bytes (normalize_bulk) and kept only when its row
provenance verifies against them (ONE cc.RetainedPayload per receipt); in-memory rows are never
trusted.

Financial statements are admitted only through a QUALIFIED CONTRACT: a non-vacuous comparison of a
bulk page against single-name payloads for every sampled name, bound to the adapter and normalizer
digests. qualify_contract() is the one function both production and verify_bundle() use.
"""

import copy
import csv
import fcntl
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from wharton import coverage_contracts as cc
from wharton import coverage_queue, screen_budget, screening

RAW_DIR = "coverage-raw"
PILOT_FILE = "coverage-pilot.json"
CONTRACT_FILE = "coverage-contract.json"
AUDIT_FILE = "coverage-lane-audit.jsonl"
ADMISSIONS_FILE = "coverage-admissions.json"
OWNER = "eodhd_screen_service"  # merged evidence stays replaceable by a fresh single-ticker fetch
FINANCIAL_KINDS = {"bank", "insurer", "etf"}
COVERAGE_MODULES = (
    "coverage_contracts",
    "coverage_queue",
    "coverage_runtime",
    "coverage_bulk",
    "coverage_events",
)
LANE_ADAPTER = {
    "bulk_market": "bulk",
    "bulk_fundamentals": "bulk",
    "calendar": "events",
    "queue": None,
}
ADAPTER_MODULE = {"bulk": "coverage_bulk", "events": "coverage_events"}
ADAPTER_API = {
    "bulk": ("request_specs", "normalize_bulk"),
    "events": ("request_specs", "refresh_jobs", "filing_event_dates"),
}
# The pilot is ONE fixed plan inside the 500-credit cap (see run_pilot).
PILOT_LANES = ("calendar", "bulk_market", "bulk_fundamentals")
PILOT_SAMPLE = 3  # qualified symbols in the bulk-fundamentals probe (100 + N credits)
PILOT_SINGLES = 3  # single-name fundamentals compared with the bulk entries (10 credits each)
SINGLE_FUNDAMENTALS_CREDITS = 10  # screen_budget floor for fundamentals/{ticker}
# (no calendar row cap: earnings windows are non-paginated with no documented cap; the real pilot
# window returned 13,284 rows. Only ACCEPTED events ever cover a ticker.)
DEFAULT_MAX_PRIORITY = 100
DEFAULT_PAGE_CEILING = 128  # runaway guard only: never a statement about candidate coverage
MIN_PAGE_CEILING = 26  # 13,000 stocks at 500 per page
# Cache validity (seconds) by request kind: a retained response is reused only while FRESH.
# An unknown endpoint or lane is never cached (0 = always refresh), never unbounded.
FRESHNESS_DEFAULTS_S = {
    "eod": 18 * 3600,  # + must be retrieved during the same UTC run day
    "bulk_fundamentals": 24 * 3600,
    "calendar_earnings": 6 * 3600,
    "calendar_other": 24 * 3600,  # calendar/splits, calendar/dividends
    "splits_dividends": 24 * 3600,  # eod-bulk-last-day type splits|dividends
    "single": 24 * 3600,  # fundamentals/{ticker}
}
_SINGLE = re.compile(r"fundamentals/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")
# Statement fields only a qualified contract may admit (and the derived ones built from them).
FINANCIAL_FIELDS = frozenset(
    "revenue net_income ebitda_annual operating_income debt_annual cash_annual ocf capex".split()
)
DERIVED_FIELDS = {
    "operating_margin": {"operating_income", "revenue"},
    "net_debt": {"debt_annual", "cash_annual"},
    "ebitda": {"ebitda_annual"},
}
ALL_FINANCIAL = FINANCIAL_FIELDS | set(DERIVED_FIELDS)
RECONCILE_STEP = (
    "A paid attempt was sent but its outcome is unknown. It is never resolved automatically: "
    "reconcile a pending with screen_budget.reconcile_full_cost(run_lock, snapshots); if the "
    "attempt provably never reached the provider, delete its entry from coverage-pilot.json after "
    "review. Its credits count as spent until then. Then re-run (reenable_lane for held lanes)."
)
SETTLED_OUTCOMES = {"refreshed", "cache_current", "price_excluded", "price_unprovable"}
_EXCHANGE = re.compile(r"[A-Z0-9]{1,8}")
_SHA = re.compile(r"[0-9a-f]{64}")
_BASIS = re.compile(r"Financials\.([A-Za-z_]+)\.yearly")
_HTTP = re.compile(r"HTTP (\d{3})")
_ENTITLEMENT = re.compile(
    r"HTTP 40[23]\b|entitle|not authori[sz]ed|subscription|upgrade your plan", re.I
)
_TOP_KEYS = set(cc.LANES) | {
    "pilot",
    "bulk_exchanges",
    "windows",
    "registry",
    "registry_overrides",
    "max_priority_tickers",
    "page_safety_ceiling",
    "freshness",
}
_PILOT_KEYS = {"enabled", "exchange", "cap_credits", "kind", "symbols"}
_CONTRACT_KEYS = {
    "contract_id",
    "endpoint",
    "template_version",
    "adapter_digest",
    "normalizer_digest",
    "supported_fields",
    "accounting_basis",
    "exercised",
    "exercised_count",
    "qualified_at",
    "sample_digests",
}


class CoverageError(ValueError):
    """A coverage lane is misconfigured or unavailable; raised before any spend."""


def _dumps(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def _scrub(text, limit=500):
    """Bounded, token-free message text for status files."""
    text = str(text)
    return "[redacted]" if cc.contains_credential(text) else text[:limit]


# ---- 1. config ---------------------------------------------------------------------------


def _disabled():
    return {
        **{lane: False for lane in cc.LANES},
        "pilot": {
            "enabled": False,
            "exchange": None,
            "cap_credits": cc.PILOT_CAP_CREDITS,
            "kind": "plan",
            "symbols": [],
        },
        "bulk_exchanges": [],
        "windows": {"calendar_days": 14},
        "max_priority_tickers": DEFAULT_MAX_PRIORITY,
        "page_safety_ceiling": DEFAULT_PAGE_CEILING,
        "freshness": {},
    }


def _available():
    return {
        name
        for name, module in ADAPTER_MODULE.items()
        if importlib.util.find_spec("wharton." + module) is not None
    }


def _pinned(block, key, base):
    pinned = block.get(key)
    if pinned is None:
        return None
    if (
        not isinstance(pinned, dict)
        or set(pinned) != {"path", "sha256"}
        or not isinstance(pinned["path"], str)
        or not pinned["path"]
        or not (isinstance(pinned["sha256"], str) and _SHA.fullmatch(pinned["sha256"]))
    ):
        raise CoverageError(f"coverage.{key} needs exactly path and a lowercase sha256")
    path = Path(pinned["path"]).expanduser()
    if base is not None and not path.is_absolute():
        path = (Path(base) / path).resolve()
    return {"path": str(path), "sha256": pinned["sha256"]}


def validate_coverage_config(config, *, available=None, base=None):
    """config['coverage'] -> normalized dict. Absent block (or None) -> disabled config.
    `available` is the set of adapter names ('bulk', 'events') to accept; default = importable.
    `base` resolves relative pinned paths (the config file's directory)."""
    block = (config or {}).get("coverage")
    out = _disabled()
    if block is None:
        return out
    if not isinstance(block, dict):
        raise CoverageError("coverage must be a mapping")
    if set(block) - _TOP_KEYS:
        raise CoverageError("unknown coverage keys: %s" % sorted(set(block) - _TOP_KEYS))
    for lane in cc.LANES:
        if lane in block:
            if not isinstance(block[lane], bool):
                raise CoverageError("coverage.%s must be true or false" % lane)
            out[lane] = block[lane]
    have = _available() if available is None else set(available)
    for lane in cc.LANES:
        adapter = LANE_ADAPTER[lane]
        if out[lane] and adapter and adapter not in have:
            raise CoverageError(
                "lane %s is enabled but wharton.%s is not available"
                % (lane, ADAPTER_MODULE[adapter])
            )
    registry = _pinned(block, "registry", base)
    if registry:
        out["registry"] = registry
    elif out["bulk_market"] or out["bulk_fundamentals"] or out["calendar"]:
        raise CoverageError("bulk and calendar lanes need coverage.registry {path, sha256}")
    overrides = _pinned(block, "registry_overrides", base)
    if overrides:
        if not registry:
            raise CoverageError("coverage.registry_overrides needs coverage.registry")
        out["registry_overrides"] = overrides
    limit = block.get("max_priority_tickers", DEFAULT_MAX_PRIORITY)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise CoverageError("coverage.max_priority_tickers must be an integer 1..500")
    out["max_priority_tickers"] = limit
    ceiling = block.get("page_safety_ceiling", DEFAULT_PAGE_CEILING)
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, int)
        or not MIN_PAGE_CEILING <= ceiling <= 5000
    ):
        raise CoverageError(
            "coverage.page_safety_ceiling must be an integer %d..5000" % MIN_PAGE_CEILING
        )
    out["page_safety_ceiling"] = ceiling
    fresh = block.get("freshness", {})
    if not isinstance(fresh, dict) or set(fresh) - set(cc.LANES):
        raise CoverageError("coverage.freshness is a mapping of lane -> {max_age_s}")
    for lane, rule in fresh.items():
        age = (
            rule.get("max_age_s") if isinstance(rule, dict) and set(rule) == {"max_age_s"} else None
        )
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise CoverageError(f"coverage.freshness.{lane} needs max_age_s, an integer >= 0")
        if age > MAX_FRESHNESS_S:
            raise CoverageError(f"coverage.freshness.{lane}.max_age_s is capped at 7 days")
    out["freshness"] = {lane: {"max_age_s": rule["max_age_s"]} for lane, rule in fresh.items()}
    pilot = block.get("pilot", {})
    if not isinstance(pilot, dict) or set(pilot) - _PILOT_KEYS:
        raise CoverageError("coverage.pilot takes only %s" % sorted(_PILOT_KEYS))
    if "enabled" in pilot and not isinstance(pilot["enabled"], bool):
        raise CoverageError("coverage.pilot.enabled must be true or false")
    cap = pilot.get("cap_credits", cc.PILOT_CAP_CREDITS)
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= cc.PILOT_CAP_CREDITS:
        raise CoverageError(
            "coverage.pilot.cap_credits must be an integer 1..%d" % cc.PILOT_CAP_CREDITS
        )
    out["pilot"].update({k: v for k, v in pilot.items() if k != "cap_credits"}, cap_credits=cap)
    exchange = out["pilot"]["exchange"]
    if exchange is not None and not (isinstance(exchange, str) and _EXCHANGE.fullmatch(exchange)):
        raise CoverageError("coverage.pilot.exchange must be one exchange code")
    symbols = out["pilot"]["symbols"]
    if (
        out["pilot"]["kind"] != "plan"
        or not isinstance(symbols, list)
        or len(symbols) > PILOT_SAMPLE
        or not all(isinstance(t, str) and t.endswith("." + str(exchange)) for t in symbols)
    ):
        raise CoverageError(
            "coverage.pilot.kind is 'plan'; symbols are <=%d tickers on the pilot exchange"
            % PILOT_SAMPLE
        )
    if out["pilot"]["enabled"]:
        if exchange is None:
            raise CoverageError("coverage.pilot.exchange must be one exchange code")
        if not any(out[lane] for lane in PILOT_LANES):
            raise CoverageError("coverage.pilot needs a bulk or calendar lane enabled")
    exchanges = block.get("bulk_exchanges", [])
    if (
        not isinstance(exchanges, list)
        or not all(isinstance(e, str) and _EXCHANGE.fullmatch(e) for e in exchanges)
        or len(set(exchanges)) != len(exchanges)
    ):
        raise CoverageError("coverage.bulk_exchanges must be a list of unique exchange codes")
    out["bulk_exchanges"] = list(exchanges)
    windows = block.get("windows", {})
    if not isinstance(windows, dict) or set(windows) - {"calendar_days"}:
        raise CoverageError("coverage.windows takes only calendar_days")
    days = windows.get("calendar_days", 14)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 31:
        raise CoverageError("coverage.windows.calendar_days must be an integer 1..31")
    out["windows"] = {"calendar_days": days}
    return out


def enabled(config, *lanes):
    """True when any of `lanes` (default: any lane) is enabled in config['coverage']."""
    return any(cc.lane_enabled(config, lane) for lane in (lanes or cc.LANES))


# ---- 1b. pinned registry (+ issuer-proof overrides) ----------------------------------------


def _read_pinned(path, sha256, what):
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise CoverageError(f"{what} file is missing or unreadable") from None
    if _sha(data) != sha256:
        raise CoverageError(f"{what} file sha256 differs from the pinned value")
    return data


@functools.lru_cache(maxsize=2)
def _parse_registry(path, sha256):  # keyed by the VERIFIED hash: parsed once per run
    from wharton import news_resolve  # lazy: pulls the news stack only when a lane needs it

    return news_resolve.load_registry(path)


@functools.lru_cache(maxsize=2)
def _parse_overrides(path, sha256):
    data = _read_pinned(path, sha256, "registry overrides").decode("utf-8")
    reader = csv.DictReader(data.splitlines())
    if reader.fieldnames != ["ticker", "issuer_id", "reason"]:
        raise CoverageError("registry overrides need exactly the columns ticker,issuer_id,reason")
    return tuple((r["ticker"].strip().upper(), r["issuer_id"].strip(), r["reason"]) for r in reader)


@functools.lru_cache(maxsize=2)
def _effective_registry(path, sha256, opath, osha):
    rows = _parse_registry(path, sha256)
    if not opath:
        return rows
    out = dict(rows)
    for ticker, issuer_id, _reason in _parse_overrides(opath, osha):
        row = rows.get(ticker)
        if row and row["issuer_id"] == issuer_id:
            out[ticker] = {**row, "issuer_proof": "unproven"}  # same 11 columns, exact keys
    return out


def load_registry(path, sha256, overrides=None):
    """Registry CSV rows (ticker -> 11-column row) after verifying the file's sha256. Missing or
    changed file raises BEFORE any spend. `overrides` = {path, sha256} of a pinned CSV
    (ticker,issuer_id,reason): a matching row's issuer_proof becomes 'unproven' (reason
    cik_ticker_mismatch = "the CIK cannot confirm the ticker", NOT "proven wrong"), so bulk
    admission holds it as unproven_issuer. Callers that attach news must read rows from here too.
    Rows are shared: callers must not mutate them."""
    _read_pinned(path, sha256, "registry")
    opath = osha = None
    if overrides:
        opath, osha = overrides["path"], overrides["sha256"]
        _read_pinned(opath, osha, "registry overrides")
    return _effective_registry(str(path), sha256, opath, osha)


def registry_rows(config):
    """Pinned registry rows for this config; {} when no registry is configured."""
    block = (config or {}).get("coverage") or {}
    pinned = block.get("registry")
    if not pinned:
        return {}
    return load_registry(pinned["path"], pinned["sha256"], block.get("registry_overrides"))


# ---- 2. adapters -------------------------------------------------------------------------


def get_adapters(config, importer=importlib.import_module):
    """Import coverage_bulk / coverage_events lazily and ONLY for enabled lanes."""
    found = {}
    for lane in cc.LANES:
        name = LANE_ADAPTER[lane]
        if not name or name in found or not cc.lane_enabled(config, lane):
            continue
        try:
            module = importer("wharton." + ADAPTER_MODULE[name])
        except ImportError:
            raise CoverageError(
                "lane %s is enabled but wharton.%s cannot be imported"
                % (lane, ADAPTER_MODULE[name])
            ) from None
        missing = [f for f in ADAPTER_API[name] if not callable(getattr(module, f, None))]
        if missing:
            raise CoverageError("adapter %s lacks %s" % (ADAPTER_MODULE[name], missing))
        found[name] = module
    return SimpleNamespace(bulk=found.get("bulk"), events=found.get("events"))


def _adapter_digest(bulk):
    """sha256 of the bulk adapter's source: its module file, else its class source."""
    path = getattr(bulk, "__file__", None)
    if path and Path(path).is_file():
        return _sha(Path(path).read_bytes())
    try:
        return _sha(inspect.getsource(type(bulk)))
    except (OSError, TypeError):
        return _sha(type(bulk).__qualname__)


@functools.lru_cache(maxsize=1)
def _normalizer_digest():
    """sha256 of the normalizer functions the bulk lane's statements go through."""
    from wharton import screen_service, screen_sources

    funcs = (
        screen_service.normalize_fundamentals,
        screen_sources.identity_metadata,
        screen_sources.listing_identity,
        screen_sources.item,
        screen_sources.stamp,
        # the helpers those call: a change in how a number or a date is read changes the digest
        screening.day,
        screening.number,
        screen_sources.numeric,
    )
    return _sha("\n".join(inspect.getsource(f) for f in funcs))


# ---- 3a. lock ownership ------------------------------------------------------------------


class RunLock:
    """Ownership of the state-root flock. `owned` is true only when THIS object holds the lock
    (it acquired it, or adopted a handle it could re-lock without blocking), and only in the
    process that took it: a forked worker inherits the object but never the ownership."""

    def __init__(self, path, handle, owned):
        self.path, self.handle, self.owned = Path(path), handle, owned
        self._pid = os.getpid()
        self._ino = os.fstat(handle.fileno()).st_ino if owned else None
        os.set_inheritable(handle.fileno(), False)  # close-on-exec: spawned workers never see it

    @classmethod
    def acquire(cls, state_root):
        root = Path(state_root)
        root.mkdir(parents=True, exist_ok=True)
        handle = open(root / ".lock", "a")  # noqa: SIM115 - the lock owns the handle
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise CoverageError("state lock is held by another process") from None
        return cls(root / ".lock", handle, True)

    @classmethod
    def adopt(cls, handle, path):
        """Wrap the handle run() already flocked. A handle that is merely open (another process
        holds the lock) fails the non-blocking re-lock and is NOT owned."""
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return cls(path, handle, False)
        return cls(path, handle, True)

    def assert_owned(self):
        if os.getpid() != self._pid:
            raise CoverageError("the run lock is not owned in this process (forked worker)")
        if not self.owned or self.handle is None or self.handle.closed:
            raise CoverageError("the run lock is not owned")
        try:
            same = os.fstat(self.handle.fileno()).st_ino == os.stat(self.path).st_ino == self._ino
        except OSError:
            same = False
        if not same:
            raise CoverageError("the run lock file was replaced or removed")
        return True

    def is_owned(self):
        try:
            return self.assert_owned()
        except CoverageError:
            return False

    def release(self):
        if os.getpid() == self._pid and self.handle is not None and not self.handle.closed:
            if self.owned:  # a forked child must not unlock the parent's shared description
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
        self.owned = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ---- 3b. retained bytes ------------------------------------------------------------------


def _atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:  # best effort: make the rename durable
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _sweep_tmp(root):
    """Remove temp files a dead process leaked (callers hold the run lock: nobody else writes)."""
    swept = 0
    for leaked in Path(root).glob(".tmp-*.part"):
        try:
            leaked.unlink()
            swept += 1
        except OSError:
            pass
    return swept


def _no_credentials(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            if cc.credential_key(key):
                raise ValueError("receipt must not carry credentials")
            _no_credentials(inner)
    elif isinstance(value, list):
        for inner in value:
            _no_credentials(inner)
    elif cc.contains_credential(value):
        raise ValueError("receipt must not carry credential text")


def _fingerprint(receipt):
    params = receipt.get("parameters") or receipt.get("params") or {}
    try:
        return cc.fingerprint(receipt["endpoint"], params)
    except (KeyError, ValueError):
        return hashlib.sha256(cc.canonical(receipt).encode()).hexdigest()


def _ts(value):
    """Provider-side retrieval time -> epoch seconds (naive = UTC); ValueError if unparseable."""
    stamp = datetime.fromisoformat(str(value))
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()


def _next_seq(root):
    counter = root / "seq.txt"
    try:
        last = int(counter.read_text())
    except (OSError, ValueError):
        last = max((_seq_of(p) for p in root.glob("*.receipt.json")), default=0)
    _atomic_bytes(counter, str(last + 1).encode())
    return last + 1


def _seq_of(path):
    try:
        return int(json.loads(path.read_bytes()).get("seq", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def store_raw(state_root, name, raw_bytes, receipt, *, lock):
    """Retain provider bytes content-addressed (coverage-raw/<sha256>.json) with a receipt
    sidecar. Needs an owned RunLock (the monotonic `seq` is assigned under it). Atomic and
    idempotent: the same bytes + request + retrieved_at keep the FIRST receipt. Returns the
    stored receipt (adds name, file, sha256, fingerprint, seq)."""
    lock.assert_owned()
    if not isinstance(raw_bytes, (bytes, bytearray)) or not isinstance(receipt, dict):
        raise ValueError("store_raw needs raw bytes and a receipt dict")
    _no_credentials(receipt)
    retrieved = receipt.get("retrieved_at")
    _ts(retrieved)  # unparseable retrieved_at is refused here, not discovered later
    raw = bytes(raw_bytes)
    sha = _sha(raw)
    root = Path(state_root) / RAW_DIR
    _sweep_tmp(root)
    raw_path = root / f"{sha}.json"
    if not raw_path.exists() or raw_path.read_bytes() != raw:  # content-addressed: repairs too
        _atomic_bytes(raw_path, raw)
    full = {**receipt, "name": name, "file": f"{sha}.json", "sha256": sha}
    full.setdefault("fingerprint", _fingerprint(full))
    side = root / f"{sha}.{full['fingerprint'][:16]}.{_sha(str(retrieved))[:8]}.receipt.json"
    if side.exists():
        try:
            return json.loads(side.read_bytes())
        except ValueError:
            pass  # corrupt sidecar: rewrite it below
    full["seq"] = _next_seq(root)
    _atomic_bytes(side, cc.canonical(full).encode())
    return full


def _receipts(state_root):
    """[(path, receipt or None, sidecar sha256 or None)] in supersession order (oldest first):
    (retrieved_at, seq). Unreadable or unorderable sidecars come last with receipt None."""
    good, bad = [], []
    for path in (Path(state_root) / RAW_DIR).glob("*.receipt.json"):
        try:
            data = path.read_bytes()
            receipt = json.loads(data)
            key = (_ts(receipt["retrieved_at"]), int(receipt["seq"]), path.name)
            good.append((key, path, receipt, _sha(data)))
        except (OSError, ValueError, TypeError, KeyError):
            bad.append((path.name, path, None, None))
    return [(p, r, d) for _, p, r, d in sorted(good, key=lambda g: g[0])] + [
        (p, r, d) for _, p, r, d in sorted(bad, key=lambda b: b[0])
    ]


def _order(receipt):
    return [_ts(receipt["retrieved_at"]), int(receipt["seq"])]


def _raw_for(state_root, receipt):
    """Retained bytes whose sha256 matches the receipt and the file name; else None."""
    sha = receipt.get("sha256")
    if not (isinstance(sha, str) and _SHA.fullmatch(sha)) or receipt.get("file") != f"{sha}.json":
        return None
    try:
        raw = (Path(state_root) / RAW_DIR / f"{sha}.json").read_bytes()
    except OSError:
        return None
    return raw if _sha(raw) == sha else None


EVIDENCE_DAYS = 45  # screening.evidence() age limit: older evidence can never be valid
MAX_FRESHNESS_S = 7 * 24 * 3600  # no policy may keep a response longer than this
SKEW_S = 300  # a receipt this far ahead of "now" is a clock step, not a fresh retrieval


def _fresh_reason(receipt, min_retrieved_at, max_age, now):
    """fresh | pre_trigger | stale_age | future_dated. max_age <= 0 never counts as fresh
    (always refresh); a receipt dated beyond the skew tolerance is never fresh (no cache-forever
    through a bad host clock)."""
    retrieved = _ts(receipt["retrieved_at"])
    if now is not None and retrieved > _ts(now) + SKEW_S:
        return "future_dated"
    if min_retrieved_at is not None and retrieved < _ts(min_retrieved_at):
        return "pre_trigger"
    if max_age is not None and (max_age <= 0 or now is None or _ts(now) - retrieved > max_age):
        return "stale_age"
    return "fresh"


def lookup_retained(state_root, fingerprint, *, min_retrieved_at=None, max_age=None, now=None):
    """-> ((receipt, raw bytes) | None, info). A hit must satisfy retrieved_at >= min_retrieved_at
    AND now - retrieved_at <= max_age; the NEWEST verified receipt of this request decides (older
    ones are older still), and an unsatisfied one is not returned. Retained bytes are never
    deleted here: a stale receipt stays on disk and is reported as `superseded_sha256`.
    info = {hit, reason: fresh|stale_age|pre_trigger|future_dated|no_receipt, retrieved_at,
    min_retrieved_at, max_age_s, superseded_sha256}."""
    info = {
        "hit": False,
        "reason": "no_receipt",
        "retrieved_at": None,
        "min_retrieved_at": None if min_retrieved_at is None else str(min_retrieved_at),
        "max_age_s": max_age,
        "superseded_sha256": None,
    }
    for _, receipt, _ in reversed(_receipts(state_root)):
        if receipt and receipt.get("fingerprint") == fingerprint:
            raw = _raw_for(state_root, receipt)
            if raw is None:
                continue
            reason = _fresh_reason(receipt, min_retrieved_at, max_age, now)
            if reason == "future_dated":
                # a clock step must not let one bogus receipt shadow every legitimate one (and
                # force a paid refetch forever): look past it, report it only if nothing else exists
                info.update(reason="future_dated", retrieved_at=receipt["retrieved_at"])
                continue
            info["retrieved_at"], info["reason"] = receipt["retrieved_at"], reason
            if reason == "fresh":
                info["hit"] = True
                return (receipt, raw), info
            info["superseded_sha256"] = receipt["sha256"]
            return None, info
    return None, info


def find_retained(state_root, fingerprint, *, min_retrieved_at=None, max_age=None, now=None):
    """(receipt, raw bytes) of the newest verified retained response, if it is fresh enough.
    With no min_retrieved_at / max_age every verified receipt qualifies (audit/qualification)."""
    return lookup_retained(
        state_root, fingerprint, min_retrieved_at=min_retrieved_at, max_age=max_age, now=now
    )[0]


def default_freshness(endpoint, params, now, overrides=None, lane=None):
    """-> {max_age_s, min_retrieved_at} for a request when the caller supplied no policy.
    eod-bulk-last-day (type eod): 18 h AND retrieved during the same UTC run day; its splits and
    dividends types, calendar splits/dividends, bulk-fundamentals pages and fundamentals/{ticker}:
    24 h; calendar/earnings: 6 h. coverage.freshness.<lane>.max_age_s overrides the age of every
    request of that lane. An endpoint or lane the table does not know gets 0: always refresh."""
    never = {"max_age_s": 0, "min_retrieved_at": None}
    if lane is not None and lane not in cc.LANES:
        return never
    params = params or {}
    if endpoint.startswith("eod-bulk-last-day/"):
        kind = "eod" if params.get("type", "eod") == "eod" else "splits_dividends"
    elif endpoint.startswith("bulk-fundamentals/"):
        kind = "bulk_fundamentals"
    elif endpoint == "calendar/earnings":
        kind = "calendar_earnings"
    elif endpoint.startswith("calendar/"):
        kind = "calendar_other"
    elif _SINGLE.fullmatch(endpoint):
        kind = "single"
    else:
        return never
    age = FRESHNESS_DEFAULTS_S[kind]
    if overrides and lane in overrides:
        age = min(overrides[lane]["max_age_s"], MAX_FRESHNESS_S)  # an override is never unbounded
    stamp = None
    if kind == "eod" and now is not None:
        day = datetime.fromisoformat(str(now))
        day = day if day.tzinfo else day.replace(tzinfo=timezone.utc)
        stamp = (
            day.astimezone(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
    return {"max_age_s": age, "min_retrieved_at": stamp}


def job_min_retrieved_at(job):
    """The earliest retrieval that may satisfy a (re)opened job, or None for a fresh job.
    Taken from the job's own meta, first present of trigger_at, revised_at, observed_at.
    observed_at is the field the queue already stores: CoverageQueue.enqueue(observed_at=...)
    records the source observation time in job meta and a newer revision/arrival moves it
    forward. Nothing is invented and coverage_queue is untouched."""
    meta = job.get("meta") or {}
    for key in ("trigger_at", "revised_at", "observed_at"):
        if meta.get(key):
            return str(meta[key])
    return None


def prune(state_root, run_lock, keep_runs=3, now=None):
    """Bound coverage-raw growth. Deletes only receipt sidecars SUPERSEDED by a newer retrieval of
    the same request fingerprint, then raw files no sidecar references. Never deletes bytes that
    an admission of the newest `keep_runs` run bundles references. With `now`, a receipt dated
    beyond the skew tolerance never counts as the newest one (the loader ignores it too), so it
    cannot cause the valid evidence it would shadow to be deleted. Needs an owned RunLock."""
    run_lock.assert_owned()
    root = Path(state_root)
    keep_receipts, keep_sources = set(), set()
    runs = sorted(  # by creation/modification time: names carry no ordering guarantee
        (p for p in (root / "runs").glob("*") if p.is_dir() and not p.name.endswith(".pending")),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    removed = {"receipts": 0, "raw": 0, "tmp": 0}

    def keep_everything(why):  # fail closed: if we cannot tell what is referenced, delete nothing
        return {**removed, "kept_all": why}

    for run_dir in runs[-keep_runs:] if keep_runs > 0 else ():
        try:
            index = json.loads((run_dir / ADMISSIONS_FILE).read_bytes())
            for a in list(index["admitted"]) + list(index.get("superseded", ())):
                keep_receipts.add(a["receipt_sha256"])
                keep_sources.add(a["source_sha256"])
            for contract in (run_dir / "coverage/contracts").glob("*.json"):
                for s in json.loads(contract.read_bytes())["sample_digests"]:
                    keep_receipts.add(s["receipt_sha256"])
                    keep_sources.add(s["raw_sha256"])
        except (OSError, ValueError, KeyError, TypeError):
            return keep_everything("unreadable_run_index")
    if (root / CONTRACT_FILE).exists():
        try:
            stored = json.loads((root / CONTRACT_FILE).read_bytes())["contracts"]
            for contract in stored.values():  # a stored contract always keeps its proof
                for sample in contract.get("sample_digests", ()):
                    keep_receipts.add(sample["receipt_sha256"])
                    keep_sources.add(sample["raw_sha256"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return keep_everything("unreadable_contracts")
    listed = _receipts(state_root)
    if any(receipt is None for _, receipt, _ in listed):
        return keep_everything("unreadable_sidecar")  # it may be the only reference to some bytes
    newest = {}
    removed["tmp"] = _sweep_tmp(root / RAW_DIR) + _sweep_tmp(root / CACHE_DIR)
    for _, receipt, _ in listed:
        if receipt and not (now is not None and _ts(receipt["retrieved_at"]) > _ts(now) + SKEW_S):
            newest[receipt.get("fingerprint")] = receipt["seq"]
    for path, receipt, digest in listed:
        if (
            receipt
            and receipt["seq"] < newest.get(receipt.get("fingerprint"), 0)
            and digest not in keep_receipts
        ):
            path.unlink()
            removed["receipts"] += 1
    referenced = {r["sha256"] for _, r, _ in _receipts(state_root) if r} | keep_sources
    cache = sorted((root / CACHE_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)
    for stale in cache[:-CACHE_KEEP]:  # bound the normalization cache
        stale.unlink()
    for raw in (root / RAW_DIR).glob("*.json"):
        if (
            not raw.name.endswith(".receipt.json")
            and raw.stem not in referenced
            and raw.name != "seq.txt"
        ):
            raw.unlink()
            removed["raw"] += 1
    return removed


def _split(result):
    """normalize_bulk -> (observations, holds); accepts a pair or {observations, row_holds}."""
    if isinstance(result, dict):
        return list(result["observations"]), list(result["row_holds"])
    observations, holds = result
    return list(observations), list(holds)


# ---- 3c. qualified financial contracts --------------------------------------------------


def _row_basis(row):
    found = sorted(set(_BASIS.findall(str(row.get("source", "")))))
    return "+".join(f"{section}.yearly" for section in found) or None


def apply_contract(evidence, contract):
    """-> (evidence kept, [(field, reason)] dropped). Statement fields are kept ONLY when a
    qualified contract supports the field (and every input of a derived field) and every row's
    basis and currency are inside the contract; with no contract all of them are dropped."""
    kept, dropped = {}, []
    supported = set((contract or {}).get("supported_fields") or ())
    bases = set((contract or {}).get("accounting_basis") or ())
    currencies = {e.get("currency") for e in (contract or {}).get("exercised") or ()}
    for key, item in evidence.items():
        if key not in ALL_FINANCIAL:
            kept[key] = item
            continue
        need = DERIVED_FIELDS.get(key, {key})
        if contract is None:
            dropped.append((key, "financial_unqualified"))
        elif not need <= supported:
            dropped.append((key, "financial_field_unsupported"))
        else:
            rows = item.get("value") if isinstance(item, dict) else None
            rows = rows if isinstance(rows, list) else [item]
            ok = bool(rows) and all(
                isinstance(r, dict)
                and r.get("currency") in currencies  # a currency the proof actually exercised
                and _row_basis(r)
                and set(_row_basis(r).split("+")) <= bases
                for r in rows
            )
            if ok:
                kept[key] = item
            else:
                dropped.append((key, "unsupported_basis"))
    return kept, dropped


def _bulk_entry(payload, ticker):
    """The bulk-fundamentals entry for a qualified ticker (exact General.Code), else None."""
    code = ticker.rsplit(".", 1)[0]
    for row in cc._rows(payload):
        if isinstance(row, dict) and (row.get("General") or {}).get("Code") == code:
            return row
    return None


def qualify_contract(adapters, registry, bulk, singles, *, qualified_at):
    """-> (contract | None, {ticker: report}, reason). The ONE qualification used by production
    (the pilot) and by verify_bundle (offline replay). `bulk` = (raw bytes, receipt) of a
    bulk-fundamentals page requested with symbols; `singles` = {ticker: (raw bytes, receipt)} for
    EVERY sampled name. Qualified only when each report is equal and non-vacuous."""
    raw, receipt = bulk
    params = receipt.get("parameters") or receipt.get("params") or {}
    endpoint = str(receipt.get("endpoint", ""))
    sample = sorted(s for s in str(params.get("symbols", "")).split(",") if s)
    version = str(params.get("version", ""))
    if not endpoint.startswith("bulk-fundamentals/") or version != cc.FUNDAMENTALS_VERSION:
        return None, {}, "not_a_versioned_bulk_fundamentals_page"
    if not sample or set(sample) != set(singles):
        return None, {}, "sample_incomplete"
    payload = json.loads(raw)
    reports = {}
    for ticker in sample:
        entry = _bulk_entry(payload, ticker)
        if entry is None or ticker not in registry:
            return None, reports, "bulk_entry_missing:" + ticker
        record = adapters.bulk.registry_record(registry[ticker])
        single_raw, _ = singles[ticker]
        try:
            reports[ticker] = adapters.bulk.equivalence_report(
                entry,
                json.loads(single_raw),
                record,
                {**receipt, "symbol": ticker, "dataset": "fundamentals"},
            )
        except (ValueError, TypeError, LookupError, AttributeError) as exc:
            return None, reports, "equivalence_error:" + type(exc).__name__
    exercised = [{**e, "ticker": t} for t in sample for e in reports[t]["exercised"]]
    if not exercised or not all(r.get("equal") is True for r in reports.values()):
        return None, reports, "not_equal"
    digests = [
        {
            "role": "bulk",
            "ticker": None,
            "raw_sha256": _sha(raw),
            "receipt_sha256": _sha(cc.canonical(receipt)),
        }
    ] + [
        {
            "role": "single",
            "ticker": t,
            "raw_sha256": _sha(singles[t][0]),
            "receipt_sha256": _sha(cc.canonical(singles[t][1])),
        }
        for t in sample
    ]
    core = {
        "endpoint": endpoint,
        "template_version": version,
        "adapter_digest": _adapter_digest(adapters.bulk),
        "normalizer_digest": _normalizer_digest(),
        "supported_fields": sorted({e["field"] for e in exercised}),
        "accounting_basis": sorted({e["basis"] for e in exercised}),
        "exercised": exercised,
        "sample_digests": digests,
    }
    contract = {
        "contract_id": _sha(cc.canonical(core))[:32],
        **core,
        "exercised_count": len(exercised),
        "qualified_at": qualified_at,
    }
    return contract, reports, "ok"


def check_contract_shape(contract):
    """ValueError unless `contract` is a full qualification record (never a boolean claim)."""
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
        raise ValueError("financial contract is not a full qualification record")
    exercised = contract["exercised"]
    if (
        not isinstance(exercised, list)
        or not exercised
        or contract["exercised_count"] != len(exercised)
        or not all(e.get("equal") is True for e in exercised)
        or not contract["supported_fields"]
        or not contract["accounting_basis"]
        or not _SHA.fullmatch(str(contract["adapter_digest"]))
        or not _SHA.fullmatch(str(contract["normalizer_digest"]))
        or not re.fullmatch(r"[0-9a-f]{32}", str(contract["contract_id"]))
    ):
        raise ValueError("financial contract exercised nothing or is malformed")


def load_contracts(state_root):
    """{contract_id: contract} qualified so far (state_root/coverage-contract.json)."""
    try:
        doc = json.loads((Path(state_root) / CONTRACT_FILE).read_bytes())
        contracts = doc["contracts"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return contracts if isinstance(contracts, dict) else {}


def store_contract(state_root, run_lock, contract):
    run_lock.assert_owned()
    check_contract_shape(contract)
    contracts = load_contracts(state_root)
    contracts[contract["contract_id"]] = contract
    _atomic_bytes(Path(state_root) / CONTRACT_FILE, _dumps({"contracts": contracts}).encode())


def contract_for(receipt, contracts, adapters):
    """The valid contract for this bulk-fundamentals receipt, or None: same endpoint, same
    template version, and the CURRENT adapter and normalizer digests (a changed one invalidates)."""
    if adapters.bulk is None or not str(receipt.get("endpoint", "")).startswith(
        "bulk-fundamentals/"
    ):
        return None
    version = str((receipt.get("parameters") or receipt.get("params") or {}).get("version", ""))
    adapter, normalizer = _adapter_digest(adapters.bulk), _normalizer_digest()
    for contract in contracts.values():
        if (
            contract.get("endpoint") == receipt["endpoint"]
            and contract.get("template_version") == version
            and contract.get("adapter_digest") == adapter
            and contract.get("normalizer_digest") == normalizer
        ):
            return contract
    return None


def financials_proven(state_root, adapters=None):
    """Compatibility view: True when at least one contract qualified (and, with adapters, is
    still valid). The authority is the contract object, never this boolean."""
    contracts = load_contracts(state_root)
    if adapters is None:
        return bool(contracts)
    return any(
        contract_for(
            {"endpoint": c["endpoint"], "parameters": {"version": c["template_version"]}},
            {c["contract_id"]: c},
            adapters,
        )
        for c in contracts.values()
    )


# ---- 3d. loading verified observations -------------------------------------------------------

_NORM_CACHE = {}  # in-process mirror of the last normalization: {key, value}


def _mapping_digest(mapping):
    """Order-independent sha256 of a {key: json-able} mapping, streamed (no giant string)."""
    digest = hashlib.sha256()
    for key in sorted(mapping):
        digest.update(json.dumps([key, mapping[key]], sort_keys=True).encode())
    return digest.hexdigest()


def _compact(value):
    """Deterministic compact JSON for the large bundle files (35k-row sized)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


CACHE_DIR = "coverage-cache"
CACHE_KEEP = 8


class _HashingWriter:
    """File-like for json.dump: streams to a binary file while hashing, no second in-memory copy."""

    def __init__(self, out):
        self.out, self.hash = out, hashlib.sha256()

    def write(self, text):
        data = text.encode()
        self.hash.update(data)
        self.out.write(data)


def _cache_read(state_root, name):
    """-> (derived, holds) or None. Entry = '<sha256 of body>\\n<json body>'; a missing, damaged
    or digest-mismatching entry is a miss. A hit refreshes the mtime so prune keeps hot entries."""
    path = Path(state_root) / CACHE_DIR / f"{name}.json"
    try:
        digest, _, body = path.read_bytes().partition(b"\n")
        if digest.decode("ascii") != hashlib.sha256(body).hexdigest():
            return None
        doc = json.loads(body)
        if (
            isinstance(doc, dict)
            and isinstance(doc.get("derived"), list)
            and isinstance(doc.get("holds"), list)
        ):
            try:
                os.utime(path)
            except OSError:
                pass
            return doc["derived"], doc["holds"]
    except (OSError, ValueError):
        pass
    return None  # missing or damaged: a miss, never an error


def _cache_write(state_root, name, derived, holds):
    path = Path(state_root) / CACHE_DIR / f"{name}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(b"0" * 64 + b"\n")  # digest placeholder, filled in after the stream
                writer = _HashingWriter(out)
                json.dump(
                    {"derived": derived, "holds": holds},
                    writer,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                out.seek(0)
                out.write(writer.hash.hexdigest().encode())
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError):
        pass  # the cache is an optimisation only


def load_verified_observations(
    state_root,
    registry,
    adapters,
    *,
    evidence=None,
    fx=None,
    contracts=None,
    not_before=None,
    floors=None,
    now=None,
    freshness=None,
):
    """Re-derive observations from the retained bulk bytes (registry = pinned registry rows) and
    keep ONLY those whose provenance verifies against those bytes (one RetainedPayload per
    receipt). Everything else is a row hold ('provenance_failed'). Statement fields are admitted
    only through a valid qualified contract (apply_contract). Each kept observation gains
    `retained` {order, retrieved_at, seq, receipt_sha256, source_sha256, contract_id}.

    Bounded work: only the NEWEST receipt of each request fingerprint is read (older ones are
    superseded), receipts retrieved before `not_before` (epoch s) are skipped, and a receipt
    retrieved before the trigger floor of its request (`floors` {fingerprint: iso}: a revised job
    whose refetch did not happen) is held as pre_trigger_receipt, never admitted. With `now`, the
    SAME freshness gate as Fetcher.fetch applies (default_freshness + `freshness` overrides): a
    receipt the fetch layer would call stale / pre-trigger / future-dated is held as
    stale_receipt and NOT admitted, so bytes just classified stale on a blocked refetch can never
    be replayed as new evidence (curated legacy facts keep their own engine contract and are
    untouched). Without `now` (audit/qualification) only the evidence window applies.
    Normalization is
    cached on disk (coverage-cache/) by receipt + adapter + contract + registry + evidence + fx."""
    observations, holds = [], []
    contracts = load_contracts(state_root) if contracts is None else contracts
    adapter_key = _adapter_digest(adapters.bulk) if adapters.bulk is not None else ""
    cache_key = (
        adapter_key,
        _normalizer_digest(),
        _mapping_digest(evidence or {}),
        _mapping_digest(registry),
        _sha(repr(sorted((fx or {}).items(), key=repr))),
    )
    floors = floors or {}

    def hold(ticker, reason, detail=None):
        holds.append(cc.row_hold(ticker, reason, detail))

    listed = _receipts(state_root)
    newest = {}
    for _, receipt, _ in listed:
        if receipt and receipt.get("kind") == "bulk" and receipt.get("fingerprint"):
            if now is not None and _ts(receipt["retrieved_at"]) > _ts(now) + SKEW_S:
                continue  # future-dated: never shadows an older fresh receipt (as in fetch)
            newest[receipt["fingerprint"]] = max(
                newest.get(receipt["fingerprint"], (0.0, 0)), tuple(_order(receipt))
            )
    for path, receipt, receipt_sha in listed:
        if receipt is None:
            hold("receipt:" + path.name, "provenance_failed", "unreadable or unorderable receipt")
            continue
        if receipt.get("kind") != "bulk":
            continue
        if newest.get(receipt.get("fingerprint")) not in (None, tuple(_order(receipt))):
            continue  # superseded by a newer retrieval of the same request
        if not_before is not None and _ts(receipt["retrieved_at"]) < not_before:
            continue  # older than any evidence could still be valid
        label = "payload:" + str(receipt.get("sha256"))[:16]
        floor = floors.get(receipt.get("fingerprint"))
        if floor and _ts(receipt["retrieved_at"]) < _ts(floor):
            hold(label, "pre_trigger_receipt", floor)
            continue
        if now is not None:
            endpoint, params, lane = (
                receipt.get("endpoint") or "",
                receipt.get("params") or receipt.get("parameters"),
                receipt.get("lane"),
            )
            if not (
                isinstance(endpoint, str)
                and (params is None or isinstance(params, dict))
                and (lane is None or isinstance(lane, str))
            ):
                hold(
                    label, "stale_receipt", "malformed_receipt"
                )  # one bad sidecar never aborts all
                continue
            policy = default_freshness(endpoint, params, now, freshness, lane)
            why = _fresh_reason(receipt, policy["min_retrieved_at"], policy["max_age_s"], now)
            if why != "fresh":
                hold(label, "stale_receipt", why)  # same gate as the fetch layer, no fallback
                continue
        raw = _raw_for(state_root, receipt)
        if raw is None:
            hold(label, "provenance_failed", "retained bytes missing or altered")
            continue
        contract = contract_for(receipt, contracts, adapters)
        key = (receipt_sha, contract["contract_id"] if contract else None, *cache_key)
        try:
            name = _sha(json.dumps(key, default=str))
            if _NORM_CACHE.get("key") != key:
                loaded = _cache_read(state_root, name)
                if loaded is None:
                    derived_, holds_ = _split(
                        adapters.bulk.normalize_bulk(
                            raw,  # the retained BYTES: provenance is bound to them
                            receipt,
                            registry,
                            fx=fx,
                            financials_proven=contract is not None,
                            **({"evidence": evidence} if evidence else {}),
                        )
                    )
                    _cache_write(state_root, name, derived_, holds_)
                    loaded = (derived_, holds_)
                _NORM_CACHE.clear()  # one entry: the big page in flight
                _NORM_CACHE.update(key=key, value=loaded)
            derived, row_holds = _NORM_CACHE["value"]
            payload = cc.RetainedPayload(raw)  # parsed and hashed ONCE for every row
        except (ValueError, TypeError, LookupError, AttributeError) as exc:
            hold(label, "normalize_failed", type(exc).__name__)
            continue
        holds.extend(row_holds)
        retained = {
            "order": _order(receipt),
            "retrieved_at": receipt["retrieved_at"],
            "seq": receipt["seq"],
            "receipt_sha256": receipt_sha,
            "source_sha256": receipt["sha256"],
            "contract_id": contract["contract_id"] if contract else None,
        }
        for obs in derived:
            try:
                payload.verify(obs["provenance"])
                if obs["ticker"] not in registry:
                    raise ValueError("ticker is not in the registry")
            except (ValueError, TypeError, LookupError) as exc:
                ticker = str(obs.get("ticker", label)) if isinstance(obs, dict) else label
                hold(ticker, "provenance_failed", str(exc)[:120])
                continue
            kept, dropped = apply_contract(obs["evidence"], contract)
            for field, reason in dropped:
                hold(obs["ticker"], reason, field)
            if kept:
                observations.append({**obs, "evidence": kept, "retained": retained})
    return observations, holds


# ---- 4. evidence merge -------------------------------------------------------------------


def apply_classification(record, evidence, base_kind=None):
    """Carry a verified bank/insurer/etf classification onto record kind/sector BEFORE evaluation.
    Raises ValueError when missing/unverified or when it contradicts a registry financial kind."""
    item = (evidence or {}).get("classification")
    value = item.get("value") if isinstance(item, dict) else None
    if not (
        isinstance(value, dict)
        and item.get("verified") is True
        and value.get("kind")
        and value.get("sector")
    ):
        raise ValueError("no verified classification in evidence")
    current = base_kind or record.get("kind", "stock")
    if current in FINANCIAL_KINDS and current != value["kind"]:
        raise ValueError(
            f"registry kind {current!r} conflicts with provider kind {value['kind']!r}"
        )
    record["kind"], record["sector"] = value["kind"], value["sector"]
    return record


def _when(item):
    return str(screening.day(item.get("observed_at")) or "")


def _size(value):
    return len(value) if isinstance(value, (list, dict)) else 1


def _unpromotable(item, cutoff):
    if not isinstance(item, dict):
        return "not_an_evidence_item"
    if item.get("verified") is not True or not item.get("source"):
        return "unverified"
    if (
        item.get("incomplete")
        or item.get("history_incomplete")
        or item.get("basis_unknown")
        or item.get("basis") == "unknown"
    ):
        return "incomplete_or_basis_unknown"  # never promoted to pass
    observed = screening.day(item.get("observed_at"))
    if observed is None or (cutoff is not None and observed > cutoff):
        return "observed_after_as_of_or_undated"
    return None


def _replaces(existing, new, order):
    """-> (rule, hold_reason). rule names why `new` may replace `existing`; (None, None) keeps
    the existing item; (None, reason) is a HOLD on conflict. Coverage-owned evidence follows the
    retrieval order, but a newer retrieval must ALSO be at least as freshly observed; foreign
    provider-owned evidence must be dominated (as fresh AND as complete); curated evidence is
    never overwritten."""
    if existing is None:
        return "new", None
    if not isinstance(existing, dict) or existing.get("owner") != OWNER:
        return None, None  # curated / human / unknown owner
    prior = existing.get("coverage_order")
    if prior is not None:
        if list(order) <= list(prior):
            return None, None  # stale replay: the newer one stands
        if _when(new) < _when(existing):
            return None, "observed_at_older_than_existing"
        return "newer_retrieval", None
    if existing.get("verified") is not True:
        return "replaces_unverified", None
    old, fresh = (
        (_when(existing), _size(existing.get("value"))),
        (_when(new), _size(new.get("value"))),
    )
    dominates = fresh[0] >= old[0] and fresh[1] >= old[1] and fresh != old
    return ("dominates_existing" if dominates else None), None


def merge_evidence(record, observations, as_of, undo=None):
    """Merge this record's observations into record['evidence'].

    Per field the NEWEST retrieval wins ((retrieved_at, seq), never value or hash); an older/late
    replay never overrides a newer one, including across calls, and a newest retrieval that is
    unpromotable (incomplete / basis-unknown / unverified) makes the field a HOLD instead of
    leaving an older item admitted. Classification (bank/insurer/etf kind + sector) follows the
    same order; a conflict at equal order, or an unresolvable classification, holds the whole
    ticker (nothing is merged, no stock default). An item already merged by an earlier pass is
    reported as admitted again, so admissions are complete after any number of passes.
    `undo`, when a list, receives (record, where, key, previous) for every change so a caller can
    roll the merge back (see rehydrate). Returns {'merged', 'kept', 'skipped', 'admitted'}."""
    report = {"merged": [], "kept": [], "skipped": [], "admitted": []}

    def log(where, key, previous):
        if undo is not None:
            undo.append((record, where, key, previous))

    cutoff = screening.day(as_of)
    if record.get("identity_quarantine"):
        report["skipped"].append(("*", "identity_quarantine"))
        return report
    by_key = {}
    for obs in observations:
        if obs["ticker"] != record["ticker"]:
            continue
        order = tuple(obs.get("retained", {}).get("order", (0.0, 0)))
        for key, item in obs["evidence"].items():
            reason = _unpromotable(item, cutoff)
            if reason == "observed_after_as_of_or_undated":
                # not admissible at this as_of: it neither wins nor withdraws an older admitted item
                report["skipped"].append((key, reason))
                continue
            by_key.setdefault(key, []).append((order, item, obs, reason))
    if "evidence" not in record:
        log("no_evidence", None, None)
    evidence = record.setdefault("evidence", {})
    picks = {}
    for key, candidates in by_key.items():
        top = max(c[0] for c in candidates)
        tied = [c for c in candidates if c[0] == top]
        if len({cc.canonical(c[1]) for c in tied}) > 1:
            report["skipped"].append((key, "conflicting_equal_order"))
            if key == "classification":  # fail closed: hold the whole ticker
                report["skipped"].append(("*", "classification_conflict"))
                return {**report, "merged": [], "kept": [], "admitted": []}
            continue
        order, item, obs, reason = tied[0]
        if reason:  # the newest retrieval cannot be promoted: HOLD, drop any older coverage item
            report["skipped"].append((key, reason))
            held = evidence.get(key)
            if isinstance(held, dict) and held.get("coverage_order") is not None:
                if list(held["coverage_order"]) < list(order):
                    log("evidence", key, evidence[key])
                    evidence.pop(key)
            continue
        picks[key] = (order, item, obs)

    def admit(key, rule, order, item, obs):
        info = obs.get("retained", {})
        report["merged"].append(key)
        report["admitted"].append(
            {
                "ticker": record["ticker"],
                "field": key,
                "source_sha256": info.get("source_sha256", obs["provenance"]["payload_sha256"]),
                "receipt_sha256": info.get("receipt_sha256"),
                "provenance": obs["provenance"],
                "observed_at": item.get("observed_at"),
                "retrieved_at": info.get("retrieved_at"),
                "item_sha256": _sha(cc.canonical(item)),
                "contract_id": info.get("contract_id") if key in ALL_FINANCIAL else None,
                "rule": rule,
            }
        )

    if "classification" in picks:
        order, item, obs = picks.pop("classification")
        done = record.get("coverage_classification")
        if (
            done
            and list(order) == list(done["order"])
            and obs["provenance"] == done.get("provenance")
        ):
            admit("classification", "already_admitted", order, item, obs)
        elif done and list(order) < list(done["order"]):
            report["kept"].append("classification")  # stale replay: the newer one stands
        else:
            try:
                base = (done or {}).get("base_kind") or record.get("kind", "stock")
                base_sector = (done or {}).get("base_sector", record.get("sector"))
                for field in ("kind", "sector", "coverage_classification"):
                    log("field", field, record.get(field, _ABSENT))
                apply_classification(record, obs["evidence"], base)
            except ValueError as exc:
                report["skipped"].append(("classification", str(exc)[:120]))
                report["skipped"].append(("*", "classification_unresolved"))
                return {**report, "merged": [], "kept": [], "admitted": []}
            record["coverage_classification"] = {
                "order": list(order),
                "kind": record["kind"],
                "sector": record["sector"],
                "base_kind": base,
                "base_sector": base_sector,
                "provenance": obs["provenance"],
            }
            admit("classification", "newer_classification" if done else "new", order, item, obs)
    for key, (order, item, obs) in sorted(picks.items()):
        existing = evidence.get(key)
        if (
            isinstance(existing, dict)
            and existing.get("owner") == OWNER
            and existing.get("coverage_order") == list(order)
            and existing.get("coverage_provenance") == obs["provenance"]
        ):
            admit(key, "already_admitted", order, item, obs)
            continue
        rule, why = _replaces(existing, item, order)
        if rule is None:
            (report["skipped"].append((key, why)) if why else report["kept"].append(key))
            continue
        log("evidence", key, evidence.get(key, _ABSENT))
        evidence[key] = {
            **copy.deepcopy(
                item
            ),  # never alias the normalization cache (_NORM_CACHE) in a live record
            "owner": OWNER,
            "coverage_provenance": copy.deepcopy(obs["provenance"]),
            "coverage_order": list(order),
        }
        admit(key, rule, order, item, obs)
    return report


_ABSENT = object()


def _rollback(undo):
    """Undo merge_evidence changes, newest first."""
    for record, where, key, previous in reversed(undo):
        if where == "no_evidence":
            record.pop("evidence", None)
        elif where == "evidence":
            if previous is _ABSENT:
                record["evidence"].pop(key, None)
            else:
                record["evidence"][key] = previous
        elif previous is _ABSENT:
            record.pop(key, None)
        else:
            record[key] = previous


def _revalidate_prices(observations, holds, as_of, now):
    """Drop every replayed `listing_price_usd` whose native session has advanced (or cannot be
    proved) at the consumption clock, as a listed row hold. Copies the observation: the rows in
    _NORM_CACHE are shared and must stay untouched."""
    from wharton import screen_sources  # lazy: screen_sources imports back through the service

    out = []
    for obs in observations:
        evidence = obs.get("evidence") or {}
        why = screen_sources.price_item_stale(evidence.get("listing_price_usd"), as_of, now)
        if not why:
            out.append(obs)
            continue
        holds.append(cc.row_hold(obs["ticker"], why, "listing_price_usd"))
        rest = {k: v for k, v in evidence.items() if k != "listing_price_usd"}
        if rest:
            out.append({**obs, "evidence": rest})
    return out


def _withdraw_unadmitted(by_ticker, admitted, undo):
    """Invariant after every pass: a live record carries a coverage-merged item ONLY if THIS pass
    admitted it. An item merged by an earlier pass whose bytes are no longer admissible (e.g. a
    page the loader now holds as pre_trigger / stale) is withdrawn, never left behind without an
    admission. Logged in `undo` so a failed freeze still rolls everything back."""
    keep = {(a["ticker"], a["field"]) for a in admitted}
    for ticker, record in by_ticker.items():
        marker = record.get("coverage_classification")
        if isinstance(marker, dict) and (ticker, "classification") not in keep:
            for field in ("kind", "sector", "coverage_classification"):
                undo.append((record, "field", field, record.get(field, _ABSENT)))
            record["kind"] = marker.get("base_kind") or record.get("kind", "stock")
            if marker.get("base_sector") is not None:
                record["sector"] = marker["base_sector"]  # the sector the record had before
            elif record.get("sector") == marker.get("sector"):
                record.pop("sector", None)
            record.pop("coverage_classification", None)
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            continue
        for key in sorted(evidence):
            item = evidence[key]
            if (
                isinstance(item, dict)
                and item.get("coverage_order") is not None
                and (ticker, key) not in keep
            ):
                undo.append((record, "evidence", key, item))
                evidence.pop(key)


def rehydrate(
    config, state_root, records, as_of, adapters=None, registry=None, floors=None, now=None
):
    """Cache-rehydration hook: retained bulk bytes -> verified observations -> record evidence,
    plus the frozen audit bundle. Returns a report (counts + holds + files); {} when no bulk lane
    is enabled. Safe to call repeatedly: every pass reports ALL its admissions. ATOMIC for the
    live records: they are merged first and rolled back if anything (including the bundle) fails,
    so a failed freeze can never leave a coverage item that has no admission. `floors` =
    {request fingerprint: min_retrieved_at} of revised jobs whose refetch did not happen. `now`
    (the run clock) switches on the shared Fetcher freshness gate for replayed bulk bytes."""
    if not enabled(config, "bulk_market", "bulk_fundamentals"):
        return {}
    adapters = adapters or get_adapters(config)
    registry = registry if registry is not None else registry_rows(config)
    by_ticker = {r["ticker"]: r for r in records}
    cached = _cached_metadata(by_ticker)  # listing_metadata the price rows are checked against
    contracts = load_contracts(state_root)
    # evidence older than the screening window can never be valid again: do not even read it
    not_before = _ts(str(screening.day(as_of))) - (EVIDENCE_DAYS + 1) * 86400
    undo = []  # every change to a live record, so a failure below can take it all back

    def load(evidence):
        found, row_holds = load_verified_observations(
            state_root,
            registry,
            adapters,
            evidence=evidence,
            contracts=contracts,
            not_before=not_before,
            floors=floors,
            now=now,
            freshness=(config.get("coverage") or {}).get("freshness"),
        )
        if now is not None:  # ALL replayed prices: a later native session voids the stored close
            found = _revalidate_prices(found, row_holds, as_of, now)
        return found, row_holds

    try:
        # Two phases when metadata is NEW: prices are normalized against the record's verified
        # listing_metadata, which the same pages may only just have supplied. One extra pass over
        # the retained bytes (the cache key includes the evidence digest) admits those prices in
        # the SAME run instead of leaving them for the next one.
        for phase in (0, 1):
            observations, holds = load(cached)
            triggers = trigger_jobs(holds)
            grouped = {}
            for obs in observations:
                if obs["ticker"] in by_ticker:
                    grouped.setdefault(obs["ticker"], []).append(obs)
            admitted, kept, skipped = [], 0, 0
            for ticker, rows in sorted(grouped.items()):
                result = merge_evidence(by_ticker[ticker], rows, as_of, undo)
                admitted += result["admitted"]
                kept += len(result["kept"])
                skipped += len(result["skipped"])
                holds += [
                    cc.row_hold(ticker, "merge_" + reason, field)
                    for field, reason in result["skipped"]
                ]
            fresh = _cached_metadata(by_ticker)
            if phase == 1 or _mapping_digest(fresh) == _mapping_digest(cached):
                break
            cached = fresh
        _withdraw_unadmitted(by_ticker, admitted, undo)
        report = {
            "observations": len(observations),
            "row_holds": len(holds),
            "merged": len(admitted),
            "kept": kept,
            "skipped": skipped,
            "trigger_jobs": triggers,
        }
        report["files"] = _bundle(state_root, config, registry, admitted, holds, cached, contracts)
    except BaseException:
        _rollback(undo)
        raise
    return report


def _cached_metadata(by_ticker):
    return {
        t: {"listing_metadata": copy.deepcopy(r["evidence"]["listing_metadata"])}
        for t, r in by_ticker.items()
        if isinstance(r.get("evidence", {}).get("listing_metadata"), dict)
        and r["evidence"]["listing_metadata"].get("verified") is True
    }


def trigger_jobs(holds):
    """Bulk 'upcoming_event' row holds (detail.trigger) -> ([hold trigger jobs], {job_id: rev}).
    A dated split/dividend after the cutoff is a queue trigger, never an observation."""
    jobs, seen = [], {}
    for hold in holds:
        detail = hold.get("detail")
        if hold.get("hold") != "upcoming_event" or not (
            isinstance(detail, dict) and detail.get("trigger") is True
        ):
            continue
        if detail.get("type") not in ("splits", "dividends") or not screening.day(
            detail.get("date")
        ):
            continue
        day = str(screening.day(detail["date"]))
        job = cc.make_job(
            "calendar",
            detail["type"],
            hold["ticker"],
            {"from": day, "to": day},
            "upcoming_event",
            0,
            status="hold",
            revision=day,
        )
        job = cc.validate_job({**job, "hold_reason": "trigger", "meta": {"trigger": True}})
        jobs.append(job)
        seen[job["job_id"]] = day
    return jobs, seen


# ---- 4b. frozen audit bundle ---------------------------------------------------------------


def _bundle(state_root, config, registry, admitted, holds, cached=None, contracts=None, note=None):
    """Frozen audit files: the index, the exact retained source bytes and receipts of every
    ADMITTED field, the qualified contracts behind admitted statements with their proof inputs
    (bulk + single raw bytes and receipts), and the registry rows needed to re-derive it all."""
    receipt_shas = {a["receipt_sha256"] for a in admitted}
    used = sorted({a["contract_id"] for a in admitted if a.get("contract_id")})
    proof = {}
    for cid in used:
        for s in (contracts or {}).get(cid, {}).get("sample_digests", ()):
            receipt_shas.add(s["receipt_sha256"])
            proof[s["receipt_sha256"]] = s
    sources, receipts = {}, {}
    for path, receipt, receipt_sha in _receipts(state_root):
        if receipt and receipt_sha in receipt_shas:
            receipts[f"coverage/receipts/{receipt_sha}.json"] = path.read_bytes().decode()
            raw = _raw_for(state_root, receipt)
            if raw is not None:
                sources[f"sources/{receipt['sha256']}.json"] = raw.decode()
    admitted = sorted(admitted, key=lambda a: (a["ticker"], a["field"], a["receipt_sha256"] or ""))
    superseded = _superseded(state_root, {a["receipt_sha256"] for a in admitted})
    tickers = {a["ticker"] for a in admitted}
    tickers |= {s["ticker"] for s in proof.values() if s.get("ticker")}
    pinned = ((config or {}).get("coverage") or {}).get("registry") or {}
    overrides = ((config or {}).get("coverage") or {}).get("registry_overrides") or {}
    files = {
        ADMISSIONS_FILE: _compact(
            {
                "version": 2,
                "status": note or "ok",
                "registry_sha256": pinned.get("sha256"),
                "registry_overrides_sha256": overrides.get("sha256"),
                "financial_contracts": used,
                "superseded": superseded,
                "admitted": admitted,
                "holds": {"total": len(holds), "listed": holds[:1000]},
            }
        ),
        "coverage/registry-rows.json": _compact(
            {t: registry[t] for t in sorted(tickers) if t in registry}
        ),
        "coverage/cached-evidence.json": _compact(
            {t: (cached or {}).get(t) for t in sorted(tickers)}
        ),
    }
    for cid in used:
        files[f"coverage/contracts/{cid}.json"] = _dumps(contracts[cid])
    return {**files, **sources, **receipts}


def _superseded(state_root, admitted_receipts):
    """Receipts of the same request an admitted receipt replaced (older retrieval, same wire
    fingerprint): kept on disk and listed (hashes + times only, no bytes) for the audit trail."""
    listed = _receipts(state_root)
    by_fp = {}
    for _, receipt, digest in listed:
        if receipt:
            by_fp.setdefault(receipt.get("fingerprint"), []).append((receipt, digest))
    out = []
    for receipt, digest in (x for group in by_fp.values() for x in group):
        if digest not in admitted_receipts:
            continue
        for old, old_digest in by_fp[receipt.get("fingerprint")]:
            if old_digest != digest and _order(old) < _order(receipt):
                out.append(
                    {
                        "receipt_sha256": old_digest,
                        "source_sha256": old["sha256"],
                        "retrieved_at": old["retrieved_at"],
                        "seq": old["seq"],
                        "superseded_by": digest,
                    }
                )
    return sorted(out, key=lambda r: (r["superseded_by"], r["seq"]))


def frozen_files(report, config=None):
    """Extra files for the frozen run directory. A bulk lane ALWAYS yields an admissions index,
    even when coverage was unavailable this run (status says so, nothing is admitted)."""
    files = dict((report or {}).get("files") or {})
    if not files and enabled(config, "bulk_market", "bulk_fundamentals"):
        files = _empty_bundle(config, "coverage_unavailable")
    return files


def _empty_bundle(config, note):
    """The admissions index of a run in which coverage admitted nothing. It reads no state, so it
    cannot fail the way a real bundle can: a bulk-lane run ALWAYS has an index to verify."""
    block = (config or {}).get("coverage") or {}
    return {
        ADMISSIONS_FILE: _compact(
            {
                "version": 2,
                "status": note,
                "registry_sha256": (block.get("registry") or {}).get("sha256"),
                "registry_overrides_sha256": (block.get("registry_overrides") or {}).get("sha256"),
                "financial_contracts": [],
                "superseded": [],
                "admitted": [],
                "holds": {"total": 0, "listed": []},
            }
        ),
        "coverage/registry-rows.json": "{}\n",
        "coverage/cached-evidence.json": "{}\n",
    }


def _safe(root, relative):
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("coverage bundle path escapes the run directory")
    return path


def _bundle_bytes(root, kind, digest):
    """Read a bundled file named by a 64-hex digest, check the digest, never trust the name."""
    if not (isinstance(digest, str) and _SHA.fullmatch(digest)):
        raise ValueError("coverage bundle digest is not 64-hex")
    try:
        data = _safe(root, f"{kind}/{digest}.json").read_bytes()
    except OSError:
        raise ValueError("coverage bundle is missing an admitted source or receipt") from None
    if _sha(data) != digest:
        raise ValueError("coverage bundle source or receipt hash mismatch")
    return data


def verify_bundle(path, config, adapters=None):
    """Offline check of a frozen run's coverage bundle, recomputing instead of trusting:
    every admitted field's source and receipt are present with matching hashes; provenance
    verifies against the bundled bytes (ONE RetainedPayload per receipt); every financial
    contract is RE-QUALIFIED from its bundled proof inputs with qualify_contract (digests must
    equal both the recomputed and the current adapter/normalizer); and the adapter re-derives
    exactly the admitted evidence. Every coverage-owned item in input.json needs an admission.
    Raises ValueError on any mismatch."""
    path = Path(path)
    try:
        index = json.loads(_safe(path, ADMISSIONS_FILE).read_bytes())
        admitted = index["admitted"]
    except (OSError, ValueError, KeyError):
        raise ValueError("coverage admissions index is missing or unreadable") from None
    inputs_by_ticker = _require_admissions_for_coverage_items(path, admitted)
    if not admitted:
        return {"admissions": 0}
    adapters = adapters or get_adapters(config)
    registry = json.loads(_safe(path, "coverage/registry-rows.json").read_bytes())
    cached = {
        t: v
        for t, v in json.loads(_safe(path, "coverage/cached-evidence.json").read_bytes()).items()
        if v
    }
    contracts = {}
    for cid in index.get("financial_contracts", ()):
        contracts[cid] = _recompute_contract(path, cid, adapters, registry)
    by_receipt = {}
    for admission in admitted:
        by_receipt.setdefault(admission["receipt_sha256"], []).append(admission)
    for receipt_sha, rows in by_receipt.items():
        receipt_bytes = _bundle_bytes(path, "coverage/receipts", receipt_sha)
        receipt = json.loads(receipt_bytes)
        raw = _bundle_bytes(path, "sources", receipt.get("sha256"))
        contract = contract_for(receipt, contracts, adapters)
        derived, _ = _split(
            adapters.bulk.normalize_bulk(
                raw,
                receipt,
                registry,
                financials_proven=contract is not None,
                **({"evidence": cached} if cached else {}),
            )
        )
        payload = cc.RetainedPayload(raw)
        by_row = {  # ONE index per receipt: verification is linear in the rows
            (o["ticker"], o["provenance"]["row_index"], o["provenance"]["row_sha256"]): o
            for o in derived
        }
        for admission in rows:
            payload.verify(admission["provenance"])
            if admission["field"] in ALL_FINANCIAL and (
                contract is None or admission.get("contract_id") != contract["contract_id"]
            ):
                raise ValueError("admitted financial field has no matching qualified contract")
            prov = admission["provenance"]
            match = by_row.get((admission["ticker"], prov["row_index"], prov["row_sha256"]))
            kept = apply_contract(match["evidence"], contract)[0] if match else {}
            item = kept.get(admission["field"])
            if item is None or _sha(cc.canonical(item)) != admission["item_sha256"]:
                raise ValueError(
                    "coverage field %s/%s cannot be re-derived from the bundle"
                    % (admission["ticker"], admission["field"])
                )
            if admission["field"] == "classification":
                marker = (inputs_by_ticker.get(admission["ticker"]) or {}).get(
                    "coverage_classification"
                )
                value = item.get("value") or {}
                if (
                    marker
                    and marker.get("provenance") == prov
                    and (marker.get("kind"), marker.get("sector"))
                    != (value.get("kind"), value.get("sector"))
                ):
                    raise ValueError(
                        f"classification of {admission['ticker']} differs from its admission"
                    )
    return {"admissions": len(admitted), "contracts": len(contracts)}


def _recompute_contract(path, cid, adapters, registry):
    """Re-qualify a bundled contract from its bundled proof inputs; reject any difference."""
    if not (isinstance(cid, str) and re.fullmatch(r"[0-9a-f]{32}", cid)):
        raise ValueError("coverage contract id is not 32-hex")
    try:
        contract = json.loads(_safe(path, f"coverage/contracts/{cid}.json").read_bytes())
    except (OSError, ValueError):
        raise ValueError("coverage contract is missing from the bundle") from None
    check_contract_shape(contract)
    if contract["contract_id"] != cid:
        raise ValueError("coverage contract id differs from its file")
    if adapters.bulk is None:
        raise ValueError("a financial contract needs the bulk adapter to be re-qualified")
    if (
        contract["adapter_digest"] != _adapter_digest(adapters.bulk)
        or contract["normalizer_digest"] != _normalizer_digest()
    ):
        raise ValueError("coverage contract digests differ from the current adapter/normalizer")
    bulk, singles = None, {}
    for sample in contract["sample_digests"]:
        raw = _bundle_bytes(path, "sources", sample["raw_sha256"])
        receipt = json.loads(_bundle_bytes(path, "coverage/receipts", sample["receipt_sha256"]))
        if sample["role"] == "bulk":
            bulk = (raw, receipt)
        else:
            singles[sample["ticker"]] = (raw, receipt)
    if bulk is None:
        raise ValueError("coverage contract proof inputs lack the bulk page")
    recomputed, _, reason = qualify_contract(
        adapters, registry, bulk, singles, qualified_at=contract["qualified_at"]
    )
    if recomputed != contract:
        raise ValueError(f"coverage contract does not re-qualify offline ({reason})")
    return contract


def _prov_key(ticker, field, prov):
    return (ticker, field, prov["row_index"], prov["row_sha256"], prov["payload_sha256"])


def _require_admissions_for_coverage_items(path, admitted):
    """Every coverage-owned evidence item (and classification marker) in input.json needs an
    admission AND must equal it: the item, stripped of owner/coverage_provenance/coverage_order,
    hashes to the admission's item_sha256. Returns {ticker: record} for the classification check."""
    try:
        inputs = json.loads(_safe(path, "input.json").read_bytes())
    except (OSError, ValueError):
        raise ValueError("frozen input.json is missing or unreadable") from None
    have = {_prov_key(a["ticker"], a["field"], a["provenance"]): a for a in admitted}
    records = list(inputs.get("universe", ())) + list(
        inputs.get("portfolio", {}).get("instruments", ())
    )
    by_ticker = {}
    for record in records:
        by_ticker[record["ticker"]] = record
        for key, item in (record.get("evidence") or {}).items():
            if isinstance(item, dict) and item.get("coverage_provenance") is not None:
                try:
                    admission = have[_prov_key(record["ticker"], key, item["coverage_provenance"])]
                except (KeyError, TypeError):
                    raise ValueError(
                        f"coverage item {record['ticker']}/{key} has no admission"
                    ) from None
                bare = {
                    k: v
                    for k, v in item.items()
                    if k not in ("owner", "coverage_provenance", "coverage_order")
                }
                if _sha(cc.canonical(bare)) != admission["item_sha256"]:
                    raise ValueError(
                        f"coverage item {record['ticker']}/{key} differs from its admission"
                    )
        done = record.get("coverage_classification")
        if done:
            try:
                have[_prov_key(record["ticker"], "classification", done["provenance"])]
            except (KeyError, TypeError):
                raise ValueError(
                    f"coverage classification of {record['ticker']} has no admission"
                ) from None
    return by_ticker


def price_state_from_evidence(record, as_of, now=None):
    """price_gate fallback when no price is cached: a coverage-merged raw close for the expected
    session. Returns a price_gate state dict, or None (the caller then keeps its normal path).
    With `now` (consumption time) the latest completed NATIVE session is re-validated: if a later
    session has completed since the receipt (e.g. after the US close the same UTC day) the stored
    close is stale and the state asks for a fresh price; retrieved_at is never altered."""
    item, valid = screening.evidence(record, "listing_price_usd", screening.day(as_of))
    value = screening.number(item.get("value"))
    if not (
        valid
        and item.get("coverage_provenance")
        and value is not None
        and item.get("currency") == "USD"
        and item.get("period") == item.get("last_completed_session") == item.get("expected_session")
        and item.get("receipt_as_of") == str(screening.day(as_of))
    ):
        return None
    if now is not None:
        from wharton import screen_sources  # lazy: screen_sources imports back through the service

        why = screen_sources.price_item_stale(item, as_of, now)
        if why == "price_stale":
            calendar = item["session_source"].removeprefix("exchange_calendars ").strip()
            return dict(
                status="unknown", price_usd=None, needs="price", reason="price_stale",
                expected_session=screen_sources.latest_completed_session(calendar, as_of, now),
            )  # fmt: skip
        if why:  # the calendar cannot say: unproved, never a silent pass
            return dict(
                status="unknown", price_usd=None, needs=None, reason=why,
                expected_session=item["expected_session"],
            )  # fmt: skip
    ok = value >= screening.PRICE_FLOOR_USD - screening.PRICE_TOLERANCE
    return dict(
        status="pass" if ok else "fail",
        price_usd=value,
        needs=None,
        reason="ok" if ok else "below_floor",
        expected_session=item["expected_session"],
        cached_as_of=str(screening.day(item.get("observed_at"))),
    )


# ---- 5. events seam ----------------------------------------------------------------------


class EventSeam:
    """Filing-event dates from retained GLOBAL earnings payloads. It covers a ticker ONLY when
    that ticker has an ACCEPTED event (validated, in window, unambiguous); everything else keeps
    its per-ticker call."""

    def __init__(self, dates, codes):
        self.dates, self.codes = dates, codes

    def covers(self, ticker):
        return ticker in self.codes

    def date(self, ticker):
        return self.dates.get(ticker)


def replace_filing_event_calls(config, state_root, as_of, adapters=None, registry=None, now=None):
    """EventSeam when the calendar lane is on and a retained global earnings payload covers
    [as_of-7d, as_of], was retrieved on/after as_of, is not flagged truncated and has fewer rows
    than the provider page cap; else None (per-ticker calls proceed). The adapter gets the pinned
    registry rows."""
    if not cc.lane_enabled(config, "calendar"):
        return None
    adapters = adapters or get_adapters(config)
    registry = registry if registry is not None else registry_rows(config)
    day = screening.day(as_of)
    need_from = str(day - timedelta(days=7))
    now = now or datetime.now(timezone.utc)
    age = default_freshness(
        "calendar/earnings",
        {},
        now,
        (config.get("coverage") or {}).get("freshness"),
        "calendar",
    )["max_age_s"]  # the same 6 h calendar policy the Fetcher applies (+ per-lane override)
    dates, codes, usable = {}, set(), False
    for _, receipt, _ in _receipts(state_root):  # oldest first: the newest date per ticker wins
        if (
            not receipt
            or receipt.get("kind") != "events"
            or receipt.get("event_type") != "earnings"
        ):
            continue
        params = receipt.get("params") or {}
        raw = _raw_for(state_root, receipt)
        if (
            raw is None
            or receipt.get("truncated")
            or not (params.get("from", "9") <= need_from and params.get("to", "") >= str(day))
            or str(receipt.get("retrieved_at", ""))[:10] < str(day)
            or _fresh_reason(receipt, None, age, now) != "fresh"
        ):
            continue
        try:
            parsed = json.loads(raw)
            rows = parsed.get("earnings") if isinstance(parsed, dict) else None
            if not isinstance(rows, list) or (isinstance(parsed, dict) and parsed.get("truncated")):
                continue  # flagged truncated / wrong shape: never trust it
            found = adapters.events.filing_event_dates(parsed, receipt, registry, as_of=str(day))
            accepted = adapters.events.accepted_filing_tickers(
                parsed, receipt, registry, as_of=str(day)
            )
        except (ValueError, TypeError, LookupError, AttributeError):
            continue  # unusable payload: fall back to the paid per-ticker call
        usable = True
        codes |= accepted  # ACCEPTED events only: raw code presence is no proof of coverage
        for ticker, when in found.items():
            dates[ticker] = max(when, dates.get(ticker, ""))
    return EventSeam(dates, codes) if usable else None


# ---- 6. queue wiring ---------------------------------------------------------------------


def open_queue(config, state_root, run_lock):
    """CoverageQueue bound to an OWNED run lock (every mutation re-proves ownership)."""
    if not cc.lane_enabled(config, "queue"):
        return None
    run_lock.assert_owned()
    return coverage_queue.CoverageQueue(state_root, run_lock.is_owned)


def _doc(queue):
    try:
        return json.loads(Path(queue.path).read_bytes())
    except FileNotFoundError:
        return {"jobs": {}, "seen": {}}


def _event_tier(job):
    """Queued event jobs carry the queue tier literally: a fresh event is 'event', a late one (the
    event day was before the payload was retrieved) a fixable 'gap'. The adapter's reason stays
    in meta.detail. Held (trigger / unresolved) jobs are untouched."""
    if job["status"] != "queued":
        return job
    tier = "gap" if job["reason"].endswith("_late") else "event"
    return cc.validate_job(
        {
            **job,
            "reason": tier,
            "meta": {**(job.get("meta") or {}), "tier": tier, "detail": job["reason"]},
        }
    )


def ingest_events(queue, state_root, registry, as_of, adapters, now, notes=None):
    """refresh_jobs for every retained events payload -> ONE atomic enqueue(jobs, seen) each.
    Idempotent through the persisted seen map. A corrupt payload is quarantined (recorded in
    `notes`) and the rest continue. Returns the number of jobs offered."""
    offered = 0
    for _, receipt, _ in _receipts(state_root):
        if not receipt or receipt.get("kind") != "events":
            continue
        raw = _raw_for(state_root, receipt)
        if raw is None:
            continue
        try:
            jobs, updates = adapters.events.refresh_jobs(
                json.loads(raw), receipt, registry, as_of=as_of, seen=_doc(queue)["seen"]
            )
            cc_jobs = [_event_tier(cc.validate_job(j)) for j in jobs]
        except (ValueError, TypeError, LookupError, AttributeError) as exc:
            if notes is not None:
                notes.append(
                    {"step": "ingest_events", "type": type(exc).__name__,
                     "reason": "quarantined_payload", "payload": str(receipt.get("sha256"))[:16]}
                )  # fmt: skip
            continue
        if cc_jobs or updates:
            # observed_at = when this calendar payload was retrieved: the job's trigger time
            queue.enqueue(cc_jobs, updates, now, observed_at=receipt["retrieved_at"])
            offered += len(cc_jobs)
    return offered


def _region_cap_key(job):
    meta = job.get("meta") or {}
    return (meta.get("region") or job["scope"].rsplit(".", 1)[-1], meta.get("cap_bucket"))


def is_ticker_scope(scope):
    """Ticker-scoped jobs (events producer) are NOT exchange pages: they become priority tickers."""
    return "." in scope or not _EXCHANGE.fullmatch(scope)


def plan_coverage(
    config,
    queue,
    now,
    remaining_credits,
    mandatory_credits,
    *,
    run_id="run",
    rotation=0,
    max_age=3600,
    max_priority=None,
):
    """Recover stale claims, pick jobs with the 20/80 age/event split and region/size fairness,
    claim them (at most `max_priority` ticker-scoped ones) and persist the age carry. Returns
    {'enabled', 'blocked'?, 'claimed': [jobs], 'stats'}. Ack every claimed job with
    finish_claimed() / execute_claimed() / finish_priority()."""
    if queue is None or not cc.lane_enabled(config, "queue"):
        return {"enabled": False, "claimed": []}
    capacity = Path(queue.path).with_name("capacity-v2.json")
    recovered = queue.recover(
        run_id, now, capacity_state_path=capacity, max_age=max_age, allow_missing_capacity=True
    )
    if "blocked" in recovered:  # unresolved provider attempt: never reclaim, never claim anew
        return {"enabled": True, "blocked": recovered["blocked"], "claimed": []}
    ids, stats = queue.select(
        remaining_credits, mandatory_credits, region_cap_key=_region_cap_key, rotation=rotation
    )
    jobs = _doc(queue)["jobs"]
    if max_priority is not None:
        kept, tickers = [], 0
        for job_id in ids:
            if is_ticker_scope(jobs[job_id]["scope"]):
                tickers += 1
                if tickers > max_priority:
                    continue  # stays queued for a later run
            kept.append(job_id)
        ids = kept
    taken = queue.claim(ids, run_id, now)
    if taken:  # an empty run (an idempotent replay) must not move the age carry or the queue file
        queue.settle(stats)
    jobs = _doc(queue)["jobs"]
    return {
        "enabled": True,
        "claimed": [jobs[i] for i in taken],
        "stats": stats,
        "recovered": recovered,
    }


TRANSIENT_DAILY = frozenset(
    {"acquisition_unreached", "acquisition_held", "coverage_halt", "budget_hold"}
)


def job_lane(job):
    """The coverage lane an exchange-scoped job runs in, or None."""
    if job["kind"] == "market":
        return "bulk_market"
    if job["kind"] == "fundamentals" and job["op"] == "bulk":
        return "bulk_fundamentals"
    return "calendar" if job["kind"] == "calendar" else None


def release_transient(config, queue, now, budget, held_lanes):
    """Release queue holds whose cause is gone, so a transient condition never parks work
    forever. Only these reasons are ever released: budget_deferred (the job fits the budget
    again), acquisition_unreached / acquisition_held / coverage_halt / budget_hold (a per-run
    condition: released on a later UTC day), stale_for_revision (the job's trigger time has
    passed), lane_held (the lane is no longer held). entitlement_unavailable, no_executor,
    invalid_job, lane_disabled, max_attempts and anything else are NEVER released here.
    Returns the released job ids."""
    if queue is None or not cc.lane_enabled(config, "queue"):
        return []
    released, counts = [], _release_counts(queue)
    try:
        cursors = load_cursors(Path(queue.path).parent)
    except CoverageError:
        cursors = None
    today = datetime.fromtimestamp(_ts(now), timezone.utc).date()
    for job_id, job in _doc(queue)["jobs"].items():
        if job["status"] != "hold":
            continue
        reason = job.get("hold_reason", "")
        if reason == "budget_deferred":
            ok = job["estimated_credits"] <= budget
        elif reason in TRANSIENT_DAILY:
            stamp = job.get("updated_at")
            ok = stamp is None or datetime.fromtimestamp(_ts(stamp), timezone.utc).date() < today
        elif reason == "stale_for_revision":
            floor = job_min_retrieved_at(job)
            ok = floor is None or _ts(floor) <= _ts(now)
        elif reason == "lane_held":
            ok = job_lane(job) not in held_lanes
        else:
            ok = False
        if not ok:
            continue
        if counts is None:
            continue  # unreadable counter file: fail closed, release nothing
        cur = cursors["jobs"].get(job_id) if cursors else None
        if cur and not cur["complete"] and cur.get("released_at_offset") != cur["offset"]:
            queue.release(job_id, now)  # the cursor moved since the last release: progress is free
            cur["released_at_offset"] = cur["offset"]
            _write_cursors(Path(queue.path).parent, cursors)
            released.append(job_id)
            continue
        if counts.get(job_id, 0) >= RELEASE_CAP:
            queue.release(job_id, now)  # a held job cannot be re-held: requeue, then park for good
            queue.hold(job_id, "release_cap", now, "release_cap")
            continue
        queue.release(job_id, now)
        counts[job_id] = counts.get(job_id, 0) + 1
        _write_release_counts(queue, counts)
        released.append(job_id)
    return released


RELEASE_CAP = 3
RELEASE_COUNTS_FILE = "coverage-release-counts.json"


def _release_counts(queue):
    """{job_id: releases so far}, or None when the file exists but is unreadable (fail closed).
    ponytail: a sidecar next to the queue file, because coverage_queue (frozen) has no free-form
    job meta to update."""
    path = Path(queue.path).with_name(RELEASE_COUNTS_FILE)
    try:
        doc = json.loads(path.read_bytes())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    ok = isinstance(doc, dict) and all(isinstance(v, int) and v >= 0 for v in doc.values())
    return doc if ok else None


def _write_release_counts(queue, counts):
    path = Path(queue.path).with_name(RELEASE_COUNTS_FILE)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    with os.fdopen(fd, "w") as out:
        json.dump(counts, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def finish_claimed(queue, run_id, now, results):
    """Ack claimed jobs after their work. results: {job_id: (status, text) | None}; None leaves
    the claim in place for recover(). status is completed|hold|failed (hold needs a reason).
    One atomic write per status."""
    groups = {"completed": [], "hold": [], "failed": []}
    for job_id, outcome in results.items():
        if outcome is not None:
            groups[outcome[0]].append((job_id, outcome[1]))
    for status, call in (
        ("completed", queue.complete_many),
        ("failed", queue.fail_many),
        ("hold", queue.hold_many),
    ):
        if groups[status]:
            call(groups[status], run_id, now)


def execute_claimed(queue, run_id, now, claimed, worker):
    """worker(job) -> (status, text) | None. Finished jobs are acked even if a later worker
    raises; a job whose worker raised (or whose process died) stays claimed and is requeued by
    recover() once the provider ledger has no unresolved attempt."""
    results = {}
    try:
        for job in claimed:
            results[job["job_id"]] = worker(job)
    finally:
        finish_claimed(queue, run_id, now, results)
    return results


def _acquisition_retrieved_at(state_root, ticker):
    """Newest retrieval (epoch s) the normal acquisition cache holds for this ticker's prices or
    fundamentals, or None."""
    newest = None
    for dataset in ("prices", "fundamentals"):
        pointer = Path(state_root) / "source-cache" / (_sha(f"{ticker}:{dataset}") + ".json")
        try:
            stamp = _ts(json.loads(pointer.read_bytes())["retrieved_at"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        newest = stamp if newest is None else max(newest, stamp)
    return newest


def finish_priority(config, state_root, run_lock, now, summary, outcomes):
    """Ack the ticker-scoped jobs run_jobs deferred to acquisition. A job is completed only when
    the ticker's acquisition outcome for THIS run is refreshed / cache_current / price_excluded /
    price_unprovable; any other outcome (held, identity_held, unreached) holds it with the reason.
    Returns {job_id: (status, text)}."""
    queue = open_queue(config, state_root, run_lock)
    if queue is None or not (summary or {}).get("priority"):
        return {}
    results = {}
    floors = summary.get("floors") or {}
    for ticker, job_ids in summary["priority"].items():
        outcome = (outcomes or {}).get(ticker)
        for job_id in job_ids:
            verdict = (
                ("completed", f"acquisition:{outcome}")
                if outcome in SETTLED_OUTCOMES
                else ("hold", f"acquisition_{outcome or 'unreached'}")
            )
            floor = floors.get(job_id)
            if verdict[0] == "completed" and floor and outcome != "refreshed":
                # served from cache: only post-trigger bytes may complete a revised job
                have = _acquisition_retrieved_at(state_root, ticker)
                if have is None or have < _ts(floor):
                    verdict = ("hold", "stale_for_revision")
            results[job_id] = verdict
    finish_claimed(queue, summary["run_id"], now, results)
    return results


def status_counts(config, state_root, report=None):
    """Counts for status.json['counts']: {} unless the queue lane is enabled."""
    if not cc.lane_enabled(config, "queue"):
        return {}
    by_status = coverage_queue.CoverageQueue(state_root, lambda: False).summary()["by_status"]
    report = report or {}
    return {
        "queue_pending": by_status.get("queued", 0),
        "queue_claimed": by_status.get("claimed", 0),
        "queue_held": by_status.get("hold", 0),
        "coverage_observations": report.get("observations", 0),
        "coverage_row_holds": report.get("row_holds", 0),
    }


def queue_step(
    config, state_root, as_of, now, run_lock, adapters=None, registry=None, report=None,
    notes=None, *, universe=None, prior=None, production=None,
):  # fmt: skip
    """Run-time hook (queue lane): the production producers (produce_jobs: exchange EOD/actions/
    fundamentals cursor jobs, global calendars, gap and age ticker tiers), bulk trigger jobs
    (upcoming splits/dividends) and, with the calendar lane, retained events. Each is one atomic
    enqueue(jobs, seen). No-op without the queue lane. `production` (a dict) receives the
    producer summary for the status file."""
    queue = open_queue(config, state_root, run_lock)
    if queue is None:
        return 0
    offered = 0
    if enabled(config, "bulk_market", "bulk_fundamentals", "calendar"):
        adapters = adapters or get_adapters(config)
        registry = registry if registry is not None else registry_rows(config)
        made = produce_jobs(
            config, state_root, queue, registry, as_of, now, adapters, notes,
            universe=universe, prior=prior,
        )  # fmt: skip
        offered += made["jobs"]
        if production is not None:
            production.update(made)
    jobs, updates = (report or {}).get("trigger_jobs") or ([], {})
    if jobs:
        queue.enqueue(jobs, updates, now)
        offered += len(jobs)
    if cc.lane_enabled(config, "calendar"):
        registry = registry if registry is not None else registry_rows(config)
        adapters = adapters or get_adapters(config)
        offered += ingest_events(queue, state_root, registry, as_of, adapters, now, notes)
    return offered


# ---- 6a. production: durable cursors, producers, due-work signature -----------------------------

CURSOR_FILE = "coverage-cursors.json"
FUNDAMENTALS_PAGE = cc.BULK_PAGE_MAX
PAGE_SENTINEL = 2  # progress/ETA only: the registered population is NEVER proof of exhaustion
GLOBAL_SCOPE = "GLOBAL"  # scope of the global calendar jobs (an exchange-shaped, non-ticker scope)
MARKET_OPS = ("eod", "splits", "dividends")
AGE_SHARE = 20  # percent of max_priority_tickers given to the age (oldest/unseen) tier
FIXABLE_FRESH_S = 24 * 3600  # evidence younger than this is not an age-tier candidate
_CURSOR_KEYS = frozenset(
    {
        "offset", "pages", "rows_seen", "complete", "incomplete_reason", "updated_at",
        "released_at_offset", "registered_population", "registered_seen", "scope", "op",
    }
)  # fmt: skip
_PAGE_KEYS = frozenset({"offset", "sha256", "rows", "first_code", "last_code", "registered"})


def _blank_cursors():
    return {"version": 1, "jobs": {}, "age": {"considered": {}}}


def check_cursors(doc):
    """Schema check of coverage-cursors.json (also run before every write); no credentials."""
    ok = (
        isinstance(doc, dict)
        and doc.get("version") == 1
        and isinstance(doc.get("jobs"), dict)
        and isinstance(doc.get("age"), dict)
        and isinstance(doc["age"].get("considered"), dict)
    )
    for job_id, cur in doc["jobs"].items() if ok else ():
        ok = (
            ok
            and isinstance(job_id, str)
            and isinstance(cur, dict)
            and set(cur) <= _CURSOR_KEYS
            and type(cur.get("offset")) is int
            and cur["offset"] >= 0
            and isinstance(cur.get("pages"), list)
            and type(cur.get("rows_seen")) is int
            and isinstance(cur.get("complete"), bool)
            and all(isinstance(p, dict) and set(p) <= _PAGE_KEYS for p in cur["pages"])
        )
    if not ok:
        raise CoverageError("coverage-cursors.json has an unexpected shape")
    try:
        _no_credentials(doc)
    except ValueError:
        raise CoverageError("coverage-cursors.json must not carry credentials") from None
    return doc


def load_cursors(state_root):
    path = Path(state_root) / CURSOR_FILE
    try:
        doc = json.loads(path.read_bytes())
    except FileNotFoundError:
        return _blank_cursors()
    except (OSError, ValueError):
        raise CoverageError("coverage-cursors.json is unreadable: nothing resumes") from None
    return check_cursors(doc)


def _write_cursors(state_root, doc):
    check_cursors(doc)
    data = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _atomic_bytes(Path(state_root) / CURSOR_FILE, data + b"\n")


def save_cursors(state_root, run_lock, doc):
    """Atomic write under the OWNED run lock."""
    run_lock.assert_owned()
    _write_cursors(state_root, doc)


def _region(ticker):
    return ticker.rsplit(".", 1)[-1] if "." in ticker else "unknown"


def _iso_or_none(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat() if stamp else None


def fundamentals_cost(config):
    """Credits per mandatory fundamentals refresh (the 80/20 reservation unit); 0 without a
    coverage-enabled config so the hook is a no-op."""
    costs = (config.get("acquisition") or {}).get("request_credit_cost", 0)
    return costs.get("fundamentals", 0) if isinstance(costs, dict) else costs


def _fund_cost(config):
    costs = (config.get("acquisition") or {}).get("request_credit_cost", 0)
    return costs.get("fundamentals", 0) if isinstance(costs, dict) else costs


def _tiered_job(kind, op, scope, window, tier, detail, credits, **meta):
    """A queued job whose `reason` IS the queue tier (coverage_queue._tier reads the literal
    age / event / gap); the specific cause is kept in meta.detail."""
    job = cc.make_job(kind, op, scope, window, tier, credits, "queued")
    job["meta"] = {"tier": tier, "detail": detail, **meta}
    return cc.validate_job(job)


def _epoch_window(config, day):
    """Fundamentals epoch: a whole number of days matching the lane's freshness (24 h => one UTC
    day), so a completed drain is not repeated inside the window in which its pages stay valid."""
    age = default_freshness(
        "bulk-fundamentals/X", {}, None, (config.get("coverage") or {}).get("freshness"),
        "bulk_fundamentals",
    )["max_age_s"]  # fmt: skip
    days = min(7, max(1, age // 86400))
    start = day.toordinal() - day.toordinal() % days
    return {
        "from": datetime.fromordinal(start).date().isoformat(),
        "to": datetime.fromordinal(start + days - 1).date().isoformat(),
    }


def fixable_gap_tickers(prior, tickers):
    """{ticker: [gate, ...]} from the previous decision ledger (results.json rows): tickers whose
    last result is blocked by an evidence gate that is only UNKNOWN (missing / stale / unverified
    field a refresh can fix). A ticker with any FAILED evidence gate is a known hard fail: it gets
    a deep refresh only on new material evidence (an event job), never from this producer."""
    out = {}
    for row in prior if isinstance(prior, list) else ():
        ticker = row.get("ticker") if isinstance(row, dict) else None
        if ticker not in tickers:
            continue
        data = [
            g
            for g in row.get("gates") or ()
            if isinstance(g, dict) and isinstance(g.get("freshness"), list) and g["freshness"]
        ]  # evidence gates (human-review gates carry a dict, never a refreshable list)
        if any(g.get("result") == "fail" for g in data):
            continue
        unknown = sorted(g["gate"] for g in data if g.get("result") == "unknown")
        if unknown:
            out[ticker] = unknown
    return out


def _round_robin_regions(ranked, limit):
    """ranked = [(key, ticker, record)] best first -> up to `limit`, one per region in turn."""
    lanes = {}
    for item in ranked:
        lanes.setdefault(_region(item[1]), []).append(item)
    picked, order = [], sorted(lanes)
    while len(picked) < limit and any(lanes.values()):
        for region in order:
            if lanes[region] and len(picked) < limit:
                picked.append(lanes[region].pop(0))
    return picked


def produce_jobs(
    config, state_root, queue, registry, as_of, now, adapters, notes=None, *,
    universe=None, prior=None, cursors=None,
):  # fmt: skip
    """The production producers (queue lane on, no manual pre-enqueue). Idempotent: job ids are
    kind + op + scope + window, so the same session / epoch / day maps to the same job and the
    queue's own no-op rules apply. Returns {jobs, exchanges, ...} for the status file.

    * exchanges (coverage.bulk_exchanges, EMPTY => nothing and an explicit
      'no_bulk_exchanges_configured' note, never a default): market eod / splits / dividends for
      the exchange's latest completed NATIVE session (no calendar mapping => note, nothing
      produced) and ONE fundamentals/bulk cursor job per exchange per epoch window.
    * global calendars (lane calendar): earnings window (NOT paginated), splits window, the
      dividends day (paginated through links.next only).
    * ticker tiers: GAP = universe tickers whose last result is blocked by a fixable evidence gap;
      AGE = the oldest / never-seen universe tickers, round-robin by region, rotated by a durable
      per-ticker 'considered' marker so the same first N are never picked while others wait.
      Event-tier jobs come from ingest_events (retained calendar receipts)."""
    notes = notes if notes is not None else []
    cov, day = config["coverage"], screening.day(as_of)
    jobs, summary = [], {"exchanges": {}, "age": 0, "gap": 0, "calendar": 0}
    market, funds = (
        cc.lane_enabled(config, "bulk_market"),
        cc.lane_enabled(config, "bulk_fundamentals"),
    )
    exchanges = list(cov.get("bulk_exchanges") or ())
    if (market or funds) and not exchanges:
        notes.append(
            {"step": "produce_jobs", "type": "Note", "reason": "no_bulk_exchanges_configured"}
        )
    if exchanges:  # fair start: later exchanges lead on other days, so none is starved by order
        shift = day.toordinal() % len(exchanges)
        exchanges = exchanges[shift:] + exchanges[:shift]
    for exchange in exchanges if (market or funds) else ():
        made = summary["exchanges"].setdefault(exchange, [])
        try:
            session = latest_native_session(exchange, now)
        except CoverageError as exc:
            notes.append(
                {"step": "produce_jobs", "type": "CoverageError", "exchange": exchange,
                 "reason": "session_calendar_unavailable", "message": _scrub(exc, 120)}
            )  # fmt: skip
            continue
        if market:
            window = {"from": session, "to": session}
            for op in MARKET_OPS:
                specs = adapters.bulk.request_specs(exchange, "market", window, data_type=op)
                credits = sum(spec["estimated_credits"] for spec in specs)
                jobs.append(
                    _tiered_job("market", op, exchange, window, "gap", f"latest_session_{op}",
                                credits, region=exchange)
                )  # fmt: skip
                made.append(f"market/{op}")
        if funds:
            # estimate = the FIRST page (100): the worker pays further pages from the run's own
            # remaining headroom; the registered population / 500 is only a progress estimate
            jobs.append(
                _tiered_job("fundamentals", "bulk", exchange, _epoch_window(config, day), "gap",
                            "fundamentals_epoch", cc.BULK_PAGE_CREDITS, region=exchange)
            )  # fmt: skip
            made.append("fundamentals/bulk")
    if cc.lane_enabled(config, "calendar") and adapters.events is not None:
        start = day - timedelta(days=7)
        span = min(
            day + timedelta(days=cov["windows"]["calendar_days"]), start + timedelta(days=31)
        )
        windows = {
            "earnings": {"from": str(start), "to": str(span)},  # one global window, no offset
            "splits": {
                "from": str(day),
                "to": str(day + timedelta(days=cov["windows"]["calendar_days"])),
            },
            "dividends": {"from": str(day), "to": str(day)},  # exact date; paged by links.next
        }
        for op, window in windows.items():
            jobs.append(
                _tiered_job("calendar", op, GLOBAL_SCOPE, window, "event", f"calendar_{op}",
                            cc.CALENDAR_CREDITS, region=GLOBAL_SCOPE)
            )  # fmt: skip
            summary["calendar"] += 1
    if universe:
        limit = (config.get("coverage") or {}).get("max_priority_tickers", DEFAULT_MAX_PRIORITY)
        age_n = max(1, limit * AGE_SHARE // 100)
        gap_n = max(0, limit - age_n)
        cost = _fund_cost(config)
        window = {"from": str(day), "to": str(day)}
        records = {r["ticker"]: r for r in universe}
        gaps = fixable_gap_tickers(prior, set(records))
        ranked = sorted(gaps, key=lambda t: (-len(gaps[t]), t))
        chosen = [item[1] for item in _round_robin_regions([(0, t, None) for t in ranked], gap_n)]
        for ticker in chosen:
            jobs.append(
                _tiered_job(
                    "evidence", "completion", ticker, window, "gap", "fixable_gap", cost,
                    region=_region(ticker), priority=len(gaps[ticker]), gates=gaps[ticker][:8],
                )
            )  # fmt: skip
        summary["gap"] = len(chosen)
        cursors = cursors if cursors is not None else load_cursors(state_root)
        considered = cursors["age"]["considered"]
        stamp, candidates = _ts(now), []
        for ticker in records:
            if ticker in chosen:
                continue
            got = _acquisition_retrieved_at(state_root, ticker)
            last = _ts(considered[ticker]) if ticker in considered else None
            newest = max((x for x in (got, last) if x is not None), default=None)
            if newest is None or stamp - newest >= FIXABLE_FRESH_S:
                candidates.append((newest or 0.0, ticker, records[ticker]))
        candidates.sort(key=lambda c: (c[0], c[1]))  # oldest / never seen first
        aged = _round_robin_regions(candidates, age_n)
        for newest, ticker, _ in aged:
            jobs.append(
                _tiered_job(
                    "evidence", "completion", ticker, window, "age", "oldest_or_unseen", cost,
                    region=_region(ticker), cursor=_iso_or_none(newest),
                )
            )  # fmt: skip
            considered[ticker] = now.isoformat()
        summary["age"] = len(aged)
        for ticker in [t for t in considered if t not in records]:  # bounded to the universe
            del considered[ticker]
        summary["age_cursor_tickers"] = len(considered)
        if aged:
            _write_cursors(state_root, cursors)
    if jobs:
        queue.enqueue(jobs, {}, now)
    summary["jobs"] = len(jobs)
    return summary


# ---- 6a-2. due-work signature (run identity) -------------------------------------------------

TRANSIENT_HOLDS = frozenset(
    {"budget_deferred", "stale_for_revision", "lane_held"} | TRANSIENT_DAILY
)


def due_work_signature(config, state_root, now, remaining_credits, mandatory=0):
    """Digest of the queue work that exists AFTER this run's production and execution, read-only
    (no spend, no queue mutation): for every queued / claimed / completed job (job_id, revision),
    for every job held with a TRANSIENT reason (same set release_transient releases) a 'due' or
    'blocked' label from the authoritative remaining credits and lane holds, and for every cursor
    its offset and complete flag. Permanent holds (entitlement, no_executor, invalid_job,
    lane_disabled, max_attempts, release_cap) and failures never enter. Volatile fields
    (timestamps, attempt counters) are not hashed. '' when the queue holds none of it.
    Completed work is deliberately included: an executed new job must change the identity (else a
    report built before it would be reused), and an unchanged replay then maps to the SAME
    identity as the run that executed it."""
    if not cc.lane_enabled(config, "queue"):
        return ""
    queue = coverage_queue.CoverageQueue(state_root, lambda: False)
    budget = max(0, remaining_credits - mandatory)
    held, today = (
        set(lane_records(state_root)),
        datetime.fromtimestamp(_ts(now), timezone.utc).date(),
    )
    try:
        cursors = load_cursors(state_root)["jobs"]
    except CoverageError:
        cursors = {}
    entries = []
    for job_id, job in sorted(_doc(queue)["jobs"].items()):
        status, reason = job["status"], job.get("hold_reason", "")
        if status in ("queued", "claimed", "completed"):
            label = "work"
        elif status == "hold" and reason in TRANSIENT_HOLDS:
            if reason == "budget_deferred":
                blocked = job["estimated_credits"] > budget
            elif reason == "stale_for_revision":
                floor = job_min_retrieved_at(job)
                blocked = floor is not None and _ts(floor) > _ts(now)
            elif reason == "lane_held":
                blocked = job_lane(job) in held
            else:
                stamp = job.get("updated_at")
                blocked = (
                    stamp is not None
                    and datetime.fromtimestamp(_ts(stamp), timezone.utc).date() >= today
                )
            label = "blocked" if blocked else "due"
        else:
            continue
        cur = cursors.get(job_id)
        entries.append(
            [job_id, job.get("revision"), label] + ([cur["offset"], cur["complete"]] if cur else [])
        )
    return _sha(json.dumps(entries, separators=(",", ":"))) if entries else ""


def cursor_summary(config, state_root):
    """Explicit per-cursor counts for status.json: pages_done, rows_seen, complete,
    incomplete_reason, and for fundamentals the registered names still unseen (population is an
    estimate: a provider universe is usually a superset of the registry)."""
    try:
        doc = load_cursors(state_root)
    except CoverageError as exc:
        return [{"error": _scrub(exc, 120)}]
    out = []
    for job_id, cur in sorted(doc["jobs"].items()):
        row = {
            "job_id": job_id[:16], "scope": cur.get("scope"), "op": cur.get("op"),
            "pages_done": len(cur["pages"]), "rows_seen": cur["rows_seen"], "offset": cur["offset"],
            "complete": cur["complete"], "incomplete_reason": cur.get("incomplete_reason"),
        }  # fmt: skip
        if cur.get("registered_population") is not None:
            seen = cur.get("registered_seen", 0)
            row.update(registered_population=cur["registered_population"], registered_seen=seen,
                       unseen=max(0, cur["registered_population"] - seen))  # fmt: skip
        out.append(row)
    return out


# ---- 6a-3. offline exchange inventory ----------------------------------------------------------


def inventory(registry):
    """Pure: registry rows -> {exchanges: {suffix: {rows, status, reason}}, supported, held,
    proposed_bulk_exchanges}. An exchange suffix is SUPPORTED when its code is a valid bulk
    exchange code AND a native session calendar exists (screen_sources.CALENDARS); every other
    suffix is HELD with an explicit reason. Rows of unsupported type (not common stock) are
    counted per exchange as `unsupported_type` and never make an exchange supported on their own."""
    from wharton import screen_sources

    exchanges = {}
    for ticker, row in registry.items():
        suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else ""
        entry = exchanges.setdefault(suffix, {"rows": 0, "common_stock": 0, "unsupported_type": 0})
        entry["rows"] += 1
        if str(row.get("type", "")).strip().lower() in ("common stock", "stock"):
            entry["common_stock"] += 1
        else:
            entry["unsupported_type"] += 1
    for suffix, entry in exchanges.items():
        if not suffix or not _EXCHANGE.fullmatch(suffix):
            entry.update(status="held", reason="invalid_exchange_code")
        elif suffix not in screen_sources.CALENDARS:
            entry.update(status="held", reason="no_calendar_mapping")
        elif not entry["common_stock"]:
            entry.update(status="held", reason="unsupported_type")
        else:
            entry.update(status="supported", reason=None)
    proposed = sorted(s for s, e in exchanges.items() if e["status"] == "supported")
    return {
        "exchanges": dict(sorted(exchanges.items())),
        "rows": len(registry),
        "supported": {s: exchanges[s]["rows"] for s in proposed},
        "held": {
            s: {"rows": e["rows"], "reason": e["reason"]}
            for s, e in sorted(exchanges.items())
            if e["status"] == "held"
        },
        "proposed_bulk_exchanges": proposed,
    }


# ---- 6b. soft failure + honest accounting status ------------------------------------------


def soft(step, sink, fn, *args, default=None, **kwargs):
    """Run a coverage step so its failure never aborts the normal screening run: ANY Exception
    (a BudgetHold, CoverageError, corrupt queue, OSError, KeyError, ...) is recorded in `sink`
    (step, type, reason) and `default` returned. Only an UNRESOLVED accounting state (guard
    `accounting_unknown`) is never masked: it re-raises so the run stops honestly."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - coverage must never abort the normal run
        if getattr(exc, "accounting_unknown", False):
            raise
        sink.append({"step": step, "type": type(exc).__name__, "reason": _scrub(exc, 200)})
        return default


def accounting_status(exc):
    """status.json fields for a guard hold with unresolved accounting (non-secret facts only)."""
    pending = getattr(exc, "pending", None)
    attempt = getattr(exc, "attempt", None)
    if not isinstance(pending, dict) and isinstance(attempt, dict):
        pending = {
            "day": attempt.get("day"),
            "endpoint": attempt.get("endpoint"),
            "possible_cost": attempt.get("reserved_cost"),
        }
    pending = (
        {k: pending.get(k) for k in ("day", "endpoint", "possible_cost")}
        if isinstance(pending, dict)
        else None
    )
    return {
        "accounting_unknown": True,
        "accounting_pending": pending,
        "accounting_message": (
            "A paid provider attempt is unresolved, so credit accounting is unknown. "
            "No further paid work runs until it is reconciled."
        ),
    }


# ---- 7. guarded fetching (pilot and queue jobs share it) ----------------------------------


def _rows(payload):
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return len(payload["data"])
    return len(payload) if isinstance(payload, (list, dict)) else 0


def _entitlement(payload):
    return (
        isinstance(payload, dict)
        and "data" not in payload
        and bool(
            _ENTITLEMENT.search(
                json.dumps({k: payload[k] for k in payload if k in ("message", "error", "errors")})
            )
        )
    )


def lane_records(state_root):
    """{lane: record} of every held coverage lane: the guard's coverage-lanes.json merged with the
    pilot ledger's per-lane status. A record is {status, reason, http_status, message,
    attempt_seq, at}; nothing here clears a hold (see reenable_lane)."""
    out = {}
    for source in (Path(state_root) / screen_budget.LANES_FILE, Path(state_root) / PILOT_FILE):
        try:
            doc = json.loads(source.read_bytes())
        except (OSError, ValueError):
            continue
        lanes = doc.get("lanes", doc) if isinstance(doc, dict) else {}
        for lane, record in lanes.items():
            if (
                lane in cc.LANES
                and isinstance(record, dict)
                and record.get("status")
                in (
                    "held",
                    "entitlement_unavailable",
                    "failed",
                    "unresolved",
                )
            ):
                out.setdefault(lane, record)
    return out


def _capacity_pending(state_root):
    """-> (known, pending). A MISSING ledger is known-empty; an unreadable or malformed one is
    UNKNOWN and must make callers refuse (never read as "nothing pending")."""
    path = Path(state_root) / "capacity-v2.json"
    try:
        doc = json.loads(path.read_bytes())
    except FileNotFoundError:
        return True, None
    except (OSError, ValueError):
        return False, None
    return (True, doc.get("pending")) if isinstance(doc, dict) else (False, None)


def reenable_lane(state_root, run_lock, lane, reason, who=None):
    """Audited re-probe path: clears ONLY this lane's hold (coverage-lanes.json and the pilot
    ledger), appends {lane, previous, reason, who, at} to coverage-lane-audit.jsonl. Needs an
    owned RunLock and a non-empty reason, and refuses while a coverage pending is unresolved.
    No approval flag exists; the audit trail is the control."""
    run_lock.assert_owned()
    if lane not in cc.LANES or not (isinstance(reason, str) and reason.strip()):
        raise CoverageError("reenable_lane needs a coverage lane and a non-empty reason")
    root = Path(state_root)
    known, pending = _capacity_pending(root)
    if not known:
        raise CoverageError("capacity-v2.json is unreadable; accounting is unknown, refusing")
    if pending is not None:
        raise CoverageError("a coverage pending is unresolved; reconcile it before re-enabling")
    previous = lane_records(root).get(lane)
    if previous is None:
        raise CoverageError(f"lane {lane} is not held")
    audit = {
        "lane": lane,
        "previous": previous,
        "reason": _scrub(reason, 500),
        "who": _scrub(who or f"pid:{os.getpid()}", 120),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    with open(root / AUDIT_FILE, "a") as out:  # AUDIT FIRST: a crash after this is recoverable
        out.write(json.dumps(audit, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    for source, wrap in ((root / screen_budget.LANES_FILE, False), (root / PILOT_FILE, True)):
        try:
            doc = json.loads(source.read_bytes())
        except (OSError, ValueError):
            continue
        lanes = doc.get("lanes") if wrap else doc
        if isinstance(lanes, dict) and lane in lanes:
            lanes.pop(lane)
            _atomic_bytes(source, _dumps(doc).encode())
    return audit


def _exact_reason(stop, entry):
    """The lane-hold reason: the stop kind + http status + the scrubbed provider message
    (<= 500 chars), never the bare stop string."""
    status = entry.get("http_status")
    head = stop + (f" http_{status}" if status else "")
    return _scrub(f"{head}: {entry.get('message', '')}".rstrip(": "), 500)


class Fetcher:
    """One guarded paid request at a time: cache-first, reserve with params, get through the
    session (final params are bound by the guard), halt-aware, observed-cost accounting, raw bytes
    retained under the run lock, typed guard errors preserved. Never builds or stores a token
    (the injected `request` adds it)."""

    def __init__(self, session_factory, request, state_root, run_lock, now, freshness=None):
        self.factory, self.request = session_factory, request
        self.root, self.lock, self.now = state_root, run_lock, now
        self.freshness = freshness  # coverage.freshness overrides {lane: {max_age_s}}
        self.session = None

    def lookup(self, spec, extra):
        """-> (hit | None, freshness info): the ONE cache decision. `extra["freshness"]` =
        {max_age_s, min_retrieved_at} wins over the default policy (the later min_retrieved_at
        wins); a missing policy never means unbounded. Nothing here reaches the wire."""
        policy = default_freshness(
            spec["endpoint"], spec["params"], self.now, self.freshness, extra.get("lane")
        )
        given = extra.get("freshness") or {}
        age = given.get("max_age_s")
        age = policy["max_age_s"] if age is None else min(age, MAX_FRESHNESS_S)  # None: default
        floors = [m for m in (given.get("min_retrieved_at"), policy["min_retrieved_at"]) if m]
        floor = max(floors, key=_ts) if floors else None
        return lookup_retained(
            self.root,
            cc.fingerprint(spec["endpoint"], spec["params"]),
            min_retrieved_at=floor,
            max_age=age,
            now=self.now,
        )

    def _hold(self, lane, reason, entry):
        """Persist a lane hold for the guard (coverage-lanes.json) next to the pilot ledger."""
        if lane:
            screen_budget.hold_lane(
                self.root, lane, reason, entry.get("http_status"), entry.get("attempt_seq")
            )

    def fetch(self, spec, extra, allowed, journal=None):
        """-> (entry, payload, stop). stop is None, or why the caller must stop (cap_reached,
        coverage_halt, accounting_unresolved, budget_hold, entitlement_unavailable, error,
        no_observation, unresolved_attempt, overspend, lane_held). `allowed` is the credits this
        call may still charge."""
        self.lock.assert_owned()
        endpoint, params, estimate = spec["endpoint"], spec["params"], spec["estimated_credits"]
        lane = extra.get("lane")
        fp = cc.fingerprint(endpoint, params)
        entry = {
            "endpoint": endpoint,
            "fingerprint": fp,
            "estimated": estimate,
            "observed": 0,
            "status": "cached",
        }
        hit, entry["freshness"] = self.lookup(spec, extra)
        if hit:  # cache first, but only while FRESH: never pay twice inside the validity window
            entry["sha256"] = hit[0]["sha256"]
            entry["retrieved_at"] = hit[0]["retrieved_at"]
            if journal:
                journal(entry)
            return entry, json.loads(hit[1]), None
        # a stale or pre-trigger receipt stays on disk and auditable; it is NEVER a fallback
        # below: every not_sent / refused path returns no payload
        if lane and lane in lane_records(self.root):
            entry["status"] = "not_sent"
            return entry, None, "lane_held"
        if estimate > allowed:
            entry["status"] = "not_sent"
            return entry, None, "cap_reached"
        try:
            self.session = self.session or self.factory()
        except screen_budget.BudgetHold as exc:  # e.g. an unresolved pending blocks the session
            entry.update(status="not_sent", error=type(exc).__name__, message=_scrub(exc))
            pending = getattr(exc, "pending", None)
            if pending:
                entry["pending"] = pending
            if journal:
                journal(entry)  # the typed reason must outlive this run
            unknown = getattr(exc, "accounting_unknown", False)
            return entry, None, "accounting_unresolved" if unknown else "budget_hold"
        session = self.session
        if getattr(session, "coverage_halt", False):
            entry["status"] = "not_sent"
            return entry, None, "coverage_halt"
        entry.update(observed=None, status="sent_unresolved")
        if journal:
            journal(entry)  # intent is durable BEFORE the paid send
        seen = len(session.capacity["observations"])
        payload, failure = None, None
        try:
            session.reserve(endpoint, estimate, params=params)
            payload = self.request(session, endpoint, dict(params), expected_type=(list, dict))
        except Exception as exc:  # noqa: BLE001 - any failure stops the caller, never retried
            failure = exc
        charged = session.capacity["observations"][seen:]
        entry["observed"] = sum(o["account_delta"] for o in charged)
        stop = None
        if isinstance(failure, screen_budget.CoverageUnresolved):  # typed: keep every field
            attempt = failure.attempt if isinstance(failure.attempt, dict) else {}
            entry.update(
                status="unresolved",
                error="CoverageUnresolved",
                reason=failure.reason,
                http_status=failure.http_status,
                attempt_seq=attempt.get("seq"),
                unresolved_credits=estimate,  # reserved credits count as unresolved, never 0
                message=_scrub(failure),
            )
            stop = "unresolved_attempt"  # the guard already held the lane and kept pending
        elif failure is not None and session.capacity.get("pending") is not None:
            entry.update(
                status="unresolved",
                error=type(failure).__name__,
                message=_scrub(failure),
                unresolved_credits=estimate,
            )
            stop = "unresolved_attempt"
        elif isinstance(failure, screen_budget.BudgetHold) and not _ENTITLEMENT.search(
            str(failure)
        ):
            entry.update(status="refused", error=type(failure).__name__, message=_scrub(failure))
            stop = "budget_hold"
        elif failure is not None:  # includes the guard's "HTTP 402: no extra calls purchased"
            message = str(failure)
            http = _HTTP.search(message)
            entry["status"] = "charged_error" if charged else "refused"
            entry["error"] = type(failure).__name__
            entry["message"] = _scrub(message)
            entry["http_status"] = int(http.group(1)) if http else None
            stop = "entitlement_unavailable" if _ENTITLEMENT.search(message) else "error"
            entry["reason"] = _exact_reason(stop, entry)
        elif _entitlement(payload):
            entry.update(status="entitlement_unavailable", message=_scrub(payload))
            stop = "entitlement_unavailable"
            entry["reason"] = _exact_reason(stop, entry)
        elif not charged:
            entry["status"], stop = "no_observation", "no_observation"
        else:
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            receipt = store_raw(
                self.root,
                extra.get("name", "coverage"),
                raw,
                {
                    **{k: v for k, v in extra.items() if k != "freshness"},
                    "endpoint": endpoint,
                    "parameters": dict(params),
                    "params": dict(params),
                    "retrieved_at": self.now.isoformat(),
                    "fingerprint": fp,
                },
                lock=self.lock,
            )
            entry.update(
                status="ok", sha256=receipt["sha256"], retrieved_at=receipt["retrieved_at"]
            )
            if entry["observed"] > estimate or getattr(session, "coverage_halt", False):
                stop = "overspend"
        if stop == "entitlement_unavailable":
            self._hold(lane, entry["reason"], entry)
        if journal:
            journal(entry)
        return entry, payload, stop


# ---- 7a. the pilot -------------------------------------------------------------------------


def _pilot_sample(config, registry, exchange):
    """<=3 qualified tickers on the pilot exchange: configured, else the first registry names
    (sorted) that are common stock with a proven issuer and a cleared duplicate flag."""
    chosen = config["coverage"]["pilot"].get("symbols") or []
    if chosen:
        return list(chosen)
    clear = (
        t
        for t, row in sorted(registry.items())
        if t.endswith("." + exchange)
        and row.get("type", "").lower() == "common stock"
        and row.get("issuer_proof", "").lower() == "proven"
        and row.get("duplicate_flag", "").lower() in ("none", "proven")
    )
    return [t for _, t in zip(range(PILOT_SAMPLE), clear)]


def _committed(requests_):
    """Credits the pilot has charged OR may have charged: an unresolved attempt counts as its
    estimate (a crashed sent_unresolved entry has no observation yet), never as 0."""
    return sum(
        r["estimated"] if r["status"] in ("sent_unresolved", "unresolved") else (r["observed"] or 0)
        for r in requests_
    )


def _lane_set(ledger, lane, status, reason, entry, now):
    ledger["lanes"][lane] = {
        "status": status,
        "reason": reason,
        "http_status": entry.get("http_status"),
        "message": _scrub(entry.get("message", "")),
        "attempt_seq": entry.get("attempt_seq"),
        "at": now.isoformat(),
    }


def latest_native_session(exchange, now):
    """ISO date of the latest session the exchange's NATIVE calendar has completed at `now`
    (Monday US -> Friday; a market already closed today -> today). An exchange with no calendar
    mapping, or a calendar that cannot say, is a hold (CoverageError), never a guess."""
    from wharton import screen_sources  # lazy: see price_state_from_evidence

    calendar = screen_sources.CALENDARS.get(exchange)
    day = (
        screen_sources.latest_completed_session(
            calendar, now.astimezone(timezone.utc).date().isoformat(), now
        )
        if calendar
        else None
    )
    if day is None:
        raise CoverageError(f"session_calendar_unavailable for exchange {exchange}: pilot held")
    return day


def _pilot_steps(config, adapters, registry, state_root, now, window, exchange):
    """-> (steps, sample, window): the fixed pilot plan, minus lanes currently held. Pure planning:
    no lock, no session, no network."""
    held = set(lane_records(state_root))
    as_of = now.astimezone(timezone.utc).date()
    if window is None:
        day = latest_native_session(exchange, now)  # never guessed "UTC yesterday"
        window = {"from": day, "to": day}
    sample = _pilot_sample(config, registry, exchange)
    base = {"as_of": str(as_of), "market_as_of": window["to"]}
    steps = []  # (label, lane, extra, spec) in execution order
    if cc.lane_enabled(config, "calendar"):
        days = config["coverage"].get("windows", {}).get("calendar_days", 14)
        span = {"from": str(as_of - timedelta(days=days)), "to": str(as_of)}
        spec = adapters.events.request_specs(span, "earnings")
        steps.append(
            ("calendar", "calendar", {"kind": "events", "event_type": "earnings", **base}, spec)
        )
    if cc.lane_enabled(config, "bulk_market"):
        for spec in adapters.bulk.request_specs(exchange, "market", window, data_type="eod"):
            steps.append(("market", "bulk_market", {"kind": "bulk", **base}, spec))
    if cc.lane_enabled(config, "bulk_fundamentals") and sample:
        for spec in adapters.bulk.request_specs(exchange, "fundamentals", window, symbols=sample):
            steps.append(("bulk", "bulk_fundamentals", {"kind": "bulk", **base}, spec))
        for ticker in sample[:PILOT_SINGLES]:
            spec = {
                "endpoint": f"fundamentals/{ticker}",
                "params": {},
                "estimated_credits": SINGLE_FUNDAMENTALS_CREDITS,
            }
            steps.append(
                (ticker, "bulk_fundamentals", {"kind": "single_fundamentals", **base}, spec)
            )
    steps = [s for s in steps if s[1] not in held]
    return steps, sample, window


def plan_pilot(config, adapters, state_root, now, *, window=None):
    """Dry run of the pilot plan: the steps, their estimates and which of them a FRESH retained
    response already covers. Reads local files only: no lock, no session, no network, no write."""
    pilot = ((config or {}).get("coverage") or {}).get("pilot") or {}
    exchange = pilot.get("exchange")
    if not (isinstance(exchange, str) and _EXCHANGE.fullmatch(exchange)):
        raise CoverageError("pilot needs one exchange code")
    registry = registry_rows(config)
    steps, sample, window = _pilot_steps(
        config, adapters, registry, state_root, now, window, exchange
    )
    fetcher = Fetcher(
        None, None, state_root, None, now, (config.get("coverage") or {}).get("freshness")
    )
    rows = []
    for label, lane, extra, spec in steps:
        hit, info = fetcher.lookup(spec, {**extra, "lane": lane})
        rows.append(
            {
                "label": label,
                "lane": lane,
                "endpoint": spec["endpoint"],
                "estimated": spec["estimated_credits"],
                "cached": hit is not None,
                "freshness": info["reason"],
            }
        )
    return {
        "enabled": pilot.get("enabled") is True,
        "cap": pilot.get("cap_credits", cc.PILOT_CAP_CREDITS),
        "exchange": exchange,
        "sample": sample,
        "steps": rows,
        "estimate": sum(r["estimated"] for r in rows),
        "uncached": sum(r["estimated"] for r in rows if not r["cached"]),
        "held_lanes": sorted(lane_records(state_root)),
    }


def run_pilot(
    session_factory, config, adapters, state_root, now, *, run_lock=None, request=None, window=None
):
    """Single-exchange pilot, ONE fixed plan inside the cap (<= cap_credits <= 500, charged):
      1. calendar/earnings global window (1)                       [lane calendar]
      2. eod-bulk-last-day page (100)                              [lane bulk_market]
      3. bulk-fundamentals page, <=3 qualified symbols (100+N)     [lane bulk_fundamentals]
         = the ENTITLEMENT CHECK, always last
      4. up to 3 single-name fundamentals (10 each) for the SAME tickers, ONLY if step 3
         succeeded, compared with the bulk entries by qualify_contract
    The whole estimate (1+100+103+30 = 234) is validated against the cap BEFORE the first paid
    call. Runs ONLY when coverage.pilot.enabled and only while it OWNS the state lock. Cache-first;
    every step is reserved through the guard and recorded in coverage-pilot.json. A failed or
    entitlement-held lane skips ONLY itself (per-lane status {status, reason, http_status,
    message, attempt_seq, at}); coverage_halt / overspend / BudgetHold / an unresolved coverage
    attempt stop everything, are recorded with their typed reason, and are reported honestly (an
    unresolved pending means accounting is unknown)."""
    pilot = ((config or {}).get("coverage") or {}).get("pilot") or {}
    if pilot.get("enabled") is not True or not any(
        cc.lane_enabled(config, lane) for lane in PILOT_LANES
    ):
        return {"ran": False, "reason": "pilot_disabled"}
    if run_lock is None:
        with RunLock.acquire(state_root) as own:
            return run_pilot(
                session_factory,
                config,
                adapters,
                state_root,
                now,
                run_lock=own,
                request=request,
                window=window,
            )
    run_lock.assert_owned()
    cap, exchange = pilot.get("cap_credits", cc.PILOT_CAP_CREDITS), pilot.get("exchange")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= cc.PILOT_CAP_CREDITS:
        raise CoverageError("pilot cap_credits must be 1..%d" % cc.PILOT_CAP_CREDITS)
    if not (isinstance(exchange, str) and _EXCHANGE.fullmatch(exchange)):
        raise CoverageError("pilot needs one exchange code")
    registry = registry_rows(config)  # pinned registry must verify before any spend
    if request is None:
        request = _default_request()
    path = Path(state_root) / PILOT_FILE
    ledger = (
        json.loads(path.read_bytes())
        if path.exists()
        else {
            "version": 2,
            "exchange": exchange,
            "kind": "plan",
            "cap_credits": cap,
            "spent": 0,
            "unresolved": 0,
            "lanes": {},
            "financial_contract": None,
            "equivalence": {},
            "stopped": None,
            "requests": [],
        }
    )
    requests_ = ledger["requests"]
    ledger["spent"] = sum(r["observed"] or 0 for r in requests_)
    ledger["unresolved"] = sum(r.get("unresolved_credits", 0) for r in requests_)
    if ledger["exchange"] != exchange:
        return {"ran": False, "reason": "pilot_exchange_changed"}
    known, pending = _capacity_pending(state_root)
    if not known:  # an unreadable ledger is UNKNOWN accounting, never "nothing pending"
        return {"ran": False, "reason": "capacity_state_unreadable"}
    if pending is not None or any(r["status"] == "sent_unresolved" for r in requests_):
        return {
            "ran": False,
            "reason": "unresolved_prior_attempt",
            "accounting_unresolved": pending is not None,
            "reconcile": RECONCILE_STEP,
        }
    steps, sample, window = _pilot_steps(
        config, adapters, registry, state_root, now, window, exchange
    )
    if not steps:
        return {"ran": False, "reason": "all_lanes_held", "lanes": dict(lane_records(state_root))}
    for label, _, _, spec in steps:
        if label in ("market", "bulk", "calendar"):  # coverage routes: contract-validated
            cc.request_spec(spec["endpoint"], spec["params"], spec["estimated_credits"])
    estimate = sum(spec["estimated_credits"] for _, _, _, spec in steps)
    fetcher = Fetcher(
        session_factory,
        request,
        state_root,
        run_lock,
        now,
        (config.get("coverage") or {}).get("freshness"),
    )
    uncached = sum(  # only a FRESH retained response is free; a stale one is paid again
        spec["estimated_credits"]
        for _, lane, extra, spec in steps
        if fetcher.lookup(spec, {**extra, "lane": lane})[0] is None  # per-lane overrides apply
    )
    if uncached + _committed(requests_) > cap:
        return {"ran": False, "reason": "plan_exceeds_cap", "estimate": estimate}

    def save():
        ledger["spent"] = sum(r["observed"] or 0 for r in requests_)
        ledger["unresolved"] = sum(r.get("unresolved_credits", 0) for r in requests_)
        _atomic_bytes(path, _dumps(ledger).encode())

    def journal(entry):
        if not any(r is entry for r in requests_):
            repeat = any(
                r["fingerprint"] == entry["fingerprint"] and r["status"] in ("ok", "cached")
                for r in requests_
            )
            if entry["status"] != "cached" or not repeat:
                requests_.append(entry)
        save()

    stop, ok, failed = None, set(), set()
    for label, lane, extra, spec in steps:
        if lane in failed or (label in sample and "bulk" not in ok):
            continue  # a failed lane skips only itself; singles need the bulk page to have worked
        allowed = cap - _committed(requests_)
        entry, _payload, why = fetcher.fetch(
            spec, {**extra, "name": f"pilot:{exchange}:{label}", "lane": lane}, allowed, journal
        )
        if why is None:
            ok.add(label)
        elif why in ("entitlement_unavailable", "error", "no_observation"):
            status = "entitlement_unavailable" if why == "entitlement_unavailable" else "failed"
            _lane_set(ledger, lane, status, entry.get("reason") or why, entry, now)
            failed.add(lane)
        elif why == "lane_held":
            failed.add(lane)
        else:  # halt / overspend / BudgetHold / unresolved / cap: stop everything, typed
            stop = why
            if why in ("unresolved_attempt", "accounting_unresolved"):
                _lane_set(ledger, lane, "unresolved", entry.get("reason") or why, entry, now)
            save()
            break
        save()
    contract_id = _try_qualify(adapters, registry, sample, state_root, run_lock, ledger, steps, now)
    if contract_id:
        ledger["financial_contract"] = contract_id
    ledger["stopped"] = stop or "complete"
    save()
    return {
        "ran": True,
        "stopped": ledger["stopped"],
        "spent": ledger["spent"],
        "unresolved": ledger["unresolved"],
        "requests": len(requests_),
        "estimate": estimate,
        "lanes": dict(ledger["lanes"]),
        "financial_contract": ledger["financial_contract"],
        "accounting_unresolved": stop in ("unresolved_attempt", "accounting_unresolved"),
    }


def _try_qualify(adapters, registry, sample, state_root, run_lock, ledger, steps, now):
    """Qualify the financial contract whenever the bulk page AND every single payload exist
    (retained), regardless of how the rest of the pilot ended. Returns the contract id or None."""
    if adapters.bulk is None or not sample:
        return None
    wanted = {label: spec for label, _, _, spec in steps}
    page = wanted.get("bulk")
    if page is None:
        return None
    bulk = find_retained(state_root, cc.fingerprint(page["endpoint"], page["params"]))
    singles = {}
    for ticker in sample[:PILOT_SINGLES]:
        spec = wanted.get(ticker)
        hit = (
            find_retained(state_root, cc.fingerprint(spec["endpoint"], spec["params"]))
            if spec
            else None
        )
        if hit:
            singles[ticker] = (hit[1], hit[0])
    if bulk is None or set(singles) != set(sample):
        return None
    contract, reports, reason = qualify_contract(
        adapters, registry, (bulk[1], bulk[0]), singles, qualified_at=now.isoformat()
    )
    ledger["equivalence"] = {**reports, "_reason": reason}
    if contract is None:
        return None
    store_contract(state_root, run_lock, contract)
    return contract["contract_id"]


# ---- 7b. queue job execution --------------------------------------------------------------


PAGED = {("fundamentals", "bulk"), ("calendar", "dividends")}  # cursor-driven, many pages per job


def job_plan(adapters, job, offset=None):
    """-> (lane, receipt extra, [specs]) for EXCHANGE-scoped jobs that map to a guarded coverage
    route, else None (no executor). Ticker-scoped jobs never reach here (they are priority
    tickers); a malformed job raises ValueError, which the worker turns into a hold."""
    kind, op, scope, window = job["kind"], job["op"], job["scope"], job["window"]
    meta = job.get("meta") or {}
    offset = meta.get("offset", 0) if offset is None else offset
    if is_ticker_scope(scope):
        return None
    if kind == "market" and adapters.bulk is not None:
        day = {"from": window["to"], "to": window["to"]}
        specs = adapters.bulk.request_specs(scope, "market", day, data_type=op)
        return "bulk_market", {"kind": "bulk"}, specs
    if kind == "fundamentals" and op == "bulk" and adapters.bulk is not None:
        specs = adapters.bulk.request_specs(scope, "fundamentals", window, offset=offset)
        return "bulk_fundamentals", {"kind": "bulk"}, specs
    if kind == "calendar" and adapters.events is not None:
        spec = adapters.events.request_specs(window, op, offset)
        return "calendar", {"kind": "events", "event_type": op}, [spec]
    return None


class JobWorker:
    """worker(job) -> (status, text) | None for execute_claimed.

    Exchange-scoped bulk/calendar jobs are fetched here through the Fetcher (route_cost only,
    cache-first, respecting `budget`, lane holds, coverage_halt and unresolved attempts: the claim
    is then left for recover()). Ticker-scoped market/fundamentals/evidence jobs become PRIORITY
    TICKERS (self.deferred {ticker: [job_id]}): their claims stay open until finish_priority()
    sees the acquisition outcome. One bad job holds only itself."""

    PRIORITY_OPS = {
        ("market", "eod"),
        ("market", "splits"),
        ("fundamentals", "single"),
        ("evidence", "completion"),
    }

    def __init__(
        self, config, adapters, fetcher, as_of, budget, universe=None, *,
        cursors=None, registry=None, deadline=None,
    ):  # fmt: skip
        self.config, self.adapters, self.fetcher = config, adapters, fetcher
        self.as_of, self.budget, self.universe = as_of, budget, universe
        self.spent, self.halted, self.unresolved = 0, None, False
        self.deferred, self.floors, self.request_floors = {}, {}, {}
        self.cursors = cursors if cursors is not None else _blank_cursors()
        self.registry = registry or {}
        self.deadline = deadline  # time.monotonic() value shared by the whole run
        self.ceiling = (config.get("coverage") or {}).get(
            "page_safety_ceiling", DEFAULT_PAGE_CEILING
        )
        self.persist = True

    def __call__(self, job):
        try:
            return self._run(job)
        except ValueError as exc:  # a malformed job holds itself; the run goes on
            return "hold", ("invalid_job:" + type(exc).__name__ + ":" + _scrub(exc, 80))[:200]

    def _run(self, job):
        if self.unresolved:
            return None
        kind, op, scope = job["kind"], job["op"], job["scope"]
        if is_ticker_scope(scope):
            if (kind, op) not in self.PRIORITY_OPS:
                return "hold", f"no_executor:{kind}/{op}"
            if self.universe is not None and scope not in self.universe:
                return "hold", "ticker_not_in_universe"
            self.deferred.setdefault(scope, []).append(job["job_id"])
            if job_min_retrieved_at(job):
                self.floors[job["job_id"]] = job_min_retrieved_at(job)
            return None
        if self.halted:
            return "hold", self.halted
        if self._late():
            return "hold", "budget_deferred"  # the shared run deadline: releasable next run
        plan = job_plan(self.adapters, job)
        if plan is None:
            return "hold", f"no_executor:{kind}/{op}"
        lane, extra, specs = plan
        if not cc.lane_enabled(self.config, lane):
            return "hold", "lane_disabled"
        extra = {
            **extra,
            "lane": lane,
            "name": f"job:{job['job_id'][:16]}",
            "as_of": str(self.as_of),
            "market_as_of": min(str(job["window"]["to"]), str(screening.day(self.as_of))),
        }
        floor = job_min_retrieved_at(job)  # a revised / re-triggered job needs post-trigger bytes
        if floor:
            extra["freshness"] = {"min_retrieved_at": floor}
        if (kind, op) in PAGED:
            return self._paginate(job, extra, floor)
        for spec in specs:
            spec = cc.request_spec(spec["endpoint"], spec["params"], spec["estimated_credits"])
            if floor:  # the loader must not admit a pre-trigger page if this refetch is blocked
                self.request_floors[cc.fingerprint(spec["endpoint"], spec["params"])] = floor
            entry, _, stop = self.fetcher.fetch(spec, extra, self.budget - self.spent)
            self.spent += entry["observed"] or 0
            stale = (
                floor
                and not stop
                and entry.get("retrieved_at")
                and _ts(entry["retrieved_at"]) < _ts(floor)
            )
            if stale:  # never complete from bytes older than the trigger
                return "hold", "stale_for_revision"
            if stop in ("unresolved_attempt", "accounting_unresolved"):
                self.unresolved = True
                return None
            if stop == "overspend":
                self.halted = "coverage_halt"
                return "completed", f"overspend credits={self.spent}"
            if stop in ("coverage_halt", "budget_hold"):
                self.halted = stop
                return "hold", stop
            if stop in ("entitlement_unavailable", "lane_held"):
                return "hold", stop  # the lane is held; other lanes' jobs keep going
            if stop == "cap_reached":
                return "hold", "budget_deferred"
            if stop:
                return "failed", stop
        return "completed", f"credits={self.spent}"

    def _late(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def _save(self):
        if self.persist:
            _write_cursors(self.fetcher.root, self.cursors)

    def _pause(self, cur, reason):
        """Out of headroom / time / pages: keep the cursor, hold releasable (budget_deferred)."""
        cur["incomplete_reason"] = reason
        self._save()
        return "hold", "budget_deferred"

    def _paginate(self, job, extra, floor):
        """ONE durable cursor job drains pages itself inside the claim: request cursor.offset
        (cache-first: a fresh retained page costs 0), persist the cursor after EACH page, stop at
        the first of: the end proof (short fundamentals page / links.next null), the guarded credit
        headroom (cap_reached), the shared deadline, the safety ceiling, a lane hold, or a page
        that looks wrong (repeated / ignored offset / malformed / not advancing: HOLD, never
        completed)."""
        op, scope = job["op"], job["scope"]
        cur = self.cursors["jobs"].setdefault(
            job["job_id"],
            {"offset": 0, "pages": [], "rows_seen": 0, "complete": False,
             "incomplete_reason": None, "updated_at": self.fetcher.now.isoformat()},
        )  # fmt: skip
        cur.update(scope=scope, op=op)
        if op == "bulk":
            cur["registered_population"] = sum(1 for t in self.registry if t.endswith("." + scope))
        if cur["complete"]:
            return "completed", "cursor_complete"
        ran = 0
        while True:
            if self._late():
                return self._pause(cur, "deadline")
            if ran >= self.ceiling:
                return self._pause(cur, "safety_ceiling")
            raw = job_plan(self.adapters, job, cur["offset"])[2][0]
            spec = cc.request_spec(raw["endpoint"], raw["params"], raw["estimated_credits"])
            if floor:
                self.request_floors[cc.fingerprint(spec["endpoint"], spec["params"])] = floor
            entry, payload, stop = self.fetcher.fetch(spec, extra, self.budget - self.spent)
            self.spent += entry["observed"] or 0
            if stop in ("unresolved_attempt", "accounting_unresolved"):
                self.unresolved = True
                cur["incomplete_reason"] = stop
                self._save()
                return None  # the claim stays for recover(); the cursor stays where it was
            if stop in (None, "overspend"):
                if (
                    floor and not stop and entry.get("retrieved_at")
                    and _ts(entry["retrieved_at"]) < _ts(floor)
                ):  # fmt: skip
                    return "hold", "stale_for_revision"
                bad = self._take_page(job, cur, entry, payload, spec)
                ran += 1
                self._save()
                if bad:
                    cur["incomplete_reason"] = bad
                    self._save()
                    return "hold", f"{bad} pages={len(cur['pages'])} rows={cur['rows_seen']}"
                if stop == "overspend":
                    self.halted = "coverage_halt"
                    return self._pause(cur, "coverage_halt")
                if cur["complete"]:
                    return "completed", f"pages={len(cur['pages'])} rows={cur['rows_seen']}"
                continue
            if stop in ("coverage_halt", "budget_hold"):
                self.halted = stop
            if stop == "cap_reached":
                return self._pause(cur, "headroom")
            cur["incomplete_reason"] = (
                "lane_hold" if stop in ("entitlement_unavailable", "lane_held") else stop
            )
            self._save()
            if stop in ("coverage_halt", "budget_hold", "entitlement_unavailable", "lane_held"):
                return "hold", stop
            return "failed", stop

    def _take_page(self, job, cur, entry, payload, spec):
        """Record one fetched page. -> None, or the exact reason the cursor must HOLD."""
        scope, op = job["scope"], job["op"]
        try:
            if op == "bulk":
                rows = cc._rows(payload)
                codes = [_row_code(r) for r in rows]
                following = _bulk_next(self.adapters, payload, entry, spec, scope, self.as_of)
            else:  # dividends: {data, meta, links}
                rows = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(rows, list):
                    raise ValueError("dividends page has no data list")
                codes = [_row_code(r) for r in rows]
                following = self.adapters.events.next_offset(payload)
        except (ValueError, TypeError, AttributeError, LookupError):
            return "malformed_page"
        sha, first = entry["sha256"], next((c for c in codes if c), None)
        if cur["pages"] and sha == cur["pages"][0]["sha256"]:
            return "ignored_offset"  # the provider served the first page again
        if any(p["sha256"] == sha for p in cur["pages"]):
            return "repeated_page"
        if first and any(p.get("first_code") == first for p in cur["pages"]):
            return "ignored_offset"
        if following is not None and following <= cur["offset"]:
            return "offset_not_advancing"
        registered = sum(1 for c in codes if c and self._ticker(c, scope) in self.registry)
        cur["pages"].append(
            {"offset": cur["offset"], "sha256": sha, "rows": len(rows), "first_code": first,
             "last_code": next((c for c in reversed(codes) if c), None), "registered": registered}
        )  # fmt: skip
        cur["rows_seen"] += len(rows)
        cur["registered_seen"] = sum(p.get("registered", 0) for p in cur["pages"])
        cur["updated_at"] = self.fetcher.now.isoformat()
        if following is None:
            cur["complete"], cur["incomplete_reason"] = True, None
        else:
            cur["offset"] = following
        return None

    @staticmethod
    def _ticker(code, scope):
        return code if "." in code else f"{code}.{scope}"


def _row_code(row):
    """The symbol code of a bulk / dividends row (General.Code, Code, code, symbol, ticker)."""
    if not isinstance(row, dict):
        return None
    general = row.get("General")
    for source in (general if isinstance(general, dict) else {}, row):
        for key in ("Code", "code", "symbol", "ticker"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _bulk_next(adapters, payload, entry, spec, exchange, as_of):
    """Next fundamentals offset after a FULL page, None after a short one (the only end proof).
    Uses coverage_bulk.next_offset on the retained bytes when the adapter has it."""
    limit = spec["params"].get("limit", FUNDAMENTALS_PAGE)
    offset = spec["params"].get("offset", 0)
    rows = cc._rows(payload)
    nxt = getattr(adapters.bulk, "next_offset", None)
    if nxt is not None:
        receipt = {
            "file": "page", "retrieved_at": entry["retrieved_at"], "as_of": str(as_of),
            "endpoint": spec["endpoint"], "parameters": dict(spec["params"]),
            "sha256": entry["sha256"],
        }  # fmt: skip
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return nxt(raw, receipt)
    if len(rows) > limit:
        raise ValueError("page exceeds its requested limit")
    return offset + limit if len(rows) == limit else None


def make_worker(config, adapters, fetcher, as_of, budget, universe=None, **kw):
    return JobWorker(config, adapters, fetcher, as_of, budget, universe, **kw)


def account_session_factory(config, state_root, job_date, request=None):
    """-> (factory, close): a guarded AccountProviderSession with ONLY the enabled coverage lanes
    open (held lanes are dropped by the guard itself). Requires accounting_mode account."""
    acquisition = config.get("acquisition") or {}
    if acquisition.get("accounting_mode") != "account":
        raise CoverageError("coverage job execution requires acquisition.accounting_mode=account")
    import requests

    from wharton import eodhd

    raw = []
    lanes = frozenset(
        lane for lane in cc.LANES if lane != "queue" and cc.lane_enabled(config, lane)
    )

    def factory():
        http = requests.Session()
        raw.append(http)
        return screen_budget.AccountProviderSession(
            http,
            job_date,
            acquisition.get("account_headroom", 25000),
            request or eodhd.request,
            [],
            capacity_path=Path(state_root) / "capacity-v2.json",
            config=acquisition,
            coverage_lanes=lanes,
        )

    return factory, lambda: [s.close() for s in raw]


def run_jobs(
    config,
    state_root,
    selected,
    run_lock,
    *,
    now,
    as_of,
    remaining_credits,
    session_factory=None,
    request=None,
    adapters=None,
    run_id=None,
    universe=None,
    deadline=None,
    registry=None,
):
    """Queue lane end to end, under an OWNED run lock: plan (20/80, fairness) -> claim ->
    guarded execution (cache-first, route_cost only, coverage_halt) -> ack. Mandatory coverage
    is reserved first: len(selected) fundamentals refreshes. Ticker-scoped jobs come back as
    `priority` {ticker: [job ids]} for acquire (still claimed; see finish_priority).
    `deadline` is the run's SHARED time.monotonic() deadline (also given to acquire): jobs and
    pagination stop at it and stay resumable. After the first round the calendar payloads fetched
    in THIS run are ingested and the ticker jobs they produce are claimed in a second round, so
    affected tickers are queued or executed in the same run. {'enabled': False} when the queue
    lane is off (nothing runs)."""
    queue = open_queue(config, state_root, run_lock)
    if queue is None:
        return {"enabled": False}
    adapters = adapters or get_adapters(config)
    costs = (config.get("acquisition") or {}).get("request_credit_cost", 0)
    fund = costs.get("fundamentals", 0) if isinstance(costs, dict) else costs
    mandatory = len(selected) * fund
    run_id = run_id or f"cov-{as_of}"
    limit = (config.get("coverage") or {}).get("max_priority_tickers", DEFAULT_MAX_PRIORITY)
    release_transient(  # work parked by a transient cause that is gone goes back in the queue
        config, queue, now, max(0, remaining_credits - mandatory), set(lane_records(state_root))
    )
    plan = plan_coverage(
        config,
        queue,
        now,
        remaining_credits,
        mandatory,
        run_id=run_id,
        rotation=len(selected),
        max_priority=limit,
    )
    if plan.get("blocked") or not plan["claimed"]:
        return {
            "enabled": True,
            "executed": 0,
            "priority": {},
            "floors": {},
            "request_floors": {},
            "run_id": run_id,
            "blocked": plan.get("blocked"),
        }
    close = lambda: None  # noqa: E731
    if session_factory is None:
        session_factory, close = account_session_factory(
            config, state_root, screening.day(as_of), request
        )
    try:
        fetcher = Fetcher(
            session_factory,
            request or _default_request(),
            state_root,
            run_lock,
            now,
            (config.get("coverage") or {}).get("freshness"),
        )
        budget = max(0, remaining_credits - mandatory)
        registry = registry if registry is not None else registry_rows(config)
        worker = make_worker(
            config, adapters, fetcher, as_of, budget, universe,
            cursors=load_cursors(state_root), registry=registry, deadline=deadline,
        )  # fmt: skip
        results = execute_claimed(queue, run_id, now, plan["claimed"], worker)
        if (
            cc.lane_enabled(config, "calendar")
            and adapters.events is not None
            and not worker.unresolved
        ):
            if ingest_events(queue, state_root, registry, as_of, adapters, now):
                more = plan_coverage(
                    config, queue, now, max(0, remaining_credits - worker.spent), mandatory,
                    run_id=run_id, rotation=len(selected), max_priority=limit,
                )  # fmt: skip
                if not more.get("blocked") and more["claimed"]:
                    results.update(execute_claimed(queue, run_id, now, more["claimed"], worker))
    finally:
        close()
    return {
        "enabled": True,
        "executed": sum(1 for r in results.values() if r is not None),
        "left_claimed": sum(1 for r in results.values() if r is None),
        "priority": worker.deferred,
        "floors": worker.floors,
        "request_floors": worker.request_floors,
        "run_id": run_id,
        "stats": plan["stats"],
        "pages": sum(len(c["pages"]) for c in worker.cursors["jobs"].values()),
        "spent": worker.spent,
    }


def _default_request():
    from wharton import eodhd

    return eodhd.request


# ---- 8. code receipt ---------------------------------------------------------------------


def code_modules(config):
    """Extra module names for the code receipt: () unless a coverage lane is enabled."""
    if not enabled(config):
        return ()
    here = Path(__file__).parent
    names = ["coverage_contracts", "coverage_queue", "coverage_runtime"]
    if (
        enabled(config, "bulk_market", "bulk_fundamentals")
        and (here / "coverage_bulk.py").is_file()
    ):
        names.append("coverage_bulk")
    if enabled(config, "calendar") and (here / "coverage_events.py").is_file():
        names.append("coverage_events")
    return tuple(names)


def checked_modules(names):
    """Validate a manifest-recorded module list (allowlist) before reading files by name."""
    names = tuple(names or ())
    if not set(names) <= set(COVERAGE_MODULES):
        raise ValueError("manifest lists a code module outside the coverage allowlist")
    return names
