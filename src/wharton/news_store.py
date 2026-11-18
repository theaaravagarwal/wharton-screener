"""Offline news store: normalize provider /news pages, dedup, lineage, cursors.

Pure stdlib, no network. Article text is untrusted data: sanitized, never fetched.
Layout under ``<root>/news/``: raw/<rid>.bin, receipts/<rid>.json,
articles/<aid>.json (all append-only), cursors.json (atomic replace).

``ingest`` and ``advance_cursor`` are read-modify-write: call them only while
holding ``news_lock`` (they are not safe to run concurrently).
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

PARSER_VERSION = "news-1"
# A short page (0 < n < limit) proves nothing until the provider's real page cap
# is measured; only an empty page at offset > 0 proves bounded exhaustion.
SHORT_PAGE_PROVES_END = False
_TITLE_LIMIT, _EXCERPT_LIMIT, _URL_LIMIT = 300, 200, 2048
_SYMBOL_LIMIT, _TAG_LIMIT, _LIST_MAX, _RAW_DATE_LIMIT, _KEY_LIMIT = 32, 64, 50, 64, 200
_RECEIPT_ID = re.compile(r"[0-9a-f]{32}")  # always fullmatch: `$` accepts a trailing newline
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# credential/tracking query params dropped from URLs; key/ref only when token-like
_DROP_PARAM = re.compile(
    r"^(utm_.*|gclid|fbclid|mc_.*|api_token|apikey|api_key|access_token|token|password|secret"
    r"|sig|signature|auth"
    r"|x-amz-signature|x-amz-credential|x-amz-security-token)$",
    re.I,
)
_MAYBE_TOKEN_PARAM = re.compile(r"^(key|ref)$", re.I)
_SECRET_PARAM_KEYS = {
    "api_token",
    "token",
    "apikey",
    "api_key",
    "access_token",
    "password",
    "secret",
}
_BLOCK = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_CREDENTIAL = re.compile(
    r"\b(api[_-]?token|access[_-]?token|token|api[_-]?key|apikey|password|secret|authorization)"
    r"\s*[=:]\s*(?:bearer\s+)?\S+",
    re.I,
)


class NewsPayloadError(ValueError):
    pass


def sha256_hex(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def atomic_write(path, data: bytes, mode: int = 0o600) -> None:
    path = Path(path)
    os.makedirs(path.parent, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def canonical_url(url) -> str | None:
    """http/https only. Drops fragment, default port, userinfo, tracking and
    credential params (``key``/``ref`` only when the value is >=16 chars, i.e.
    token-like); sorts the remaining query."""
    if not isinstance(url, str):
        return None
    url = url.strip()
    if (
        not url
        or len(url) > _URL_LIMIT
        or any(c.isspace() or unicodedata.category(c) == "Cc" for c in url)
    ):
        return None
    try:
        parts = urlsplit(url)
        scheme, host, port = parts.scheme.lower(), parts.hostname, parts.port
    except ValueError:
        return None
    if scheme not in ("http", "https") or not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is not None and port != (80 if scheme == "http" else 443):
        host = f"{host}:{port}"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _DROP_PARAM.match(k) and not (_MAYBE_TOKEN_PARAM.match(k) and len(v) >= 16)
    )
    out = f"{scheme}://{host}{parts.path or '/'}"
    return out + ("?" + urlencode(query) if query else "")


def sanitize_text(s, limit: int) -> str:
    """NFC; entities unescaped (<=3 passes) before well-formed tags are stripped,
    so ``&lt;b&gt;`` cannot survive as ``<b>``; literal ``a < b`` is kept. Control,
    bidi/zero-width and surrogate chars dropped; whitespace collapsed; credential
    pairs like ``api_token=...`` redacted; then truncated."""
    if not isinstance(s, str):
        return ""
    for _ in range(3):
        u = html.unescape(s)
        if u == s:
            break
        s = u
    for _ in range(3):  # stripping can splice a new tag together
        t = _TAG.sub(" ", _BLOCK.sub(" ", s))
        if t == s:
            break
        s = t
    s = unicodedata.normalize("NFC", s)
    out = []
    for c in s:
        cat = unicodedata.category(c)
        if c.isspace() or cat in ("Zl", "Zp"):
            out.append(" ")
        elif cat not in ("Cc", "Cf", "Cs"):  # control, bidi/zero-width (Cf), surrogates
            out.append(c)
    text = _CREDENTIAL.sub("[redacted]", " ".join("".join(out).split()))
    return text[:limit]


def utc(s) -> datetime:
    """Strict tz-aware ISO timestamp -> UTC datetime; ValueError otherwise."""
    if not isinstance(s, str) or len(s) > 64:
        raise ValueError(f"bad timestamp {s!r}")
    try:
        dt = datetime.fromisoformat(s.strip())
        if dt.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise ValueError(f"bad timestamp {s!r}") from None


_utc = utc  # transitional alias; import `utc`


def _iso(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return (
        dt.strftime("%Y-%m-%dT%H:%M:%S")
        + (f".{dt.microsecond:06d}" if dt.microsecond else "")
        + "Z"
    )


def _day(s) -> date:
    if not isinstance(s, str) or not _DATE.fullmatch(s):
        raise ValueError(f"bad date {s!r}; want YYYY-MM-DD")
    return date.fromisoformat(s)


def parse_published(raw) -> tuple[str | None, bool]:
    if not isinstance(raw, str):
        return None, False
    try:
        dt = datetime.fromisoformat(raw.strip())
        known = dt.tzinfo is not None
        return _iso(
            (dt if known else dt.replace(tzinfo=timezone.utc)).replace(microsecond=0)
        ), known
    except (ValueError, OverflowError):
        return None, False


def _title_norm(title: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", title).lower()).split())


def _key(a: dict) -> str:
    """Lineage key: canonical URL, else title+published (URL-less items)."""
    return (
        a["canonical_url"]
        or f"title:{a['title_norm_sha256']}|{a['published_at'] or a['published_at_raw']}"
    )


def _str_list(v, cap: int) -> list[str]:
    if not isinstance(v, list):
        return []
    out = [t for x in v if isinstance(x, str) and (t := sanitize_text(x, cap))]
    return out[:_LIST_MAX]


def _sentiment(v) -> dict | None:
    if not isinstance(v, dict):
        return None
    out = {}
    for k in ("polarity", "neg", "neu", "pos"):
        x = v.get(k)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            try:
                f = float(x)
            except OverflowError:
                continue
            if math.isfinite(f):
                out[k] = f
    return out or None


def _normalize_item(item: dict, position: int, receipt_id, first_seen_at: str) -> dict:
    title = sanitize_text(item.get("title"), _TITLE_LIMIT)
    content = item.get("content")
    content = unicodedata.normalize("NFC", content) if isinstance(content, str) else ""
    raw_date = sanitize_text(item.get("date"), _RAW_DATE_LIMIT)
    published, known = parse_published(raw_date)
    curl = canonical_url(item.get("link"))
    tnorm = _title_norm(title)
    symbols, tags = (
        _str_list(item.get("symbols"), _SYMBOL_LIMIT),
        _str_list(item.get("tags"), _TAG_LIMIT),
    )
    a = {
        "canonical_url": curl,
        "url_sha256": sha256_hex(curl or ""),
        "title": title,
        "title_norm_sha256": sha256_hex(tnorm),
        "content_sha256": sha256_hex(content),
        "excerpt": sanitize_text(content, _EXCERPT_LIMIT),
        "published_at_raw": raw_date,
        "published_at": published,
        "published_tz_known": known,
        "first_seen_at": first_seen_at,
        "published_after_first_seen": bool(published) and _utc(published) > _utc(first_seen_at),
        "symbols": symbols,
        "tags": tags,
        "sentiment": _sentiment(item.get("sentiment")),
        "receipt_id": receipt_id,
        "position": position,
        "supersedes": None,
        "revision": 0,
    }
    # observation identity: every semantic field, so a retag/retitle is a new row
    a["observation_id"] = sha256_hex(
        canonical_json(
            [
                _key(a),
                a["content_sha256"],
                a["title_norm_sha256"],
                raw_date,
                sorted(symbols),
                sorted(tags),
            ]
        )
    )
    a["article_id"] = a["observation_id"]
    # untitled or undated items never merge into a family: own family
    a["family_id"] = (
        sha256_hex(tnorm + "|" + published[:10]) if tnorm and published else a["observation_id"]
    )
    return a


def _parse(raw: bytes, receipt: dict, first_seen_at: str) -> tuple[int, list[dict]]:
    first_seen_at = _iso(_utc(first_seen_at))
    try:
        items = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise NewsPayloadError(f"undecodable news payload: {exc}") from exc
    if not isinstance(items, list):
        raise NewsPayloadError("news payload is not a list")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(_normalize_item(item, i, receipt.get("receipt_id"), first_seen_at))
        except Exception:  # one bad item never sinks the page
            continue
    return len(items), out


def normalize_page(raw: bytes, *, receipt: dict, first_seen_at: str) -> list[dict]:
    return _parse(raw, receipt, first_seen_at)[1]


def news_dir(root) -> Path:
    """``<root>/news``: the only directory the news store writes under."""
    return Path(root) / "news"


_news = news_dir  # transitional alias; import `news_dir`


def _load_articles(root) -> list[dict]:
    d = _news(root) / "articles"
    return [json.loads(p.read_text("utf-8")) for p in d.glob("*.json")] if d.is_dir() else []


_LINEAGE_FIELDS = ("article_id", "revision", "observation_id", "family_id")


def _count_articles(root) -> int:
    d = _news(root) / "articles"
    return sum(e.name.endswith(".json") for e in os.scandir(d)) if d.is_dir() else 0


def _write_index(root, latest: dict, n: int) -> None:
    atomic_write(
        _news(root) / "index.json", canonical_json({"version": 1, "n": n, "latest": latest})
    )


def _lineage(root) -> tuple[dict, bool, int]:
    """(lineage key -> latest article summary, rebuilt?, stored article count).

    ``news/index.json`` is only a cache: it is trusted iff it parses, has the
    expected shape and its article count equals the number of article files
    (listing names, no reads). Anything else (missing, corrupt, stale after a
    crash between article and index writes) is rebuilt from the articles."""
    n = _count_articles(root)
    try:
        doc = json.loads((_news(root) / "index.json").read_text("utf-8"))
        latest = doc["latest"]
        if (
            doc["version"] == 1
            and doc["n"] == n
            and isinstance(latest, dict)
            and all(
                isinstance(v, dict) and all(f in v for f in _LINEAGE_FIELDS)
                for v in latest.values()
            )
        ):
            return latest, False, n
    except (OSError, ValueError, KeyError, TypeError):
        pass
    latest = {}
    for a in _load_articles(root):
        k = _key(a)
        if k not in latest or a["revision"] > latest[k]["revision"]:
            latest[k] = {f: a[f] for f in _LINEAGE_FIELDS}
    return latest, True, n


def _safe_receipt(receipt: dict) -> dict:
    params = receipt.get("params")
    if isinstance(params, dict):
        receipt = {
            **receipt,
            "params": {k: v for k, v in params.items() if str(k).lower() not in _SECRET_PARAM_KEYS},
        }
    return receipt


def ingest(root, raw: bytes, receipt: dict, *, first_seen_at: str) -> dict:
    """Store raw bytes, receipt and articles; return counts.

    ``items`` raw items on the page, ``skipped`` malformed items dropped,
    ``new`` unseen key, ``revised`` known canonical key with any difference from
    its LATEST observation (new row, supersedes it, revision+1; so A->B->A is
    two revisions), ``duplicate`` identical to the latest observation.

    Raw + receipt are written before parsing; an undecodable/non-list payload,
    or a non-empty page whose every item was skipped (so it can never read as
    items-complete), keeps both (receipt outcome forced to "error") and then
    raises NewsPayloadError. Raises ValueError (nothing written) for a receipt_id that
    is not 32 hex chars, a response_sha256 that is not sha256(raw), a receipt
    already stored with a different response_sha256, or an unparseable
    first_seen_at. api_token-like keys are stripped from stored receipt params.

    Not concurrency-safe: call while holding news_lock.
    """
    rid, sha = receipt.get("receipt_id"), sha256_hex(raw)
    if not isinstance(rid, str) or not _RECEIPT_ID.fullmatch(rid):
        raise ValueError("receipt_id must be 32 lowercase hex chars")
    if receipt.get("response_sha256") != sha:
        raise ValueError("receipt response_sha256 does not match raw bytes")
    first_seen_at = _iso(_utc(first_seen_at))
    base = _news(root)
    rpath, wpath = base / "receipts" / f"{rid}.json", base / "raw" / f"{rid}.bin"
    receipt_known = rpath.exists()
    if receipt_known and json.loads(rpath.read_text("utf-8")).get("response_sha256") != sha:
        raise ValueError("receipt already stored with different response_sha256")
    if wpath.exists() and sha256_hex(wpath.read_bytes()) != sha:
        raise ValueError("raw already stored with different bytes")
    for sub in ("", "raw", "receipts", "articles"):
        os.makedirs(base / sub, mode=0o700, exist_ok=True)
    if not wpath.exists():
        atomic_write(wpath, raw)
    stored = _safe_receipt(receipt)
    try:
        n_items, articles = _parse(raw, receipt, first_seen_at)
        if n_items and not articles:
            raise NewsPayloadError(f"all {n_items} items on the page are malformed")
    except NewsPayloadError:
        if not receipt_known:
            atomic_write(rpath, canonical_json({**stored, "outcome": "error"}))
        raise
    if not receipt_known:
        atomic_write(rpath, canonical_json(stored))
    latest, rebuilt, n_stored = _lineage(root)
    written = 0
    counts = {
        "new": 0,
        "duplicate": 0,
        "revised": 0,
        "skipped": n_items - len(articles),
        "items": n_items,
    }
    for a in articles:
        prior = latest.get(_key(a))
        if prior and prior["observation_id"] == a["observation_id"]:
            counts["duplicate"] += 1
            continue
        if prior:
            a["supersedes"], a["revision"] = prior["article_id"], prior["revision"] + 1
            a["article_id"] = sha256_hex(
                f"{a['observation_id']}\n{a['revision']}"
            )  # distinct id for reverts
            if (
                a["family_id"] == a["observation_id"]
            ):  # own-family items keep their family across revisions
                a["family_id"] = prior["family_id"]
            counts["revised"] += 1
        else:
            counts["new"] += 1
        atomic_write(base / "articles" / f"{a['article_id']}.json", canonical_json(a))
        latest[_key(a)] = {f: a[f] for f in _LINEAGE_FIELDS}
        written += 1
    if written or rebuilt:
        _write_index(root, latest, written + n_stored)
    return counts


def articles_as_of(root, cutoff_iso: str) -> list[dict]:
    """Lookahead guard: only articles first seen at or before the cutoff.
    Timestamps are parsed as UTC datetimes (never compared as strings);
    ValueError on an unparseable cutoff or stored first_seen_at."""
    cutoff = _utc(cutoff_iso)
    rows = [(_utc(a["first_seen_at"]), a) for a in _load_articles(root)]
    return [
        a
        for dt, a in sorted(
            ((d, a) for d, a in rows if d <= cutoff), key=lambda r: (r[0], r[1]["article_id"])
        )
    ]


def families(articles) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in articles:
        out.setdefault(a["family_id"], []).append(a["article_id"])
    return out


def _cursors(root) -> dict:
    p = _news(root) / "cursors.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _check_key(query_key) -> str:
    if (
        not isinstance(query_key, str)
        or not query_key
        or len(query_key) > _KEY_LIMIT
        or any(unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in query_key)
    ):
        raise ValueError(
            "query_key must be a non-empty string <=200 chars without control characters"
        )
    return query_key


def load_cursor(root, query_key: str) -> dict | None:
    return _cursors(root).get(_check_key(query_key))


def advance_cursor(
    root,
    query_key: str,
    *,
    complete_to: str,
    pages_done: int,
    exhausted: bool,
    truncated: bool,
    today: str | None = None,
) -> dict:
    """Move ``complete_to`` forward only on an exhausted, untruncated run.

    ``complete_to`` must be YYYY-MM-DD and not after ``today`` (default: today
    UTC); ValueError otherwise, so a bad value can never poison the cursor.
    Never moves backward. Not concurrency-safe: call while holding news_lock.
    """
    _check_key(query_key)
    day = _day(complete_to)
    if day > (_day(today) if today else datetime.now(timezone.utc).date()):
        raise ValueError(f"complete_to {complete_to} is in the future")
    cursors = _cursors(root)
    cur = dict(cursors.get(query_key) or {"complete_to": None})
    if exhausted and not truncated:
        old = cur["complete_to"]
        cur["complete_to"] = max(day, _day(old)).isoformat() if old else day.isoformat()
    cur.update(pages_done=pages_done, exhausted=exhausted, truncated=truncated)
    cursors[query_key] = cur
    os.makedirs(_news(root), mode=0o700, exist_ok=True)
    atomic_write(_news(root) / "cursors.json", canonical_json(cursors))
    return cur


def next_window(
    root, query_key: str, today: str, *, overlap_days: int = 1, default_lookback_days: int = 2
) -> tuple[str, str]:
    """Inclusive (from, to) dates; overlap re-reads late arrivals behind the cursor."""
    t = _day(today)
    cur = load_cursor(root, query_key)
    if cur and cur.get("complete_to"):
        start = _day(cur["complete_to"]) - timedelta(days=overlap_days)
    else:
        start = t - timedelta(days=default_lookback_days)
    return min(start, t).isoformat(), t.isoformat()


def page_outcome(n_items: int, limit: int, offset: int) -> str:
    """Shape of a successfully fetched page: 'empty' (n==0, offset==0);
    'complete' only for an empty page at offset>0 (or a short page when
    SHORT_PAGE_PROVES_END); otherwise 'partial' (more may exist). Failed calls
    never reach here: the caller records an error receipt instead."""
    if n_items == 0 and offset == 0:
        return "empty"
    if (n_items == 0 and offset > 0) or (SHORT_PAGE_PROVES_END and n_items < limit):
        return "complete"
    return "partial"
