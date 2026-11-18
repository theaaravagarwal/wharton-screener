"""Durable evidence-completion queue: one atomic JSON sidecar, stdlib only.

File: state_root/coverage-queue.json = {version, jobs{job_id: job}, seen{job_id: revision},
claims{job_id: {run_id, claimed_at}}, age_carry, pruned{jobs, seen}}. Every mutation rewrites
the WHOLE document through one unique temp file (mkstemp in the same dir) + fsync + os.replace
(+ best-effort directory fsync), so jobs and seen can never diverge and a crash leaves either
the old or the new document. A corrupt file raises QueueCorrupt; it is never silently reset.
Leftover temp files from a dead process are swept on the next write and never renamed in.

Concurrency: all mutations of any CoverageQueue in this process serialise on one module-level
RLock (load-modify-write is atomic across threads). Across processes the shared state_root/.lock
flock held by run() is the only guard: `lock_check` is REQUIRED and must return true while the
caller holds it; every mutator calls it first and raises QueueLockError otherwise.
summary()/select() are read-only and need no lock. This module never writes capacity-v2.json;
recover() only reads its `pending` field.

Seen map: keys are JOB IDS and values are the (aggregate) revision recorded for that job. enqueue
refuses a seen key that is not a job id in the document/call, and a revision that the job does
not hold as `revision` or `pending_revision`: a seen revision always has durable work behind it.
Revisions are never dropped: one arriving for a claimed job is stored as top-level
`pending_revision` and survives ack() and recover() as a requeue at that revision.

Job meta read here (all optional): region, cap_bucket (fairness strata), priority (number,
higher first), event_at (ISO, newer first), cursor (ISO last-seen, older first; missing = never
seen), known_fail + new_evidence (a known failure is eligible only with new evidence).
Reasons: "event" and "gap" form the 80% pool, "age" the 20% pool.

Age starvation: the document keeps `age_carry`, the unspent age entitlement of earlier runs. An
age job dearer than one run's 20% pool becomes affordable once carry + pool_20 covers it; the
extra comes out of that run's event/gap pool. select() stays read-only and returns the new carry
in its stats; settle(stats) persists it.

Recovery: an un-acked claim whose paid call may already have succeeded is requeued like any
stale claim; the cache-first fetch path (not this queue) is what prevents a second charge.
"""

import json
import os
import re
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone

from wharton import coverage_contracts as cc

QUEUE_VERSION = 2  # 2: seen keys are job ids; adds age_carry, settle_seq, pruned
QUEUE_FILE = "coverage-queue.json"
_TMP_PREFIX = ".%s." % QUEUE_FILE
_TMP_SUFFIX = ".tmp"
_TIER_RANK = {"event": 0, "gap": 1, "age": 2}
_REQUIRED = frozenset(
    {"job_id", "kind", "op", "scope", "window", "reason", "estimated_credits", "status"}
)
_MUTEX = threading.RLock()  # ponytail: one lock for all queues in the process; fine for one owner
_UNSET = object()


class QueueCorrupt(ValueError):
    """The sidecar exists but cannot be trusted; refusing to continue (no silent reset)."""


class QueueLockError(RuntimeError):
    """A mutator was called without the shared run lock."""


def _ts(value):
    """ISO date/datetime -> epoch seconds (naive = UTC). None -> 0.0; unparsable -> QueueCorrupt."""
    if value is None:
        return 0.0
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise QueueCorrupt("unparsable timestamp %r" % (value,)) from exc
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def _iso(now):
    return now.isoformat() if isinstance(now, datetime) else str(now)


# the contract matcher wants "bearer: x"; header-style text needs the space form too
_HEADER_TEXT = re.compile(r"(?i)\bbearer\s+[\w.~+/=-]{8,}|\bauthorization\s*[:=]\s*\S+")


def _credential_text(text):
    return cc.contains_credential(text) or bool(_HEADER_TEXT.search(text))


def _scan_keys(obj):
    """No credential-like or holding-payload key, and no credential text in any string value,
    anywhere inside meta structures."""
    if isinstance(obj, str):
        if _credential_text(obj):
            raise ValueError("job meta must not carry credential text")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if cc.credential_key(key) or _credential_text(str(key)):
                raise ValueError("job meta must not carry credentials or holding payloads")
            _scan_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            _scan_keys(value)


def _meta(job):
    return job.get("meta") or {}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _tier(job):
    return job["reason"] if job["reason"] in _TIER_RANK else "gap"  # ponytail: unknown reason = gap


def _key(job):
    """Deterministic priority inside a tier; job_id is the final tiebreak."""
    meta = _meta(job)
    if _tier(job) == "age":
        return (_ts(meta.get("cursor")), job["job_id"])  # oldest / never seen first
    return (-_num(meta.get("priority")), -_ts(meta.get("event_at")), job["job_id"])


def _stratum(job):
    meta = _meta(job)
    return (meta.get("region"), meta.get("cap_bucket"))


def _norm(stratum):
    parts = stratum if isinstance(stratum, tuple) else (stratum,)
    return tuple("" if p is None else str(p) for p in parts)


def _round_robin(jobs, budget, stratum_key, rotation):
    """One job per stratum per round (best first), skipping jobs that do not fit `budget`.
    Returns (picked, budget_left)."""
    strata = {}
    for job in sorted(jobs, key=_key):
        strata.setdefault(_norm(stratum_key(job)), deque()).append(job)
    names = sorted(strata)
    if names:
        shift = rotation % len(names)
        names = names[shift:] + names[:shift]
    picked = []
    while any(strata[n] for n in names):
        for name in names:
            queue = strata[name]
            while queue:
                job = queue.popleft()
                if job["estimated_credits"] <= budget:
                    picked.append(job)
                    budget -= job["estimated_credits"]
                    break
    return picked, budget


def _empty():
    return {
        "version": QUEUE_VERSION,
        "jobs": {},
        "seen": {},
        "claims": {},
        "age_carry": 0,
        "settle_seq": 0,  # monotonic: makes a replayed (ABA) stats object detectable
        "pruned": {"jobs": 0, "seen": 0, "failed": 0},
    }


class CoverageQueue:
    def __init__(self, state_root, lock_check):
        if not callable(lock_check):
            raise TypeError("lock_check is required: a callable true while the run lock is held")
        self.path = os.path.join(os.fspath(state_root), QUEUE_FILE)
        self._lock_check = lock_check

    # ---- storage ---------------------------------------------------------------------------

    def _load(self):
        try:
            with open(self.path, "rb") as handle:
                doc = json.loads(handle.read().decode("utf-8"))
        except FileNotFoundError:
            return _empty()
        except (OSError, ValueError) as exc:  # includes JSONDecodeError / UnicodeDecodeError
            raise QueueCorrupt("unreadable queue file %s: %s" % (self.path, exc)) from exc
        self._check(doc)
        return doc

    @staticmethod
    def _check(doc):
        """Cheap structural check (no per-job hashing). Full validate_job runs on every job a
        mutation adds or changes. ponytail: a hand-edited job body that keeps a valid shape is
        not re-hashed on load; add a periodic full audit if hand edits become a concern."""
        try:
            if not isinstance(doc, dict):
                raise ValueError("document is not an object")
            if doc.get("version") != QUEUE_VERSION:
                raise QueueCorrupt(
                    "queue sidecar has format version %r, expected %r (old format). Migration: "
                    "stop the runner, move coverage-queue.json aside as audit evidence and start "
                    "from an empty queue; it is rebuilt by the next event/calendar refresh. Seen "
                    "keys changed from event ids to job ids, so old entries cannot be carried "
                    "over by hand." % (doc.get("version"), QUEUE_VERSION)
                )
            if set(doc) != set(_empty()):
                raise ValueError("document fields")
            if not all(isinstance(doc[k], dict) for k in ("jobs", "seen", "claims", "pruned")):
                raise ValueError("jobs/seen/claims/pruned must be objects")
            carry = doc["age_carry"]
            if isinstance(carry, bool) or not isinstance(carry, int) or carry < 0:
                raise ValueError("age_carry must be a nonnegative integer")
            seq = doc["settle_seq"]
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
                raise ValueError("settle_seq must be a nonnegative integer")
            if set(doc["pruned"]) != {"jobs", "seen", "failed"} or not all(
                isinstance(v, int) and v >= 0 for v in doc["pruned"].values()
            ):
                raise ValueError("pruned counts")
            statuses = cc.JOB_STATUSES
            claimed = set()
            for job_id, job in doc["jobs"].items():
                if (
                    not isinstance(job, dict)
                    or not _REQUIRED.issubset(job)
                    or job["job_id"] != job_id
                    or job["status"] not in statuses
                    or not isinstance(job["reason"], str)
                    or isinstance(job["estimated_credits"], bool)
                    or not isinstance(job["estimated_credits"], int)
                ):
                    raise ValueError("malformed job %s" % job_id)
                if job["status"] == "claimed":
                    claimed.add(job_id)
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in doc["seen"].items()):
                raise ValueError("seen must map job ids to revision strings")
            if not set(doc["seen"]) <= set(doc["jobs"]):
                raise ValueError("seen keys must be job ids of existing jobs")
            if claimed != set(doc["claims"]):
                raise ValueError("claims do not match claimed jobs")
            for job_id, claim in doc["claims"].items():
                if not (
                    isinstance(claim, dict)
                    and set(claim) == {"run_id", "claimed_at"}
                    and isinstance(claim["run_id"], str)
                    and isinstance(claim["claimed_at"], str)
                ):
                    raise ValueError("malformed claim %s" % job_id)
                _ts(claim["claimed_at"])
        except QueueCorrupt:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise QueueCorrupt("invalid queue document: %s" % (exc,)) from exc

    def _write(self, doc):
        """Unique temp file in the same dir, flush + fsync, atomic replace. The directory fsync
        afterwards is best-effort: once os.replace returned the new document IS committed."""
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(folder):  # temp files of a dead writer: sweep, never adopt
            if name.startswith(_TMP_PREFIX) and name.endswith(_TMP_SUFFIX):
                try:
                    os.unlink(os.path.join(folder, name))
                except OSError:
                    pass
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(cc.canonical(doc) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        try:
            dir_fd = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    def _require_lock(self):
        if not self._lock_check():
            raise QueueLockError("coverage queue mutation requires the shared run lock")

    def _mutate(self, change):
        """lock check -> load -> change(doc) -> (result, changed) -> one atomic write if changed.
        Serialised process-wide so concurrent load-modify-write cycles cannot lose updates."""
        with _MUTEX:
            self._require_lock()
            doc = self._load()
            result, changed = change(doc)
            if changed:
                self._write(doc)
            return result

    # ---- mutators --------------------------------------------------------------------------

    def enqueue(self, jobs, seen_updates=None, now=None, observed_at=None):
        """Apply jobs AND seen updates ({job_id: revision}) in ONE replace. Existing ids go
        through coverage_contracts.reopen (same revision = no-op; queued takes the new revision
        in place; claimed stores pending_revision; finished requeues). A seen key must be a job
        id in the document or this call and the job must hold the revision (as revision or
        pending_revision); otherwise the call raises before anything is written.

        Ordering: with no observed_at, the contract's reopen semantics apply and the LATEST
        ARRIVAL WINS: a replay of an earlier digest is indistinguishable from a new arrival and
        is applied as one (a revision equal to the current one is a no-op). Pass observed_at (an
        ISO time of the source observation) to make arrivals ordered: it is stored in the job's
        meta, and an arrival older than the job's stored observed_at is ignored together with
        its seen update. Returns the job ids ignored for that reason."""
        jobs, seen_updates = list(jobs), dict(seen_updates or {})
        if observed_at is not None:
            observed_at = _iso(observed_at)
            _ts(observed_at)  # reject an unparsable ordering stamp up front
        for job in jobs:
            cc.validate_job(job)
            _scan_keys(job.get("meta"))
            if job["status"] not in ("queued", "hold"):
                raise ValueError("only queued or hold jobs can be enqueued")
        for job_id, revision in seen_updates.items():
            if not (isinstance(job_id, str) and isinstance(revision, str) and revision):
                raise ValueError("seen updates map job ids to non-empty revision strings")

        def change(doc):
            changed, ignored = False, []
            stamp = {"updated_at": _iso(now)} if now is not None else {}
            obs = {"observed_at": observed_at} if observed_at is not None else {}
            for job in jobs:
                have = doc["jobs"].get(job["job_id"])
                if have is None:
                    meta = {**_meta(job), **obs}
                    fresh = {**job, **({"meta": meta} if meta else {})}
                    doc["jobs"][job["job_id"]] = {**fresh, **stamp}
                    changed = True
                    continue
                revision = job.get("revision")
                if revision is None:
                    continue
                last = _meta(have).get("observed_at")
                if obs and last is not None and _ts(observed_at) < _ts(last):
                    ignored.append(job["job_id"])  # an older observation: never a downgrade
                    continue
                new = cc.reopen(have, revision)
                if new == have:
                    continue
                if have["status"] == "queued":  # refreshed in place: newest facts win
                    new = {
                        **new,
                        **{k: job[k] for k in ("reason", "estimated_credits", "meta") if k in job},
                    }
                if obs:
                    new = {**new, "meta": {**_meta(new), **obs}}
                doc["jobs"][job["job_id"]] = cc.validate_job({**new, **stamp})
                changed = True
            for job_id, revision in seen_updates.items():
                if job_id in ignored:
                    continue
                have = doc["jobs"].get(job_id)
                if have is None:
                    raise ValueError("seen key %s is not a job id in the queue" % job_id)
                if revision not in (have.get("revision"), have.get("pending_revision")):
                    raise ValueError("job %s does not hold seen revision %s" % (job_id, revision))
                if doc["seen"].get(job_id) != revision:
                    doc["seen"][job_id] = revision
                    changed = True
            return ignored, changed

        return self._mutate(change)

    def claim(self, job_ids, run_id, now):
        """Claim the given jobs in ONE write. Ids are deduplicated (first occurrence order); an
        unknown id raises; ids that are not queued are skipped; a known_fail job without
        meta.new_evidence raises. Returns the ids actually claimed."""
        ids = list(dict.fromkeys(job_ids))

        def change(doc):
            took = []
            for job_id in ids:
                job = doc["jobs"].get(job_id)
                if job is None:
                    raise KeyError("unknown job id %s" % job_id)
                if job["status"] != "queued":
                    continue
                if _meta(job).get("known_fail") and _meta(job).get("new_evidence") is not True:
                    raise ValueError("job %s is a known failure without new evidence" % job_id)
                job.update(
                    status="claimed", attempts=job.get("attempts", 0) + 1, updated_at=_iso(now)
                )
                doc["claims"][job_id] = {"run_id": run_id, "claimed_at": _iso(now)}
                took.append(job_id)
            return took, bool(took)

        return self._mutate(change)

    def _finish_many(self, items, run_id, now, status, text=None):
        """items: job ids or (job_id, text) pairs, one load and one write. text is the detail
        (completed/failed) or the hold_reason (hold). All or nothing."""
        pairs = dict((i, None) if isinstance(i, str) else tuple(i) for i in items)

        def change(doc):
            for job_id, own in pairs.items():
                note = text if own is None else own
                job = doc["jobs"].get(job_id)
                if job is None:
                    raise KeyError(job_id)
                if job["status"] == "claimed":
                    if run_id is None or doc["claims"][job_id]["run_id"] != run_id:
                        raise ValueError("job %s is claimed by another run" % job_id)
                    del doc["claims"][job_id]
                    # ack requeues instead of finishing when a revision arrived while claimed
                    job = {**cc.ack(job, status, note if status == "hold" else None)}
                    if job["status"] == status and status != "hold" and note is not None:
                        job["detail"] = str(note)
                elif status == "hold" and job["status"] == "queued":
                    job = {**job, "status": "hold", "hold_reason": str(note)}
                else:
                    raise ValueError("job %s is not claimed" % job_id)
                if status == "hold" and not note:
                    raise ValueError("hold needs a reason")
                job["updated_at"] = _iso(now)
                doc["jobs"][job_id] = cc.validate_job(job)
            return None, bool(pairs)

        self._mutate(change)

    def complete_many(self, items, run_id, now, detail=None):
        self._finish_many(items, run_id, now, "completed", detail)

    def fail_many(self, items, run_id, now, detail=None):
        self._finish_many(items, run_id, now, "failed", detail)

    def hold_many(self, items, run_id, now, hold_reason=None):
        """Hold claimed jobs (run_id must match) or queued ones (run_id ignored)."""
        self._finish_many(items, run_id, now, "hold", hold_reason)

    def complete(self, job_id, run_id, now, detail=None):
        self.complete_many([job_id], run_id, now, detail)

    def fail(self, job_id, run_id, now, detail=None):
        self.fail_many([job_id], run_id, now, detail)

    def hold(self, job_id, run_id, now, hold_reason):
        self.hold_many([job_id], run_id, now, hold_reason)

    def release(self, job_id, now):
        """Operator/budget release: hold -> queued. The hold stays visible until then."""

        def change(doc):
            job = doc["jobs"][job_id]
            if job["status"] != "hold":
                raise ValueError("job %s is not on hold" % job_id)
            job = {k: v for k, v in job.items() if k != "hold_reason"}
            doc["jobs"][job_id] = cc.validate_job(
                {**job, "status": "queued", "updated_at": _iso(now)}
            )
            return None, True

        self._mutate(change)

    def settle(self, stats):
        """Persist the age carry a select() proposed. Raises on stats from a stale selection
        (age_carry or the monotonic settle_seq moved, so an ABA replay is caught) and on a carry
        outside [0, min(4 * pool_20, age_carry + pool_20)]. ponytail: stats are in-process data,
        so pool_20 is trusted; forging both numbers is deliberate misuse, not an accident."""

        def change(doc):
            if doc["age_carry"] != stats["age_carry"] or doc["settle_seq"] != stats["settle_seq"]:
                raise ValueError("stale selection stats: queue settled since select()")
            carry, pool = stats["age_carry_next"], stats["pool_20"]
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (carry, pool)):
                raise ValueError("age_carry_next and pool_20 must be nonnegative integers")
            if carry > min(4 * pool, doc["age_carry"] + pool):
                raise ValueError("age_carry_next %d exceeds the allowed carry" % carry)
            doc["age_carry"] = carry
            doc["settle_seq"] += 1
            return carry, True

        return self._mutate(change)

    def prune(self, now, keep_seconds, max_failed_age=None):
        """Drop COMPLETED jobs last updated more than keep_seconds ago, with their seen entries.
        Never touches queued/claimed/hold jobs, and never a job whose seen revision is not the
        one it finished at. FAILED jobs are kept forever unless max_failed_age (seconds) is
        given; then those older than it are dropped too and counted in summary()["pruned"]
        ["failed"] so the loss stays visible. Re-arrival of a pruned revision is treated as
        unseen, so keep the windows longer than the event lane's replay horizon."""
        cutoff = _ts(now) - keep_seconds
        failed_cutoff = None if max_failed_age is None else _ts(now) - max_failed_age

        def old(job, limit):
            return limit is not None and "updated_at" in job and _ts(job["updated_at"]) < limit

        def change(doc):
            gone = [
                i
                for i, j in doc["jobs"].items()
                if (
                    (j["status"] == "completed" and old(j, cutoff))
                    or (j["status"] == "failed" and old(j, failed_cutoff))
                )
                and doc["seen"].get(i, j.get("revision")) == j.get("revision")
            ]
            seen_gone = failed_gone = 0
            for job_id in gone:
                failed_gone += doc["jobs"][job_id]["status"] == "failed"
                del doc["jobs"][job_id]
                seen_gone += doc["seen"].pop(job_id, None) is not None
            doc["pruned"]["jobs"] += len(gone)
            doc["pruned"]["seen"] += seen_gone
            doc["pruned"]["failed"] += failed_gone
            result = {"jobs": len(gone), "seen": seen_gone, "failed": failed_gone}
            return result, bool(gone)

        return self._mutate(change)

    def recover(
        self,
        run_id,
        now,
        capacity_state_path=None,
        pending=_UNSET,
        max_age=3600,
        max_attempts=5,
        allow_missing_capacity=False,
    ):
        """Requeue claims older than max_age seconds, ONLY when the provider ledger has no
        unresolved `pending`. Reads capacity-v2.json (never writes it) unless `pending` is
        passed explicitly (None = resolved). Blocked -> {"blocked": "provider_pending"}; a
        missing ledger fails closed ({"blocked": "capacity_state_missing"}) unless
        allow_missing_capacity=True. A stale claim already tried max_attempts times becomes a
        hold with reason "max_attempts" (poison job) unless a newer revision is pending."""
        self._require_lock()
        if capacity_state_path is not None:
            try:
                with open(capacity_state_path, "rb") as handle:
                    ledger = json.loads(handle.read().decode("utf-8"))
                if not isinstance(ledger, dict):
                    raise ValueError("capacity state is not an object")
                if "pending" not in ledger:  # unknown is not resolved: fail closed
                    return {"blocked": "capacity_pending_unknown"}
                pending = ledger["pending"]
            except FileNotFoundError:
                if not allow_missing_capacity:
                    return {"blocked": "capacity_state_missing"}
                pending = None
            except (OSError, ValueError, AttributeError, TypeError) as exc:
                raise QueueCorrupt("unreadable capacity state: %s" % (exc,)) from exc
        elif pending is _UNSET:
            raise ValueError("recover needs capacity_state_path or an explicit pending")
        if pending is not None:
            return {"blocked": "provider_pending"}
        cutoff = _ts(now) - max_age

        def change(doc):
            stale = sorted(i for i, c in doc["claims"].items() if _ts(c["claimed_at"]) <= cutoff)
            held = []
            for job_id in stale:
                del doc["claims"][job_id]
                old = doc["jobs"][job_id]
                job = {k: v for k, v in old.items() if k != "pending_revision"}
                if old.get("pending_revision"):  # a deferred revision is new work
                    job.update(status="queued", revision=old["pending_revision"], attempts=0)
                elif job.get("attempts", 0) >= max_attempts:
                    job.update(status="hold", hold_reason="max_attempts")
                    held.append(job_id)
                else:
                    job.update(status="queued")
                job["updated_at"] = _iso(now)
                doc["jobs"][job_id] = cc.validate_job(job)
            return {
                "requeued": [i for i in stale if i not in held],
                "held": held,
                "run_id": run_id,
            }, bool(stale)

        return self._mutate(change)

    # ---- read-only -------------------------------------------------------------------------

    def select(self, budget_credits, mandatory_credits, region_cap_key=None, rotation=0):
        """Pick queued jobs for the credits left after mandatory coverage + priority news.
        R = budget - mandatory; 20% of R (plus any age carry) is the age pool, the rest the
        event/gap pool, and a pool's unused credits transfer to the other. Never exceeds R.
        Non-selected jobs stay queued. Returns (ordered job ids, stats); stats["age_carry_next"]
        is what settle() would persist. ponytail: the carry follows what was SELECTED, not what
        finally ran; settle after the run if exact accounting matters."""
        stratum_key = region_cap_key or _stratum
        doc = self._load()
        queued = [j for j in doc["jobs"].values() if j["status"] == "queued"]
        eligible = [
            j
            for j in queued
            if not _meta(j).get("known_fail") or _meta(j).get("new_evidence") is True
        ]
        remaining = max(0, budget_credits - mandatory_credits)
        pool_20 = remaining * 20 // 100
        pool_80 = remaining - pool_20
        carry, cap = doc["age_carry"], 4 * pool_20
        avail = pool_20 + carry
        taken = set()

        def draw(tiers, budget):
            picked = []
            for tier in tiers:  # strict tier order, round-robin across strata inside a tier
                pool = [j for j in eligible if _tier(j) == tier and j["job_id"] not in taken]
                got, budget = _round_robin(pool, budget, stratum_key, rotation)
                taken.update(j["job_id"] for j in got)
                picked += got
            return picked, budget

        # Head-of-line age job: the oldest-cursor age job that could ever be funded (within
        # pool_20 + the carry cap). If it is dearer than one run's pool_20 it is served only
        # from carry, so younger/cheaper age jobs must not eat that carry or the pool_20 that
        # banks toward it: they get nothing until the head is affordable, then only pool_20.
        ages = [
            j for j in eligible if _tier(j) == "age" and j["estimated_credits"] <= pool_20 + cap
        ]
        head = min(ages, key=_key, default=None)
        head_picked, age_budget, banked = [], min(remaining, avail), 0
        if head is not None and head["estimated_credits"] > pool_20:
            cost = head["estimated_credits"]
            if cost <= min(remaining, avail):
                head_picked = [head]
                taken.add(head["job_id"])
                age_budget = min(avail - cost, pool_20, remaining - cost)
            else:
                age_budget, banked = 0, pool_20  # bank: pool_20 is held back, not spent
        event_budget = (
            remaining - banked - age_budget - sum(j["estimated_credits"] for j in head_picked)
        )

        main, left_events = draw(("event", "gap"), event_budget)
        aged, left_age = draw(("age",), age_budget)
        # unused capacity crosses over: leftover age budget -> events/gaps, leftover events -> age
        extra_main, _ = draw(("event", "gap"), left_age)
        extra_aged, _ = draw(("age",), left_events)
        extra = extra_main + extra_aged
        aged = head_picked + aged
        chosen = main + aged + extra
        # the carry is charged only for age work paid from the age budget, not for transfers
        age_spent = sum(j["estimated_credits"] for j in aged)
        stats = {
            "pool_20": pool_20,
            "pool_80": pool_80,
            "transferred": sum(j["estimated_credits"] for j in extra),
            "deferred": len(queued) - len(chosen),
            "spent": sum(j["estimated_credits"] for j in chosen),
            "age_carry": carry,
            "settle_seq": doc["settle_seq"],
            "age_budget": age_budget + sum(j["estimated_credits"] for j in head_picked),
            "age_spent": age_spent,
            "age_carry_next": min(cap, max(0, avail - age_spent)),
        }
        return [j["job_id"] for j in chosen], stats

    def summary(self, now=None):
        doc = self._load()
        out = {
            "total": len(doc["jobs"]),
            "seen": len(doc["seen"]),
            "age_carry": doc["age_carry"],
            "pruned": dict(doc["pruned"]),
            "settle_seq": doc["settle_seq"],
        }
        for field in ("kind", "op", "status", "reason"):
            counts = {}
            for job in doc["jobs"].values():
                counts[job[field]] = counts.get(job[field], 0) + 1
            out["by_" + field] = dict(sorted(counts.items()))
        held = {}
        for job in doc["jobs"].values():
            if job["status"] == "hold":
                reason = job.get("hold_reason", "")
                held[reason] = held.get(reason, 0) + 1
        out["held_by_reason"] = dict(sorted(held.items()))
        stamps = [
            j["updated_at"]
            for j in doc["jobs"].values()
            if j["status"] == "queued" and "updated_at" in j
        ]
        oldest = min(stamps, key=_ts) if stamps else None
        out["oldest_queued_at"] = oldest
        out["oldest_queued_age_s"] = (
            max(0.0, _ts(now) - _ts(oldest)) if oldest and now is not None else None
        )
        return out
