import json
import time
from datetime import UTC, datetime

import pytest
import requests

from wharton import http_data, screen_budget
from wharton.screen_budget import AccountProviderSession, BudgetHold


class Response:
    headers = {}

    def __init__(self, value, status=200):
        self.value, self.status_code = value, status

    def json(self):
        return self.value


def request(session, endpoint, params, *, expected_type=dict):
    return http_data.get_json(
        session, "https://eodhd.com/api/" + endpoint, params, "EODHD", expected_type=expected_type
    )


class Backend:
    def __init__(self, counts, paid=None):
        self.counts, self.calls, self.paid = iter(counts), [], paid or Response({})

    def get(self, url, **kwargs):
        endpoint = url.split("/api/")[1]
        self.calls.append(endpoint)
        if endpoint == "user":
            return Response(
                {
                    "apiRequests": next(self.counts),
                    "dailyRateLimit": 100000,
                    "apiRequestsDate": str(datetime.now(UTC).date()),
                }
            )
        if isinstance(self.paid, Exception):
            raise self.paid
        return self.paid


def session(tmp_path, backend, monkeypatch):
    monkeypatch.setattr("wharton.screen_budget.time.sleep", lambda _: None)
    result = AccountProviderSession(
        backend,
        datetime.now(UTC).date(),
        25000,
        request,
        [],
        capacity_path=tmp_path / "capacity-v2.json",
        config={"daily_credit_limit": 75000, "daily_working_credit_limit": 50000},
    )
    result.reserve("fundamentals/A.US", 10, "missing_identity")
    return result


def test_observed_account_counter_and_one_attempt(tmp_path, monkeypatch):
    backend = Backend([25209, 25219])
    s = session(tmp_path, backend, monkeypatch)
    assert request(s, "fundamentals/A.US", {}) == {}
    state = json.loads((tmp_path / "capacity-v2.json").read_text())
    assert backend.calls == ["user", "fundamentals/A.US", "user"]
    assert state["pending"] is None
    assert state["observations"][-1]["account_delta"] == 10
    assert state["observations"][-1]["account_credits"] == 25219


def test_timeout_is_durable_and_restart_cannot_retry(tmp_path, monkeypatch):
    s = session(tmp_path, Backend([25209], requests.Timeout()), monkeypatch)
    with pytest.raises(BudgetHold, match="outcome unknown"):
        request(s, "fundamentals/A.US", {})
    assert json.loads((tmp_path / "capacity-v2.json").read_text())["pending"]["possible_cost"] == 10
    with pytest.raises(BudgetHold, match="unresolved"):
        session(tmp_path, Backend([25219]), monkeypatch)


def test_regression_holds_pending_without_refund(tmp_path, monkeypatch):
    s = session(tmp_path, Backend([25209, 25208]), monkeypatch)
    with pytest.raises(BudgetHold, match="post-attempt"):
        request(s, "fundamentals/A.US", {})
    assert json.loads((tmp_path / "capacity-v2.json").read_text())["pending"]["possible_cost"] == 10


def test_unknown_schema_fails_closed(tmp_path, monkeypatch):
    (tmp_path / "capacity-v2.json").write_text('{"version": 3}')
    with pytest.raises(BudgetHold, match="invalid capacity"):
        session(tmp_path, Backend([]), monkeypatch)


def test_completed_500_retries_only_after_post_snapshot(tmp_path, monkeypatch):
    class RetryBackend(Backend):
        def get(self, url, **kwargs):
            if url.endswith("/fundamentals/A.US"):
                self.calls.append("fundamentals/A.US")
                return Response({}, 500 if self.calls.count("fundamentals/A.US") == 1 else 200)
            return super().get(url, **kwargs)

    monkeypatch.setattr("wharton.http_data.time.sleep", lambda _: None)
    backend = RetryBackend([25209, 25219, 25219, 25229])
    s = session(tmp_path, backend, monkeypatch)
    assert request(s, "fundamentals/A.US", {}) == {}
    assert backend.calls == [
        "user",
        "fundamentals/A.US",
        "user",
        "user",
        "fundamentals/A.US",
        "user",
    ]
    assert len(json.loads((tmp_path / "capacity-v2.json").read_text())["observations"]) == 4


def test_working_limit_and_account_headroom_checked_per_attempt(tmp_path, monkeypatch):
    backend = Backend([49995])
    s = session(tmp_path, backend, monkeypatch)
    s.reserve("fundamentals/A.US", 10, "routine_refresh")
    with pytest.raises(BudgetHold, match="project budget"):
        request(s, "fundamentals/A.US", {})
    assert backend.calls == ["user"]
    assert (
        json.loads((tmp_path / "capacity-v2.json").read_text())["observations"][-1][
            "account_credits"
        ]
        == 49995
    )


def test_unchanged_post_counter_holds_possible_charge_and_blocks_retry(tmp_path, monkeypatch):
    backend = Backend([74990, 74990])
    s = session(tmp_path, backend, monkeypatch)
    with pytest.raises(BudgetHold, match="post-attempt"):
        request(s, "fundamentals/A.US", {})
    state = json.loads((tmp_path / "capacity-v2.json").read_text())
    assert state["pending"]["possible_cost"] == 10
    with pytest.raises(BudgetHold, match="unresolved"):
        session(tmp_path, Backend([75000]), monkeypatch)


def test_preflight_at_limit_is_persisted_and_later_regression_holds(tmp_path, monkeypatch):
    s = session(tmp_path, Backend([50000]), monkeypatch)
    s.reserve("fundamentals/A.US", 10, "routine_refresh")
    with pytest.raises(BudgetHold, match="project budget"):
        request(s, "fundamentals/A.US", {})
    s = session(tmp_path, Backend([49999]), monkeypatch)
    with pytest.raises(BudgetHold, match="regressed"):
        request(s, "fundamentals/A.US", {})


def test_uncharged_500_is_not_retried(tmp_path, monkeypatch):
    backend = Backend([100, 100], Response({}, 500))
    s = session(tmp_path, backend, monkeypatch)
    with pytest.raises(BudgetHold, match="post-attempt"):
        request(s, "fundamentals/A.US", {})
    assert backend.calls == ["user", "fundamentals/A.US", "user"]


# ---- S1 deadline safety, S2 pacing, S3 archive ----


class DeadlineBackend(Backend):
    """Moves the session deadline into the past right after the call list equals `after`."""

    def __init__(self, counts, *a, after=("user", "fundamentals/A.US"), **k):
        super().__init__(counts, *a, **k)
        self.session, self.after = None, list(after)

    def get(self, url, **kwargs):
        result = super().get(url, **kwargs)
        if self.session is not None and self.calls == self.after:
            self.session.deadline = 0.0
        return result


def test_deadline_passing_after_preflight_stops_before_pending(tmp_path, monkeypatch):
    backend = DeadlineBackend([100, 110], after=("user",))
    s = session(tmp_path, backend, monkeypatch)
    backend.session, s.deadline = s, time.monotonic() + 3600
    with pytest.raises(BudgetHold, match="network time budget exhausted"):
        request(s, "fundamentals/A.US", {})
    assert backend.calls == ["user"]  # no paid GET once the deadline has passed
    assert json.loads((tmp_path / "capacity-v2.json").read_text())["pending"] is None


def test_deadline_passing_mid_attempt_still_records_and_clears_pending(tmp_path, monkeypatch):
    backend = DeadlineBackend([100, 110])
    s = session(tmp_path, backend, monkeypatch)
    backend.session, s.deadline = s, time.monotonic() + 3600
    assert request(s, "fundamentals/A.US", {}) == {}
    state = json.loads((tmp_path / "capacity-v2.json").read_text())
    assert state["pending"] is None
    assert state["observations"][-1]["account_delta"] == 10
    assert backend.calls == ["user", "fundamentals/A.US", "user"]


def test_deadline_already_passed_makes_no_http_call_and_no_pending(tmp_path, monkeypatch):
    backend = Backend([100, 110])
    s = session(tmp_path, backend, monkeypatch)
    s.deadline = 0.0
    with pytest.raises(BudgetHold, match="network time budget exhausted"):
        request(s, "fundamentals/A.US", {})
    assert backend.calls == []
    assert s.capacity["pending"] is None
    assert not (tmp_path / "capacity-v2.json").exists()


def test_deadline_still_stops_free_sends_outside_an_attempt(tmp_path, monkeypatch):
    s = session(tmp_path, Backend([100]), monkeypatch)
    s.deadline = 0.0
    with pytest.raises(BudgetHold, match="network time budget exhausted"):
        request(s, "user", {})


def test_request_gap_is_75ms(tmp_path, monkeypatch):
    sleeps = []
    s = session(tmp_path, Backend([100]), monkeypatch)
    monkeypatch.setattr("wharton.screen_budget.time.sleep", sleeps.append)
    assert screen_budget.MIN_REQUEST_GAP == 0.075
    s.last_request = time.monotonic()
    request(s, "user", {})
    assert len(sleeps) == 1 and 0.05 < sleeps[0] <= 0.075


def _obs(day, n):
    return [
        dict(day=day, account_credits=n + i, account_delta=1, endpoint="x", possible_cost=1)
        for i in range(n)
    ]


def _seed(tmp_path, today):
    old = _obs("2020-01-01", 3) + _obs("2020-01-02", 2)
    cur = _obs(str(today), 2)
    state = {"version": 2, "pending": None, "observations": old + cur}
    (tmp_path / "capacity-v2.json").write_text(json.dumps(state))
    return state


def test_prior_days_are_archived_losslessly_and_second_run_is_noop(tmp_path, monkeypatch):
    today = datetime.now(UTC).date()
    original = _seed(tmp_path, today)
    session(tmp_path, Backend([]), monkeypatch)
    live = json.loads((tmp_path / "capacity-v2.json").read_text())
    assert [o["day"] for o in live["observations"]] == [str(today)] * 2
    arch = tmp_path / "capacity-archive"
    merged = []
    for day in ("2020-01-01", "2020-01-02"):
        merged += json.loads((arch / f"{day}.json").read_text())
    assert merged + live["observations"] == original["observations"]
    before = {p.name: p.read_bytes() for p in arch.iterdir()}
    session(tmp_path, Backend([]), monkeypatch)
    assert {p.name: p.read_bytes() for p in arch.iterdir()} == before


def test_archive_exact_prefix_merges_and_conflict_holds(tmp_path, monkeypatch):
    today = datetime.now(UTC).date()
    _seed(tmp_path, today)
    arch = tmp_path / "capacity-archive"
    arch.mkdir()
    prefix = _obs("2020-01-01", 3)[:2]  # crashed earlier mid-way: exact prefix
    (arch / "2020-01-01.json").write_text(json.dumps(prefix, sort_keys=True))
    session(tmp_path, Backend([]), monkeypatch)
    assert json.loads((arch / "2020-01-01.json").read_text()) == _obs("2020-01-01", 3)
    _seed(tmp_path, today)
    bad = _obs("2020-01-01", 3)
    bad[0]["account_credits"] = 999
    (arch / "2020-01-01.json").write_text(json.dumps(bad, sort_keys=True))
    with pytest.raises(BudgetHold, match="archive"):
        session(tmp_path, Backend([]), monkeypatch)


def test_spend_readers_unchanged_by_archiving(tmp_path, monkeypatch):
    from wharton import news_fetch

    today = datetime.now(UTC).date()
    state = _seed(tmp_path, today)
    day = str(today)
    before = news_fetch._spend_and_screening(state, day)
    session(tmp_path, Backend([]), monkeypatch)
    live = json.loads((tmp_path / "capacity-v2.json").read_text())
    assert news_fetch._spend_and_screening(live, day) == before
