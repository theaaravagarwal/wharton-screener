import json
import os

import pytest

from wharton import screen_budget
from wharton.standalone import configure, credentials, initialize, inventory_records, self_test


def test_offline_setup_preserves_config(tmp_path):
    path = configure(tmp_path / "with spaces")
    path.write_text("preserved")
    assert configure(path.parent).read_text() == "preserved"
    assert self_test()["offline"]


def test_init_requires_explicit_live(tmp_path):
    with pytest.raises(ValueError, match="--live"):
        initialize(tmp_path, ["US"], tmp_path / "missing")


def test_credentials_conflict_and_permissions(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("EODHD_API_KEY=synthetic-local\n")
    path.chmod(0o600)
    monkeypatch.setenv("EODHD_API_KEY", "synthetic-other")
    with pytest.raises(ValueError, match="Conflicting"):
        credentials(path)
    monkeypatch.delenv("EODHD_API_KEY")
    credentials(path)
    assert os.environ["EODHD_API_KEY"] == "synthetic-local"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        credentials(path)


def test_inventory_filters_and_never_admits_evidence():
    rows = [
        {"Code": "SYNTH", "Type": "Common Stock"},
        {"Code": "ETF", "Type": "ETF"},
        {"Code": "../bad", "Type": "Common Stock"},
    ]
    records = inventory_records(rows + rows, "US")
    assert len(records) == 1
    assert records[0]["evidence"] == {}
    assert records[0]["discovery"]["issuer_proof"] == "unverified"
    assert screen_budget.endpoint_cost("exchange-symbol-list/US") == 1
    with pytest.raises(ValueError):
        screen_budget.endpoint_cost("exchange-symbol-list/../user")


def test_initialization_reuses_receipts_and_keeps_acquisition_off(tmp_path, monkeypatch):
    from wharton import standalone

    source = tmp_path / "key.env"
    source.write_text("EODHD_API_KEY=synthetic-test-key\n")
    source.chmod(0o600)
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    paid = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def reserve(self, endpoint, cost):
            assert cost == 1
            paid.append(endpoint)

    monkeypatch.setattr(standalone.screen_budget, "AccountProviderSession", Session)
    monkeypatch.setattr(
        standalone.eodhd, "request", lambda *a, **k: [{"Code": "SYNTH", "Type": "Common Stock"}]
    )
    root = tmp_path / "new machine"
    first = initialize(root, ["US"], source, live=True)
    private = json.loads((root / "input.json").read_text())
    private["portfolio"]["user_note"] = "preserve private holdings context"
    private["universe"][0]["evidence"] = {"manual": {"value": "preserved"}}
    private["custom_context"] = {"must": "survive"}
    (root / "input.json").write_text(json.dumps(private))
    second = initialize(root, ["US"], source, live=True)
    assert first == second
    assert len(paid) == 1
    assert not first["acquisition_enabled"]
    payload = json.loads((root / "input.json").read_text())
    assert payload == private
    import yaml

    assert "as_of" not in yaml.safe_load((root / "screen.yaml").read_text())
    assert payload["universe"][0]["ticker"] == "SYNTH.US"
    from wharton.screen_service import run, verify

    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("Unexpected network"))
    report = run(root / "screen.yaml")
    verified = verify(report.parent)
    assert verified
    results = json.loads((report.parent / "results.json").read_text())
    assert len(results) == 1
    assert results[0]["stage"] != "review_ready"
    receipt = next((root / "state").glob("inventory-*.json"))
    data = json.loads(receipt.read_text())
    data["payload"][0]["Code"] = "TAMPERED"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        initialize(root, ["US"], source, live=True)
    assert len(paid) == 1


def test_self_test_has_no_repository_fixture_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).endswith("configs/screening-example-input.json"):
            pytest.fail("Read repo fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert self_test()["offline"]


def test_inventory_isin_binds_only_matching_fundamentals():
    from wharton.screen_sources import SourceIdentityConflict, identity_metadata

    records = inventory_records(
        [{"Code": "SYNTH", "Type": "Common Stock", "Isin": "US0000000000"}], "US"
    )
    record = records[0]
    assert record["identifiers"] == {"isin": "US0000000000"}
    assert record["evidence"] == {}
    payload = {"General": {"Code": "SYNTH", "Exchange": "NYSE", "ISIN": "US0000000000"}}
    receipt = {
        "retrieved_at": "2026-11-17T12:00:00+00:00",
        "sha256": "synthetic",
        "file": "synthetic.raw.json",
        "source": "synthetic",
        "ticker": "SYNTH.US",
    }
    evidence = identity_metadata(payload, record, receipt)
    assert evidence["issuer_identity"]["verified"]
    payload["General"]["ISIN"] = "US0000000018"
    with pytest.raises(SourceIdentityConflict):
        identity_metadata(payload, record, receipt)


def test_inventory_conflicting_identifiers_rejected():
    with pytest.raises(ValueError, match="Conflicting inventory"):
        inventory_records(
            [
                {"Code": "SYNTH", "Type": "Common Stock", "Isin": "US0000000000"},
                {"Code": "SYNTH", "Type": "Common Stock", "Isin": "US0000000018"},
            ],
            "US",
        )


def test_real_account_guard_keeps_cumulative_init_cap(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import requests

    from wharton import standalone

    class Response:
        headers = {}

        def __init__(self, value, status=200):
            self.value, self.status_code = value, status

        def json(self):
            return self.value

    class Backend:
        def __init__(self):
            self.count = 0
            self.paid = 0
            self.mode = "retry"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            if url.endswith("/user"):
                return Response(
                    dict(
                        apiRequests=self.count,
                        dailyRateLimit=100000,
                        apiRequestsDate=str(datetime.now(UTC).date()),
                    )
                )
            self.paid += 1
            self.count += 1
            if self.mode == "timeout":
                raise requests.Timeout()
            status = 500 if self.mode == "retry" and self.paid == 1 else 200
            return Response([{"Code": "SYNTH", "Type": "Common Stock"}], status)

    backend = Backend()
    source = tmp_path / "key.env"
    source.write_text("EODHD_API_KEY=synthetic-test-key\n")
    source.chmod(0o600)
    monkeypatch.setenv("EODHD_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(standalone.requests, "Session", lambda: backend)
    monkeypatch.setattr("wharton.screen_budget.time.sleep", lambda *args: None)
    monkeypatch.setattr("wharton.http_data.time.sleep", lambda *args: None)
    root = tmp_path / "retry"
    initialize(root, ["US"], source, live=True, max_credits=2)
    assert backend.paid == 2
    capacity = json.loads((root / "state/capacity-v2.json").read_text())
    assert sum(row["account_delta"] for row in capacity["observations"]) == 2
    with pytest.raises(screen_budget.BudgetHold, match="Initialization credit cap"):
        initialize(root, ["XETRA"], source, live=True, max_credits=2)
    assert backend.paid == 2

    backend.mode = "timeout"
    held = tmp_path / "timeout"
    with pytest.raises(screen_budget.BudgetHold, match="outcome unknown"):
        initialize(held, ["US"], source, live=True, max_credits=2)
    paid = backend.paid
    with pytest.raises(screen_budget.BudgetHold, match="unresolved"):
        initialize(held, ["US"], source, live=True, max_credits=2)
    assert backend.paid == paid
    pending = json.loads((held / "state/capacity-v2.json").read_text())["pending"]
    assert pending["possible_cost"] == 1

    backend = Backend()
    capped_retry = tmp_path / "capped-retry"
    with pytest.raises(screen_budget.BudgetHold, match="Initialization credit cap"):
        initialize(capped_retry, ["US"], source, live=True, max_credits=1)
    assert backend.paid == 1
    with pytest.raises(screen_budget.BudgetHold, match="Initialization credit cap"):
        initialize(capped_retry, ["US"], source, live=True, max_credits=1)
    assert backend.paid == 1
