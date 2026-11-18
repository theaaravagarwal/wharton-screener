import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from wharton import runtime_verify
from wharton.public_cli import app


def test_offline_does_not_run_live_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_verify.standalone, "self_test", lambda: {"offline": True})
    monkeypatch.setattr(runtime_verify.standalone, "initialize", lambda *a, **k: pytest.fail())
    path = tmp_path / "receipt.json"
    assert runtime_verify.verify_install(path)["passed"]
    assert json.loads(path.read_text())["checks"] == {"offline": {"offline": True}}


def test_gpu_failure_has_nonzero_cli_and_safe_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_verify.standalone, "self_test", lambda: {"offline": True})
    def fail(device):
        raise RuntimeError("secret-token-must-not-be-written")
    monkeypatch.setitem(sys.modules, "wharton.gpu", SimpleNamespace(verify_gpu=fail))
    path = tmp_path / "receipt.json"
    result = CliRunner().invoke(app, ["verify-install", "--require-gpu", "--output", str(path)])
    assert result.exit_code == 1
    assert "reinstall.sh" in result.output
    assert "secret-token" not in path.read_text() + result.output
    assert json.loads(path.read_text())["failed_check"] == "gpu"


def test_live_explicit_bounded_and_receipted(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_verify.standalone, "self_test", lambda: {"offline": True})
    monkeypatch.setattr(runtime_verify, "live_account_probe",
                        lambda path: {"fresh_authentication": True})
    calls = []
    def init(*args, **kwargs):
        calls.append(kwargs)
        return {"candidates": 42}
    monkeypatch.setattr(runtime_verify.standalone, "initialize", init)
    result = runtime_verify.verify_install(tmp_path / "receipt.json", live=True)
    assert calls == [{"live": True, "max_credits": 1}]
    assert result["checks"]["live_api"]["candidates"] == 42


def test_live_probe_runs_despite_cached_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_verify.standalone, "self_test", lambda: {"offline": True})
    called = []
    monkeypatch.setattr(runtime_verify, "live_account_probe",
                        lambda path: called.append("network") or {"fresh_authentication": True})
    monkeypatch.setattr(runtime_verify.standalone, "initialize",
                        lambda *a, **k: {"candidates": 7})
    result = runtime_verify.verify_install(tmp_path / "receipt.json", live=True)
    assert called == ["network"]
    assert result["checks"]["live_api"]["fresh_authentication"]
