"""GPU risk math must agree with CPU; explicit acceleration never silently falls back."""

from types import SimpleNamespace

import numpy as np
import pytest

from wharton import gpu, screening


def test_missing_optional_dependency_fails(monkeypatch):
    def missing(name):
        raise ImportError("synthetic missing dependency")

    monkeypatch.setattr(gpu.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="optional dependency"):
        gpu.verify_gpu("mps")


def test_unavailable_gpu_fails_even_for_empty_screen(monkeypatch):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(gpu.importlib, "import_module", lambda _: torch)
    with pytest.raises(RuntimeError, match="unavailable; no CPU fallback"):
        screening.screen({"universe": [], "portfolio": {"numeric_backend": "mps"}}, "2026-11-17")


def test_mps_fallback_setting_fails(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="forbids fallback"):
        gpu.require_device("mps")


def test_financial_reference_preserved_and_disagreement_rejected(monkeypatch):
    returns = np.random.default_rng(17).normal(0, 0.01, size=(40, 3))
    reference = np.cov(returns, rowvar=False, ddof=1)
    monkeypatch.setattr(gpu, "gpu_covariance", lambda a, d: (reference.astype(np.float32), {}))
    np.testing.assert_array_equal(gpu.portfolio_covariance(returns, "mps"), reference)
    monkeypatch.setattr(gpu, "gpu_covariance", lambda a, d: (reference * 2, {}))
    with pytest.raises(RuntimeError, match="disagrees"):
        gpu.portfolio_covariance(returns, "mps")


def test_actual_screen_uses_gpu_financial_seam(monkeypatch):
    from test_screening_service import AS_OF, risk_inputs

    inputs, _ = risk_inputs()
    expected = screening.screen(inputs, AS_OF)
    calls = []
    monkeypatch.setattr(gpu, "require_device", lambda _: None)

    def covariance(returns, device):
        calls.append(device)
        return np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))

    monkeypatch.setattr(gpu, "portfolio_covariance", covariance)
    inputs["portfolio"]["numeric_backend"] = "mps"
    actual = screening.screen(inputs, AS_OF)
    assert calls and set(calls) == {"mps"}
    assert actual == expected


@pytest.mark.parametrize("returns", [[], [[1]], [[1], [float("nan")]]])
def test_bad_return_matrices_fail_before_device_use(returns):
    with pytest.raises(ValueError, match="returns"):
        gpu.gpu_covariance(returns, "mps")
