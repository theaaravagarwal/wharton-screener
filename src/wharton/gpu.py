"""Optional GPU sample covariance with authoritative float64 financial gate checks.

MPS supports float32 only. GPU output must agree with NumPy float64; gates retain
that reference so hardware rounding cannot change an investment decision.
"""

import importlib
import os
import time
from statistics import median

import numpy as np


def require_device(device):
    if device not in {"cuda", "mps"}:
        raise ValueError("GPU device must explicitly be 'cuda' or 'mps'")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK: GPU verification forbids fallback")
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "GPU backend requires the optional dependency: uv sync --extra gpu"
        ) from error
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        raise RuntimeError(f"Requested GPU device {device} is unavailable; no CPU fallback")
    return torch


def _matrix(returns):
    array = np.asarray(returns, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("returns require at least two observations and one instrument")
    if not np.isfinite(array).all():
        raise ValueError("returns must contain only finite observations")
    return array


def gpu_covariance(returns, device):
    """Execute real financial covariance on the requested accelerator; never CPU fallback."""
    array = _matrix(returns)
    torch = require_device(device)
    synchronize = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize
    dtype = torch.float64 if device == "cuda" else torch.float32
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
        if tensor.device.type != device:
            raise RuntimeError("GPU allocation unexpectedly used another device")
        centered = tensor - tensor.mean(dim=0)
        covariance = centered.transpose(0, 1) @ centered / (len(array) - 1)
        synchronize()
        result = covariance.cpu().numpy().astype(np.float64)
    elapsed = time.perf_counter() - started
    if not np.isfinite(result).all():
        raise RuntimeError("GPU covariance produced nonfinite values")
    return result, dict(
        device=device,
        tensor_device=str(tensor.device),
        dtype=str(dtype),
        seconds=elapsed,
        observations=len(array),
        instruments=array.shape[1],
    )


def portfolio_covariance(returns, device):
    """Check accelerated financial computation, preserving exact CPU gate decisions."""
    array = _matrix(returns)
    accelerated, _ = gpu_covariance(array, device)
    reference = np.atleast_2d(np.cov(array, rowvar=False, ddof=1))
    if not np.allclose(accelerated, reference, rtol=1e-4, atol=1e-9):
        raise RuntimeError("GPU financial covariance disagrees with CPU float64 reference")
    return reference


def resolve_device(device):
    if device == "auto":
        try:
            require_device("cuda")
            return "cuda"
        except RuntimeError:
            require_device("mps")
            return "mps"
    require_device(device)
    return device


def verify_gpu(device="auto"):
    """Run synchronized portfolio risk math and fail loudly on unavailable/mismatched GPU."""
    device = resolve_device(device)
    returns = np.random.default_rng(7401).normal(0.0002, 0.01, size=(4096, 64))
    weights = np.full(64, 1 / 64)
    gpu_covariance(returns, device)  # Warm up kernels before measured runs.
    cpu_times, gpu_times = [], []
    for _ in range(3):
        start = time.perf_counter()
        reference = np.cov(returns, rowvar=False, ddof=1)
        cpu_times.append(time.perf_counter() - start)
        accelerated, receipt = gpu_covariance(returns, device)
        gpu_times.append(receipt["seconds"])
        if not np.allclose(accelerated, reference, rtol=1e-4, atol=1e-9):
            raise RuntimeError("GPU covariance verification failed")
    cpu_total = float(weights @ reference @ weights)
    gpu_total = float(weights @ accelerated @ weights)
    cpu_euler = weights * (reference @ weights) / cpu_total
    gpu_euler = weights * (accelerated @ weights) / gpu_total
    if not np.allclose(cpu_euler, gpu_euler, rtol=1e-4, atol=1e-7):
        raise RuntimeError("GPU portfolio variance contribution verification failed")
    return dict(
        receipt,
        verified=True,
        available=True,
        passed=True,
        metric="sample covariance and Euler variance contributions",
        cpu_variance=cpu_total,
        gpu_variance=gpu_total,
        max_covariance_error=float(np.max(np.abs(accelerated - reference))),
        max_contribution_error=float(np.max(np.abs(gpu_euler - cpu_euler))),
        cpu_median_seconds=median(cpu_times),
        gpu_median_seconds=median(gpu_times),
        gpu_faster=median(gpu_times) < median(cpu_times),
        gates="CPU float64 authoritative; GPU covariance independently checked",
        synthetic=True,
        torch_version=str(require_device(device).__version__),
    )
