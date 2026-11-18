"""Explicit local installation checks with durable, credential-free receipts."""

import platform
from datetime import UTC, datetime
from pathlib import Path

import requests

from wharton import eodhd, screen_budget, screen_service, standalone


def live_account_probe(env_file):
    """Authenticate every live check, even when inventory receipts are cached."""
    standalone.credentials(env_file)
    with requests.Session() as transport:
        account = eodhd.request(transport, "user", {})
    safe, effective, remaining = screen_budget.account_usage(
        account, datetime.now(UTC).date(), 25000
    )
    return {"fresh_authentication": True, "account_credits": effective,
            "base_capacity_after_headroom": remaining,
            "provider_usage_date": safe["apiRequestsDate"]}


def verify_install(output, *, require_gpu=False, device="auto", live=False,
                   root=Path("runs/standalone"), env_file=Path(".env")):
    output = Path(output)
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError("Verification output must not contain symlinks")
    receipt = {"at": datetime.now(UTC).isoformat(), "platform": platform.system(),
               "machine": platform.machine(), "checks": {}, "passed": False}
    current = "offline"
    try:
        receipt["checks"][current] = standalone.self_test()
        if require_gpu:
            current = "gpu"
            from wharton.gpu import verify_gpu
            receipt["checks"][current] = verify_gpu(device)
        if live:
            current = "live_api"
            account = live_account_probe(env_file)
            result = standalone.initialize(root, ["US"], env_file, live=True, max_credits=1)
            receipt["checks"][current] = {"candidates": result["candidates"],
                                          "acquisition_enabled": False,
                                          "inventory_may_be_cached": True, **account}
        receipt["passed"] = True
    except Exception as error:
        # Exception messages can contain provider URLs; retain only safe diagnostic fields.
        receipt["failed_check"] = current
        receipt["error_type"] = type(error).__name__
        receipt["recovery"] = "Install the reviewed update, then run ./reinstall.sh"
        screen_service.atomic(output, receipt)
        raise RuntimeError(
            f"Installation check {current} failed. Receipt: {output}. "
            "Install the reviewed update and rerun ./reinstall.sh; "
            "credentials and runs are preserved."
        ) from None
    screen_service.atomic(output, receipt)
    return receipt
