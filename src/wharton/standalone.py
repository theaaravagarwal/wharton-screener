"""Standalone initialization. Offline setup never loads credentials or calls providers."""

import fcntl
import hashlib
import json
import os
import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import requests
import yaml
from dotenv import dotenv_values

from wharton import eodhd, screen_budget, screen_service


def configure(root):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Standalone paths must not contain symlinks")
    root.mkdir(parents=True, exist_ok=True)
    config = root / "screen.yaml"
    if config.is_symlink():
        raise ValueError("Standalone configuration must not be a symlink")
    if not config.exists():
        value = dict(
            input="input.json",
            state_root="state",
            batch_size=1000,
            acquisition=dict(
                enabled=False,
                accounting_mode="account",
                mode="daily",
                initial_credit_limit=40000,
                daily_credit_limit=75000,
                daily_working_credit_limit=75000,
                account_headroom=25000,
                request_credit_cost=dict(fundamentals=10, prices=1),
            ),
        )
        config.write_text(yaml.safe_dump(value))
    return config


def credentials(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("Credential file must be private (chmod 600) and not a symlink")
    key = dotenv_values(path, interpolate=False).get("EODHD_API_KEY")
    if not key or any(c in key for c in "\r\n\x00"):
        raise ValueError("Credential file needs a one-line EODHD_API_KEY")
    existing = os.environ.get("EODHD_API_KEY")
    if existing and existing != key:
        raise ValueError("Conflicting EODHD credential sources")
    os.environ["EODHD_API_KEY"] = key


def inventory_records(payload, exchange):
    records = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Invalid exchange inventory row")
        if row.get("Type") != "Common Stock":
            continue
        code = row.get("Code")
        if not isinstance(code, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", code
        ):
            continue
        ticker = eodhd.symbol(f"{code}.{exchange}")
        isin = row.get("Isin")
        identifiers = {}
        if isinstance(isin, str) and re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin):
            # Inventory binding is only a seed; fundamentals must match it before admission.
            identifiers["isin"] = isin
        # Listing identity remains provisional when the inventory supplies no identifier.
        if ticker in records and records[ticker]["identifiers"] != identifiers:
            raise ValueError("Conflicting inventory identifiers for one ticker")
        records[ticker] = dict(
            ticker=ticker,
            issuer_id=f"isin:{isin}" if identifiers else f"listing:{ticker}",
            kind="stock",
            name=str(row.get("Name", ticker)),
            region="unknown",
            identifiers=identifiers,
            evidence={},
            reviews={},
            discovery={"issuer_proof": "unverified", "exchange": exchange},
        )
    return [records[k] for k in sorted(records)]


def initialize(root, exchanges, env_file, *, live=False, max_credits=500):
    if not live:
        raise ValueError("Live discovery requires --live; setup and self-test are offline")
    if type(max_credits) is not int or not 1 <= max_credits <= 500:
        raise ValueError("Initialization credit cap must be 1..500")
    exchanges = sorted(set(exchanges))
    if not exchanges or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in exchanges):
        raise ValueError("Specify valid exchange codes")
    if len(exchanges) > max_credits:
        raise ValueError("Exchange count exceeds packet credit cap")
    credentials(env_file)
    config_path = configure(root)
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text())
    state = root / "state"
    if config.get("state_root") != "state" or config.get("input") != "input.json":
        raise ValueError("Standalone configuration paths must remain local")
    if state.is_symlink():
        raise ValueError("Standalone state must not be a symlink")
    state.mkdir(exist_ok=True)
    receipts = []
    day = datetime.now(UTC).date()
    budget = config["acquisition"]
    screen_budget.validate_budget(budget)
    if (
        budget.get("accounting_mode") != "account"
        or budget["daily_credit_limit"] > 75000
        or budget.get("daily_working_credit_limit", 75000) > 75000
        or budget.get("account_headroom", 0) < 25000
    ):
        raise ValueError("Unsafe account budget configuration")

    class PacketSession(screen_budget.AccountProviderSession):
        def _paid_attempt(self, *args, **kwargs):
            # Capacity observations survive retries, crashes, and repeated init invocations.
            spent = sum(
                row["account_delta"]
                for row in self.capacity["observations"]
                if row["day"] == str(day) and row["endpoint"].startswith("exchange-symbol-list/")
            )
            if spent + self.reserved_cost > max_credits:
                raise screen_budget.BudgetHold("Initialization credit cap reached")
            return super()._paid_attempt(*args, **kwargs)

    with (state / ".lock").open("a") as lock, requests.Session() as transport:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        input_path = root / "input.json"
        if input_path.is_symlink():
            raise ValueError("Standalone input must not be a symlink")
        if input_path.exists():
            inputs = json.loads(input_path.read_text())
            if not isinstance(inputs, dict) or not isinstance(inputs.get("universe"), list):
                raise ValueError("Existing input is not a screening universe; refusing overwrite")
            old = inputs["universe"]
            if any(
                not isinstance(r, dict) or not r.get("ticker") or not r.get("issuer_id")
                for r in old
            ):
                raise ValueError("Existing input contains malformed records; refusing overwrite")
        else:
            inputs = dict(
                universe=[],
                active=[],
                portfolio=dict(benchmark="VT.US", existing_weights={}, holdings=[], weights={}),
            )
        session = PacketSession(
            transport,
            day,
            25000,
            eodhd.request,
            receipts,
            capacity_path=state / "capacity-v2.json",
            config=budget,
        )
        records = []
        for exchange in exchanges:
            receipt_path = state / f"inventory-{exchange}-{day}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                payload = receipt["payload"]
                if (
                    hashlib.sha256(screen_service.encoded(payload).encode()).hexdigest()
                    != receipt["sha256"]
                ):
                    raise ValueError("Inventory receipt hash mismatch")
            else:
                endpoint = f"exchange-symbol-list/{exchange}"
                session.reserve(endpoint, screen_budget.endpoint_cost(endpoint))
                payload = eodhd.request(session, endpoint, {}, expected_type=list)
                screen_service.atomic(
                    receipt_path,
                    dict(
                        payload=payload,
                        exchange=exchange,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        sha256=hashlib.sha256(screen_service.encoded(payload).encode()).hexdigest(),
                    ),
                )
            records.extend(inventory_records(payload, exchange))
        if not records:
            raise ValueError("No common-stock candidates in exchange inventories")
        tickers = {r["ticker"] for r in inputs["universe"]}
        issuers = {r["issuer_id"] for r in inputs["universe"]}
        for record in records:
            if record["ticker"] in tickers or record["issuer_id"] in issuers:
                continue
            inputs["universe"].append(record)
            tickers.add(record["ticker"])
            issuers.add(record["issuer_id"])
        screen_service.atomic(input_path, inputs)
        config.pop("as_of", None)
        # Discovery never enables unbounded acquisition as a side effect.
        config["acquisition"]["enabled"] = False
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(config))
        os.replace(temporary, config_path)
    return {"config": str(config_path), "candidates": len(records), "acquisition_enabled": False}


def self_test():
    """Run synthetic financial gates while denying network access."""
    from wharton.screening import screen

    inputs = {
        "synthetic": True,
        "universe": inventory_records([{"Code": "SYNTH", "Type": "Common Stock"}], "US"),
        "portfolio": {},
    }
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Offline test used network")
    ):
        result = screen(inputs, "2026-10-28")
    if len(result) != 1 or result[0].get("stage") == "review_ready":
        raise AssertionError("Synthetic missing evidence must not qualify")
    return {"offline": True, "synthetic": True, "result_type": type(result).__name__}
