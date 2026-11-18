"""Read-only EODHD research downloads; native-currency snapshots stay out of backtests."""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wharton.credentials import load_credentials
from wharton.http_data import get_json


def symbol(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", value
    ):
        raise ValueError("Use an EODHD exchange-qualified symbol, e.g. AAPL.US or SAP.XETRA")
    return value


def us_symbol(value: str) -> str:
    qualified = value if "." in value else f"{value}.US"
    symbol(qualified)
    if not qualified.endswith(".US"):
        raise ValueError(
            "EODHD backtests currently require US listings; use 'wharton eodhd' for global "
            "research snapshots. Foreign backtests need FX and exchange-calendar support."
        )
    return qualified


def date_window(start: str, end: str) -> dict:
    try:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(first) or pd.isna(last) or first.tz or last.tz or first >= last:
            raise ValueError
        if first != first.normalize() or last != last.normalize():
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Specify date-only start before exclusive end") from None
    return {"from": str(first.date()), "to": str((last - pd.Timedelta(days=1)).date())}


def request(session, endpoint: str, params: dict, *, expected_type=dict):
    load_credentials()
    token = os.environ.get("EODHD_API_KEY", "").strip()
    if not token:
        raise ValueError("Set EODHD_API_KEY in .env or your environment")
    # EODHD requires a query token. The shared HTTP boundary suppresses URLs and bodies.
    return get_json(
        session, f"https://eodhd.com/api/{endpoint}",
        {**params, "api_token": token, "fmt": "json"}, "EODHD",
        expected_type=expected_type,
    )


def general_data(session, ticker: str) -> dict:
    payload = request(session, f"fundamentals/{symbol(ticker)}", {})
    general = payload.get("General")
    if not isinstance(general, dict) or not general.get("CurrencyCode"):
        raise ValueError("EODHD returned no instrument currency metadata")
    code = ticker.rsplit(".", 1)[0]
    # Require exact provider identity, including case; never normalize or infer missing codes.
    if general.get("Code") != code:
        raise ValueError("EODHD instrument code does not match the requested symbol")
    return payload


def adjusted_series(payload: list, ticker: str, start: str, end: str) -> pd.Series:
    try:
        frame = pd.DataFrame(payload)
        if not frame["date"].map(
            lambda value: isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
        ).all():
            raise ValueError
        if frame["adjusted_close"].map(lambda value: isinstance(value, bool)).any():
            raise ValueError
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], format="%Y-%m-%d"))
        values = pd.to_numeric(frame["adjusted_close"], errors="raise").to_numpy()
        if dates.has_duplicates or dates.hasnans or dates.tz is not None:
            raise ValueError
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError
        series = pd.Series(values, index=dates, name=ticker).sort_index()
        series = series.loc[(series.index >= start) & (series.index < end)]
        if series.empty:
            raise ValueError
        return series
    except (KeyError, TypeError, ValueError):
        raise ValueError("EODHD returned invalid adjusted-price history") from None


class EODHDProvider:
    """USD US listings only; the existing loader enforces the XNYS session calendar."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        qualified = [us_symbol(ticker) for ticker in tickers]
        window = date_window(start, end)
        columns = []
        snapshots = []
        with requests.Session() as session:
            for ticker, code in zip(tickers, qualified, strict=True):
                fundamentals = general_data(session, code)
                general = fundamentals["General"]
                if general["CurrencyCode"] != "USD" or general.get("Type") not in {
                    "Common Stock", "ETF",
                }:
                    raise ValueError("EODHD backtests require USD-denominated US stocks or ETFs")
                payload = request(
                    session, f"eod/{code}", {**window, "period": "d", "order": "a"},
                    expected_type=list,
                )
                series = adjusted_series(payload, ticker, start, end)
                snapshots.append(
                    str(_save_snapshot(code, "prices", start, end, payload, fundamentals))
                )
                columns.append(series)
        frame = pd.concat(columns, axis=1).sort_index()
        frame.attrs["source_snapshots"] = snapshots
        return frame


def download_snapshot(ticker: str, dataset: str, start: str, end: str) -> Path:
    """Save one dated research snapshot; fundamentals are current, never historical signals."""
    symbol(ticker)
    if dataset not in {"prices", "fundamentals", "news"}:
        raise ValueError("dataset must be prices, fundamentals, or news")
    window = date_window(start, end)
    with requests.Session() as session:
        fundamentals = general_data(session, ticker)
        if dataset == "prices":
            payload = request(
                session, f"eod/{ticker}", {**window, "period": "d", "order": "a"},
                expected_type=list,
            )
        elif dataset == "news":
            payload = request(
                session, "news", {**window, "s": ticker, "limit": 100, "offset": 0},
                expected_type=list,
            )
        else:
            payload = fundamentals
    return _save_snapshot(ticker, dataset, start, end, payload, fundamentals)


def _save_snapshot(
    ticker: str, dataset: str, start: str, end: str, raw, fundamentals: dict
) -> Path:
    payload = raw
    if dataset == "prices":
        series = adjusted_series(raw, ticker, start, end)
        payload = [{"date": str(day.date()), "adjusted_close": value}
                   for day, value in series.items()]
    retrieved = datetime.now(UTC)
    output = Path("data/eodhd") / f"{ticker}-{dataset}-{retrieved:%Y%m%dT%H%M%S%fZ}"
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    files = {
        "data.json": encoded,
        "raw.json": json.dumps(raw, indent=2, allow_nan=False),
        "fundamentals.json": json.dumps(fundamentals, indent=2, allow_nan=False),
    }
    metadata = {
        "provider": "eodhd", "symbol": ticker, "dataset": dataset,
        "currency": fundamentals["General"]["CurrencyCode"],
        "start": start if dataset != "fundamentals" else None,
        "end_exclusive": end if dataset != "fundamentals" else None,
        "retrieved_at": retrieved.isoformat(),
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "files": {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()
        },
        "usage": "Native currency research snapshot; no FX conversion or exchange-calendar "
        "validation. Fundamentals are current as retrieved, not point-in-time backtest inputs. "
        "News is the first 100 matching articles, not a complete archive.",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (output / name).write_text(value)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output
