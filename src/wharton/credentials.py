"""Explicit credential import and allowlisted .env loading; never display values."""

import os
from pathlib import Path

from dotenv import dotenv_values

NAMES = ("TRADER_ALPACA_KEY_ID", "TRADER_ALPACA_SECRET_KEY", "TRADER_FRED_API_KEY")


def load_credentials():
    """Load only supported secrets from local .env; existing environment takes precedence."""
    path = Path(".env")
    if path.is_file():
        values = dotenv_values(path, interpolate=False)
        for name in (*NAMES, "TIINGO_API_KEY", "EODHD_API_KEY"):
            if values.get(name):
                os.environ.setdefault(name, values[name])


def import_trader_credentials(source):
    """Import allowlisted settings from an explicitly supplied local private file."""
    destination = Path(".env")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Local .env already exists; refusing to overwrite credentials")
    source = Path(source)
    if source.is_symlink() or source.stat().st_mode & 0o077:
        raise ValueError("Source credential file must be private and not a symlink")
    values = dotenv_values(source, interpolate=False)
    if any(not values.get(name) for name in NAMES):
        raise ValueError("Source file is missing required Alpaca/FRED settings")
    if any(any(char in values[name] for char in "\r\n\x00") for name in NAMES):
        raise ValueError("Unexpected multiline credential; nothing imported")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        for name in NAMES:
            value = values[name].replace("\\", "\\\\").replace("'", "\\'")
            output.write(f"{name}='{value}'\n")
    return "Imported local Alpaca/FRED settings into .env (permissions 0600)."
