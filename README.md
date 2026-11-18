# Wharton stock screener

A standalone, long-only research screener with explicit evidence gaps, reproducible
calculations, guarded API acquisition, and offline verification. No live trades.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
./setup.sh
```

Supports macOS/Linux; Windows uses WSL2. Setup installs locked Python 3.12 dependencies
and runs a synthetic offline check. Add your entitled EODHD key to the ignored local
`.env` file, keeping its permissions at `600`. Never commit or share credentials.
See [standalone setup](docs/standalone-setup.md) for explicit live initialization,
account limits, daily screening and replay verification. One paid runner per account.

No prior research, acquired data, portfolio, API entitlement or remote host is included.
Missing identity/accounting/portfolio evidence stays missing; successful software checks
are not evidence of investment suitability or permission to trade.

CPU workers are bounded by available CPU/RAM. Benchmark before raising concurrency.
No GPU is required. The local source contains no web server or SSH interface.

Code: [MIT](LICENSE). See [security](SECURITY.md) and [third-party notices](THIRD_PARTY.md).
