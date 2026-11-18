# Return screening results privately

Screening already keeps published runs under the configured state directory. Each run
contains results, gate decisions, a report, source hashes and frozen inputs. Keep that
state directory between updates; reinstalling source must not delete it.

Create one return archive after a screen has completed:

```sh
uv run --locked --no-dev python scripts/package_results.py PATH_TO_PUBLISHED_RUN --output output/results-return.zip
```

The command replays and verifies the run offline before publishing the ZIP. It refuses
an existing output path. Use another filename for a later return. Send this ZIP privately;
the command does not upload or send anything.

The archive includes selected result tables and report, a generated diagnostic event,
a replay receipt, original artifact hashes and hashes for every returned file. It excludes
raw provider evidence, private input/configuration, `.env`, arbitrary logs and API keys.
Known credential fields, query tokens and loaded environment secrets are redacted. Review
results before sharing because portfolios and research conclusions can themselves be private.
Provider data rights still apply; this archive is not intended for public redistribution.

Verify a received ZIP without extracting or executing it:

```sh
uv run --locked --no-dev python -c 'from wharton.result_bundle import verify_bundle; print(verify_bundle("output/results-return.zip"))'
```

Tampered, duplicate, unexpected, symlink and unsafe-path members fail verification.
Uncompressed archives above 256 MiB are rejected. Hash verification detects corruption;
it does not authenticate the sender. The returned archive omits raw inputs, so it cannot
independently repeat financial calculations on the receiving computer.
