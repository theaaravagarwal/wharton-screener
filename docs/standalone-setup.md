# Standalone setup

Supports macOS and Linux, with Python 3.12 managed by uv. Windows uses WSL2.
Install uv, then run `./setup.sh`. This installs dependencies and runs synthetic
checks without provider access. Existing `.env` and run state are preserved.

Put an entitled `EODHD_API_KEY` in local `.env`, with `chmod 600 .env`.
Never send keys through Git, command arguments, logs, or public issues.
EODHD personal subscriptions do not permit sharing a key between people.

Discover a fresh universe without SSH or old-host data:

```sh
uv run --locked --no-dev wharton init-screen --live --exchange US
```

Add `--exchange XETRA` or other supported exchanges as needed. Inventory requests
cost [one call each](https://eodhd.com/financial-apis/api-limits); initialization caps charged credits at 500 and preserves
accounting state. Interrupted runs reuse same-day hashed inventory receipts.
The account guard keeps the 75,000 daily limit and 25,000 base headroom.
Only one machine may acquire paid data for this account. Unresolved requests
must be reconciled; never delete accounting state to bypass a hold.

Discovery writes `runs/standalone/input.json` and `screen.yaml`. Acquisition
remains disabled. To refresh evidence, deliberately set `acquisition.enabled`
to `true` in that local YAML, then run:

```sh
uv run --locked --no-dev wharton screen --config runs/standalone/screen.yaml
uv run --locked --no-dev wharton screen-status runs/standalone/state
uv run --locked --no-dev wharton verify-screen PATH_PRINTED_BY_SCREEN
```

Inventories create provisional listing identities, not verified legal issuers.
The existing normalizer establishes evidence or emits holds. Missing primary
sources, issuer classification, holdings and human reviews remain visible;
a successful run does not mean a stock qualifies or is approved to trade.
No acquired data belongs in the public source repository.

Initialization merges new listings while preserving existing evidence, reviews, holdings,
and other input fields. Generated configuration does not pin an acquisition date; each
screen resolves the current UTC day, unless `screen --as-of` is explicitly supplied.
Inventory ISINs seed claimed identity only. The existing fundamentals normalizer must
match the qualified listing and ISIN before identity admission. Missing identifiers
stay held; contradictory inventory identifiers stop initialization.

## Optional encrypted private credential handoff

The public source contains no keys or encrypted credentials. Install the optional
`handoff` dependency only when building or unlocking a private credential bundle.

On the sender computer, with a private `.env` (permissions 0600):

```sh
uv run --locked --extra handoff python scripts/build_private_handoff.py
```

The helper encrypts only `EODHD_API_KEY` into
`output/private-handoff/eodhd.env.enc`. Its generated 8-character password is saved
separately in `output/private-handoff-password.txt`, also with permissions 0600.
Never put that password file in the bundle. Transfer the encrypted file privately;
send the password through a separate channel. Do not publish either file.

On the recipient computer, from the source checkout:

```sh
./setup.sh --encrypted-key /path/to/eodhd.env.enc
```

Enter the password at the hidden terminal prompt. No password argument or
environment variable is accepted. Unlock authenticates the ciphertext before
atomically creating `.env` with permissions 0600. Existing `.env` files are never
overwritten; if setup previously created an empty placeholder, remove that empty
file deliberately before unlocking. Wrong passwords and modified ciphertext leave
no destination credentials. The generated password cannot be recovered from the
encrypted file; retain it securely until the recipient verifies successful unlock.

Encryption uses [cryptography Fernet](https://cryptography.io/en/latest/fernet/)
with a random salt and fixed Scrypt password derivation (128 MiB working memory).
It protects credentials in transit; the unlocked local `.env` remains a secret.
Provider account and data entitlements still apply.

## Mac mini / Mac Studio verified installation

On an Apple Silicon Mac, run `./setup.sh --gpu --live` to install the optional Metal backend and require successful offline, GPU and guarded live API checks. For the private encrypted handoff, add `--encrypted-key eodhd.env.enc`; enter the separately supplied password at the hidden prompt. An existing `.env` is preserved.

Checks write `runs/verification/install.json` and exit nonzero on failure. After installing the reviewed source update, rerun `./reinstall.sh --gpu --live`; credentials and run state remain. Intel Macs can run CPU mode; explicitly requiring unavailable Metal fails. Linux is tested separately and is not required for a Mac install. GPU correctness is verified against CPU float64; small workloads may run faster on CPU.

Send back a private result archive with `wharton package-results RUN_DIRECTORY output/results.zip`; see [returning results](results-return.md). No automatic transmission occurs.

## Transfer privately over SSH

Copy the private bundle over your existing SSH connection:

```sh
scp output/wharton-mac-private-8char.zip username@mac-mini:~/Downloads/
```

Send the password separately. SSH access must already be configured on the receiving Mac; the installer itself does not require SSH. The eight-character password contains upper/lowercase letters, digits and symbols; it provides less resistance to offline guessing than a longer password. Never publish the encrypted key or password in GitHub.
