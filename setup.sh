#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
command -v uv >/dev/null 2>&1 || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ then rerun ./setup.sh'; exit 1; }
encrypted_key=
gpu=
live=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --encrypted-key) [ "$#" -ge 2 ] || exit 2; encrypted_key=$2; shift 2 ;;
        --gpu) gpu=1; shift ;;
        --live) live=1; shift ;;
        *) echo 'Usage: ./setup.sh [--encrypted-key PATH] [--gpu] [--live]'; exit 2 ;;
    esac
done
uv python install 3.12
set -- --locked --no-dev
[ -z "$gpu" ] || set -- "$@" --extra gpu
[ -z "$encrypted_key" ] || set -- "$@" --extra handoff
uv sync "$@"
if [ -n "$encrypted_key" ] && [ ! -e .env ]; then
    uv run "$@" python -m wharton.handoff "$encrypted_key"
fi
if [ ! -e .env ]; then
    (umask 077; set -C; printf 'EODHD_API_KEY=\n' > .env)
fi
uv run "$@" wharton doctor
if [ -n "$gpu" ] && [ -n "$live" ]; then
    uv run "$@" wharton verify-install --require-gpu --live
elif [ -n "$gpu" ]; then
    uv run "$@" wharton verify-install --require-gpu
elif [ -n "$live" ]; then
    uv run "$@" wharton verify-install --live
else
    uv run "$@" wharton verify-install
fi
printf '\nSetup verified. Results remain under runs/. See docs/standalone-setup.md.\n'
