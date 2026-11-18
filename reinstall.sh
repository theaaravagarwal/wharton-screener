#!/bin/sh
# Reinstall dependencies from the reviewed local source; never reset source or run state.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
./setup.sh "$@"
printf '\nReinstall complete. Existing credentials and run state were preserved.\n'
