"""Build only private encrypted credentials; password goes outside the bundle directory."""

import argparse
import json
from pathlib import Path

from wharton.handoff import build_private_handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("output/private-handoff/eodhd.env.enc"))
    parser.add_argument("--password-file", type=Path,
                        default=Path("output/private-handoff-password.txt"))
    args = parser.parse_args()
    try:
        print(json.dumps(build_private_handoff(args.source, args.output, args.password_file)))
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
