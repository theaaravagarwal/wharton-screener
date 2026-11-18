"""Package one explicitly selected screen run for private return, without sending it."""

import argparse
from pathlib import Path

from wharton.result_bundle import package_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_results(args.run, args.output))


if __name__ == "__main__":
    main()
