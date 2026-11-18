"""Build a new source-only repository; never copy history or unlisted files."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(
    r"/Users/" + r"aa(?:/|\b)|/home/" + r"software/|theaa"
    + r"@|100\.64\.0\.(?:1|3|7|9)\b"
)


def paths_from_manifest(manifest):
    paths = [line.strip() for line in manifest.splitlines() if line.strip()
             and not line.startswith("#")]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate public manifest path")
    for path in paths:
        name = PurePosixPath(path)
        if name.is_absolute() or ".." in name.parts or str(name) != path:
            raise ValueError("Unsafe public manifest path")
        if name.parts[0] in {".git", "data", "runs", "output", "control", ".worktrees"}:
            raise ValueError("Private root in public manifest")
        if name.name.startswith(".env") and name.name != ".env.example":
            raise ValueError("Credential file in public manifest")
    return paths


def export(output, root=ROOT):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Public output must be a new directory")
    paths = paths_from_manifest((root / "public-source.txt").read_text())
    contents = {}
    for name in paths:
        source = root / name
        if source.is_symlink() or not source.is_file() or any(
            part.is_symlink() for part in source.parents if part != root.parent
        ):
            raise ValueError(f"Public source must be a regular file: {name}")
        body = source.read_bytes()
        if PRIVATE.search(body.decode("utf-8")):
            raise ValueError(f"Private machine detail in public source: {name}")
        contents[name] = body
    contents["README.md"] = contents["public/README.md"]
    contents["src/wharton/cli.py"] = contents["src/wharton/public_cli.py"]
    scanner = shutil.which("gitleaks")
    if not scanner:
        raise ValueError("Gitleaks must be installed before creating a public export")
    output.mkdir(parents=True)
    try:
        for name, body in contents.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            target.chmod((root / name).stat().st_mode & 0o777)
        receipt = [{"path": name, "sha256": hashlib.sha256(body).hexdigest()}
                   for name, body in contents.items()]
        (output / "public-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
        subprocess.run(["git", "init", "-q", "--initial-branch=main", str(output)], check=True)
        subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "add", "--", "."],
                       cwd=output, check=True)
        subprocess.run(["python3", "scripts/check_publication.py"], cwd=output, check=True,
                       stdout=subprocess.DEVNULL)
        # Scanner output stays private even if a future scanner version weakens redaction.
        subprocess.run([scanner, "dir", "--redact", "--no-banner", str(output)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "-c", "user.name=Source Release", "-c",
                        "user.email=source-release@example.invalid", "-c", "core.hooksPath=",
                        "-c", "commit.gpgsign=false",
                        "commit", "-qm", "Initial reviewed source release"], cwd=output, check=True)
        subprocess.run([scanner, "git", "--redact", "--no-banner", str(output)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=output)
        if count.strip() != b"1":
            raise ValueError("Public repository must contain exactly one clean root commit")
    except Exception:
        # Leave failed output for private inspection; never retry into it or publish it.
        raise RuntimeError("Public export failed verification; keep output private") from None
    return {"files": len(contents), "commits": 1, "history_secret_scan": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output)))
