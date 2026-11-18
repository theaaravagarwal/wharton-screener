"""Private, credential-free result returns; raw evidence stays on the screening computer."""

import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

RESULT_FILES = frozenset({
    "results.json", "gates.csv", "report.html", "coverage.json", "daily-changes.json",
    "identity-quarantines.json", "manual-review-queue.json",
})
RECEIPTS = frozenset({"verification.json", "runtime.json"})
MAX_BUNDLE_BYTES = 256 * 1024**2


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("bundle paths must not contain symlinks")
    return path


def _member(name):
    if not isinstance(name, str):
        raise ValueError("unsafe bundle member")
    path = PurePosixPath(name)
    if (not name or "\\" in name
            or path.is_absolute() or ".." in path.parts or str(path) != name):
        raise ValueError("unsafe bundle member")


def _sanitize(body):
    text = body.decode("utf-8")
    for key, value in os.environ.items():
        if value and re.search(r"(TOKEN|SECRET|PASSWORD|API_KEY|KEY_ID)", key, re.I):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", text)
    # Source citations can contain provider query credentials, including HTML-escaped queries.
    text = re.sub(
        r"(?i)((?:api[_-]?(?:key|token)|access_token|authorization|password|secret)"
        r"(?:&quot;|[\"'])?\s*(?:=|:|%3[dD])\s*(?:&quot;|[\"'])?)"
        r"[^\s&\"'<>\\,}]+", r"\1[REDACTED]", text,
    )
    return text.encode()


def package_results(run, output):
    """Offline replay one explicit run, then atomically create a private result ZIP.

    Refuses overwrites. The return archive is hash-verifiable, but deliberately omits the
    private inputs/raw data required to repeat financial evaluation on another computer.
    """
    from wharton import screen_service

    run, output = _path(run), _path(output)
    if output.exists():
        raise FileExistsError(output)
    original_manifest = _path(run / "manifest.json").read_bytes()
    manifest = json.loads(original_manifest)
    for name in manifest["files"]:
        _member(name)
        source = _path(run / name)
        if not source.is_relative_to(run) or not source.is_file():
            raise ValueError("unsafe source artifact")
    verification = screen_service.verify(run)
    files, originals = {}, {}
    for name in sorted(RESULT_FILES & manifest["files"].keys()):
        body = _path(run / name).read_bytes()
        if _sha(body) != manifest["files"][name]:
            raise ValueError("source artifact changed during packaging")
        originals[name] = _sha(body)
        files[name] = _sanitize(body)
    if not {"results.json", "gates.csv", "report.html"} <= files.keys():
        raise ValueError("screen run omits required results")
    if (run / "manifest.json").read_bytes() != original_manifest:
        raise ValueError("source manifest changed during packaging")
    files["verification.json"] = _json(verification)
    files["runtime.json"] = _json({
        "event": "offline_replay_and_private_return_packaged",
        "at": datetime.now(UTC).isoformat(),
        "arbitrary_runtime_logs": "excluded",
    })
    files["manifest.json"] = _json({
        "version": 1, "purpose": "private_result_return", "run_id": run.name,
        "as_of": manifest["as_of"], "synthetic": manifest["synthetic"],
        "source_manifest_sha256": _sha(original_manifest),
        "source_artifacts_sha256": originals,
        "files": {name: _sha(body) for name, body in files.items()},
        "limitations": ["Sanitized results; raw sources and private inputs excluded",
                        "Hashes establish integrity, not authenticity or investment approval"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".results-return-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, body in files.items():
                    archive.writestr(name, body)
            stream.flush()
            os.fsync(stream.fileno())
        verify_bundle(temporary)
        os.link(temporary, output)  # Atomic publication without replacing an existing return.
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def verify_bundle(path):
    """Verify a return ZIP without extracting or executing any member."""
    with ZipFile(_path(path)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("duplicate or missing bundle members")
        if sum(item.file_size for item in archive.infolist()) > MAX_BUNDLE_BYTES:
            raise ValueError("bundle exceeds 256 MiB uncompressed limit")
        for item in archive.infolist():
            _member(item.filename)
            if (item.filename not in RESULT_FILES | RECEIPTS | {"manifest.json"}
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("unexpected bundle member")
        manifest_body = archive.read("manifest.json")
        manifest = json.loads(manifest_body)
        if (manifest.get("version") != 1
                or set(manifest["files"]) != set(names) - {"manifest.json"}
                or not {"results.json", "gates.csv", "report.html", *RECEIPTS}
                <= set(manifest["files"])):
            raise ValueError("invalid bundle manifest")
        for name, expected in manifest["files"].items():
            if _sha(archive.read(name)) != expected:
                raise ValueError("bundle artifact hash mismatch")
    return {"verified": True, "files": len(names),
            "manifest_sha256": _sha(manifest_body)}
