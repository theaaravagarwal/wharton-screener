"""Public exports cannot pick up working-tree data, symlinks, or old Git history."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "export_public", Path(__file__).parents[1] / "scripts/export_public.py"
)
export_public = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_public)


@pytest.mark.parametrize("path", ["../key", "/tmp/key", ".git/config", ".env", "runs/a.py"])
def test_manifest_rejects_private_or_escaping_paths(path):
    with pytest.raises(ValueError):
        export_public.paths_from_manifest(path)


def test_manifest_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        export_public.paths_from_manifest("src/app.py\nsrc/app.py\n")


def test_export_rejects_private_machine_details_before_writing(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "public-source.txt").write_text("source.py\n")
    (root / "source.py").write_text('host = "100.64.0.' + '7"\n')
    output = tmp_path / "public"
    with pytest.raises(ValueError, match="Private machine"):
        export_public.export(output, root)
    assert not output.exists()


def test_export_rejects_symlink_even_inside_source(tmp_path):
    (tmp_path / "public-source.txt").write_text("source.py\n")
    (tmp_path / "real.py").write_text("safe = True\n")
    (tmp_path / "source.py").symlink_to(tmp_path / "real.py")
    with pytest.raises(ValueError, match="regular file"):
        export_public.export(tmp_path / "public", tmp_path)


def test_export_never_overwrites_existing_output(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        export_public.export(tmp_path)


def test_push_requires_secret_scan_of_commit_history(tmp_path, monkeypatch):
    import os
    import subprocess

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    Path("safe.py").write_text("safe = True\n")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=Synthetic", "-c",
                    "user.email=synthetic@example.invalid", "commit", "-qm", "source"], check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    scanner = tmp_path / "gitleaks"
    scanner.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > scanner-arguments\nexit 7\n')
    scanner.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    hook = Path(__file__).parents[1] / ".githooks/pre-push"
    result = subprocess.run([str(hook)], text=True, capture_output=True,
                            input=f'refs/heads/main {head} refs/heads/main {"0" * 40}\n')
    assert result.returncode == 7
    arguments = Path("scanner-arguments").read_text().splitlines()
    assert arguments[0] == "git"
    assert f"--log-opts={head}" in arguments
    assert Path(arguments[-1]) == tmp_path
