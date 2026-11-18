import json
from zipfile import ZipFile

import pytest
from test_screening_service import config

from wharton import result_bundle, screen_service


def test_real_replay_bundle_excludes_secrets_and_raw_inputs(tmp_path, monkeypatch):
    report = screen_service.run(config(tmp_path))
    run = report.parent
    (run / '.env').write_text('EODHD_API_KEY=do-not-copy')
    (run / 'debug.log').write_text('private-log-do-not-copy')
    output = tmp_path / 'return.zip'
    assert result_bundle.package_results(run, output) == output
    assert result_bundle.verify_bundle(output)['verified']
    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {'results.json', 'report.html', 'gates.csv', 'verification.json'} <= names
        assert not {'.env', 'input.json', 'config.json', 'debug.log'} & names
        assert b'do-not-copy' not in b''.join(archive.read(n) for n in names)
    assert output.stat().st_mode & 0o077 == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        result_bundle.package_results(run, output)
    assert output.read_bytes() == before


def test_sanitize_credentials_preserves_json(monkeypatch):
    monkeypatch.setenv('EODHD_API_KEY', 'synthetic-key-value')
    body = json.dumps({'source': 'https://example.invalid/?api_token=other-secret&fmt=json',
                       'value': 'synthetic-key-value', 'authorization': 'Bearer opaque-token'})
    cleaned = result_bundle._sanitize(body.encode())
    assert b'synthetic-key-value' not in cleaned
    assert b'other-secret' not in cleaned
    assert b'opaque-token' not in cleaned
    json.loads(cleaned)


def test_tampered_return_rejected(tmp_path):
    run = screen_service.run(config(tmp_path)).parent
    original = result_bundle.package_results(run, tmp_path / 'original.zip')
    changed = tmp_path / 'changed.zip'
    with ZipFile(original) as source, ZipFile(changed, 'w') as target:
        for item in source.infolist():
            target.writestr(item, b'[]' if item.filename == 'results.json' else source.read(item))
    with pytest.raises(ValueError, match='hash mismatch'):
        result_bundle.verify_bundle(changed)


@pytest.mark.parametrize('name', ['../escape', '/absolute', r'..\escape', 'nested/../escape'])
def test_unsafe_archive_members_rejected(tmp_path, name):
    archive = tmp_path / 'unsafe.zip'
    with ZipFile(archive, 'w') as target:
        target.writestr('manifest.json', '{}')
        target.writestr(name, 'bad')
    with pytest.raises(ValueError, match='unsafe'):
        result_bundle.verify_bundle(archive)


def test_symlink_source_rejected(tmp_path):
    run = screen_service.run(config(tmp_path)).parent
    output = tmp_path / 'return.zip'
    source = run / 'results.json'
    copied = tmp_path / 'real-results.json'
    copied.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(copied)
    with pytest.raises(ValueError, match='symlinks'):
        result_bundle.package_results(run, output)
    assert not output.exists()


def test_source_tamper_rejected(tmp_path):
    run = screen_service.run(config(tmp_path)).parent
    (run / 'results.json').write_text('[]')
    with pytest.raises(ValueError, match='hash mismatch'):
        result_bundle.package_results(run, tmp_path / 'return.zip')


def test_oversized_bundle_rejected_before_read(tmp_path, monkeypatch):
    archive = tmp_path / 'large.zip'
    with ZipFile(archive, 'w') as target:
        target.writestr('manifest.json', '{}')
    monkeypatch.setattr(result_bundle, 'MAX_BUNDLE_BYTES', 1)
    with pytest.raises(ValueError, match='limit'):
        result_bundle.verify_bundle(archive)


def test_package_cli_serializes_archive_path(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import wharton.result_bundle as module
    from wharton.public_cli import app
    target = tmp_path / "result.zip"
    monkeypatch.setattr(module, "package_results", lambda run, output: target)
    result = CliRunner().invoke(app, ["package-results", str(tmp_path), str(target)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["archive"] == str(target)
