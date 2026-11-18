"""Publication checks use the index, including force-added and staged-only data."""

import importlib.util
import json
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'publication', Path(__file__).parents[1] / 'scripts/check_publication.py'
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_publication_rejects_data_even_when_working_copy_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    Path('.gitignore').write_text('private/\ncustom-packet.json\n')
    Path('private').mkdir()
    Path('private/packet.json').write_text('{}')
    Path('renamed.csv').write_text('provider,price\nSYNTH,123\n')
    Path('research.ipynb').write_text(json.dumps({'cells': [
        {'outputs': [{'text': 'private values'}], 'execution_count': 1}
    ]}))
    Path('custom-packet.json').write_text('{}')
    Path('safe.py').write_text('print("safe source")\n')
    subprocess.run(['git', 'add', '-f', '.'], check=True)
    Path('research.ipynb').write_text('{"cells": []}')
    Path('.gitignore').write_text('')
    errors = publication.check_index()
    assert any('private/packet.json' in error for error in errors)
    assert any('custom-packet.json' in error for error in errors)
    assert any('renamed.csv' in error for error in errors)
    assert any('research.ipynb' in error for error in errors)
    assert not any('safe.py' in error for error in errors)


def test_publication_requires_synthetic_fixtures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    folder = Path('tests/fixtures')
    folder.mkdir(parents=True)
    (folder / 'raw.json').write_text('{"data": []}')
    (folder / 'synthetic.json').write_text('{"synthetic": true, "data": []}')
    subprocess.run(['git', 'add', '.'], check=True)
    assert publication.check_index() == [
        'tests/fixtures/raw.json: fixture must explicitly declare synthetic data'
    ]


def test_push_checks_outbound_commit_without_changing_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    Path('leaked.csv').write_text('SYNTH,123\n')
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run([
        'git', '-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid',
        'commit', '-qm', 'synthetic commit'
    ], check=True)
    subprocess.run(['git', 'rm', '--cached', '-q', 'leaked.csv'], check=True)
    subprocess.run([
        'git', '-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.invalid',
        'commit', '-qm', 'clean tip with a dirty ancestor'
    ], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    index_before = Path('.git/index').read_bytes()
    hook = Path(__file__).parents[1] / '.githooks/pre-push'
    result = subprocess.run(
        [str(hook)], input=f'refs/heads/test {sha} refs/heads/test {"0" * 40}\n',
        text=True, capture_output=True
    )
    assert result.returncode != 0
    assert 'leaked.csv: local-only material' in result.stdout
    assert Path('.git/index').read_bytes() == index_before


def test_push_rejects_blob_and_tree_tags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(['git', 'init', '-q'], check=True)
    hook = Path(__file__).parents[1] / '.githooks/pre-push'
    for kind, content in [('blob', b'SYNTH,123'), ('tree', b'')]:
        sha = subprocess.check_output(
            ['git', 'hash-object', '-w', '-t', kind, '--stdin'], input=content
        ).decode().strip()
        result = subprocess.run(
            [str(hook)], input=f'refs/tags/test {sha} refs/tags/test {"0" * 40}\n',
            text=True, capture_output=True
        )
        assert result.returncode != 0
        assert not Path('.git/index').exists()
