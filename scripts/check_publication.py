"""Check staged source publication; never read ignored working-tree data."""

import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

DATA_SUFFIXES = {'.csv', '.parquet', '.feather', '.arrow', '.db', '.sqlite', '.sqlite3', '.pdf',
                 '.jsonl', '.ndjson', '.gz', '.zst'}
LOCAL_ROOTS = {'data', 'runs', 'output', 'dist', 'control', '.worktrees', '.claude', '.agent',
               '.scratch'}


def is_news_store(path):
    # News state may sit under any state_root; synthetic fixtures keep the fixture rule.
    parts = PurePosixPath(path).parts
    if parts[:3] == ('tests', 'fixtures', 'news'):
        return False
    return any(p.lower() == 'news' for p in parts)


def staged_ignored_paths(entries):
    # Git evaluates staged ignore rules in isolation, never the edited working copy.
    local_vars = subprocess.check_output(['git', 'rev-parse', '--local-env-vars']).decode().split()
    env = {key: value for key, value in os.environ.items() if key not in local_vars}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    paths = []
    with tempfile.TemporaryDirectory(prefix='publication-ignore-') as directory:
        subprocess.run(['git', 'init', '-q', directory], env=env, check=True)
        for entry in filter(None, entries):
            metadata, path = entry.split(b'\t', 1)
            paths.append(path)
            mode, blob, stage = metadata.decode().split()
            if PurePosixPath(os.fsdecode(path)).name == '.gitignore' and stage == '0':
                target = Path(directory) / os.fsdecode(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(['git', 'cat-file', 'blob', blob]))
        if not paths:
            return set()
        result = subprocess.run(
            ['git', '-c', 'core.excludesFile=' + os.devnull, 'check-ignore', '--stdin', '-z'],
            cwd=directory, env=env, input=b'\0'.join(paths) + b'\0', capture_output=True
        )
        if result.returncode not in (0, 1):
            raise RuntimeError('Could not evaluate staged ignore policy')
        return set(result.stdout.split(b'\0'))


def check_index():
    failures = []
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).split(b'\0')
    ignored_paths = staged_ignored_paths(entries)
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b'\t', 1)
        mode, blob, stage = metadata.decode().split()
        path = raw_path.decode('utf-8', errors='surrogateescape')
        name = PurePosixPath(path)
        if stage != '0':
            failures.append(path + ': unresolved conflict')
            continue
        ignored = raw_path in ignored_paths
        if (ignored or name.parts[0] in LOCAL_ROOTS or name.suffix.lower() in DATA_SUFFIXES
                or is_news_store(path)
                or (name.name.startswith('.env') and name.name != '.env.example')):
            failures.append(path + ': local-only material')
            continue
        if mode != '100644' and mode != '100755':
            failures.append(path + ': unsupported publication file mode')
            continue
        content = subprocess.check_output(['git', 'cat-file', 'blob', blob])
        if len(content) > 1024 * 1024:
            failures.append(path + ': exceeds 1 MiB source-file limit')
        if name.suffix == '.json' and path.startswith('tests/fixtures/'):
            payload = json.loads(content)
            if not isinstance(payload, dict) or payload.get('synthetic') is not True:
                failures.append(path + ': fixture must explicitly declare synthetic data')
        if name.suffix == '.ipynb':
            notebook = json.loads(content)
            if any(cell.get('outputs') or cell.get('execution_count') is not None
                   for cell in notebook.get('cells', [])):
                failures.append(path + ': notebook contains executed output')
    return failures


if __name__ == '__main__':
    errors = check_index()
    print('\n'.join(errors) if errors else 'Staged publication policy: OK')
    raise SystemExit(bool(errors))
