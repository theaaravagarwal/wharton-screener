from types import SimpleNamespace

import pytest

from wharton import resources


def test_worker_limit_reserves_memory_and_accepts_large_hosts(monkeypatch):
    monkeypatch.setattr(resources, "available_cpus", lambda: 64)
    monkeypatch.setattr(
        resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=60 * 1024**3)
    )
    monkeypatch.setattr(
        resources.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert resources.worker_limit("auto") == 32
    assert resources.worker_limit(12) == 12
    assert resources.worker_limit(100, cap=8) == 8


def test_cpu_affinity_fallback(monkeypatch):
    monkeypatch.delattr(resources.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 3)
    monkeypatch.setattr(
        resources.Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert resources.available_cpus() == 3


@pytest.mark.parametrize("value", [True, 0, -1, None, "2"])
def test_invalid_workers(value):
    with pytest.raises(ValueError, match="positive integer"):
        resources.worker_limit(value)


def test_cgroup_bounds(monkeypatch):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    monkeypatch.setattr(resources, "cgroup_paths", lambda: [resources.Path("/synthetic")])
    values = {
        "cpu.max": "200000 100000",
        "memory.max": str(5 * 1024**3),
        "memory.current": str(2 * 1024**3),
    }
    monkeypatch.setattr(resources.Path, "read_text", lambda path: values[path.name])
    monkeypatch.setattr(
        resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=60 * 1024**3)
    )
    assert resources.available_cpus() == 2
    assert resources.worker_limit("auto") == 1


def test_benchmark_timeout_keeps_serial(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/benchmark_screening.py"
    spec = importlib.util.spec_from_file_location("benchmark_screening", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.benchmark({"universe": []}, "2026-10-28", seconds=0)
    assert report == {"complete": False, "winner": 1, "timings": [], "input_sha256": None}


def test_screen_accepts_auto_and_large_worker_request():
    from wharton import screening

    assert screening.screen({"universe": []}, "2026-10-28", workers="auto") == []
    assert screening.screen({"universe": []}, "2026-10-28", workers=32) == []


def test_nested_cgroup_uses_ancestors(monkeypatch):
    from pathlib import Path

    files = {
        "/proc/self/cgroup": "0::/user.slice/task.scope\n",
        "/proc/self/mountinfo": "30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/cgroup/user.slice/task.scope/cpu.max": "max 100000",
        "/sys/fs/cgroup/user.slice/cpu.max": "200000 100000",
        "/sys/fs/cgroup/cpu.max": "800000 100000",
        "/sys/fs/cgroup/user.slice/task.scope/memory.max": "max",
        "/sys/fs/cgroup/user.slice/memory.max": "10000",
        "/sys/fs/cgroup/user.slice/memory.current": "2000",
        "/sys/fs/cgroup/memory.max": "max",
    }

    def read(path):
        if str(path) not in files:
            raise OSError()
        return files[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    monkeypatch.setattr(
        resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=999999)
    )
    assert resources.available_cpus() == 2
    assert resources.available_memory() == 8000


def benchmark_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/benchmark_screening.py"
    spec = importlib.util.spec_from_file_location("benchmark_screening", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_cannot_choose_winner_and_can_reject_it():
    module = benchmark_module()
    rows = [
        dict(requested=v, panel=p, seconds=t, peak_rss_bytes=1)
        for v, p, t in [
            (1, "tuning", 10),
            (1, "holdout", 10),
            (2, "tuning", 8),
            (2, "holdout", 11),
            (4, "tuning", 9),
            (4, "holdout", 1),
        ]
    ]
    assert module.select_winner(rows, [1, 2, 4], 100)[0] == 1
    rows[3]["seconds"] = 9
    assert module.select_winner(rows, [1, 2, 4], 100)[0] == 2
    assert module.select_winner(rows, [1, 2, 4], 1)[0] == 1


def test_supervisor_kills_timeout_and_retains_partial(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    module = benchmark_module()
    source, output = tmp_path / "input.json", tmp_path / "receipt.json"
    source.write_text('{"universe": []}')
    launch = subprocess.Popen
    children = []

    def sleepy(*args, **kwargs):
        child = launch([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", sleepy)
    report = module.supervised(source, "2026-10-28", output, seconds=0.1)
    assert report["timeout"] and not report["complete"] and report["winner"] == 1
    assert children[0].poll() is not None
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))


def test_benchmark_persists_each_completed_row(tmp_path, monkeypatch):
    import json

    module = benchmark_module()
    writes = []
    original = module.write_receipt

    def write(path, report):
        original(path, report)
        writes.append(len(json.loads(path.read_text())["timings"]))

    monkeypatch.setattr(module, "write_receipt", write)
    monkeypatch.setattr(module, "worker_limit", lambda value: 1)
    output = tmp_path / "receipt.json"
    report = module.benchmark({"universe": []}, "2026-10-28", output=output)
    assert report["complete"]
    assert writes[0] == 0
    assert writes[1:-1] == list(range(1, len(report["timings"]) + 1))
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))
