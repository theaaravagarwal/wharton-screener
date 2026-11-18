"""Conservative CPU and RAM bounds shared by local evaluation paths."""

import math
import os
import threading
from pathlib import Path

import psutil


def cgroup_paths():
    """Own cgroup and ancestors, respecting a mounted cgroup namespace root."""
    try:
        group = next(
            row.split(":", 2)[2]
            for row in Path("/proc/self/cgroup").read_text().splitlines()
            if row.startswith("0::")
        )
        for row in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = row.split(" - ", 1)
            if after.split()[0] != "cgroup2":
                continue
            fields = before.split()
            root, mount = Path(fields[3]), Path(fields[4])
            current = mount / Path(group).relative_to(root)
            if ".." in current.parts:
                continue
            return [current, *list(current.parents)[: len(current.parts) - len(mount.parts)]]
    except (OSError, ValueError, StopIteration):
        pass
    return [Path("/sys/fs/cgroup")]


def available_cpus():
    affinity = getattr(os, "sched_getaffinity", None)
    try:
        count = len(affinity(0)) if affinity else (os.cpu_count() or 1)
    except OSError:
        count = os.cpu_count() or 1
    for group in cgroup_paths():
        try:
            quota, period = (group / "cpu.max").read_text().split()
            if quota != "max":
                count = min(count, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, count)


def available_memory():
    available = psutil.virtual_memory().available
    for group in cgroup_paths():
        try:
            maximum = (group / "memory.max").read_text().strip()
            if maximum != "max":
                current = int((group / "memory.current").read_text())
                available = min(available, max(0, int(maximum) - current))
        except (OSError, ValueError):
            pass
    return available


def worker_limit(requested="auto", cap=None):
    if requested != "auto" and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
    ):
        raise ValueError("evaluation workers must be 'auto' or a positive integer")
    available = available_memory()
    # shortcut: reserve 20% and estimate 1.5 GiB per worker; refine with measured footprints.
    memory = max(1, int(available * 0.8 / (1.5 * 1024**3)))
    limit = min(available_cpus(), memory)
    if requested != "auto":
        limit = min(limit, requested)
    if cap is not None:
        limit = min(limit, cap)
    return max(1, limit)


def limit_threads():
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


class PeakMemory:
    """Sample aggregate parent/child RSS every 50ms, including process startup."""

    def __init__(self):
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        parent = psutil.Process()
        while not self.done.is_set():
            total = 0
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self.done.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
