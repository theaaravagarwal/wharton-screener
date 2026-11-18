"""Offline full-screen benchmark; never acquires data or changes screening criteria."""

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from wharton import screening
from wharton.resources import PeakMemory, available_cpus, available_memory, worker_limit


def write_receipt(output, report):
    if output is not None:
        output = Path(output)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)


def select_winner(rows, variants, memory_budget):
    tuning = {
        v: statistics.median(
            r["seconds"] for r in rows if r["requested"] == v and r["panel"] == "tuning"
        )
        for v in variants
    }
    candidate = min(tuning, key=tuning.get)
    holdout = {
        v: statistics.median(
            r["seconds"] for r in rows if r["requested"] == v and r["panel"] == "holdout"
        )
        for v in (1, candidate)
    }
    safe_memory = all(
        r["peak_rss_bytes"] < memory_budget for r in rows if r["requested"] == candidate
    )
    winner = (
        candidate
        if (
            safe_memory and tuning[candidate] < tuning[1] * 0.9 and holdout[candidate] <= holdout[1]
        )
        else 1
    )
    return winner, tuning


def benchmark(inputs, as_of, *, seconds=1800, output=None, input_sha256=None):
    started = time.monotonic()
    rows, reference = [], {}
    memory_budget = available_memory() * 0.8

    def partial():
        return dict(complete=False, winner=1, timings=rows, input_sha256=input_sha256)

    write_receipt(output, partial())
    # Freeze the final fifth before tuning; holdout is evaluated as its own cohort.
    split = max(1, len(inputs["universe"]) * 4 // 5)
    panels = {"tuning": inputs["universe"][:split], "holdout": inputs["universe"][split:]}
    variants = list(dict.fromkeys([1, 2, 4, 8, worker_limit("auto")]))
    for repeat in range(3):
        for requested in variants:
            for name, records in panels.items():
                if time.monotonic() - started >= seconds:
                    return partial()
                panel = dict(inputs, universe=records)
                workers = worker_limit(requested)
                start = time.perf_counter()
                cpu = time.process_time()
                with PeakMemory() as memory, threadpool_limits(limits=1):
                    result = screening.screen(panel, as_of, workers=workers)
                digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
                if name in reference and reference[name] != digest:
                    raise ValueError("full screening results differ across worker configurations")
                reference[name] = digest
                rows.append(
                    dict(
                        panel=name,
                        requested=requested,
                        workers=workers,
                        repeat=repeat + 1,
                        seconds=time.perf_counter() - start,
                        parent_cpu_seconds=time.process_time() - cpu,
                        peak_rss_bytes=memory.peak,
                        result_sha256=digest,
                    )
                )
                write_receipt(output, partial())
    winner, medians = select_winner(rows, variants, memory_budget)
    report = dict(
        complete=True,
        winner=winner,
        medians=medians,
        timings=rows,
        cpus=available_cpus(),
        available_ram_bytes=available_memory(),
        input_sha256=input_sha256,
        scope="full screening including portfolio gates; excludes acquisition and export",
    )

    write_receipt(output, report)
    return report


def supervised(input_path, as_of, output, *, seconds=1800):
    """Bound the entire run, including all executor descendants, on supported Unix hosts."""
    raw = Path(input_path).read_bytes()
    initial = dict(
        complete=False, winner=1, timings=[], input_sha256=hashlib.sha256(raw).hexdigest()
    )
    write_receipt(output, initial)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(input_path),
        "--as-of",
        as_of,
        "--output",
        str(output),
        "--worker",
    ]
    child = subprocess.Popen(
        command, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        _, error = child.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        report = json.loads(Path(output).read_text())
        report.update(complete=False, winner=1, timeout=True)
        write_receipt(output, report)
        return report
    if child.returncode:
        raise RuntimeError("benchmark child failed; partial receipt retained")
    return json.loads(Path(output).read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if args.worker:
        benchmark(
            json.loads(raw),
            args.as_of,
            output=args.output,
            input_sha256=hashlib.sha256(raw).hexdigest(),
        )
    else:
        supervised(args.input, args.as_of, args.output)
