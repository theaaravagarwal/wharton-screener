"""CPU evaluation stays deterministic and never owns provider acquisition."""

import copy
import json

import pytest
from test_screening_service import AS_OF, config, record, risk_inputs

from wharton import screen_service as service
from wharton import screening


@pytest.mark.parametrize("selected", [None, {"TEST.US", "BANK.US", "MISSING.US"}])
def test_parallel_matches_serial_with_portfolio_and_adverse_evidence(selected):
    inputs, _ = risk_inputs()
    inputs["market_as_of"] = AS_OF
    inputs["universe"].extend(
        [
            record("BANK.US", "bank"),
            record("INSURER.US", "insurer"),
            record("FUND.US", "etf"),
            record("MISSING.US"),
            record("QUARANTINE.US"),
            *[record(f"PEER{i}.US") for i in range(25)],
        ]
    )
    inputs["universe"][6]["evidence"] = {}
    inputs["universe"][7]["identity_quarantine"] = {
        "reason": "synthetic issuer conflict",
        "source": "synthetic:test",
        "as_of": AS_OF,
    }
    original = copy.deepcopy(inputs)
    serial = screening.screen(inputs, AS_OF, selected, workers=1)
    parallel = screening.screen(inputs, AS_OF, selected, workers=4)
    assert json.dumps(parallel, sort_keys=True) == json.dumps(serial, sort_keys=True)
    assert inputs == original


@pytest.mark.parametrize("workers", [1, 4])
def test_evaluation_error_propagates(workers):
    with pytest.raises(ValueError, match="as_of must be an ISO date"):
        screening.screen({"universe": [record()]}, "invalid", workers=workers)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4", None])
def test_workers_and_config_reject_invalid_values(tmp_path, workers):
    with pytest.raises(ValueError, match="positive integer"):
        screening.screen({"universe": []}, AS_OF, workers=workers)
    path = config(tmp_path)
    settings = json.loads(path.read_text())
    settings["evaluation_workers"] = workers
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="positive integer"):
        service.load_config(path)


def test_config_defaults_and_service_forward_workers(tmp_path, monkeypatch):
    path = config(tmp_path)
    assert service.load_config(path)["evaluation_workers"] == 1
    settings = json.loads(path.read_text())
    settings["evaluation_workers"] = 4
    path.write_text(json.dumps(settings))
    assert service.load_config(path)["evaluation_workers"] == 4
    calls = []

    def stop(inputs, as_of, selected, *, workers):
        calls.append(workers)
        raise RuntimeError("stop before artifact construction")

    monkeypatch.setattr(screening, "screen", stop)
    with pytest.raises(RuntimeError, match="stop before artifact"):
        service.artifacts({}, settings, set(), {})
    assert calls == [4]


def test_parallel_submissions_are_bounded_and_keep_input_order(monkeypatch):
    batch_sizes = []
    monkeypatch.setattr(screening, "worker_limit", lambda workers: workers)

    class Pool:
        def __init__(self, *, max_workers, initializer):
            assert max_workers == 4

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, records, cutoffs, market_cutoffs, *, chunksize):
            assert chunksize == 8
            batch_sizes.append(len(records))
            return map(function, records, cutoffs, market_cutoffs)

    monkeypatch.setattr(screening, "ProcessPoolExecutor", Pool)
    records = [dict(ticker=f"ROW{i}.US", issuer_id=f"ROW{i}") for i in range(133)]
    inputs = {"universe": records}
    assert screening.screen(inputs, AS_OF, workers=4) == screening.screen(inputs, AS_OF)
    assert batch_sizes == [128, 5]


def test_empty_universe_does_not_start_pool(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("empty universe needs no worker processes")

    monkeypatch.setattr(screening, "ProcessPoolExecutor", unexpected)
    assert screening.screen({"universe": []}, AS_OF, workers=4) == []
