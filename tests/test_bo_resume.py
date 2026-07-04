"""Tests for the BayesOpt resume logic in adbo/exp.py.

These tests exercise the pure-numpy resume helpers (``rebuild_initial_data``,
``count_completed_runs``, ``run_exists``) against fake experiment directories
laid out in the exact on-disk format produced by an interrupted
``adbo.exp.run_bayesopt_exp`` session:

    <exp_dir>/bo-config.json          run configuration (incl. "constraints")
    <exp_dir>/experiments.json        ledger written incrementally after each
                                      successful ADCIRC evaluation, keyed by
                                      call number ("0", "1", ...), each record
                                      {"": <run folder>, "res": <max water
                                      level, m>, <param>: <real value>, ...}
    <exp_dir>/exp_0000/config.yaml    per-run wrap config saved by
                                      adforce.config.save_config (plus ADCIRC
                                      outputs such as maxele.63.nc, which the
                                      resume scan does not need to read)

NOTE on import cost: ``adbo.exp`` imports tensorflow/gpflow/trieste at module
level (measured ~10 s in the t1 environment), which is paid once per test
session and is acceptable. The functions under test are pure numpy at call
time. ``pytest.importorskip`` is used so that environments without the heavy
stack skip rather than error.
"""

import json
import os

import numpy as np
import pytest

exp = pytest.importorskip(
    "adbo.exp", reason="adbo.exp needs tensorflow/gpflow/trieste/adforce installed"
)

CONSTRAINTS = {
    "order": ["angle", "trans_speed"],
    "angle": {"min": -80.0, "max": 80.0, "units": "degrees"},
    "trans_speed": {"min": 0.0, "max": 15.0, "units": "m s**-1"},
}

# (angle, trans_speed) in real space and observed max water level [m]
COMPLETED_RUNS = [
    {"angle": -40.0, "trans_speed": 3.0, "res": 6.5},
    {"angle": 0.0, "trans_speed": 7.5, "res": 7.25},
    {"angle": 60.0, "trans_speed": 12.0, "res": 5.0},
]


def _make_run_folder(exp_dir: str, call: int) -> str:
    """Create an exp_NNNN run folder with the per-run config file that
    adforce.config.save_config leaves behind after a completed run."""
    run_folder = os.path.join(exp_dir, f"exp_{call:04}")
    os.makedirs(run_folder, exist_ok=True)
    with open(os.path.join(run_folder, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("files:\n  run_folder: " + run_folder + "\n")
    return run_folder


def _write_ledger(exp_dir: str, records: dict) -> None:
    with open(os.path.join(exp_dir, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)


def _build_experiment(exp_dir: str, runs: list, bo_config: bool = True) -> None:
    """Build a fake (partially) completed experiment directory."""
    os.makedirs(exp_dir, exist_ok=True)
    if bo_config:
        with open(os.path.join(exp_dir, "bo-config.json"), "w", encoding="utf-8") as f:
            json.dump({"constraints": CONSTRAINTS, "exp_dir": exp_dir}, f)
    ledger = {}
    for call, run in enumerate(runs):
        run_folder = _make_run_folder(exp_dir, call)
        ledger[str(call)] = {"": run_folder, **run}
    _write_ledger(exp_dir, ledger)


def _expected_scaled_queries(runs: list) -> np.ndarray:
    order = CONSTRAINTS["order"]
    mins = np.array([CONSTRAINTS[p]["min"] for p in order])
    diffs = np.array([CONSTRAINTS[p]["max"] - CONSTRAINTS[p]["min"] for p in order])
    real = np.array([[run[p] for p in order] for run in runs])
    return (real - mins) / diffs


def test_rebuild_three_completed_one_incomplete(tmp_path, capsys) -> None:
    """3 completed runs are rebuilt in call order; the incomplete 4th folder
    (crashed mid-ADCIRC, so never reached the ledger) is skipped with a warning."""
    exp_dir = str(tmp_path / "bo_test")
    _build_experiment(exp_dir, COMPLETED_RUNS)
    _make_run_folder(exp_dir, 3)  # incomplete: folder + config.yaml, no ledger entry

    queries, observations = exp.rebuild_initial_data(exp_dir, constraints=CONSTRAINTS)

    assert queries.shape == (3, 2)
    assert observations.shape == (3, 1)
    assert queries.dtype == np.float64 and observations.dtype == np.float64
    # values and ordering (by call number), in the scaled [0, 1] space
    np.testing.assert_allclose(queries, _expected_scaled_queries(COMPLETED_RUNS))
    assert np.all((queries >= 0.0) & (queries <= 1.0))
    # observations carry the objective's negated sign convention
    np.testing.assert_allclose(
        observations, -np.array([[run["res"]] for run in COMPLETED_RUNS])
    )
    out = capsys.readouterr().out
    assert "exp_0003" in out
    assert "arning" in out  # "Warning"/"warning"


def test_rebuild_empty_and_missing_dir(tmp_path) -> None:
    """Empty directory and non-existent directory both give empty arrays with
    the right trailing dimensions."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    for direc in (str(empty_dir), str(tmp_path / "does_not_exist")):
        queries, observations = exp.rebuild_initial_data(
            direc, constraints=CONSTRAINTS
        )
        assert queries.shape == (0, 2)
        assert observations.shape == (0, 1)


def test_rebuild_reads_constraints_from_bo_config(tmp_path) -> None:
    """Without an explicit constraints argument, the constraints saved in
    bo-config.json are used (pure-json read, no tf/trieste involved)."""
    exp_dir = str(tmp_path / "bo_test")
    _build_experiment(exp_dir, COMPLETED_RUNS, bo_config=True)

    queries, observations = exp.rebuild_initial_data(exp_dir)

    np.testing.assert_allclose(queries, _expected_scaled_queries(COMPLETED_RUNS))
    np.testing.assert_allclose(
        observations, -np.array([[run["res"]] for run in COMPLETED_RUNS])
    )


def test_rebuild_stops_at_corrupt_record(tmp_path, capsys) -> None:
    """A ledger record with a missing parameter truncates the completed prefix
    (call numbering must stay contiguous) and prints a warning."""
    exp_dir = str(tmp_path / "bo_test")
    corrupt = {"angle": 10.0, "res": 4.0}  # "trans_speed" missing
    _build_experiment(exp_dir, COMPLETED_RUNS[:2] + [corrupt])

    queries, observations = exp.rebuild_initial_data(exp_dir, constraints=CONSTRAINTS)

    assert queries.shape == (2, 2)
    np.testing.assert_allclose(queries, _expected_scaled_queries(COMPLETED_RUNS[:2]))
    np.testing.assert_allclose(
        observations, -np.array([[run["res"]] for run in COMPLETED_RUNS[:2]])
    )
    out = capsys.readouterr().out
    assert "arning" in out
    assert "2" in out  # names the offending ledger entry


def test_rebuild_stops_at_gap_in_ledger(tmp_path, capsys) -> None:
    """A gap in the ledger (entry 1 missing) means only the contiguous prefix
    is reused; later folders are reported as skipped."""
    exp_dir = str(tmp_path / "bo_test")
    _build_experiment(exp_dir, COMPLETED_RUNS)
    ledger_path = os.path.join(exp_dir, "experiments.json")
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    del ledger["1"]
    _write_ledger(exp_dir, ledger)

    queries, observations = exp.rebuild_initial_data(exp_dir, constraints=CONSTRAINTS)

    assert queries.shape == (1, 2)
    np.testing.assert_allclose(queries, _expected_scaled_queries(COMPLETED_RUNS[:1]))
    out = capsys.readouterr().out
    assert "exp_0001" in out and "exp_0002" in out


def test_run_exists_partial_completion_logic(tmp_path) -> None:
    """run_exists: completed >= expected means done; 0 < completed < expected
    means resumable (returns False so run_bayesopt_exp resumes)."""
    root = str(tmp_path)
    exp_name = "bo_partial"
    exp_dir = os.path.join(root, exp_name)

    # no experiment at all
    assert exp.run_exists(exp_name, num_runs=4, root_exp_direc=root) is False

    # partially completed: 3 of 4 runs -> resumable, not done
    _build_experiment(exp_dir, COMPLETED_RUNS)
    assert exp.count_completed_runs(exp_dir) == 3
    assert exp.run_exists(exp_name, num_runs=4, root_exp_direc=root) is False

    # exactly complete
    assert exp.run_exists(exp_name, num_runs=3, root_exp_direc=root) is True

    # more completed than expected (>= semantics, not exact equality)
    assert exp.run_exists(exp_name, num_runs=2, root_exp_direc=root) is True
