"""Machine-local env overrides for the argparse->adforce config path.

The adbo drivers (exp.py, sweep_vmax.py) are argparse entry points: they build
their adforce config through adforce.wrap.get_default_config() rather than the
adforce.wrap Hydra CLI, so `files.*`/`slurm.*` dot-overrides cannot reach them.
get_default_config() therefore honours three environment variables
(ADCIRC_EXE_PATH, ADCIRC_NP, WORSTSURGE_MODULES) so those runs can be pointed
at a non-ARCHER2 machine (cloud VM/container). These tests pin that contract
in both directions: env set -> applied; env unset -> ARCHER2 defaults intact.
"""

import pytest

pytestmark = pytest.mark.usefixtures("monkeypatch")

ENV_VARS = ("ADCIRC_EXE_PATH", "ADCIRC_NP", "WORSTSURGE_MODULES")


def _clean(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_defaults_unchanged_without_env(monkeypatch):
    """No env vars -> the composed config is the untouched ARCHER2 default."""
    _clean(monkeypatch)
    from adforce.wrap import get_default_config

    cfg = get_default_config()
    assert cfg.slurm.tasks_per_node == 128
    assert cfg.slurm.reserved_cpus == 20
    assert "PrgEnv-gnu" in cfg.slurm.modules
    assert cfg.files.exe_path.startswith("/work/")  # ARCHER2 install path


def test_env_overrides_applied(monkeypatch):
    """Each env var lands on the right config field."""
    _clean(monkeypatch)
    monkeypatch.setenv("ADCIRC_EXE_PATH", "/opt/adcirc/work")
    monkeypatch.setenv("ADCIRC_NP", "16")
    monkeypatch.setenv("WORSTSURGE_MODULES", "")
    from adforce.wrap import get_default_config

    cfg = get_default_config()
    assert cfg.files.exe_path == "/opt/adcirc/work"
    assert cfg.slurm.tasks_per_node == 16
    assert cfg.slurm.reserved_cpus == 0
    # empty string -> the `if modules_to_load:` block in adforce.subprocess
    # is skipped, so no `module load` is attempted on machines without one
    assert cfg.slurm.modules == ""


def test_np_formula_through_build_wrap_config(monkeypatch, tmp_path):
    """End-to-end: the argparse pipeline (build_wrap_config) yields np = ADCIRC_NP
    for the mid resolution ((16 - 0) * 1 node), with the exe path applied."""
    _clean(monkeypatch)
    monkeypatch.setenv("ADCIRC_EXE_PATH", "/opt/adcirc/work")
    monkeypatch.setenv("ADCIRC_NP", "16")
    monkeypatch.setenv("WORSTSURGE_MODULES", "")
    from adbo.wrap_utils import build_wrap_config

    cfg = {
        "obs_lon": -90.0715,
        "obs_lat": 29.9511,
        "resolution": "mid",
        "profile_name": "2015_new_orleans_profile_r4i1p1f1",
    }
    wrap_cfg = build_wrap_config(cfg, {"angle": 0.0}, str(tmp_path))
    res = wrap_cfg.adcirc["resolution"].value
    np_ranks = (
        wrap_cfg.slurm.tasks_per_node - wrap_cfg.slurm.reserved_cpus
    ) * wrap_cfg.slurm.options[res].nodes
    assert np_ranks == 16
    assert wrap_cfg.files.exe_path == "/opt/adcirc/work"
    assert wrap_cfg.slurm.modules == ""
    assert wrap_cfg.files.run_folder == str(tmp_path)
