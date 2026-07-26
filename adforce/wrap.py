"""Wrap the adcirc call."""

import os, shutil
import numpy as np
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sithom.time import timeit
from .constants import CONFIG_PATH, SETUP_PATH
from .fort22 import create_fort22
from .slurm import setoff_slurm_job_and_wait
from .subprocess import setoff_subprocess_job_and_wait
from .mesh import xr_loader
from .config import save_config


class DryObservationPoint(ValueError):
    """The ADCIRC run succeeded (zeta_max valid over essentially the whole
    mesh) but the observation node itself never wetted. Surge optimizers may
    legitimately score this as 0.0 m."""


class AdcircRunFailure(RuntimeError):
    """zeta_max is invalid across (most of) the mesh: the run crashed, blew
    up, or its output was truncated (e.g. disk full). Must never be scored as
    a physical result. Deliberately not a ValueError so that callers catching
    DryObservationPoint cannot swallow it by accident."""


# healthy runs on the mid mesh have valid zeta_max at ~100% of nodes (measured
# 1.000 on 31435-node runs); a crashed/truncated run leaves fill values nearly
# everywhere, so 0.5 separates the two regimes with a wide margin either side
MIN_VALID_ZETA_FRACTION = 0.5


def observe_max_point(cfg: DictConfig) -> float:
    """Observe the ADCIRC model.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        float: max water level at the node nearest the observation point.

    Raises:
        DryObservationPoint: the run is healthy but the observation node
            stayed dry (zeta_max is NaN/fill there only).
        AdcircRunFailure: zeta_max is NaN/fill over most of the mesh, i.e.
            the ADCIRC run itself failed.
    """
    mele_ds = xr_loader(os.path.join(cfg.files.run_folder, "maxele.63.nc"))
    # print("mele_ds", mele_ds)
    xs = mele_ds.x.values
    ys = mele_ds.y.values
    at_obs_loc = cfg.adcirc.attempted_observation_location.value

    distsq = (xs - at_obs_loc[0]) ** 2 + (ys - at_obs_loc[1]) ** 2
    min_p = np.argmin(distsq)
    maxele = mele_ds.zeta_max.isel(node=min_p).values
    point = (xs[min_p], ys[min_p])
    cfg.adcirc["actual_observation_location"]["value"][0] = float(point[0])
    cfg.adcirc["actual_observation_location"]["value"][1] = float(point[1])

    print(
        "point info:",
        mele_ds.isel(node=min_p)["depth"],
        "\n depth at point: ",
        mele_ds.isel(node=min_p)["depth"].values,
        " m",
    )

    # guard against NaN/netCDF fill values (e.g. 9.96921e36) from failed or dry runs
    if not np.isfinite(maxele) or abs(maxele) >= 100:
        zeta_all = np.asarray(mele_ds.zeta_max.values).ravel()
        valid_fraction = float(
            np.mean(np.isfinite(zeta_all) & (np.abs(zeta_all) < 100))
        )
        where = (
            f"invalid max water level {maxele} m at node {min_p} "
            f"(lon, lat) = ({point[0]}, {point[1]}) "
            f"for run folder {cfg.files.run_folder} "
            f"(zeta_max valid at {valid_fraction:.1%} of nodes)"
        )
        if valid_fraction < MIN_VALID_ZETA_FRACTION:
            raise AdcircRunFailure(
                f"{where}: the ADCIRC run failed (crash/blow-up/truncated "
                "output) — refusing to treat this as a physical result."
            )
        raise DryObservationPoint(
            f"{where}: the run is healthy but the observation node never "
            "wetted; zero surge at the point is the honest interpretation."
        )

    return maxele


RSTIMINC_S = 1200  # SWAN coupling interval [s]; must equal fort.26 COMPUTE dt


def _enable_swan_in_fort15(path: str, rstiminc: int = RSTIMINC_S) -> None:
    """Patch a fort.15 for SWAN coupling in place: NWS += 300 (e.g. 13 -> 313,
    20 -> 320) and append RSTIMINC to the WTIMINC line (coupling interval,
    seconds; RSTIMINC/DT must be a positive integer or ADCIRC SIGFPEs at the
    first wave call). Idempotent: an NWS already >= 300 is left alone."""
    with open(path) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if "! NWS" in line and "WTIMINC" not in line:
            head, _, comment = line.partition("!")
            nws = int(head.split()[0])
            if nws < 300:
                head = head.replace(str(nws), str(nws + 300), 1)
            line = head + "!" + comment
        elif "! WTIMINC" in line:
            head, _, comment = line.partition("!")
            tokens = head.split()
            if str(rstiminc) not in tokens[1:]:
                pad = len(head) - len(head.rstrip()) - len(f" {rstiminc}")
                head = head.rstrip() + f" {rstiminc}" + " " * max(pad, 1)
            line = head + "!" + comment
        out.append(line)
    with open(path, "w") as f:
        f.writelines(out)


def stage_input_files(
    run_folder: str, resolution: str = "mid", tide: bool = False, swan: bool = False
) -> None:
    """Copy the per-option ADCIRC input deck into ``run_folder``.

    * ``resolution``: chooses ``fort.14.{res}``/``fort.13.{res}`` and the
      ``fort.15.{res}.{tide|notide}`` template (only "low" and "mid" ship in
      the repo; "high" lived on ARCHER2).
    * ``tide``: picks the tidal template. NOTE: the tidal templates carry a
      fixed cold-start/RNDAY window (2005-08-19 + 13 days); the forcing grid
      (cfg.grid) must cover it.
    * ``swan``: patches fort.15 for coupling (NWS += 300, RSTIMINC) and
      stages ``fort.26`` + ``swaninit`` (which must name fort.26, not the
      SWAN default INPUT). The run must then use ``padcswan`` from a
      SWAN-coupled build (adforce.subprocess selects it from this flag).
    """
    templates = {
        "fort.15": f"fort.15.{resolution}.{'tide' if tide else 'notide'}",
        "fort.13": f"fort.13.{resolution}",
        "fort.14": f"fort.14.{resolution}",
    }
    if swan:
        templates["fort.26"] = "fort.26"
        templates["swaninit"] = "swaninit"
    for dst, src in templates.items():
        src_path = os.path.join(SETUP_PATH, src)
        if not os.path.exists(src_path):
            raise FileNotFoundError(
                f"ADCIRC setup file {src} not found in {SETUP_PATH} "
                f"(resolution={resolution!r}, tide={tide}, swan={swan}); "
                "only low/mid templates ship with the repo."
            )
        shutil.copy(src_path, os.path.join(run_folder, dst))
    if swan:
        _enable_swan_in_fort15(os.path.join(run_folder, "fort.15"))


@timeit
def idealized_tc_observe(cfg: DictConfig) -> float:
    """Wrap the adcirc call.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        float: max water level at observation point.
    """
    os.makedirs(cfg.files.run_folder, exist_ok=True)
    # transfer relevant ADCIRC setup files per the adcirc options
    # (resolution / tide / swan -- see stage_input_files)
    stage_input_files(
        cfg.files.run_folder,
        resolution=cfg.adcirc.resolution.value,
        tide=bool(cfg.adcirc.tide.value),
        swan=bool(getattr(cfg.adcirc, "swan", None) and cfg.adcirc.swan.value),
    )

    print("ADFORCE cfg:", cfg)

    # save config file
    save_config(cfg)
    # create forcing files
    create_fort22(cfg.files.run_folder, cfg.grid, cfg.tc)
    # run ADCIRC
    if cfg.use_slurm:
        setoff_slurm_job_and_wait(cfg.files.run_folder, cfg)
    else:
        setoff_subprocess_job_and_wait(cfg.files.run_folder, cfg)
    # observe ADCIRC
    maxele = observe_max_point(cfg)
    # save config file
    save_config(cfg)
    print("max height at obs point: ", maxele, " m\n\n")

    if cfg.ani:
        from adforce.ani import plot_heights_and_winds

        plot_heights_and_winds(os.path.join(cfg.files.run_folder), step_size=10)

    if cfg.files.low_storage:
        # Delete the bulky input + full-field time-series outputs (~1 GB per
        # run) and the PE* partition dirs, keeping maxele.63.nc & other max*
        # summaries (the science) and the small setup/log files. Doing this
        # here, after a *successful* observe, is race-free — an external
        # janitor loop sweeping exp dirs on a timer once deleted the ACTIVE
        # run's PE dirs mid-startup and produced all-NaN maxele files (see
        # AdcircRunFailure / rerun/results/acquisition_mes_vs_ei.md). A raised
        # observe_max_point above skips this, leaving failures for forensics.
        freed = 0
        for name in (
            "fort.22.nc",
            "fort.63.nc",
            "fort.64.nc",
            "fort.73.nc",
            "fort.74.nc",
        ):
            path = os.path.join(cfg.files.run_folder, name)
            if os.path.exists(path):
                freed += os.path.getsize(path)
                os.remove(path)
        n_pe = 0
        for entry in os.listdir(cfg.files.run_folder):
            path = os.path.join(cfg.files.run_folder, entry)
            if entry.startswith("PE") and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                n_pe += 1
        print(
            f"low_storage: freed {freed / 1e9:.2f} GB + {n_pe} PE dirs "
            f"in {cfg.files.run_folder}"
        )
    return maxele


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="wrap_config")
def main(cfg: DictConfig) -> float:
    """
    Command line call

    Args:
        cfg (DictConfig): Full config file.

    Returns:
        float: float.
    """
    cfg["name"] = str(
        cfg["name"]
    )  # ensure name is a string (hydra may parse it as int)
    cfg.files["run_folder"] = os.path.join(str(cfg.files.exp_path), str(cfg.name))
    return idealized_tc_observe(cfg)


def get_default_config() -> OmegaConf:
    with initialize(
        config_path=os.path.relpath(CONFIG_PATH, os.path.dirname(__file__))
    ):
        # Compose the configuration as if you were running from the command line.
        cfg = compose(config_name="wrap_config")
    # Machine-local overrides from the environment. The adbo drivers (exp.py,
    # sweep_vmax.py) are argparse entry points that compose the config HERE
    # rather than via the adforce.wrap Hydra CLI, so `files.*`/`slurm.*`
    # dot-overrides cannot reach them; these variables are the supported way
    # to point such runs at a non-ARCHER2 machine (e.g. a cloud VM/container).
    # Each is applied only when set, so ARCHER2 behaviour is unchanged.
    if os.environ.get("ADCIRC_EXE_PATH"):
        # directory containing the adcprep/padcirc binaries
        cfg.files.exe_path = os.environ["ADCIRC_EXE_PATH"]
        print(f"env override: files.exe_path={cfg.files.exe_path} (ADCIRC_EXE_PATH)")
    if os.environ.get("ADCIRC_NP"):  # empty string treated as unset
        # MPI ranks: np = (tasks_per_node - reserved_cpus) * nodes, so with
        # reserved_cpus=0 and nodes=1 (the mid resolution the observe loop
        # asserts) this is the rank count.
        try:
            cfg.slurm.tasks_per_node = int(os.environ["ADCIRC_NP"])
        except ValueError as e:
            raise ValueError(
                f"ADCIRC_NP must be an integer MPI rank count, "
                f"got {os.environ['ADCIRC_NP']!r}"
            ) from e
        cfg.slurm.reserved_cpus = 0
        print(
            f"env override: slurm.tasks_per_node={cfg.slurm.tasks_per_node}, "
            "slurm.reserved_cpus=0 (ADCIRC_NP)"
        )
    if "WORSTSURGE_MODULES" in os.environ:
        # HPC modules to load before running ADCIRC; set to "" (empty) to
        # disable the `module load` step entirely on machines without one.
        cfg.slurm.modules = os.environ["WORSTSURGE_MODULES"]
        print(f"env override: slurm.modules={cfg.slurm.modules!r} (WORSTSURGE_MODULES)")
    return cfg


if __name__ == "__main__":
    # python -m adforce.wrap name=changed_calendar_wrap
    # python -m adforce.wrap name="2100" tc.profile_name.value="2100_new_orleans_profile_r4i1p1f1"
    main()
