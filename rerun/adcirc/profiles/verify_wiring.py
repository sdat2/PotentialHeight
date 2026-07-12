"""Verify the profile-JSON -> fort.22.nc wiring end-to-end (no ADCIRC needed).

Answers "will `tc.profile_name.value=/path/to/profile.json` wire in okay?" by
composing the REAL adforce Hydra config with the EXACT override strings that
rerun/adcirc/run_on_gcp.sh passes, then calling the real
``adforce.fort22.create_fort22`` and asserting on the resulting fort.22.nc:

  1. bare-name branch  (tc.profile_name.value=2015_new_orleans_profile_r4i1p1f1,
     resolved against w22/data) — the branch the original paper runs used;
  2. literal-path branch (tc.profile_name.value=/abs/path/profile_fixed_2015.json)
     — the branch the materiality check and the adbo tradeoff sweep use;
  3. old-vs-fixed forcing actually DIFFERS (the ~3% rmax change reaches the
     wind field), and each file's surface winds/pressures are consistent with
     its source profile (v_reduc * max(VV); min pressure ~ profile min p).

Run:  PYTHONPATH=<repo> python rerun/adcirc/profiles/verify_wiring.py [workdir]
(workdir defaults to a scratch tmpdir; ~1-2 GB RAM transient while gridding.)
"""

import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from hydra import compose, initialize_config_dir
from adforce.fort22 import create_fort22

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_REPO, "adforce", "config")
FAILURES = []


def check(label: str, cond: bool, detail: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}: {detail}")
    if not cond:
        FAILURES.append(label)


def make_fort22(workdir: str, run_name: str, profile_value: str) -> str:
    """Compose the real wrap config with the run_on_gcp.sh-style override, run create_fort22."""
    run_folder = os.path.join(workdir, run_name)
    os.makedirs(run_folder, exist_ok=True)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="wrap_config",
            overrides=[f"name={run_name}",
                       f"tc.profile_name.value={profile_value}"],
        )
    create_fort22(run_folder, cfg.grid, cfg.tc)
    out = os.path.join(run_folder, "fort.22.nc")
    assert os.path.exists(out), f"fort.22.nc not written for {run_name}"
    return out


def summarize(nc_file: str) -> dict:
    import netCDF4 as nc
    ds = nc.Dataset(nc_file)
    tc1 = ds.groups["TC1"]
    spd = np.sqrt(np.asarray(tc1["U10"][:]) ** 2 + np.asarray(tc1["V10"][:]) ** 2)
    out = {
        "groups": sorted(ds.groups.keys()),
        "max_wind": float(np.nanmax(spd)),
        "min_psfc": float(np.nanmin(np.asarray(tc1["PSFC"][:]))),
    }
    ds.close()
    return out


def main() -> None:
    workdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="wiring_")
    print(f"workdir: {workdir}\n")
    prof_fixed = json.load(open(os.path.join(HERE, "profile_fixed_2015.json")))
    prof_old = json.load(open(os.path.join(HERE, "profile_old_2015.json")))
    v_reduc = 0.8  # fort22 default when tc config carries none-overridden value

    print("[1] bare-name branch (as the original paper runs used):")
    f_name = make_fort22(workdir, "wiring_name", "2015_new_orleans_profile_r4i1p1f1")
    s_name = summarize(f_name)
    check("groups", s_name["groups"] == ["Main", "TC1"], f"{s_name['groups']}")
    vexp = v_reduc * float(np.max(prof_old["VV"]))
    check("surface wind ~ v_reduc*max(VV)", abs(s_name["max_wind"] - vexp) < 0.15 * vexp,
          f"{s_name['max_wind']:.1f} vs expected ~{vexp:.1f} m/s")

    print("[2] literal-path branch (materiality check / tradeoff sweep):")
    f_fixed = make_fort22(workdir, "wiring_fixed",
                          os.path.join(HERE, "profile_fixed_2015.json"))
    s_fixed = summarize(f_fixed)
    vexp_f = v_reduc * float(np.max(prof_fixed["VV"]))
    check("surface wind ~ v_reduc*max(VV)", abs(s_fixed["max_wind"] - vexp_f) < 0.15 * vexp_f,
          f"{s_fixed['max_wind']:.1f} vs expected ~{vexp_f:.1f} m/s")
    check("min psfc ~ profile min p", abs(s_fixed["min_psfc"] - min(prof_fixed["p"])) < 5.0,
          f"{s_fixed['min_psfc']:.1f} vs profile {min(prof_fixed['p']):.1f} hPa")

    print("[3] old vs fixed forcing differs (the fix reaches the wind field):")
    f_old = make_fort22(workdir, "wiring_old",
                        os.path.join(HERE, "profile_old_2015.json"))
    s_old = summarize(f_old)
    import netCDF4 as nc
    a = nc.Dataset(f_old); b = nc.Dataset(f_fixed)
    u_old = np.asarray(a.groups["TC1"]["U10"][:]); u_fix = np.asarray(b.groups["TC1"]["U10"][:])
    rel = float(np.nanmax(np.abs(u_fix - u_old)) / np.nanmax(np.abs(u_old)))
    a.close(); b.close()
    check("fields differ", rel > 0.005, f"max relative U10 difference {100*rel:.1f}%")
    check("same intensity both eras", abs(s_old["max_wind"] - s_fixed["max_wind"]) < 1.0,
          f"{s_old['max_wind']:.1f} vs {s_fixed['max_wind']:.1f} m/s (fix changes size, not Vp)")

    print()
    if FAILURES:
        print(f"WIRING TEST FAILED: {FAILURES}")
        sys.exit(1)
    print("WIRING TEST PASSED — both branches produce correct fort.22.nc forcing.")


if __name__ == "__main__":
    main()
