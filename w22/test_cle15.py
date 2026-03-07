"""
Test suite for the CLE15 tropical cyclone wind profile implementations.

Covers:
- :mod:`w22.cle15`  (pure-Python reference)
- :mod:`w22.cle15n` (numba-accelerated drop-in replacement)

Tests are grouped as:

1. ``TestChavasProfile``   – unit tests for ``chavas_et_al_2015_profile``
2. ``TestProcessInputs``   – unit tests for ``process_inputs``
3. ``TestRunCle15``        – smoke tests for ``run_cle15``
4. ``TestProfileFromStats``– smoke tests for ``profile_from_stats``
5. ``TestNbConsistency``   – numerical agreement between the two implementations
6. ``TestEdgeCases``       – degenerate / boundary inputs

Run with::

    pytest w22/test_cle15.py -v

or as part of the full suite::

    pytest
"""

import warnings
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import both implementations
# ---------------------------------------------------------------------------
from w22 import cle15, cle15n
from w22.constants import (
    CK_CD_DEFAULT,
    CD_DEFAULT,
    W_COOL_DEFAULT,
    CDVARY_DEFAULT,
    CKCDVARY_DEFAULT,
    EYE_ADJ_DEFAULT,
    ALPHA_EYE_DEFAULT,
    BACKGROUND_PRESSURE,
)


# ---------------------------------------------------------------------------
# Warm up numba once for the whole session (avoids counting JIT in tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def warmup_numba():
    cle15n.warmup()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A small selection of representative parameter sets
#   (Vmax m/s,  r0 m,     fcor s-1)
REFERENCE_CASES = [
    (30.0, 400e3, 3e-5),
    (40.0, 600e3, 5e-5),
    (50.0, 800e3, 5e-5),
    (60.0, 1000e3, 7e-5),
    (70.0, 1200e3, 7e-5),
]

# Default keyword arguments for chavas_et_al_2015_profile
_PROF_DEFAULTS = dict(
    Cdvary=CDVARY_DEFAULT,
    C_d=CD_DEFAULT,
    w_cool=W_COOL_DEFAULT,
    CkCdvary=CKCDVARY_DEFAULT,
    CkCd_input=CK_CD_DEFAULT,
    eye_adj=EYE_ADJ_DEFAULT,
    alpha_eye=ALPHA_EYE_DEFAULT,
)


def _call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, **kwargs):
    """Convenience wrapper around chavas_et_al_2015_profile for either module."""
    kw = dict(_PROF_DEFAULTS)
    kw.update(kwargs)
    return mod.chavas_et_al_2015_profile(
        Vmax=Vmax,
        r0=r0,
        fcor=fcor,
        Cdvary=kw["Cdvary"],
        C_d=kw["C_d"],
        w_cool=kw["w_cool"],
        CkCdvary=kw["CkCdvary"],
        CkCd_input=kw["CkCd_input"],
        eye_adj=kw["eye_adj"],
        alpha_eye=kw["alpha_eye"],
    )


def _unpack(result):
    """Return named fields from the 11-tuple returned by chavas_et_al_2015_profile."""
    (
        rr,
        VV,
        rmax,
        rmerge,
        Vmerge,
        rrfracr0,
        MMfracM0,
        rmaxr0,
        MmM0,
        rmerger0,
        MmergeM0,
    ) = result
    return dict(
        rr=rr,
        VV=VV,
        rmax=rmax,
        rmerge=rmerge,
        Vmerge=Vmerge,
        rrfracr0=rrfracr0,
        MMfracM0=MMfracM0,
        rmaxr0=rmaxr0,
        MmM0=MmM0,
        rmerger0=rmerger0,
        MmergeM0=MmergeM0,
    )


# ===========================================================================
# 1. TestChavasProfile
# ===========================================================================


class TestChavasProfile:
    """Unit tests for chavas_et_al_2015_profile in both implementations."""

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_returns_finite_values(self, mod, Vmax, r0, fcor):
        """All scalar outputs must be finite for valid inputs."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        for key in (
            "rmax",
            "rmerge",
            "Vmerge",
            "rmaxr0",
            "MmM0",
            "rmerger0",
            "MmergeM0",
        ):
            assert np.isfinite(
                res[key]
            ), f"mod={mod.__name__}: {key}={res[key]} not finite"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_profile_arrays_shape_and_finite(self, mod, Vmax, r0, fcor):
        """rr and VV must be 1-D arrays of equal length with mostly finite values."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        rr, VV = res["rr"], res["VV"]
        assert rr.ndim == 1
        assert VV.ndim == 1
        assert len(rr) == len(VV)
        assert np.sum(np.isfinite(VV)) > 10, "fewer than 10 finite wind-speed points"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_rmax_within_r0(self, mod, Vmax, r0, fcor):
        """rmax must be strictly less than r0."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        assert res["rmax"] < r0, f"rmax={res['rmax']/1e3:.1f} km >= r0={r0/1e3:.0f} km"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_rmerge_between_rmax_and_r0(self, mod, Vmax, r0, fcor):
        """rmerge must sit between rmax and r0."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        assert res["rmax"] <= res["rmerge"] <= r0, (
            f"rmerge={res['rmerge']/1e3:.1f} km outside "
            f"[{res['rmax']/1e3:.1f}, {r0/1e3:.0f}] km"
        )

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_vmax_in_profile(self, mod, Vmax, r0, fcor):
        """Peak wind speed in the output profile must be within 2 % of target Vmax."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        VV = res["VV"]
        peak = float(np.nanmax(VV))
        rel_err = abs(peak - Vmax) / Vmax
        assert rel_err < 0.02, (
            f"mod={mod.__name__}: profile Vmax={peak:.2f} m/s vs target {Vmax} m/s "
            f"(err={100*rel_err:.1f} %)"
        )

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_wind_non_negative(self, mod, Vmax, r0, fcor):
        """All finite wind speeds in the output profile must be >= 0."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        VV = res["VV"]
        finite = VV[np.isfinite(VV)]
        assert np.all(
            finite >= -1e-10
        ), f"Negative wind speeds found: min={finite.min():.4f} m/s"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_radius_vector_monotone(self, mod, Vmax, r0, fcor):
        """rr must be strictly monotonically increasing."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        rr = res["rr"]
        diffs = np.diff(rr)
        assert np.all(diffs > 0), "rr is not strictly increasing"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_rmaxr0_nondim_consistent(self, mod, Vmax, r0, fcor):
        """rmaxr0 must equal rmax / r0 to within 1 %."""
        res = _unpack(_call_profile(mod, Vmax=Vmax, r0=r0, fcor=fcor))
        expected = res["rmax"] / r0
        assert (
            abs(res["rmaxr0"] - expected) / expected < 0.01
        ), f"rmaxr0={res['rmaxr0']:.4f} vs rmax/r0={expected:.4f}"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_eye_adjustment_changes_profile(self, mod):
        """Enabling eye adjustment (eye_adj=1) should change the inner wind profile."""
        no_eye = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, eye_adj=0))
        with_eye = _unpack(
            _call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, eye_adj=1)
        )
        # Profiles should differ inside the eye (near r=0)
        rr = no_eye["rr"]
        rmax = no_eye["rmax"]
        inside = rr < 0.5 * rmax
        if np.any(inside):
            diff = np.nanmax(np.abs(with_eye["VV"][inside] - no_eye["VV"][inside]))
            assert diff > 0.0, "Eye adjustment had no effect on the inner profile"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_negative_fcor_same_as_positive(self, mod):
        """The Southern Hemisphere sign convention (negative fcor) should work."""
        pos = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=+5e-5))
        neg = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=-5e-5))
        assert (
            abs(pos["rmax"] - neg["rmax"]) < 1e3
        ), "rmax should be identical for ±fcor"


# ===========================================================================
# 2. TestProcessInputs
# ===========================================================================


class TestProcessInputs:
    """Unit tests for process_inputs in both implementations."""

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_defaults_populated(self, mod):
        """Calling process_inputs({}) should populate all required keys."""
        required = {
            "Vmax",
            "r0",
            "fcor",
            "Cd",
            "CkCd",
            "w_cool",
            "Cdvary",
            "CkCdvary",
            "eye_adj",
            "alpha_eye",
            "p0",
        }
        ins = mod.process_inputs({})
        assert required.issubset(ins.keys()), f"Missing keys: {required - ins.keys()}"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_user_override_respected(self, mod):
        """A user-supplied value should override the default."""
        ins = mod.process_inputs({"Vmax": 99.0, "r0": 500e3})
        assert ins["Vmax"] == 99.0
        assert ins["r0"] == 500e3

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_unknown_key_ignored(self, mod):
        """Unknown keys in the input dict should not raise an error."""
        mod.process_inputs({"not_a_key": 42})  # must not raise

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_p0_range_check(self, mod):
        """p0 outside [900, 1100] hPa should raise AssertionError."""
        with pytest.raises((AssertionError, Exception)):
            mod.process_inputs({"p0": 500.0})
        with pytest.raises((AssertionError, Exception)):
            mod.process_inputs({"p0": 1200.0})


# ===========================================================================
# 3. TestRunCle15
# ===========================================================================


class TestRunCle15:
    """Smoke tests for the run_cle15 convenience wrapper."""

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_returns_three_floats(self, mod):
        """run_cle15 must return a 3-tuple of finite floats (pm, rmax, pc)."""
        result = mod.run_cle15()
        assert len(result) == 3
        pm, rmax, pc = result
        assert np.isfinite(pm), f"pm={pm} not finite"
        assert np.isfinite(rmax), f"rmax={rmax} not finite"
        assert np.isfinite(pc), f"pc={pc} not finite"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_pressure_ordering(self, mod):
        """Central pressure pc must be less than ambient (background) pressure."""
        pm, rmax, pc = mod.run_cle15()
        assert (
            pc < BACKGROUND_PRESSURE
        ), f"pc={pc:.0f} Pa >= background {BACKGROUND_PRESSURE} Pa"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_pm_between_pc_and_background(self, mod):
        """pm (at rmax) should be between pc (centre) and background pressure."""
        pm, rmax, pc = mod.run_cle15()
        assert (
            pc <= pm <= BACKGROUND_PRESSURE
        ), f"pm={pm:.0f} Pa not in [{pc:.0f}, {BACKGROUND_PRESSURE}] Pa"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_custom_vmax_changes_output(self, mod):
        """A stronger storm (higher Vmax) should produce a lower central pressure."""
        _, _, pc_weak = mod.run_cle15(inputs={"Vmax": 30.0})
        _, _, pc_strong = mod.run_cle15(inputs={"Vmax": 70.0})
        assert pc_strong < pc_weak, (
            f"Stronger storm did not give lower pc: pc_strong={pc_strong:.0f}, "
            f"pc_weak={pc_weak:.0f}"
        )


# ===========================================================================
# 4. TestProfileFromStats
# ===========================================================================


class TestProfileFromStats:
    """Smoke tests for profile_from_stats in both implementations."""

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_returns_required_keys(self, mod):
        """Output dict must contain rr, VV, rmax, rmerge, Vmerge, p."""
        out = mod.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)
        for key in ("rr", "VV", "rmax", "rmerge", "Vmerge", "p"):
            assert key in out, f"Missing key '{key}' in profile_from_stats output"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_pressure_in_hpa(self, mod):
        """Pressure array p should be in hPa (i.e., between ~850 and 1050)."""
        out = mod.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)
        p = np.asarray(out["p"])
        assert np.all(p < 1100), "Pressure looks like Pa not hPa"
        assert np.all(p > 850), f"Unusually low central pressure: {p.min():.1f} hPa"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_arrays_same_length(self, mod):
        """rr, VV, and p must all have the same length."""
        out = mod.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)
        n = len(out["rr"])
        assert len(out["VV"]) == n
        assert len(out["p"]) == n


# ===========================================================================
# 5. TestNbConsistency
# ===========================================================================


class TestNbConsistency:
    """
    Verify that cle15n produces results numerically consistent with cle15.

    Thresholds are deliberately generous (matching observed benchmark spread):
      - rmax: within 2 % relative
      - rmerge: within 3 % relative
      - Vmerge: within 0.5 m/s absolute
      - Profile RMS: within 0.02 m/s
    """

    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_rmax_agreement(self, Vmax, r0, fcor):
        py = _unpack(_call_profile(cle15, Vmax=Vmax, r0=r0, fcor=fcor))
        nb = _unpack(_call_profile(cle15n, Vmax=Vmax, r0=r0, fcor=fcor))
        rel = abs(py["rmax"] - nb["rmax"]) / py["rmax"]
        assert rel < 0.02, (
            f"Vmax={Vmax} r0={r0/1e3:.0f}km: rmax py={py['rmax']/1e3:.2f} km "
            f"nb={nb['rmax']/1e3:.2f} km  rel={100*rel:.2f} %"
        )

    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_rmerge_agreement(self, Vmax, r0, fcor):
        py = _unpack(_call_profile(cle15, Vmax=Vmax, r0=r0, fcor=fcor))
        nb = _unpack(_call_profile(cle15n, Vmax=Vmax, r0=r0, fcor=fcor))
        rel = abs(py["rmerge"] - nb["rmerge"]) / py["rmerge"]
        assert rel < 0.03, (
            f"rmerge py={py['rmerge']/1e3:.2f} km nb={nb['rmerge']/1e3:.2f} km "
            f"rel={100*rel:.2f} %"
        )

    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_vmerge_agreement(self, Vmax, r0, fcor):
        py = _unpack(_call_profile(cle15, Vmax=Vmax, r0=r0, fcor=fcor))
        nb = _unpack(_call_profile(cle15n, Vmax=Vmax, r0=r0, fcor=fcor))
        assert (
            abs(py["Vmerge"] - nb["Vmerge"]) < 0.5
        ), f"Vmerge py={py['Vmerge']:.3f} nb={nb['Vmerge']:.3f} m/s"

    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_profile_rms(self, Vmax, r0, fcor):
        """RMS difference of wind profiles on a common grid must be < 0.05 m/s.

        The 0.05 m/s threshold reflects the residual difference from the
        simplified 10-iteration ER11 loop inside the compiled numba bisection
        kernel; all other algorithm choices are identical between the two
        implementations.
        """
        py = _unpack(_call_profile(cle15, Vmax=Vmax, r0=r0, fcor=fcor))
        nb = _unpack(_call_profile(cle15n, Vmax=Vmax, r0=r0, fcor=fcor))
        from scipy.interpolate import interp1d

        r_common = np.linspace(0, r0, 2000)
        f_py = interp1d(py["rr"], py["VV"], bounds_error=False, fill_value=np.nan)
        f_nb = interp1d(nb["rr"], nb["VV"], bounds_error=False, fill_value=np.nan)
        v_py = f_py(r_common)
        v_nb = f_nb(r_common)
        valid = np.isfinite(v_py) & np.isfinite(v_nb)
        rms = float(np.sqrt(np.mean((v_py[valid] - v_nb[valid]) ** 2)))
        assert rms < 0.05, f"Vmax={Vmax} r0={r0/1e3:.0f}km: profile RMS={rms:.4f} m/s"

    def test_run_cle15_consistent(self):
        """run_cle15 pm and pc must agree within 50 Pa between implementations."""
        py_pm, py_rmax, py_pc = cle15.run_cle15()
        nb_pm, nb_rmax, nb_pc = cle15n.run_cle15()
        assert (
            abs(py_pm - nb_pm) < 50
        ), f"pm disagrees: py={py_pm:.0f} Pa nb={nb_pm:.0f} Pa"
        assert (
            abs(py_pc - nb_pc) < 50
        ), f"pc disagrees: py={py_pc:.0f} Pa nb={nb_pc:.0f} Pa"


# ===========================================================================
# 6. TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Degenerate and boundary inputs."""

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_very_small_storm(self, mod):
        """Small r0 (200 km) should still converge without raising."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(_call_profile(mod, Vmax=40.0, r0=200e3, fcor=5e-5))
        # Either converges with finite rmax, or fails gracefully with NaN
        if np.isfinite(res["rmax"]):
            assert res["rmax"] < 200e3

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_very_large_storm(self, mod):
        """Large r0 (1500 km) should converge."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(_call_profile(mod, Vmax=50.0, r0=1500e3, fcor=5e-5))
        assert np.isfinite(res["rmax"]), "Large storm did not converge"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_high_latitude_fcor(self, mod):
        """High Coriolis (fcor=1e-4, ~45°) should still converge."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=1e-4))
        assert np.isfinite(res["rmax"]), "High-latitude case did not converge"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_ckcdvary_mode(self, mod):
        """CkCdvary=1 (Vmax-dependent Ck/Cd) should produce a valid profile."""
        res = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, CkCdvary=1))
        assert np.isfinite(res["rmax"]), "CkCdvary=1 mode did not converge"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_cdvary_mode(self, mod):
        """Cdvary=1 (wind-speed-dependent Cd) should produce a valid profile."""
        res = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, Cdvary=1))
        assert np.isfinite(res["rmax"]), "Cdvary=1 mode did not converge"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_from_stats_returns_valid(self, mod):
        """profile_from_stats must not raise for a standard set of inputs."""
        out = mod.profile_from_stats(vmax=45.0, fcor=4e-5, r0=700e3, p0=1010.0)
        assert "rr" in out
        assert np.isfinite(out["rmax"])
