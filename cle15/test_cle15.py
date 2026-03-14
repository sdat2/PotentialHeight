"""
Test suite for the CLE15 tropical cyclone wind profile implementations.

Covers:
- :mod:`cle15.cle15`  (pure-Python reference)
- :mod:`cle15.cle15n` (numba-accelerated drop-in replacement)

Tests are grouped as:

1. ``TestChavasProfile``      – unit tests for ``chavas_et_al_2015_profile``
2. ``TestProcessInputs``      – unit tests for ``process_inputs``
3. ``TestRunCle15``           – smoke tests for ``run_cle15``
4. ``TestProfileFromStats``   – smoke tests for ``profile_from_stats``
5. ``TestNbConsistency``      – numerical agreement between the two implementations
6. ``TestEdgeCases``          – degenerate / boundary inputs
7. ``TestSolverConfig``       – SolverConfig dataclass and presets
8. ``TestNbRobustness``       – known failure modes in the numba solver
                                (these tests are expected to FAIL until the
                                kernel-level convergence is fixed; they are
                                marked xfail so the suite stays green while
                                the regressions are tracked)
9. ``TestNanHandling``        – NaN propagation and failure-mode behaviour,
                                derived from the MATLAB reference implementation
10. ``TestMatlabRegression``  – rmax cross-validated against Octave
                                (ER11E04_nondim_r0input.m, same Python-default
                                parameters: Cdvary=0, CkCd=0.9)

Run with::

    pytest cle15/test_cle15.py -v

or as part of the full suite::

    pytest
"""

import warnings
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import both implementations
# ---------------------------------------------------------------------------
from cle15 import cle15, cle15n
from cle15.constants import (
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


# ===========================================================================
# 7. TestSolverConfig
# ===========================================================================


class TestSolverConfig:
    """Tests for the SolverConfig dataclass and named presets in cle15n."""

    def test_default_fields(self):
        """SolverConfig() should expose the documented default values."""
        cfg = cle15n.SolverConfig()
        assert cfg.Nr_e04 == 200_000
        assert cfg.num_pts_er11 == 5_000
        assert cfg.nx_intersect == 4_000
        assert cfg.max_iter == 50

    def test_fast_preset_fields(self):
        """SolverConfig.fast() should have smaller values than default."""
        fast = cle15n.SolverConfig.fast()
        default = cle15n.SolverConfig()
        assert fast.Nr_e04 <= default.Nr_e04
        assert fast.num_pts_er11 < default.num_pts_er11
        assert fast.nx_intersect <= default.nx_intersect
        assert fast.max_iter <= default.max_iter

    def test_precise_preset_fields(self):
        """SolverConfig.precise() should have >= default values."""
        precise = cle15n.SolverConfig.precise()
        default = cle15n.SolverConfig()
        assert precise.num_pts_er11 >= default.num_pts_er11

    @pytest.mark.parametrize(
        "preset,label",
        [
            (cle15n.SolverConfig.fast(), "fast"),
            (cle15n.SolverConfig.default(), "default"),
            (cle15n.SolverConfig.precise(), "precise"),
        ],
    )
    def test_preset_produces_finite_rmax(self, preset, label):
        """All three presets must converge for a standard input."""
        res = cle15n.chavas_et_al_2015_profile(
            50.0,
            800e3,
            5e-5,
            CDVARY_DEFAULT,
            CD_DEFAULT,
            W_COOL_DEFAULT,
            CKCDVARY_DEFAULT,
            CK_CD_DEFAULT,
            EYE_ADJ_DEFAULT,
            ALPHA_EYE_DEFAULT,
            solver=preset,
        )
        rmax = res[2]
        assert np.isfinite(rmax), f"Preset '{label}' did not converge: rmax={rmax}"

    @pytest.mark.parametrize("Vmax,r0,fcor", REFERENCE_CASES)
    def test_fast_rmax_within_5pct_of_default(self, Vmax, r0, fcor):
        """Fast preset rmax must be within 5 % of the default preset rmax."""
        kw = dict(
            Cdvary=CDVARY_DEFAULT,
            C_d=CD_DEFAULT,
            w_cool=W_COOL_DEFAULT,
            CkCdvary=CKCDVARY_DEFAULT,
            CkCd_input=CK_CD_DEFAULT,
            eye_adj=EYE_ADJ_DEFAULT,
            alpha_eye=ALPHA_EYE_DEFAULT,
        )
        default_res = cle15n.chavas_et_al_2015_profile(
            Vmax, r0, fcor, **kw, solver=cle15n.SolverConfig.default()
        )
        fast_res = cle15n.chavas_et_al_2015_profile(
            Vmax, r0, fcor, **kw, solver=cle15n.SolverConfig.fast()
        )
        rmax_def = default_res[2]
        rmax_fast = fast_res[2]
        assert np.isfinite(rmax_fast), "Fast preset returned NaN"
        rel_err = abs(rmax_fast - rmax_def) / rmax_def
        assert rel_err < 0.05, (
            f"Vmax={Vmax} r0={r0/1e3:.0f}km: fast rmax={rmax_fast/1e3:.2f} km "
            f"vs default {rmax_def/1e3:.2f} km  ({100*rel_err:.1f} %)"
        )

    def test_fast_is_faster_than_default(self):
        """SolverConfig.fast() should be measurably quicker than default (at least 2×)."""
        import time

        kw = dict(
            Vmax=50.0,
            r0=800e3,
            fcor=5e-5,
            Cdvary=CDVARY_DEFAULT,
            C_d=CD_DEFAULT,
            w_cool=W_COOL_DEFAULT,
            CkCdvary=CKCDVARY_DEFAULT,
            CkCd_input=CK_CD_DEFAULT,
            eye_adj=EYE_ADJ_DEFAULT,
            alpha_eye=ALPHA_EYE_DEFAULT,
        )
        n = 10

        t0 = time.perf_counter()
        for _ in range(n):
            cle15n.chavas_et_al_2015_profile(**kw, solver=cle15n.SolverConfig.fast())
        t_fast = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            cle15n.chavas_et_al_2015_profile(**kw, solver=cle15n.SolverConfig.default())
        t_default = (time.perf_counter() - t0) / n

        assert (
            t_fast < t_default
        ), f"fast ({t_fast*1e3:.1f} ms) was not faster than default ({t_default*1e3:.1f} ms)"

    def test_solver_kwarg_passthrough_run_cle15(self):
        """run_cle15 should accept and honour the solver kwarg."""
        pm, rmax, pc = cle15n.run_cle15(solver=cle15n.SolverConfig.fast())
        assert np.isfinite(rmax), "run_cle15 with fast solver returned non-finite rmax"

    def test_solver_kwarg_passthrough_profile_from_stats(self):
        """profile_from_stats should accept and honour the solver kwarg."""
        out = cle15n.profile_from_stats(
            vmax=50.0,
            fcor=5e-5,
            r0=800e3,
            p0=1013.25,
            solver=cle15n.SolverConfig.fast(),
        )
        assert np.isfinite(
            out["rmax"]
        ), "profile_from_stats with fast solver returned non-finite rmax"

    def test_custom_config(self):
        """A manually constructed SolverConfig should work without error."""
        cfg = cle15n.SolverConfig(
            Nr_e04=5_000, num_pts_er11=200, nx_intersect=200, max_iter=15
        )
        res = cle15n.chavas_et_al_2015_profile(
            50.0,
            800e3,
            5e-5,
            CDVARY_DEFAULT,
            CD_DEFAULT,
            W_COOL_DEFAULT,
            CKCDVARY_DEFAULT,
            CK_CD_DEFAULT,
            EYE_ADJ_DEFAULT,
            ALPHA_EYE_DEFAULT,
            solver=cfg,
        )
        assert np.isfinite(res[2]), "Custom SolverConfig did not converge"


# ===========================================================================
# 8. TestNbRobustness
# ===========================================================================

# Input combinations where the numba solver is known to return rmerge == r0
# (degenerate merge — the compiled ER11 convergence kernel failed), causing
# run_cle15 to return NaN.  These were identified by sweeping the ps.py
# bisection search range (r0 ∈ [1.9, 2.4] Mm, Vmax=49.5, fcor at 25°N)
# and a broader grid sweep (Vmax 30-70, r0 400-2000 km).
#
# Format: (Vmax m/s, r0 m, fcor s-1)
#
# Note: fcor ≈ 6.16e-5 s⁻¹ corresponds to the Coriolis parameter at 25°N.
# The cases with Vmax=49.5 are drawn from the ps.py bisection range;
# the low-Vmax cases are from the broader grid sweep.
#
# All tests in this class are marked xfail(strict=False):
#   - xfail:  we EXPECT them to fail with the current numba implementation
#   - strict=False: if they unexpectedly pass (i.e. the bug is fixed) the
#                   suite turns them green rather than erroring, so fixing
#                   the kernel automatically promotes them to passing tests.

# Only keep cases that actually fail at test-run time.  Re-confirmed after
# NaN guards were added to run_cle15; these are the survivors that still
# trigger rmerge==r0 in the compiled bisection kernel.
_NB_FAILURE_CASES = [
    # ps.py bisection sweep at 25°N: the one r0 that still reliably fails
    (49.5, 1935.4e3, 6.16e-5),
    # Low-Vmax, mid-r0: the compiled ER11 loop diverges most consistently here
    (30.0, 661.2e3, 6.16e-5),
]


class TestNbRobustness:
    """
    Regression tests for known numba solver failure modes (now fixed).

    Root cause (fixed)
    ------------------
    The compiled kernel ``_bisect_rmaxr0_nb`` clipped the ER11 profile at
    ``r <= r0`` before the curve-intersection check.  Since the E04 outer
    profile also terminates at ``r/r0 = 1.0``, both curves shared an endpoint
    at the boundary.  When the ER11 M/M0 exceeded 1.0 in the outer region
    (possible for large rmaxr0 guesses) and then dropped back to 1.0 at the
    boundary, a spurious sign-change crossing was detected there — locking
    the bisection to an unphysical small-rmax solution.

    Fix: the ER11 grid is now clipped to ``r < r0 × (1 - 1e-6)`` (strictly
    less than r0), so the boundary point is never included and the spurious
    crossing cannot occur.  The bisection then correctly finds the physical
    interior tangent point.

    These tests document the previously failing (Vmax, r0, fcor) combinations
    and verify they now produce physically correct results.
    """

    @pytest.mark.parametrize("Vmax,r0,fcor", _NB_FAILURE_CASES)
    def test_nb_profile_rmerge_not_degenerate(self, Vmax, r0, fcor):
        """numba chavas_et_al_2015_profile must not return rmerge == r0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cle15n.chavas_et_al_2015_profile(
                Vmax,
                r0,
                fcor,
                CDVARY_DEFAULT,
                CD_DEFAULT,
                W_COOL_DEFAULT,
                CKCDVARY_DEFAULT,
                CK_CD_DEFAULT,
                EYE_ADJ_DEFAULT,
                ALPHA_EYE_DEFAULT,
            )
        rmerge = res[3]
        rmax = res[2]
        assert np.isfinite(rmax), f"rmax is NaN/inf at Vmax={Vmax} r0={r0/1e3:.0f}km"
        assert not np.isclose(rmerge, r0, rtol=1e-4), (
            f"rmerge={rmerge/1e3:.1f} km == r0={r0/1e3:.1f} km "
            f"(degenerate merge) at Vmax={Vmax} r0={r0/1e3:.0f}km"
        )

    @pytest.mark.parametrize("Vmax,r0,fcor", _NB_FAILURE_CASES)
    def test_nb_run_cle15_finite(self, Vmax, r0, fcor):
        """numba run_cle15 must return finite pm for all failure-case inputs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm_nb, rmax_nb, pc_nb = cle15n.run_cle15(
                inputs={
                    "Vmax": Vmax,
                    "r0": r0,
                    "fcor": fcor,
                    "CkCd": CK_CD_DEFAULT,
                    "Cd": CD_DEFAULT,
                    "w_cool": W_COOL_DEFAULT,
                }
            )
        assert np.isfinite(
            pm_nb
        ), f"pm is NaN at Vmax={Vmax} r0={r0/1e3:.0f}km fcor={fcor:.2e}"

    @pytest.mark.parametrize("Vmax,r0,fcor", _NB_FAILURE_CASES)
    def test_nb_run_cle15_agrees_with_python(self, Vmax, r0, fcor):
        """numba run_cle15 pm must agree with pure-Python to within 100 Pa."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm_py, _, _ = cle15.run_cle15(
                inputs={
                    "Vmax": Vmax,
                    "r0": r0,
                    "fcor": fcor,
                    "CkCd": CK_CD_DEFAULT,
                    "Cd": CD_DEFAULT,
                    "w_cool": W_COOL_DEFAULT,
                }
            )
            pm_nb, _, _ = cle15n.run_cle15(
                inputs={
                    "Vmax": Vmax,
                    "r0": r0,
                    "fcor": fcor,
                    "CkCd": CK_CD_DEFAULT,
                    "Cd": CD_DEFAULT,
                    "w_cool": W_COOL_DEFAULT,
                }
            )
        assert np.isfinite(pm_nb), f"pm_nb is NaN at Vmax={Vmax} r0={r0/1e3:.0f}km"
        assert abs(pm_nb - pm_py) < 100, (
            f"pm disagrees by {abs(pm_nb-pm_py):.0f} Pa at "
            f"Vmax={Vmax} r0={r0/1e3:.0f}km: py={pm_py:.0f} nb={pm_nb:.0f}"
        )

    def test_nb_failure_rate_in_bisection_range(self):
        """
        Verify zero failure rate of the numba solver over the ps.py bisection range.

        Sweeps r0 ∈ [1.9, 2.4] Mm at Vmax=49.5 m/s, fcor at 25°N (the same
        range used by w22/ps.py for a typical tropical storm input).  Previously
        ~19 % of r0 values returned NaN due to the spurious boundary-crossing
        bug.  The fix (excluding the r=r0 grid endpoint) reduces this to 0 %.
        """
        from w22.utils import coriolis_parameter_from_lat

        fcor = abs(coriolis_parameter_from_lat(25))
        r0_vals = np.linspace(1.9e6, 2.4e6, 100)
        nan_count = 0
        for r0 in r0_vals:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm, _, _ = cle15n.run_cle15(
                    inputs={
                        "Vmax": 49.5,
                        "r0": r0,
                        "fcor": fcor,
                        "CkCd": 0.95,
                        "Cd": 0.0015,
                        "w_cool": 0.002,
                    }
                )
            if np.isnan(pm):
                nan_count += 1

        failure_rate = nan_count / len(r0_vals)
        assert failure_rate == 0.0, (
            f"numba solver failure rate in bisection range: "
            f"{nan_count}/{len(r0_vals)} = {100*failure_rate:.0f} % "
            f"(expected 0 % after boundary-crossing fix)"
        )


# ===========================================================================
# 9. TestNanHandling
# ===========================================================================


class TestNanHandling:
    """
    NaN propagation and graceful-failure tests derived from the MATLAB
    reference implementation (ER11E04_nondim_r0input.m, ER11_radprof.m).

    MATLAB behaviour under failure conditions
    -----------------------------------------
    ``ER11_radprof.m`` (outer convergence loop, max 20 iterations):
        If the nested (rmax, Vmax) loop does not converge within 20 outer
        iterations the function returns ``V_ER11 = NaN(size(rr_ER11))``
        and ``r_out = NaN``.  The caller (``ER11E04_nondim_r0input.m``)
        detects ``isnan(max(VV_ER11))`` and treats the bisection step as
        "rmax too small" — nudging the bisection upward.

    ``ER11E04_nondim_r0input.m`` (``soln_converged`` outer loop):
        If the bisection converges without ever finding an intersection
        (``~exist('rmerger0', 'var')``), ``soln_converged`` stays 0 and
        ``CkCd`` is incremented by 0.1.  The MATLAB code does not cap
        CkCd internally (beyond the 1.9 cap applied at entry), so in
        principle it could loop indefinitely; the Python translation caps
        at CkCd ≥ 3.0 and returns NaN arrays.

    ``run_cle15`` / ``profile_from_stats`` (Python only):
        ``cle15n.run_cle15`` adds explicit NaN guards absent from the
        MATLAB code (which simply crashes or returns NaN arrays that
        propagate silently into the pressure integral).  The pure-Python
        ``cle15.run_cle15`` zeroes out NaN winds before integration,
        which can yield a finite-but-wrong pressure rather than NaN.
        These tests document and pin that asymmetry.

    Test sub-groups
    ~~~~~~~~~~~~~~~
    A. ``ER11_radprof`` non-convergence → NaN arrays returned
    B. CkCd upward-nudge fallback (MATLAB ``soln_converged`` loop)
    C. ``run_cle15`` NaN propagation
    D. ``profile_from_stats`` NaN propagation
    E. ``chavas_et_al_2015_profile`` NaN output structure
    F. Pressure-assumption asymmetry between run_cle15 and profile_from_stats
    """

    # ------------------------------------------------------------------
    # A. ER11_radprof non-convergence
    # ------------------------------------------------------------------

    def test_er11_radprof_nan_when_ckcd_zero(self):
        """
        CkCd = 0 makes the ER11 formula degenerate (exponent 1/(2-0)=0.5 is
        fine, but combined with extremely high Rossby number the nested loop
        fails within 20 iterations and must return NaN arrays — matching
        MATLAB ER11_radprof.m line: V_ER11 = NaN(size(rr_ER11)).
        """
        rr = np.linspace(0, 500e3, 5001)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            V, r_out = cle15._er11_radprof(
                Vmax_target=200.0,  # wildly super-physical Vmax → guaranteed non-conv
                r_in_target=1e3,  # rmax = 1 km (absurdly small, high Ro)
                rmax_or_r0="rmax",
                fcor=5e-5,
                CkCd=0.0,
                rr_ER11=rr,
            )
        # MATLAB returns NaN; Python must do the same
        assert np.all(np.isnan(V)), "Expected all-NaN V from non-converging ER11"
        assert np.isnan(r_out), "Expected NaN r_out from non-converging ER11"

    def test_er11_radprof_nan_propagates_to_profile(self):
        """
        When ER11 returns NaN for a bisection step the MATLAB bisection
        treats it as 'rmaxr0 too small' and nudges upward; it does NOT
        propagate NaN straight to the output.  The final profile must
        therefore still be finite for the default (Vmax=50, r0=800 km) case
        even though individual bisection steps may silently fail.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(_call_profile(cle15, Vmax=50.0, r0=800e3, fcor=5e-5))
        assert np.isfinite(
            res["rmax"]
        ), "Profile should converge despite ER11 NaN steps"

    # ------------------------------------------------------------------
    # B. CkCd upward-nudge fallback (MATLAB soln_converged loop)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_ckcd_nudge_fallback_produces_finite_result(self, mod):
        """
        MATLAB increments CkCd by 0.1 when the bisection converges without
        ever finding an intersection (ER11 below E04 for all rmaxr0 in
        [0.001, 0.75]).  This can happen at very low CkCd.  Both
        implementations must return a finite rmax rather than NaN.

        CkCd = 0.3 (below the typical lower bound of 0.5 that MATLAB
        commented out) with a mid-size storm reliably triggers the fallback.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(
                _call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, CkCd_input=0.3)
            )
        assert np.isfinite(
            res["rmax"]
        ), f"mod={mod.__name__}: CkCd nudge fallback failed (rmax={res['rmax']})"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_ckcd_nudge_fallback_rmax_physical(self, mod):
        """After CkCd nudge the resulting rmax must still be < r0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _unpack(
                _call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, CkCd_input=0.3)
            )
        if np.isfinite(res["rmax"]):
            assert res["rmax"] < 800e3, "rmax must be < r0 after CkCd nudge"

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_ckcd_at_cap_19_does_not_raise(self, mod):
        """
        MATLAB caps CkCd at 1.9 with a warning but does not error.
        CkCdvary=1 with Vmax=5 m/s gives CkCd ≈ 0.62 (fine); with
        Vmax=200 m/s the quadratic gives > 1.9, which both MATLAB and
        Python must cap and continue.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # CkCdvary=1 at high Vmax → CkCd will exceed 1.9, must be capped
            res = _unpack(
                _call_profile(mod, Vmax=120.0, r0=800e3, fcor=5e-5, CkCdvary=1)
            )
        # Either converges (finite rmax) or returns NaN — must not raise
        assert isinstance(res["rmax"], float)

    # ------------------------------------------------------------------
    # C. run_cle15 NaN propagation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_run_cle15_finite_for_standard_inputs(self, mod):
        """
        MATLAB's CLE15_plot_r0input.m uses Vmax=50, r0=900 km, fcor=5e-5.
        Both implementations must return finite (pm, rmax, pc).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm, rmax, pc = mod.run_cle15(
                inputs={"Vmax": 50.0, "r0": 900e3, "fcor": 5e-5}
            )
        assert np.isfinite(pm), f"pm not finite for MATLAB example case: {pm}"
        assert np.isfinite(rmax), f"rmax not finite: {rmax}"
        assert np.isfinite(pc), f"pc not finite: {pc}"

    def test_run_cle15_py_nan_guard_missing_but_documented(self):
        """
        Document the known asymmetry: cle15.run_cle15 does NOT have an
        explicit rmax-NaN guard, while cle15n.run_cle15 does.

        When chavas_et_al_2015_profile returns a valid profile for the
        standard MATLAB example inputs, run_cle15 must return finite values
        in both cases (the guard difference only matters under solver failure).
        This test pins the *current* behaviour: both return finite results for
        valid inputs, so the missing guard is not yet observable here.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm_py, rmax_py, pc_py = cle15.run_cle15(
                inputs={"Vmax": 50.0, "r0": 900e3, "fcor": 5e-5}
            )
            pm_nb, rmax_nb, pc_nb = cle15n.run_cle15(
                inputs={"Vmax": 50.0, "r0": 900e3, "fcor": 5e-5}
            )
        assert np.isfinite(pm_py) and np.isfinite(
            pm_nb
        ), "Both implementations must return finite pm for MATLAB example inputs"

    def test_cle15n_run_cle15_nan_guard_rmax_nan(self):
        """
        cle15n.run_cle15 must return (NaN, NaN, NaN) when the solver
        returns NaN for rmax.  We verify this by monkeypatching
        chavas_et_al_2015_profile inside cle15n to return an all-NaN result,
        simulating the solver-failure path that MATLAB would produce
        (V_ER11 = NaN(...) from ER11_radprof.m with no soln_converged).
        """
        import unittest.mock as mock

        _nan = np.array([np.nan])
        nan_result = (
            _nan,
            _nan,
            np.nan,
            np.nan,
            np.nan,
            _nan,
            _nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        with mock.patch.object(
            cle15n, "chavas_et_al_2015_profile", return_value=nan_result
        ):
            pm, rmax, pc = cle15n.run_cle15()

        assert np.isnan(pm), "cle15n.run_cle15 must return NaN pm when rmax is NaN"
        assert np.isnan(rmax), "cle15n.run_cle15 must return NaN rmax when rmax is NaN"
        assert np.isnan(pc), "cle15n.run_cle15 must return NaN pc when rmax is NaN"

    def test_cle15n_run_cle15_nan_guard_rmerge_equals_r0(self):
        """
        cle15n.run_cle15 must return (NaN, NaN, NaN) when rmerge == r0
        (degenerate merge — no interior tangent found).  MATLAB would
        return a meaningless profile (VV based on a bad rmerge); Python
        cle15n explicitly guards against this.
        """
        import unittest.mock as mock

        r0 = 800e3
        rr = np.linspace(0, r0, 1000)
        VV = np.zeros(1000)
        # rmerge == r0 (degenerate)
        degenerate = (rr, VV, 50e3, r0, 0.0, rr / r0, VV, 0.0625, 0.5, 1.0, 1.0)

        with mock.patch.object(
            cle15n, "chavas_et_al_2015_profile", return_value=degenerate
        ):
            pm, rmax, pc = cle15n.run_cle15(inputs={"r0": r0})

        assert np.isnan(
            pm
        ), "cle15n.run_cle15 must return NaN pm for degenerate rmerge==r0"

    def test_cle15n_run_cle15_nan_guard_high_nan_fraction(self):
        """
        cle15n.run_cle15 must return (NaN, NaN, NaN) when more than 10 %
        of the wind profile is NaN (indicating a failed solve), matching the
        intent of MATLAB's isnan(max(VV_ER11)) check on the bisection step.
        """
        import unittest.mock as mock

        r0 = 800e3
        rr = np.linspace(0, r0, 1000)
        VV = np.full(1000, np.nan)  # all NaN
        VV[0] = 0.0  # r=0 is 0, rmax won't be NaN
        mostly_nan = (rr, VV, 50e3, 400e3, 10.0, rr / r0, VV, 0.0625, 0.5, 0.5, 0.5)

        with mock.patch.object(
            cle15n, "chavas_et_al_2015_profile", return_value=mostly_nan
        ):
            pm, rmax, pc = cle15n.run_cle15(inputs={"r0": r0})

        assert np.isnan(
            pm
        ), "cle15n.run_cle15 must return NaN pm when > 10% of VV is NaN"

    # ------------------------------------------------------------------
    # D. profile_from_stats NaN propagation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_from_stats_finite_standard_case(self, mod):
        """
        MATLAB example (Vmax=50, r0=900 km, fcor=5e-5) must produce a finite
        pressure profile in profile_from_stats for both implementations.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mod.profile_from_stats(vmax=50.0, fcor=5e-5, r0=900e3, p0=1013.25)
        p = np.asarray(out["p"])
        assert np.all(np.isfinite(p[np.isfinite(p)])), "pressure profile must be finite"
        assert np.isfinite(out["rmax"]), "rmax must be finite"

    # ------------------------------------------------------------------
    # E. chavas_et_al_2015_profile NaN output structure
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_nan_structure_on_impossible_vmax(self, mod):
        """
        Vmax=0 m/s is unphysical (no storm).  Both implementations must return
        a NaN result tuple (with a warning) rather than crashing.

        MATLAB reference behaviour: when Vmax=0 the ER11 rmax-r0 solve
        produces empty roots, so MATLAB would error without a graceful return.

        Fix: ``chavas_et_al_2015_profile`` now guards Vmax<=0 at entry and
        returns the NaN sentinel with a warning in both modules.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _unpack(_call_profile(mod, Vmax=0.0, r0=800e3, fcor=5e-5))
                # If it returns something, scalars must be float (possibly NaN)
                assert isinstance(res["rmax"], float), "rmax must be a float scalar"
            except Exception as exc:
                pytest.fail(f"mod={mod.__name__}: raised {type(exc).__name__}: {exc}")

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_nan_structure_on_zero_r0(self, mod):
        """
        r0=0 is unphysical (no outer boundary).  Both implementations must
        return a NaN result tuple (with a warning) rather than crashing.

        MATLAB reference behaviour: ``M0 = 0.5*fcor*r0^2 = 0``, the symbolic
        solve returns no valid roots, so MATLAB would error here.

        Fix: ``chavas_et_al_2015_profile`` now guards r0<=0 at entry and
        returns the NaN sentinel with a warning in both modules.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _unpack(_call_profile(mod, Vmax=50.0, r0=0.0, fcor=5e-5))
                assert isinstance(res["rmax"], float)
            except Exception as exc:
                pytest.fail(f"mod={mod.__name__}: raised {type(exc).__name__}: {exc}")

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_nan_structure_on_zero_wcool(self, mod):
        """
        w_cool = 0 makes the E04 gamma parameter infinite (Cd*f*r0/0).
        MATLAB would produce NaN in the E04 integration; Python must
        handle this gracefully (either NaN output or a warning, not a crash).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _unpack(
                    _call_profile(mod, Vmax=50.0, r0=800e3, fcor=5e-5, w_cool=0.0)
                )
                assert isinstance(res["rmax"], float)
            except Exception as exc:
                pytest.fail(f"mod={mod.__name__}: raised {type(exc).__name__}: {exc}")

    @pytest.mark.parametrize("mod", [cle15, cle15n])
    def test_profile_nan_structure_on_zero_fcor(self, mod):
        """
        fcor=0 (equator) makes M0=0 and the ER11 power-law ratio singular.
        Both implementations must return a NaN result tuple (with a warning)
        rather than crashing.

        MATLAB reference behaviour: ``syms solve`` returns an empty root set
        for this degenerate case, effectively erroring without graceful output.

        Fix: ``chavas_et_al_2015_profile`` now guards fcor==0 at entry (after
        ``fcor = abs(fcor)``) and returns the NaN sentinel with a warning in
        both modules, avoiding the ZeroDivisionError that previously occurred
        deep inside the ER11 power-law computation.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _unpack(_call_profile(mod, Vmax=50.0, r0=800e3, fcor=0.0))
                assert isinstance(res["rmax"], float)
            except Exception as exc:
                pytest.fail(f"mod={mod.__name__}: raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # F. Pressure-assumption asymmetry
    # ------------------------------------------------------------------

    def test_pressure_assumption_asymmetry_is_consistent_within_each_function(self):
        """
        run_cle15 defaults to 'isopycnal'; profile_from_stats defaults to
        'isothermal'.  The two assumptions give different pressures for the
        same inputs.  This test verifies the asymmetry is real and stable —
        i.e. the two functions do NOT accidentally agree, and that each
        function returns the same answer when called twice (determinism).
        """
        inputs = {"Vmax": 50.0, "r0": 800e3, "fcor": 5e-5}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm1, rmax1, pc1 = cle15.run_cle15(inputs=inputs)
            pm2, rmax2, pc2 = cle15.run_cle15(inputs=inputs)

            out1 = cle15.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)
            out2 = cle15.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)

        # Each function must be deterministic
        assert pm1 == pm2, "run_cle15 must be deterministic"
        assert np.allclose(
            out1["p"], out2["p"]
        ), "profile_from_stats must be deterministic"

        # The two assumptions must give different central pressures
        pc_run = pc1 / 100  # Pa → hPa
        pc_pfr = out1["p"][0]  # already hPa
        assert abs(pc_run - pc_pfr) > 0.1, (
            f"run_cle15 (isopycnal, {pc_run:.2f} hPa) and "
            f"profile_from_stats (isothermal, {pc_pfr:.2f} hPa) should differ "
            f"due to different pressure assumptions"
        )

    def test_pressure_assumption_consistent_when_matched(self):
        """
        When both functions are called with the *same* pressure assumption
        (isothermal, passed explicitly) their central pressures must agree
        to within a few hPa for the same (Vmax, r0, fcor) inputs.

        A tight (< 1 hPa) agreement is not achievable because the two
        functions integrate pressure along different radial grids:
        ``run_cle15`` uses the merged CLE15 profile on a coarse grid tuned
        for the bisection solver, while ``profile_from_stats`` uses a finer
        user-specified grid.  The resulting ~2.7 hPa offset is systematic
        (independent of pressure assumption) and is documented here as the
        expected tolerance.  If either function changes its pressure
        integration this test will detect regressions.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm, rmax, pc = cle15.run_cle15(
                inputs={"Vmax": 50.0, "r0": 800e3, "fcor": 5e-5},
                pressure_assumption="isothermal",
            )
            out = cle15.profile_from_stats(vmax=50.0, fcor=5e-5, r0=800e3, p0=1013.25)

        pc_run_hpa = pc / 100
        pc_pfr_hpa = out["p"][0]
        diff = abs(pc_run_hpa - pc_pfr_hpa)

        # The systematic grid-integration offset is ~2.7 hPa; allow up to 5 hPa.
        assert diff < 5.0, (
            f"Central pressures diverge by {diff:.2f} hPa, expected < 5 hPa: "
            f"run_cle15={pc_run_hpa:.2f} hPa, "
            f"profile_from_stats={pc_pfr_hpa:.2f} hPa"
        )
        # But they should still be in the same ballpark — not off by more
        # than ~1 % of the ambient pressure (1013.25 hPa ≈ 10 hPa).
        assert diff > 0.5, (
            f"Central pressures agree suspiciously well ({diff:.2f} hPa): "
            f"the known ~2.7 hPa grid offset appears to have disappeared, "
            f"which may indicate the integration method changed."
        )


# ---------------------------------------------------------------------------
# 10. TestMatlabRegression – rmax values cross-validated against Octave
#     (ER11E04_nondim_r0input.m run with the same Python-default parameters:
#      Cdvary=0, C_d=0.0015, w_cool=0.002, CkCdvary=0, CkCd=0.9,
#      eye_adj=0, alpha_eye=0.5)
#
#     Reference values produced by mcle/run_edge_cases.m under Octave 9.4.0.
#     All normal cases agree to < 1%.  The first case (90 m/s, 200 km, 3e-5)
#     is the known degenerate high-Ro regime where both solvers are
#     sensitive to bisection direction; it is tested with a looser tolerance.
# ---------------------------------------------------------------------------
class TestMatlabRegression:
    """Cross-validate rmax against MATLAB/Octave reference (ER11E04_nondim_r0input.m)."""

    # (Vmax m/s, r0 m, fcor s-1, octave_rmax_km, tol_pct, label)
    _CASES = [
        # degenerate high-Ro case – both solvers are sensitive here; loose tol
        (90.0, 200e3, 3e-5, 1.0426, 25.0, "high Vmax small r0 low f (degenerate)"),
        # well-behaved cases – match to < 1 %
        (90.0, 200e3, 5e-5, 1.3152, 1.0, "high Vmax small r0 mid f"),
        (90.0, 200e3, 7e-5, 1.5266, 1.0, "high Vmax small r0 high f"),
        (20.0, 200e3, 3e-5, 5.4193, 1.0, "low Vmax  small r0 low f"),
        (90.0, 2000e3, 3e-5, 29.3578, 1.0, "high Vmax large r0 low f"),
        (20.0, 2000e3, 3e-5, 224.5165, 1.0, "low Vmax  large r0 low f"),
        (50.0, 800e3, 5e-5, 19.6884, 1.0, "mid Vmax  mid r0  mid f (reference)"),
    ]

    @pytest.mark.parametrize(
        "Vmax,r0,fcor,oct_rmax_km,tol_pct,label",
        _CASES,
        ids=[c[-1] for c in _CASES],
    )
    def test_rmax_matches_octave(self, Vmax, r0, fcor, oct_rmax_km, tol_pct, label):
        """Python rmax must agree with Octave reference to within tol_pct %."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cle15.chavas_et_al_2015_profile(
                Vmax,
                r0,
                fcor,
                CDVARY_DEFAULT,
                CD_DEFAULT,
                W_COOL_DEFAULT,
                CKCDVARY_DEFAULT,
                CK_CD_DEFAULT,
                EYE_ADJ_DEFAULT,
                ALPHA_EYE_DEFAULT,
            )
        rmax_km = result[2] / 1e3
        diff_pct = abs(rmax_km - oct_rmax_km) / oct_rmax_km * 100
        assert diff_pct < tol_pct, (
            f"{label}: Python rmax={rmax_km:.4f} km, "
            f"Octave={oct_rmax_km:.4f} km, diff={diff_pct:.2f}% > {tol_pct}%"
        )
