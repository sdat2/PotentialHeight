"""Characterization tests for the pure helpers of the SurgeNet training-data generator.

These pin down the current behavior of the pure (no adcircpy, no subprocess)
functions that were split out of ``adforce/generate_training_data.py`` into the
``adforce.training`` package:

- ``clean_radii`` and ``_decode_char_array`` (``adforce.training.atcf``)
- ``Storm`` and ``calculate_simulation_window`` (``adforce.training.storms``)
- ``calculate_cfl_timestep`` (``adforce.training.cfl``; exercised with a
  duck-typed fake mesh, so no adcircpy or mesh files are needed)

Expected values were captured by running these same assertions against the
original monolithic ``adforce/generate_training_data.py`` before the move.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# After the split these live in adforce.training.*; the compatibility shim
# adforce.generate_training_data re-exports the same names.
from adforce.training.storms import Storm, calculate_simulation_window
from adforce.training.atcf import _decode_char_array, clean_radii
from adforce.training.cfl import calculate_cfl_timestep


# ---------------------------------------------------------------------------
# clean_radii
# ---------------------------------------------------------------------------


class TestCleanRadii:
    def test_simple_mirroring(self):
        assert clean_radii([60.0, 60.0, 0.0, 0.0]) == [60.0, 60.0, 60.0, 60.0]

    def test_unmirrorable_pair_symmetrizes_to_max(self):
        assert clean_radii([15.0, 0.0, 15.0, 0.0]) == [15.0, 15.0, 15.0, 15.0]

    def test_single_value_symmetrizes_to_max(self):
        assert clean_radii([15.0, 0.0, 0.0, 0.0]) == [15.0, 15.0, 15.0, 15.0]

    def test_all_zeros_passthrough(self):
        assert clean_radii([0.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 0.0]

    def test_fully_defined_passthrough(self):
        assert clean_radii([60.0, 60.0, 15.0, 25.0]) == [60.0, 60.0, 15.0, 25.0]

    def test_opposite_quadrant_mirroring_both_pairs(self):
        # NE<-SW and NW<-SE mirroring resolves all zeros without symmetrizing.
        assert clean_radii([0.0, 25.0, 30.0, 0.0]) == [30.0, 25.0, 30.0, 25.0]

    def test_int_inputs(self):
        assert clean_radii([100, 100, 75, 75]) == [100, 100, 75, 75]
        assert clean_radii([0, 0, 0, 40]) == [40, 40, 40, 40]

    def test_mutates_input_in_place_when_mixed(self):
        # Characterization: the input list is modified in place for "mixed"
        # (some-zero, some-nonzero) inputs; callers aliasing it see changes.
        radii = [60.0, 60.0, 0.0, 0.0]
        result = clean_radii(radii)
        assert radii == [60.0, 60.0, 60.0, 60.0]
        assert result is radii

    def test_returns_same_object_when_clean(self):
        radii = [10.0, 20.0, 30.0, 40.0]
        assert clean_radii(radii) is radii


# ---------------------------------------------------------------------------
# _decode_char_array
# ---------------------------------------------------------------------------


class TestDecodeCharArray:
    def test_byte_char_array(self):
        assert _decode_char_array(np.array([b"A", b"L"], dtype="|S1")) == "AL"

    def test_byte_char_array_trailing_spaces(self):
        arr = np.array([b"G", b"E", b"R", b"T", b" ", b" "], dtype="|S1")
        assert _decode_char_array(arr) == "GERT"

    def test_byte_char_array_with_nulls(self):
        arr = np.array([b"T", b"S", b"", b""], dtype="|S1")
        assert _decode_char_array(arr) == "TS"

    def test_plain_str(self):
        assert _decode_char_array("KATRINA  ") == "KATRINA"

    def test_str_with_nulls(self):
        assert _decode_char_array("KAT\x00\x00 ") == "KAT"

    def test_plain_bytes(self):
        assert _decode_char_array(b"IDA\x00\x00 ") == "IDA"

    def test_fixed_width_bytes_array(self):
        # 0-d |S10 array: tobytes() path with null padding stripped.
        assert _decode_char_array(np.array(b"KATRINA", dtype="|S10")) == "KATRINA"

    def test_iterable_fallback(self):
        assert _decode_char_array([b"A", "L", b"1"]) == "AL1"

    def test_failure_returns_unknown(self):
        # Characterization: any exception during decoding yields "UNKNOWN".
        assert _decode_char_array(12345) == "UNKNOWN"
        assert _decode_char_array(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Storm / calculate_simulation_window
# ---------------------------------------------------------------------------


def _make_storm(times):
    # Storm expects raw IBTrACS bytes for sid/name (it calls .decode()).
    return Storm(sid=b"2005236N23285", name=b"KATRINA", time=times)


class TestStorm:
    def test_decodes_bytes_and_takes_year_from_first_time(self):
        times = [datetime(2005, 8, 23, 18), datetime(2005, 8, 31, 6)]
        storm = _make_storm(times)
        assert storm.id == "2005236N23285"
        assert storm.name == "KATRINA"
        assert storm.year == 2005
        assert storm.time is times

    def test_empty_time_list_gives_year_none(self):
        storm = _make_storm([])
        assert storm.year is None


class TestCalculateSimulationWindow:
    def test_defaults_return_first_and_last_track_times(self):
        times = [
            datetime(2005, 8, 23, 18),
            datetime(2005, 8, 28, 12),
            datetime(2005, 8, 31, 6),
        ]
        start, end = calculate_simulation_window(_make_storm(times))
        assert start == datetime(2005, 8, 23, 18)
        assert end == datetime(2005, 8, 31, 6)

    def test_spinup_and_extra_days_extend_window(self):
        times = [datetime(2020, 7, 1, 0), datetime(2020, 7, 3, 12)]
        start, end = calculate_simulation_window(
            _make_storm(times), spinup_days=2.0, extra_days=1.5
        )
        assert start == datetime(2020, 6, 29, 0)
        assert end == datetime(2020, 7, 5, 0)

    def test_fractional_spinup(self):
        times = [datetime(2020, 7, 1, 0), datetime(2020, 7, 2, 0)]
        start, end = calculate_simulation_window(_make_storm(times), spinup_days=0.5)
        assert start == datetime(2020, 6, 30, 12)
        assert end == datetime(2020, 7, 2, 0)


# ---------------------------------------------------------------------------
# calculate_cfl_timestep (duck-typed fake mesh; no adcircpy required)
# ---------------------------------------------------------------------------


class FakeMesh:
    """Minimal stand-in for adcircpy.AdcircMesh.

    calculate_cfl_timestep only touches ``node_distances_in_meters`` (dict of
    node -> {neighbor: distance}) and ``values`` (pandas DataFrame of depths,
    positive down), so a tiny fake is enough to characterize the math.
    """

    def __init__(self, distances, depths):
        self._distances = distances
        self._depths = depths

    @property
    def node_distances_in_meters(self):
        return self._distances

    @property
    def values(self):
        return self._depths


class TestCalculateCflTimestep:
    def _mesh(self):
        distances = {
            "1": {"2": 500.0, "3": 800.0},
            "2": {"1": 500.0, "3": 1200.0},
            "3": {},  # empty neighbor dicts are skipped
        }
        depths = pd.DataFrame({"depth": [10.0, 250.0, 40.0]})
        return FakeMesh(distances, depths)

    def test_default_arguments(self):
        # dt = cfl_target * min_dx / (maxvel + sqrt(g * max_h))
        dt = calculate_cfl_timestep(self._mesh())
        expected = 0.7 * 500.0 / (5.0 + math.sqrt(9.81 * 250.0))
        assert dt == pytest.approx(expected)

    def test_custom_arguments(self):
        dt = calculate_cfl_timestep(self._mesh(), cfl_target=0.5, maxvel=10.0, g=9.8)
        expected = 0.5 * 500.0 / (10.0 + math.sqrt(9.8 * 250.0))
        assert dt == pytest.approx(expected)

    def test_negative_max_depth_uses_absolute_value(self, capsys):
        mesh = FakeMesh({"1": {"2": 100.0}}, pd.DataFrame({"depth": [-25.0, -100.0]}))
        dt = calculate_cfl_timestep(mesh)
        expected = 0.7 * 100.0 / (5.0 + math.sqrt(9.81 * 25.0))
        assert dt == pytest.approx(expected)
        assert "Using absolute value" in capsys.readouterr().out

    def test_empty_distances_raises_value_error(self):
        mesh = FakeMesh({}, pd.DataFrame({"depth": [10.0]}))
        with pytest.raises(ValueError, match="empty"):
            calculate_cfl_timestep(mesh)

    def test_missing_distance_property_raises_attribute_error(self):
        class NoDistances:
            @property
            def values(self):
                return pd.DataFrame({"depth": [10.0]})

        with pytest.raises(AttributeError, match="node_distances_in_meters"):
            calculate_cfl_timestep(NoDistances())

    def test_non_dataframe_values_raises_value_error(self):
        mesh = FakeMesh({"1": {"2": 100.0}}, np.array([10.0]))
        with pytest.raises(ValueError, match="mesh depths"):
            calculate_cfl_timestep(mesh)
