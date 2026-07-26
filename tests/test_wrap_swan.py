"""Tests for the adforce input-deck options: resolution / tide / swan.

Covers ``adforce.wrap.stage_input_files`` (per-option template selection,
introduced when the hardcoded mid/no-tide asserts were removed) and
``_enable_swan_in_fort15`` (the SWAN coupling patch: NWS += 300, RSTIMINC on
the WTIMINC line), run against the REAL templates in adforce/setup so a
template regression (missing file, renamed comment tag) fails here rather
than mid-campaign.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adforce.wrap import (
    RSTIMINC_S,
    SETUP_PATH,
    _enable_swan_in_fort15,
    stage_input_files,
)


def _fort15_lines(path):
    with open(path) as f:
        return f.readlines()


def _value_of(lines, tag):
    for line in lines:
        if tag in line:
            return line.partition("!")[0].split()
    raise AssertionError(f"{tag} line not found")


@pytest.mark.parametrize("resolution", ["low", "mid"])
@pytest.mark.parametrize("tide", [False, True])
def test_stage_selects_templates(tmp_path, resolution, tide):
    stage_input_files(str(tmp_path), resolution=resolution, tide=tide, swan=False)
    for f in ("fort.13", "fort.14", "fort.15"):
        assert (tmp_path / f).exists()
    ref = os.path.join(
        SETUP_PATH, f"fort.15.{resolution}.{'tide' if tide else 'notide'}"
    )
    assert (tmp_path / "fort.15").read_text() == open(ref).read()
    assert not (tmp_path / "fort.26").exists()


def test_stage_swan_files_and_patch(tmp_path):
    stage_input_files(str(tmp_path), resolution="mid", tide=False, swan=True)
    assert (tmp_path / "fort.26").exists()
    swaninit = (tmp_path / "swaninit").read_text()
    assert "fort.26" in swaninit  # SWAN must read fort.26, not INPUT
    lines = _fort15_lines(tmp_path / "fort.15")
    assert _value_of(lines, "! NWS")[0] == "313"  # 13 + 300
    wt = _value_of(lines, "! WTIMINC")
    assert wt[-1] == str(RSTIMINC_S) and len(wt) >= 2  # RSTIMINC appended


def test_swan_patch_idempotent(tmp_path):
    src = os.path.join(SETUP_PATH, "fort.15.mid.notide")
    dst = tmp_path / "fort.15"
    shutil.copy(src, dst)
    _enable_swan_in_fort15(str(dst))
    once = dst.read_text()
    _enable_swan_in_fort15(str(dst))
    assert dst.read_text() == once  # no 613, no duplicated RSTIMINC


def test_unknown_resolution_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="high"):
        stage_input_files(str(tmp_path), resolution="high")
