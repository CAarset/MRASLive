"""The diagnostics behind the paper's numbers (scripts/): they run on a real
checkpoint and say what they are claimed to say."""

import json

import numpy as np
import pytest

from mras.live import build, run_live
from scripts import error_regions, flow_check, noise_terms


@pytest.fixture(scope="module")
def noisy_checkpoint(tmp_path_factory):
    stem = tmp_path_factory.mktemp("runs") / "noisy"  # no dots: a suffix
    run_live(build("burgers_2d", h=0.5, dt=0.05, tend=0.6), 0.05, path=stem,
             movie=False)
    return stem.with_suffix(".h5")


def test_error_regions_partition_the_removable_error(noisy_checkpoint, tmp_path):
    report = error_regions.main([str(noisy_checkpoint), "--near", "inlet:0.5",
                                 "--out", str(tmp_path)])
    shares = [row["share_of_gap2"] for row in report["regions"]]
    areas = [row["area_share"] for row in report["regions"]]
    assert sum(shares) == pytest.approx(1.0) and sum(areas) == pytest.approx(1.0)
    # the Pythagorean split: error^2 = removable^2 + floor^2
    assert report["error"] ** 2 == pytest.approx(
        report["removable"] ** 2 + report["floor"] ** 2)
    assert report["regions"][0]["region"] == "cut cells"
    assert (tmp_path / f"{noisy_checkpoint.stem}.json").exists()


def test_noise_terms_measure_the_saved_datums_error(noisy_checkpoint, tmp_path):
    rows = noise_terms.main([str(noisy_checkpoint), "--times", "0.4", "0.6",
                             "--out", str(tmp_path)])
    assert len(rows) == 2
    for row in rows:
        assert row["noise_L2_relative"] > 0
        assert all(np.isfinite(v) and v >= 0 for k, v in row.items()
                   if k != "t")
        assert "boundary wall|barrier" in row  # 2D Burgers' slip surfaces


def test_flow_check_compares_refinements(tmp_path):
    report = flow_check.main(["burgers_2d", "--h", "0.5", "--tend", "0.3",
                              "--marks", "0.3", "--flux", "inlet", "outlet",
                              "--threads", "2", "--out", str(tmp_path)])
    row = report["marks"]["0.3"]
    for key in ("half_dt", "double_h"):
        l2, h1 = row[key]
        assert 0 <= l2 < 1 and 0 <= h1 < 1
    assert row["flux"]["inlet"] < 0  # inflow: against the outward normal
    assert json.loads((tmp_path / "burgers_2d.json").read_text())["name"] == "burgers_2d"
