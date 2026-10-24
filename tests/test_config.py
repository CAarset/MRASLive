"""Tests for the layered configuration: unset fields never reach the problem.

An unset RunConfig field means "inherit" (the example DEFAULTS, or the
constructor's own defaults at the bottom), never "override with a number",
so a YAML only carries what the user actually wants to change.
"""

import pytest
from pydantic import ValidationError

from mras.config import RunConfig, build, load_yaml


def test_everything_unset_by_default():
    cfg = RunConfig()
    assert cfg.h is None and cfg.dt is None and cfg.tend is None
    assert cfg.k_u is None and cfg.seed is None and cfg.out is None
    assert cfg.noise_levels is None and cfg.times == []


def test_unknown_keys_are_kept_for_the_constructor():
    cfg = RunConfig(data_h=0.05, tsteps=7)
    assert cfg.data_h == 0.05 and cfg.tsteps == 7


def test_validation_rejects_bad_values():
    with pytest.raises(ValidationError):
        RunConfig(dt=0.0)
    with pytest.raises(ValidationError):
        RunConfig(k_u=0)
    with pytest.raises(ValidationError):
        RunConfig(noise_levels=[])
    with pytest.raises(ValidationError):
        RunConfig(noise_levels=[-0.05])


def test_load_yaml_single_mapping(tmp_path):
    p = tmp_path / "one.yaml"
    p.write_text("h: 0.1\ndt: 0.01\nnoise_levels: [0.0, 0.05]\n")
    defaults, runs = load_yaml(p)
    assert defaults == {}
    assert runs == [{"h": 0.1, "dt": 0.01, "noise_levels": [0.0, 0.05]}]


def test_load_yaml_sequence_is_a_sweep(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(
        "- {h: 0.2, seed: 1}\n"
        "- {h: 0.1, seed: 2}\n")
    defaults, runs = load_yaml(p)
    assert defaults == {}
    assert [(r["h"], r["seed"]) for r in runs] == [(0.2, 1), (0.1, 2)]


def test_load_yaml_defaults_and_runs(tmp_path):
    p = tmp_path / "both.yaml"
    p.write_text(
        "defaults: {tend: 2.0, noise_levels: [0.0, 0.1]}\n"
        "runs:\n"
        "  - {h: 0.2}\n"
        "  - {h: 0.1, noise_levels: [0.2]}\n")
    defaults, runs = load_yaml(p)
    assert defaults == {"tend": 2.0, "noise_levels": [0.0, 0.1]}
    assert runs[0] == {"h": 0.2}
    assert runs[1]["noise_levels"] == [0.2]


def test_load_yaml_documents_accumulate(tmp_path):
    p = tmp_path / "multi.yaml"
    p.write_text("defaults: {tend: 1.0}\n---\nruns: [{seed: 3}, {seed: 4}]\n")
    defaults, runs = load_yaml(p)
    assert defaults == {"tend": 1.0}
    assert [r["seed"] for r in runs] == [3, 4]


def test_build_forwards_only_set_fields_and_extras():
    seen = {}

    class Dummy:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    cfg = RunConfig(h=0.1, dt=0.01, tend=1.0, data_h=0.05,
                    noise_levels=[0.05], out="here")
    prob = build(Dummy, cfg, extra_flag=True)
    # seed/k_u unset: not passed; orchestration fields consumed by the driver.
    assert seen == {"h": 0.1, "dt": 0.01, "tend": 1.0, "data_h": 0.05,
                    "extra_flag": True}
    assert prob is not None


def test_counters_from_times():
    cfg = RunConfig(dt=0.01, times=[0.0, 0.5, 2.0])
    assert cfg.counters() == [1, 51, 201]


def test_counters_need_dt():
    with pytest.raises(ValueError, match="needs dt"):
        RunConfig(times=[0.5]).counters()
