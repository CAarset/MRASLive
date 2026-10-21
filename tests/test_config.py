"""Tests for the YAML/pydantic configuration manager."""

import pytest
from pydantic import ValidationError

from mras.config import RunConfig, build, load_yaml


def test_defaults():
    cfg = RunConfig()
    assert cfg.levels == [0.0]
    assert cfg.dt == 1e-3
    assert cfg.h is None


def test_validation_rejects_unknown_and_bad_values():
    with pytest.raises(ValidationError):
        RunConfig(nuisance=1)
    with pytest.raises(ValidationError):
        RunConfig(dt=0.0)
    with pytest.raises(ValidationError):
        RunConfig(k_u=0)


def test_load_yaml_single_mapping(tmp_path):
    p = tmp_path / "one.yaml"
    p.write_text("h: 0.1\ndt: 0.01\nlevels: [0.0, 0.05]\n")
    cfgs = load_yaml(p)
    assert len(cfgs) == 1
    assert cfgs[0].h == 0.1 and cfgs[0].levels == [0.0, 0.05]
    assert cfgs[0].seed == 0  # default filled in


def test_load_yaml_sequence_is_a_sweep(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(
        "- {h: 0.2, seed: 1}\n"
        "- {h: 0.1, seed: 2}\n")
    cfgs = load_yaml(p)
    assert [(c.h, c.seed) for c in cfgs] == [(0.2, 1), (0.1, 2)]


def test_load_yaml_defaults_and_runs(tmp_path):
    p = tmp_path / "both.yaml"
    p.write_text(
        "defaults: {tend: 2.0, levels: [0.0, 0.1]}\n"
        "runs:\n"
        "  - {h: 0.2}\n"
        "  - {h: 0.1, levels: [0.2]}\n")
    cfgs = load_yaml(p)
    assert len(cfgs) == 2
    assert cfgs[0].tend == 2.0 and cfgs[0].levels == [0.0, 0.1]
    assert cfgs[1].h == 0.1 and cfgs[1].levels == [0.2]


def test_build_passes_typed_kwargs():
    seen = {}

    class Dummy:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    cfg = RunConfig(h=0.1, dt=0.01, tend=1.0, levels=[0.05], out="here")
    prob = build(Dummy, cfg, extra_flag=True)
    assert seen == {"h": 0.1, "dt": 0.01, "tend": 1.0,
                    "k_u": 3, "seed": 0, "extra_flag": True}
    assert prob is not None


def test_counters_from_times():
    cfg = RunConfig(dt=0.01, times=[0.0, 0.5, 2.0])
    assert cfg.counters() == [1, 51, 201]
