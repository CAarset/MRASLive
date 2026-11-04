"""Validated run configurations in YAML, managed with pydantic.

Every field is optional: an unset field inherits the example's ``DEFAULTS``
(the problem class's own keyword defaults being the floor), so the YAML is
the control switch that tweaks values, never a file that must restate them.
Unknown keys are kept and forwarded to the problem constructor, which lets
problem-specific knobs (``q_h``, ``tsteps``, ...) live in the YAML too.

A configuration file is either

* a single mapping  -- one run,
* a sequence of mappings -- a sweep, or
* a mapping with ``defaults`` and ``runs`` keys -- a sweep with shared
  settings; each entry of ``runs`` is merged over ``defaults``.

    defaults: {tend: 5.0, noise_levels: [0.0, 0.05]}
    runs:
      - {h: 0.2, seed: 1}
      - {h: 0.1, seed: 2, noise_levels: [0.2]}

Out-of-range values are rejected at validation time; unknown names reach
the constructor, which is the one that knows its own keywords.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    """One run specification; ``None`` means "inherit", not "default here"."""

    model_config = ConfigDict(extra="allow")

    h: float | None = Field(None, gt=0, description="design mesh width")
    dt: float | None = Field(None, gt=0)
    tend: float | None = Field(None, gt=0)
    k_u: int | None = Field(None, ge=1, le=8,
                            description="polynomial order of U_h")
    k_q: int | None = Field(None, ge=0, le=8,
                            description="polynomial order of Q_h (L2)")
    seed: int | None = None
    noise_levels: list[float] | None = Field(
        None, description="relative noise levels, fractions (0.05 = 5%)")
    times: list[float] = Field(default_factory=list,
                               description="snapshot times for figures")
    out: str | None = None

    @field_validator("noise_levels")
    @classmethod
    def _noise_levels_nonnegative(cls, v):
        if v is not None and (not v or any(x < 0 for x in v)):
            raise ValueError("noise_levels must be non-empty and non-negative")
        return v

    def steps(self):
        """The steps n (the states at t = n dt) closest to the requested
        times."""
        if self.dt is None:
            raise ValueError("steps() needs dt; the constructor's own "
                             "series_steps covers the standard snapshots")
        return sorted({round(t / self.dt) for t in self.times})


def load_yaml(path) -> tuple[dict, list[dict]]:
    """Parse a YAML configuration file into (defaults, one dict per run).

    The dicts are raw: merging happens per example, whose ``DEFAULTS`` sit
    under the YAML defaults in the layering, and validation happens when
    the merged mapping becomes a RunConfig.
    """
    text = Path(path).read_text()
    defaults: dict = {}
    runs: list[dict] = []
    for document in yaml.safe_load_all(text):
        if document is None:
            continue
        if isinstance(document, list):
            runs.extend(document)
        elif isinstance(document, dict) and {"defaults", "runs"} & document.keys():
            defaults.update(document.get("defaults") or {})
            runs.extend(document.get("runs") or [])
        else:
            runs.append(document)
    return defaults, runs


def build(problem_cls, cfg: RunConfig, **kwargs):
    """Instantiate a problem class from a config plus fixed extra keywords.

    The orchestration fields (``noise_levels``, ``times``, ``out``) are consumed by
    the run driver, not by the problem constructor; unset fields never reach
    it either, so the class's own signature stays the source of truth.
    """
    fields = cfg.model_dump(exclude_none=True,
                            exclude={"out", "times", "noise_levels"})
    fields.update(kwargs)
    return problem_cls(**fields)
