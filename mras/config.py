"""Validated run configurations in YAML, managed with pydantic.

A configuration file is either

* a single mapping  -- one run,
* a sequence of mappings -- a sweep, or
* a mapping with ``defaults`` and ``runs`` keys -- a sweep with shared
  settings; each entry of ``runs`` is merged over ``defaults``.

    defaults: {tend: 5.0, noise_levels: [0.0, 0.05]}
    runs:
      - {h: 0.2, seed: 1}
      - {h: 0.1, seed: 2, noise_levels: [0.2]}

Unknown keys and out-of-range values are rejected at load time.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    """One validated run specification (mesh, time stepping, noise levels)."""

    model_config = ConfigDict(extra="forbid")

    h: float | None = Field(None, gt=0, description="design mesh width")
    dt: float = Field(1e-3, gt=0)
    tend: float = Field(5.0, gt=0)
    k_u: int = Field(3, ge=1, le=8, description="polynomial order of U_h")
    seed: int = 0
    noise_levels: list[float] = Field(default_factory=lambda: [0.0])
    times: list[float] = Field(default_factory=list,
                               description="snapshot times for figures")
    out: str | None = None

    @field_validator("noise_levels")
    @classmethod
    def _noise_levels_nonnegative(cls, v):
        if not v or any(x < 0 for x in v):
            raise ValueError("noise_levels must be non-empty and non-negative")
        return v

    def counters(self):
        """Step indices (``State.counter`` convention: state of ``t`` at
        index counter - 1) closest to the requested times."""
        return sorted({max(1, round(t / self.dt) + 1) for t in self.times})


def load_yaml(path) -> list[RunConfig]:
    """Parse a YAML configuration file into one validated RunConfig per run."""
    text = Path(path).read_text()
    configs = []
    for document in yaml.safe_load_all(text):
        if document is None:
            continue
        if isinstance(document, list):
            configs.extend(RunConfig(**entry) for entry in document)
        elif isinstance(document, dict) and "runs" in document:
            defaults = document.get("defaults") or {}
            configs.extend(RunConfig(**{**defaults, **entry})
                           for entry in document["runs"])
        else:
            configs.append(RunConfig(**document))
    return configs


def build(problem_cls, cfg: RunConfig, **kwargs):
    """Instantiate a problem class from a config plus fixed extra keywords.

    The orchestration fields (``noise_levels``, ``times``, ``out``) are consumed by
    the run driver, not by the problem constructor.
    """
    fields = cfg.model_dump(exclude_none=True,
                            exclude={"out", "times", "noise_levels"})
    fields.update(kwargs)
    return problem_cls(**fields)
