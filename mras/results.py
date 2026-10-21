"""Structured results of one noise-level run and their persistence."""

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LevelResult:
    """Error histories and solution snapshots for one (problem, level) run."""

    problem: str
    noise_level: float
    error_u: np.ndarray
    error_q: np.ndarray
    saved_gfu: np.ndarray
    saved_gfq: np.ndarray
    series_gfu: np.ndarray
    series_gfq: np.ndarray
    meta: dict = field(default_factory=dict)
    # Final solver state of the level; lets a multi-level run carry
    # cross-level data (z_old, reference states) into the next level.
    state: object = None

    @property
    def final_error_u(self):
        return float(self.error_u[-1])

    @property
    def final_error_q(self):
        return float(self.error_q[-1])

    def save(self, directory):
        """Write trajectories to one npz and headline numbers to one json."""
        stem = f"{self.problem}_{self.noise_level:g}".replace(".", "p")
        npz = f"{directory}/{stem}.npz"
        np.savez_compressed(
            npz, error_u=self.error_u, error_q=self.error_q,
            saved_gfu=self.saved_gfu, saved_gfq=self.saved_gfq,
            series_gfu=self.series_gfu, series_gfq=self.series_gfq)
        summary = {"problem": self.problem, "noise_level": self.noise_level,
                   "steps": len(self.error_u),
                   "final_error_u": self.final_error_u,
                   "final_error_q": self.final_error_q, **self.meta}
        with open(f"{directory}/{stem}.json", "w") as fh:
            json.dump(summary, fh, indent=1)
        return npz
