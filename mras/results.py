"""Structured results of one noise-level run and their persistence."""

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LevelResult:
    """Error histories and solution snapshots for one (problem, level) run."""

    problem: str
    noise_level: float
    error_u: np.ndarray | None
    # Relative parameter reconstruction error ||qhat - q||_L2 / ||q||_L2.
    error_q: np.ndarray | None
    saved_gfu: np.ndarray
    saved_gfq: np.ndarray
    series_gfu: np.ndarray
    series_gfq: np.ndarray
    meta: dict = field(default_factory=dict)
    # Final solver state of the level, kept for post-run field figures.
    state: object = None
    # ||u - z|| per step: reported for every run, error histories only
    # when a ground truth is available.
    misfit_u: np.ndarray | None = None

    @property
    def final_error_u(self):
        return None if self.error_u is None else float(self.error_u[-1])

    @property
    def final_error_q(self):
        return None if self.error_q is None else float(self.error_q[-1])

    def save(self, directory):
        """Write trajectories to one npz and headline numbers to one json."""
        stem = f"{self.problem}_{self.noise_level:g}".replace(".", "p")
        npz = f"{directory}/{stem}.npz"
        arrays = {"saved_gfu": self.saved_gfu, "saved_gfq": self.saved_gfq,
                  "series_gfu": self.series_gfu, "series_gfq": self.series_gfq}
        if self.error_u is not None:
            arrays["error_u"] = self.error_u
        if self.error_q is not None:
            arrays["error_q"] = self.error_q
        if self.misfit_u is not None:
            arrays["misfit_u"] = self.misfit_u
        np.savez_compressed(npz, **arrays)
        summary = {"problem": self.problem, "noise_level": self.noise_level,
                   "steps": len(self.misfit_u
                                if self.misfit_u is not None else self.error_u),
                   "final_error_u": self.final_error_u,
                   "final_error_q": self.final_error_q,
                   "final_misfit_u": None if self.misfit_u is None
                   else float(self.misfit_u[-1]), **self.meta}
        with open(f"{directory}/{stem}.json", "w") as fh:
            json.dump(summary, fh, indent=1)
        return npz
