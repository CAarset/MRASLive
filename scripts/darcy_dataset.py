#!/usr/bin/env python
"""Build an NGSolve-side Darcy dataset and flag how it compares to PDEBench.

The paper runs on PDEBench 2D_DarcyFlow_beta100.0_Train.hdf5; inspection of
sample 0 showed a 10:1 contrast, not the 100:1 the filename promises. This
script writes a synthetic set with the same
hdf5 layout (/nu, N x 128 x 128) so every script accepts it via --hdf5,
then optionally runs the same small MRAS setup on PDEBench and on the
synthetic fields to produce a side-by-side table. The comparison is a
validation flag on the paper's dataset choice, not a replacement.

    cd /home/aarset/Documents/MRAS
    .venv/bin/python scripts/darcy_dataset.py --quick   # needs ~5 min, low mem

Outputs: results/darcy_dataset/darcy_ngsolve.h5 and comparison.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mras.problems.darcy import DarcyProblem

N = 128


def synth_field(kind, contrast, radius=0.18, centre=(0.5, 0.5)):
    """Piecewise-constant field: contrast inside the inclusion, 1 outside."""
    y, x = np.mgrid[0:1:N * 1j, 0:1:N * 1j]
    if kind == "disk":
        inside = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 < radius ** 2
    else:
        side = radius * 1.6
        inside = (np.abs(x - centre[0]) < side) & (np.abs(y - centre[1]) < side)
    return np.where(inside, contrast, 1.0).astype(np.float64)


def build(path):
    samples = [("disk", 0.1), ("disk", 0.01), ("square", 0.1), ("square", 0.01)]
    nu = np.stack([synth_field(k, c) for k, c in samples])
    path.parent.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("/nu", data=nu.astype(np.float32))
    print(f"wrote {len(samples)} samples to {path}")
    return samples


def run_case(label, **kwargs):
    prob = DarcyProblem(**kwargs)
    result = prob.run(levels=[0.05], progress=False)[-1]
    return {"case": label, "final_error_u": result.final_error_u,
            "final_error_q": result.final_error_q}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/darcy_dataset")
    ap.add_argument("--quick", action="store_true",
                    help="also run the small side-by-side MRAS comparison")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h5 = out / "darcy_ngsolve.h5"
    samples = build(h5)

    if not args.quick:
        print("use --quick for the MRAS side-by-side comparison")
        return

    micro = dict(h=0.2, dt=1e-3, tend=0.02, seed=0)
    rows = [run_case("pdebench sample 0 (paper dataset)", **micro)]
    for i, (kind, contrast) in enumerate(samples):
        rows.append(run_case(f"synthetic {kind} contrast 1:{int(1 / contrast)}",
                             hdf5=h5, sample=i, **micro))

    flag = ("Paper dataset check: identical MRAS setup on the paper's PDEBench "
            "field vs centred NGSolve-built fields of matching contrast. Large "
            "gaps mean the inclusion shape and placement of the raw sample "
            "drive the reported errors; look at this before trusting table "
            "entries.")
    report = {"note": flag, "rows": rows}
    with open(out / "comparison.json", "w") as fh:
        json.dump(report, fh, indent=1)
    for r in rows:
        print(f"{r['case']:<42} u {r['final_error_u']:.4f}  q {r['final_error_q']:.4f}")
    print(f"comparison written to {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
