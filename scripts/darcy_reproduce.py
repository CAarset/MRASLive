#!/usr/bin/env python
"""Reproduce the paper's Darcy results (Fig 4, Table 6 h-row).

One level per process so a crash loses at most one level; finished levels
are skipped when the output file already exists, so rerunning the same
command resumes. Keep OMP_NUM_THREADS low on a busy machine.

    cd /home/aarset/Documents/MRAS
    OMP_NUM_THREADS=4 .venv/bin/python scripts/darcy_reproduce.py --out results/darcy
    # or a subset:  --levels 0 0.05
    # wiring check (seconds):  --micro

Outputs per level: <out>/darcy_<level>.npz (error histories + snapshots)
and <out>/darcy_<level>.json (final errors); a combined <out>/table.json.

Expected behaviour (Tables 6-7, h=0.04, T=5): the clean level decays toward
machine precision while err_q converges; noisy levels settle at a plateau of
the order of the noise level.
"""

import argparse
import json
import time
from pathlib import Path

from mras.problems.darcy import DarcyProblem

# Paper Fig 4 time instants and Table 6 noise columns, as step indices.
FIG_TIMES = [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]
TABLE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.50]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=float, default=0.04, help="mesh width (Table 6 row: 0.04)")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--tend", type=float, default=5.0)
    ap.add_argument("--sample", type=int, default=0, help="PDEBench nu index")
    ap.add_argument("--seed", type=int, default=0, help="base seed; level-dependent draw")
    ap.add_argument("--levels", type=float, nargs="+", default=TABLE_LEVELS)
    ap.add_argument("--out", default="results/darcy")
    ap.add_argument("--micro", action="store_true",
                    help="tiny setup to check wiring, not the paper numbers")
    args = ap.parse_args()

    h = 0.2 if args.micro else args.h
    tend = 0.02 if args.micro else args.tend
    levels = [0.0, 0.05] if args.micro else args.levels
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    series_counters = [1] + [round(t / args.dt) for t in FIG_TIMES[1:]]
    series_counters = [c for c in series_counters if c <= round(tend / args.dt)]
    summary = {"h": h, "dt": args.dt, "tend": tend, "sample": args.sample,
               "seed": args.seed, "levels": {}}

    for level in levels:
        stem = f"darcy_{level:g}".replace(".", "p")
        if (out / f"{stem}.json").exists():
            print(f"level {level:g}: already done, skipping")
            continue
        print(f"level {level:g}: h={h} dt={args.dt} tend={tend} ...", flush=True)
        t0 = time.time()
        prob = DarcyProblem(h=h, dt=args.dt, tend=tend, sample=args.sample,
                            seed=args.seed)
        prob.series_counters = tuple(series_counters)
        result = prob.run(levels=[level], progress=True)[-1]
        result.save(out)
        print(f"level {level:g}: err_u={result.final_error_u:.6f} "
              f"err_q={result.final_error_q:.6f} in {time.time() - t0:.0f}s",
              flush=True)
        summary["levels"][f"{level:g}"] = {
            "final_error_u": result.final_error_u,
            "final_error_q": result.final_error_q}
        with open(out / "table.json", "w") as fh:
            json.dump(summary, fh, indent=1)

    print(f"\nsummary in {out / 'table.json'}")
    print("compare with Table 6 (h=0.04 noise study)")


if __name__ == "__main__":
    main()
