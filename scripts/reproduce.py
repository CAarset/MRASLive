#!/usr/bin/env python
"""Reproduce the Fisher / potential / Allen-Cahn results (Fig 5-11, Tables 6-7).

One level per process so a crash loses at most one level; finished levels
are skipped when the output file already exists, so rerunning the same
command resumes. Keep OMP_NUM_THREADS low on a busy machine.

    cd /home/aarset/Documents/MRAS
    OMP_NUM_THREADS=4 .venv/bin/python scripts/reproduce.py --problem fisher
    # mesh/level overrides:   --h 0.05 --levels 0 0.05
    # wiring check (seconds):  --micro

Outputs per level: <out>/<problem>_<level>.npz (error histories + snapshots)
and <out>/<problem>_<level>.json (final errors); a combined <out>/table.json.
"""

import argparse
import json
import time
from pathlib import Path

from mras.problems.allen_cahn import AllenCahnProblem
from mras.problems.fisher import FisherProblem
from mras.problems.potential import PotentialProblem

# Per-example paper configuration: run length, main-text noise levels and
# figure time instants (converted to step indices).  Additional levels from
# the Tables 6-7 studies are available via --levels.
PROBLEMS = {
    "fisher": {"cls": FisherProblem, "tend": 10.0, "levels": [0.0, 0.03],
               "times": [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]},
    "potential": {"cls": PotentialProblem, "tend": 5.0, "levels": [0.0, 0.05],
                  "times": [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]},
    "allen_cahn": {"cls": AllenCahnProblem, "tend": 5.0,
                   "levels": [0.0, 0.05, 0.10, 0.20],
                   "times": [0.001, 0.05, 0.075, 0.2, 0.5, 1.5, 5.0]},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    ap.add_argument("--h", type=float)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--tend", type=float)
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="noise levels (default: the example's main-text set)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--micro", action="store_true",
                    help="tiny setup to check wiring, not the paper numbers")
    args = ap.parse_args()

    spec = PROBLEMS[args.problem]
    h = 0.2 if args.micro else (args.h or 0.1)
    tend = 0.02 if args.micro else args.tend  # None -> paper run length
    levels = [0.0, 0.05] if args.micro else (args.levels or spec["levels"])
    out = Path(args.out or f"results/{args.problem}")
    out.mkdir(parents=True, exist_ok=True)

    t_end = tend if tend is not None else spec["tend"]
    series_counters = [1] + [round(t / args.dt) for t in spec["times"][1:]]
    series_counters = [c for c in series_counters if c <= round(t_end / args.dt)]
    summary = {"problem": args.problem, "h": h, "dt": args.dt,
               "tend": t_end, "seed": args.seed, "levels": {}}

    for level in levels:
        stem = f"{args.problem}_{level:g}".replace(".", "p")
        if (out / f"{stem}.json").exists():
            print(f"level {level:g}: already done, skipping")
            continue
        print(f"level {level:g}: h={h} dt={args.dt} tend={t_end} ...",
              flush=True)
        t0 = time.time()
        kwargs = {"h": h, "dt": args.dt, "seed": args.seed}
        if tend is not None:
            kwargs["tend"] = tend
        prob = spec["cls"](**kwargs)
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
    print("compare with Tables 6-7 (space/time and noise studies)")


if __name__ == "__main__":
    main()
