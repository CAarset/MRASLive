"""Asymptotics of the Arrhenius reconstruction on a very long horizon.

Answers one question with data: does the relative q-error keep decaying
(the cascade theory says exponentially, rate ~ r^2/lambda_1(C), toward
the P0 representation floor ~0.85%), or does it plateau?  Runs one
member of the pair at shipped design parameters to `--tend` and prints
the trajectory plus a log-linear decay fit on the final window.

usage: python scripts/asymptotics_arrhenius.py [linear|exp]


CLEANUP-FLAG (pre-push): exploration/reproduction harness, off
the user path; delete before push (docs/CLEANUP_FLAGS.md).
"""
import sys
import time

import numpy as np

from mras.problems.arrhenius import ArrheniusLinearProblem, ArrheniusProblem

MEMBERS = {"linear": ArrheniusLinearProblem, "exp": ArrheniusProblem}


def main(member="exp", tend=100.0):
    problem = MEMBERS[member](tend=tend)
    t0 = time.perf_counter()
    res = problem.run_level(0, progress=False)
    err = np.asarray(res.error_q)
    t = np.arange(len(err)) * problem.dt

    print(f"[asympt] {member}: {len(err)} samples in "
          f"{time.perf_counter() - t0:.0f}s  "
          f"(dt={problem.dt}, tend={tend})", flush=True)
    for k in range(0, len(err) - 1, int(2 / problem.dt) or 1):
        print(f"[asympt] t={t[k]:6.1f}  err={err[k]:.5f}", flush=True)
    print(f"[asympt] t={t[-1]:6.1f}  err={err[-1]:.5f}", flush=True)

    # log-linear decay fit on the final half: asymptotic rate per tu
    tail = t >= 0.5 * tend
    slope = np.polyfit(t[tail], np.log(err[tail]), 1)[0]
    last = err[int(0.97 * len(err)):]
    print(f"[asympt] {member}: tail decay rate {1 - np.exp(slope):.3%}/tu, "
          f"final-window min {last.min():.4f} / max {last.max():.4f}",
          flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["exp"]))
