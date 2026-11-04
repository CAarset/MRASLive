#!/usr/bin/env python
"""Automated verification checklist for every example, past or future.

Runs ``mras.verify.check_example`` for all registered examples at the
micro (fast) configurations and writes one Markdown report to
``results/checklist.md``.  Each row is OK / FAIL / UNVERIFIABLE /
DEVIATION with the measured constant or an honest reason nothing can be
confirmed - see ``mras/verify.py`` for the check semantics and
``docs/DEVIATIONS.md`` for the documented deviations.

``--after SUMMARY`` instead checks the persistent excitation of finished
runs over the horizons they took -- each example's longest ``t_end`` in a
``scripts/run_examples.py`` summary -- adds it to the summary and prints
its table (the check run_examples does itself after each example).

Usage:  python scripts/checklist.py [name ...]
        python scripts/checklist.py --after results/summary.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mras.verify import _registry, as_markdown, check_example  # noqa: E402
from scripts.run_examples import excitation, table  # noqa: E402


def after(summary):
    rows = json.loads(summary.read_text())
    runs = {}
    for r in rows:
        runs.setdefault(r["example"], []).append(r)
    for name, group in runs.items():
        excitation(name, group)
    summary.write_text(json.dumps(rows, indent=1))
    print(table(rows))


def main():
    ap = argparse.ArgumentParser(description="The theory checklist.")
    ap.add_argument("names", nargs="*", metavar="example",
                    help="examples to check (default: all registered)")
    ap.add_argument("--after", type=pathlib.Path, metavar="SUMMARY",
                    help="persistent excitation of the finished runs of a "
                         "run_examples summary, over the horizons they took")
    args = ap.parse_args()
    if args.after is not None:
        after(args.after)
        return
    names = args.names or sorted(_registry())
    out = ["# Verification checklist (all examples)", ""]
    for name in names:
        print(f"[checklist] {name} ...", flush=True)
        out.append(as_markdown(name, check_example(name)))
        out.append("")
    target = pathlib.Path(__file__).resolve().parents[1] / "results"
    target.mkdir(exist_ok=True)
    (target / "checklist.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[checklist] wrote {target / 'checklist.md'}", flush=True)


if __name__ == "__main__":
    main()
