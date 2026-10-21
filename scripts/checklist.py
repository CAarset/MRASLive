#!/usr/bin/env python
"""Automated verification checklist for every example, past or future.

Runs ``mras.verify.check_example`` for all registered examples at the
micro (fast) configurations and writes one Markdown report to
``results/checklist.md``.  Each row is OK / FAIL / UNVERIFIABLE /
DEVIATION with the measured constant or an honest reason nothing can be
confirmed - see ``mras/verify.py`` for the check semantics and
``docs/DEVIATIONS.md`` for the documented deviations.

Usage:  python scripts/checklist.py [name ...]
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mras.verify import _registry, as_markdown, check_example  # noqa: E402


def main():
    names = sys.argv[1:] or sorted(_registry())
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
