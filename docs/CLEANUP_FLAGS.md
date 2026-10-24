# Cleanup flags — what comes out before the push

Everything a user touches is on one path: `mras/` (package),
`examples/blank_problem.py` (the template to copy), the four paper
examples, the Arrhenius pair, the Burgers pipe, the theory checklist
(`mras/verify.py`, `scripts/checklist.py`), and the tests.  Files flagged
below are **exploration and reproduction harness** — they exist to answer
questions about the numerics or to regenerate the published figures, and
they stay off the user path.  Each carries a machine-greppable marker:

```bash
grep -rn "CLEANUP-FLAG" --include="*.py" .   # everything flagged
```

| file | why it exists | why it is flagged |
|---|---|---|
| `scripts/reproduce.py` | multi-hour Fisher/potential/Allen–Cahn campaigns (Figs 5–11, Tables 6–7) | campaign harness for paper figures, not for the interface |
| `scripts/darcy_reproduce.py` | same, Darcy/PDEBench (Fig 4, Table 6) | same |
| `scripts/asymptotics_arrhenius.py` | long-horizon decay study of the Arrhenius error | one-off analysis; its answer lives in the paper text now |
| `scripts/darcy_dataset.py` | synthetic Darcy fields of matched contrast vs PDEBench | data-provenance exploration; conclusion recorded in `docs/DEVIATIONS.md` |
| `examples/helmholtz_in_parameter.py` | Helmholtz "identifies nothing" placeholder | superseded by the scalar-q rework of `docs/ASSUMPTIONS.md` section 6 |

Not flagged (user path or the paper-figure toolchain that consumes stored
`results/` artefacts): `mras/` incl. `stepper.py` and `data/`, the
`examples/` problem files (`darcy.py` … `helmholtz` excepted) plus
`examples/blank_problem.py` and `examples/darcy_decaying.py`,
`scripts/checklist.py`, `scripts/demo_arrhenius.py`,
`scripts/make_figures.py`, `mras_live.ipynb`, `configs/`, `tests/`.

## Removal sweep (when the flags are pulled)

Deleting the flagged files is not self-contained; the push-cleanup commit
must also

1. drop the `"helmholtz"` row of `EXPECTED` in `tests/test_verify.py` and
   the Helmholtz case in `tests/test_adjoint.py`; the catalog and the
   checklist registry pick the example up from its file, so deleting
   `examples/helmholtz_in_parameter.py` removes it from both;
2. drop the "Reproduce the paper's figures" table from `README.md` (it
   routes through the two reproduce scripts) — or keep the scripts until
   the paper revision ships, which is the current intent;
3. rerun `.venv/bin/python -m pytest tests/ -q`.
