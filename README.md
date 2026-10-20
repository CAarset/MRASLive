# Adaptive Data Assimilation – Code

Code (Python modules and Jupyter notebooks, built on [NGSolve](https://ngsolve.org/)) for adaptive finite-element data assimilation experiments on several PDE problems.

This repository mirrors the dataset published on **GRO.data (Göttingen Research Online)**:

> DOI: [10.25625/HWM9EJ](https://doi.org/10.25625/HWM9EJ)

The GRO record is the citable, archived version. Please cite it via the DOI above.

## Contents

| Problem | Notebook | Supporting modules |
|---|---|---|
| Darcy flow | `adaptive_darcy_joint_gitlab.ipynb` | `domain_darcy.py`, `functions_darcy.py`, `implicit_darcy.py`, `implicit_update_darcy.py`, `download_darcy.py` |
| Nonlinear in state (Fisher) | `adaptive_nonlinear_in_state_joint.ipynb` | `domain_fisher.py`, `functions_fisher.py`, `implicit_fisher.py`, `implicit_update_fisher.py` |
| Modified Allen–Cahn | `adaptive_modified_allen_cahn_gro.ipynb` | `domain_ac.py`, `functions_ac.py` |
| Nonlinear c problem | `adaptive_nonlinear_c_problem_joint_new.ipynb` | `domain_nc.py`, `functions_nc.py` |

Notebook file names differ from the GRO record only in that spaces were replaced by underscores.

## Requirements

Python 3 with: `ngsolve` (incl. `netgen`), `numpy`, `scipy`, `matplotlib`, `ipywidgets`, `jupyter`.
The Darcy and Fisher notebooks additionally use `pandas`, `h5py`, `tqdm`, `torchvision` and `scikit-sparse` (`sksparse`).

## Notes

- The Darcy and Fisher notebooks load reference data from [PDEBench](https://github.com/pdebench/PDEBench). `download_darcy.py` expects PDEBench's `pdebench_data_urls.csv` in the working directory; it is not part of this repository.
- The Fisher notebook imports a `download_fisher` module that is not included in the GRO record either.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) – see `LICENSE`.
