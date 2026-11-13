"""PDEBench 2D Darcy flow data access."""

from pathlib import Path

DEFAULT_HDF5 = (Path(__file__).resolve().parents[2]
                / "data" / "2D_DarcyFlow_beta100.0_Train.hdf5")


def load_nu_grid(hdf5=None, sample=0):
    """Return the (N, N) permeability field of one PDEBench sample."""
    import h5py
    path = DEFAULT_HDF5 if hdf5 is None else Path(hdf5)
    with h5py.File(path, "r") as f:
        nu = f["/nu"][sample]
    return nu.squeeze()
