"""Seeded Gaussian FEM noise for the data feed."""

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from ngsolve import H1, BilinearForm, GridFunction, dx


def seeded_rng(seed, *tags):
    """Independent random stream fixed by the key ``(seed, *tags)``.

    Anything drawn through this generator reproduces bit-for-bit whenever the
    same key is used, independent of any other draws made before or after.
    """
    return np.random.default_rng([int(seed), *(int(t) for t in tags)])


class GaussianFEMNoise:
    r"""Gaussian FEM white noise with exact relative L2 amplitude.

    A draw produces the datum ``z = gf + e`` where the perturbation ``e`` is the
    discrete Gaussian field in ``U_h`` reproducing the defining covariance of
    Gaussian white noise on the trial space,

    .. math::

        \mathbb{E}\,[(e, \varphi)(e, \psi)] = (\varphi, \psi)_{L^2}
        \qquad \text{for all } \varphi, \psi \in U_h,

    i.e. ``e = M^{-1/2} \eta`` with ``eta ~ N(0, I)`` on the free degrees of
    freedom and ``M`` the mass matrix.  ``M^{-1/2}`` is applied by polynomial
    filtering of the diagonally scaled mass operator
    ``S = D^{-1/2} M D^{-1/2}``, ``D = diag(M)``,

    .. math::

        e = D^{-1/2}\, p(S)\, \eta, \qquad p(\lambda) \approx \lambda^{-1/2},

    with the Chebyshev series ``p`` evaluated through a Clenshaw three-term
    recurrence; one draw then costs a few dozen O(nnz) mass-matrix mat-vecs and
    no factorization (Beckmann 2006; Ainsworth, Li & Liang 2017).  Diagonal
    scaling is what makes the iteration cheap: the raw mass matrix in the
    hierarchical P3 basis has a condition number of order 1e5, which drops to
    ~230 for ``S``, so ``p`` meets a relative spectral error of ``tol`` within
    a few dozen terms.

    Afterwards the draw is renormalized with the exact mass norm, so that
    ``||e||_L2 == noise_level * ||gf||_L2`` holds identically: the polynomial filter
    only perturbs the spectral shape of the noise, never its amplitude.

    The noise lives entirely in ``U_h`` and vanishes on Dirichlet-constrained
    degrees of freedom, so ``z`` inherits the boundary values of ``gf``.

    All randomness is keyed by ``(seed, noise_level, counter)``: equal keys give
    bit-identical draws regardless of call order or intervening draws.
    ``noise_level == 0`` returns an exact copy of ``gf`` without consuming
    randomness.

    Parameters
    ----------
    fes : FESpace
        Scalar space ``U_h`` the noise lives in (typically the H1 space of the
        state variable).
    seed : int
        Non-negative root of all draw streams.
    tol : float
        Target relative error of the filter p(lambda)*sqrt(lambda) -> 1 over the
        spectrum of S.
    """

    def __init__(self, fes, seed=0, tol=5e-4):
        self.fes = fes
        self.seed = int(seed)
        self.idx = np.flatnonzero(np.fromiter(fes.FreeDofs(), dtype=bool))

        u, v = fes.TnT()
        self._bf = BilinearForm(fes)
        self._bf += u * v * dx
        self._bf.Assemble()
        self._mm = self._bf.mat

        rows, cols, vals = self._mm.COO()
        rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
        diag = np.zeros(fes.ndof)
        off = rows == cols
        diag[rows[off]] = vals[off]
        if np.any(diag[self.idx] <= 0.0):
            raise ValueError("mass matrix must have positive diagonal")
        self._dis2 = np.zeros(fes.ndof)
        self._dis2[self.idx] = 1.0 / np.sqrt(diag[self.idx])

        lo, hi = self._lanczos_bounds()
        if not lo > 0.0:
            raise RuntimeError("scaled mass operator is not positive definite")
        self._fit(0.7 * lo, 1.2 * hi, tol)

    # -- operator construction ------------------------------------------------

    def _scaled_matvec(self, x):
        """v -> D^{-1/2} M D^{-1/2} v, restricted to the free dofs."""
        w = np.array(self._mm.matvec(self._dis2 * x))
        return self._dis2 * w

    def _lanczos_bounds(self, k=150):
        """Extreme Ritz values of S by Lanczos with full reorthogonalization.

        For problems with at most ``k`` free dofs the reduction is complete and
        the returned bounds are the exact spectrum of S.
        """
        k = int(min(self.idx.size, k))
        rng = seeded_rng(self.seed, 77_819)
        q = np.zeros(self.fes.ndof)
        q[self.idx] = rng.standard_normal(self.idx.size)
        q /= np.linalg.norm(q[self.idx])
        Q = np.zeros((self.fes.ndof, k))
        alphas, betas, q_prev = [], [], np.zeros_like(q)
        beta = 0.0
        for j in range(k):
            Q[:, j] = q
            w = self._scaled_matvec(q)
            alpha = float(q @ w)
            w -= alpha * q + beta * q_prev
            for _ in range(2):  # full reorthogonalization for stable endpoints
                w -= Q[:, : j + 1] @ (Q[:, : j + 1].T @ w)
            beta = float(np.linalg.norm(w[self.idx]))
            alphas.append(alpha)
            betas.append(beta)
            if beta < 1e-12 * max(alphas):
                break
            q_prev, q = q, w / beta
        n = len(alphas)
        T = np.diag(alphas)
        if n > 1:
            T += np.diag(betas[: n - 1], 1) + np.diag(betas[: n - 1], -1)
        theta = np.linalg.eigvalsh(T)
        return float(theta[0]), float(theta[-1])

    def _fit(self, a, b, tol):
        """Chebyshev coefficients of lambda -> lambda^{-1/2} on [a, b]."""
        deg = 8
        while True:
            p = _cheb.Chebyshev.interpolate(lambda z: z ** -0.5, deg, domain=[a, b])
            xs = np.linspace(a, b, 4001)
            err = float(np.abs(1.0 - p(xs) * np.sqrt(xs)).max())
            if err <= tol or deg >= 512:
                break
            deg *= 2
        if not err <= tol:
            raise RuntimeError(f"could not fit lambda^(-1/2) to tol {tol} "
                               f"(best relative error {err:.3e} at degree {deg})")
        self.coef = np.ascontiguousarray(p.coef, dtype=np.float64)
        self.delta = 0.5 * (a + b)
        self.gamma = 0.5 * (b - a)

    # -- sampling ---------------------------------------------------------------

    # Draws are keyed by (seed, noise level, counter): nothing to seed or
    # carry across a restart.
    def start(self):
        pass

    def rng_state(self):
        return None

    def set_rng_state(self, state):
        pass

    def _mass_norm(self, v):
        w = np.array(self._mm.matvec(v))
        return float(np.sqrt(max(v[self.idx] @ w[self.idx], 0.0)))

    def draw(self, gf, noise_level, counter):
        """Return a fresh GridFunction ``z = gf + e`` whose noise ``e`` has
        relative L2 norm ``noise_level``."""
        if noise_level < 0:
            raise ValueError(f"noise level must be non-negative, got {noise_level}")
        z = GridFunction(self.fes)
        gfv = np.array(gf.vec.FV().NumPy())
        if noise_level == 0:
            z.vec.FV().NumPy()[:] = gfv
            return z

        rng = seeded_rng(self.seed, int(round(noise_level * 1e6)), int(counter))
        eta = np.zeros(self.fes.ndof)
        eta[self.idx] = rng.standard_normal(self.idx.size)

        # Clenshaw recurrence for p(S) eta with Chebyshev argument (S - delta I)/gamma.
        b1 = np.zeros_like(eta)
        b2 = np.zeros_like(eta)
        for c in self.coef[:0:-1]:
            t = self._scaled_matvec(b1)
            t -= self.delta * b1
            t /= self.gamma
            b1, b2 = 2.0 * t - b2 + c * eta, b1
        t = self._scaled_matvec(b1)
        t -= self.delta * b1
        t /= self.gamma
        e = self._dis2 * (t - b2 + self.coef[0] * eta)

        norm_e = self._mass_norm(e)
        if norm_e > 0.0:
            e *= noise_level * self._mass_norm(gfv) / norm_e
        z.vec.FV().NumPy()[:] = gfv + e
        return z


NOISE_LENGTH = 0.05  # Matern correlation length: this fraction of the diagonal
# The Robin coefficient of the Matern field's boundary condition du/dn + beta u
# = 0, as a fraction of kappa = sqrt(8 nu) / length: 1/1.42 keeps the variance
# close to one up to the boundary for nu = 1 in 2D (Roininen, Huttunen and
# Lasanen 2014), ngsrng's default until it made beta a required argument.
ROBIN_FACTOR = 1 / 1.42


class MaternFEMNoise:
    r"""Gaussian noise with a Matern covariance in space, of exact relative L2
    amplitude: the datum ``z = gf + e``, ``||e||_L2 = noise_level ||gf||_L2``,
    as :class:`GaussianFEMNoise` and the paper scale it (its Section 4).

    The field is ngsrng's Matern Gaussian on U_h (Lindgren-Rue SPDE, ``nu``
    the smoothness, ``length`` the correlation length, by default
    ``NOISE_LENGTH`` of the domain's diagonal; Robin boundary treatment, so
    the variance is about constant up to the boundary), drawn by ngsrng
    directly, zero on the Dirichlet-constrained dofs; a vector space takes
    one independent field per component, all in one draw.  The draws follow
    torch's global random state, as ngsrng's do: :meth:`start` seeds it
    with ``seed`` (a run calls it before its first datum), and
    :meth:`rng_state` / :meth:`set_rng_state` carry it across a restart.

    The MRAS theory assumes nothing of the noise but its size (a bound in
    L^p(0, infinity; Z), Kaltenbacher-Nguyen 2021 Section 2.2): smoother
    noise only makes the smoothing error smaller.
    """

    def __init__(self, fes, seed=0, length=None, nu=1.0):
        import ngsrng
        import torch
        self._torch = torch
        self.fes = fes
        self.seed = int(seed)
        mesh = fes.mesh
        points = np.array([v.point for v in mesh.vertices])
        diagonal = float(np.linalg.norm(points.max(axis=0)
                                        - points.min(axis=0)))
        self.length = float(length) if length else NOISE_LENGTH * diagonal
        self.nu = float(nu)
        self.dim = GridFunction(fes).dim
        scalar = fes if self.dim == 1 else H1(
            mesh, order=fes.globalorder, dirichlet=self._dirichlet(fes))
        self._n = scalar.ndof
        rng = ngsrng.default_rng(
            scalar, dtype=torch.float64,
            correlation=ngsrng.matern_correlation(
                self.length, nu=self.nu, boundary="robin",
                beta=ROBIN_FACTOR * np.sqrt(8 * self.nu) / self.length))
        self._gaussian = rng.gaussian()
        u, v = fes.TnT()
        mass = BilinearForm(fes)
        mass += u * v * dx
        mass.Assemble()
        self._mm = mass.mat

    @staticmethod
    def _dirichlet(fes):
        """The boundaries whose dofs ``fes`` constrains, as a regex."""
        mesh = fes.mesh
        free = np.fromiter(fes.FreeDofs(), dtype=bool)
        names = []
        for name in sorted(set(mesh.GetBoundaries())):
            dofs = np.array(fes.GetDofs(mesh.Boundaries(name)), dtype=bool)
            if dofs.any() and not (dofs & free).any():
                names.append(name)
        return "|".join(names) if names else None

    def start(self):
        """Seed the draws of a run (torch's global generator)."""
        self._torch.manual_seed(self.seed)

    def rng_state(self):
        return self._torch.get_rng_state().numpy().copy()

    def set_rng_state(self, state):
        self._torch.set_rng_state(self._torch.from_numpy(
            np.asarray(state, dtype=np.uint8)))

    def _mass_norm(self, v):
        w = np.array(self._mm.matvec(v))
        return float(np.sqrt(max(v @ w, 0.0)))

    def draw(self, gf, noise_level, counter=None):
        """Return a fresh GridFunction ``z = gf + e`` whose noise ``e`` has
        relative L2 norm ``noise_level`` (the next draw of the stream;
        ``counter`` is not used)."""
        if noise_level < 0:
            raise ValueError(f"noise level must be non-negative, got {noise_level}")
        z = GridFunction(self.fes)
        gfv = np.array(gf.vec.FV().NumPy())
        if noise_level == 0:
            z.vec.FV().NumPy()[:] = gfv
            return z
        e = np.empty(self.fes.ndof)
        # one sample per component, written in place: component k holds
        # dofs k n ... (k + 1) n - 1 of a vector space
        self._gaussian.sample_into(e.reshape(self.dim, self._n))
        norm_e = self._mass_norm(e)
        if norm_e > 0.0:
            e *= noise_level * self._mass_norm(gfv) / norm_e
        z.vec.FV().NumPy()[:] = gfv + e
        return z
