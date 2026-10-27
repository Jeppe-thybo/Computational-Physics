"""NumPy pair potentials, atomic clusters and local optimisers (weeks 3-7).

The PyTorch versions in :mod:`comphy.potentials` are used from week 8 on, when
gradients come from autograd. Here forces are written out analytically.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist


class PairPotential:
    """Energy and forces from a pair potential ``V(r)`` and its derivative.

    Parameters
    ----------
    box : array of shape (2,), optional
        Side lengths of a periodic cell. Distances then use the minimum-image
        convention.
    """

    def __init__(self, box=None):
        self.box = None if box is None else np.asarray(box, dtype=float)

    def V(self, r):
        raise NotImplementedError

    def dV_dr(self, r):
        raise NotImplementedError

    def _differences(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]          # r_i - r_j
        if self.box is not None:
            diff -= self.box * np.round(diff / self.box)
        return diff

    def energy(self, pos):
        pos = np.asarray(pos, dtype=float)
        if self.box is None:
            return np.sum(self.V(pdist(pos)))
        r = np.linalg.norm(self._differences(pos), axis=-1)
        return np.sum(self.V(r[np.triu_indices(len(pos), k=1)]))

    def forces(self, pos):
        pos = np.asarray(pos, dtype=float)
        diff = self._differences(pos)
        r = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(r, 1.0)                         # avoid 0/0 on the diagonal
        coeff = self.dV_dr(r) / r
        np.fill_diagonal(coeff, 0.0)
        return -np.sum(coeff[..., None] * diff, axis=1)


class LennardJones(PairPotential):
    r"""Lennard-Jones potential :math:`V(r) = 4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]`.

    The defaults put the minimum at ``r = 1`` with depth ``-eps0 = -5``.
    """

    def __init__(self, eps0=5.0, sigma=2 ** (-1 / 6), box=None):
        super().__init__(box)
        self.eps0, self.sigma = eps0, sigma

    def V(self, r):
        s6 = (self.sigma / r) ** 6
        return 4 * self.eps0 * (s6**2 - s6)

    def dV_dr(self, r):
        s6 = (self.sigma / r) ** 6
        return -4 * self.eps0 * (12 * s6**2 - 6 * s6) / r


class LennardJonesGauss(PairPotential):
    r"""Lennard-Jones-Gauss potential
    :math:`V(r) = r^{-12} - 2r^{-6} - \varepsilon e^{-(r-r_0)^2/2\sigma^2}`."""

    def __init__(self, eps=1.0, r0=1.7, sigma_squared=0.02, box=None):
        super().__init__(box)
        self.eps, self.r0, self.sigma_squared = eps, r0, sigma_squared

    def V(self, r):
        return r**-12 - 2 * r**-6 - self.eps * np.exp(-(r - self.r0) ** 2 / (2 * self.sigma_squared))

    def dV_dr(self, r):
        gauss = self.eps * np.exp(-(r - self.r0) ** 2 / (2 * self.sigma_squared))
        return -12 * r**-13 + 12 * r**-7 + gauss * (r - self.r0) / self.sigma_squared


class Cluster:
    """A set of atoms, some of which may be held fixed ("static")."""

    def __init__(self, calc, positions, static=None):
        self.calc = calc
        self.positions = np.array(positions, dtype=float)
        n = len(self.positions)
        self.static = np.zeros(n, bool) if static is None else np.asarray(static, bool)

    def __len__(self):
        return len(self.positions)

    @property
    def dynamic(self):
        return np.flatnonzero(~self.static)

    @property
    def energy(self):
        return self.calc.energy(self.positions)

    def forces(self):
        f = self.calc.forces(self.positions)
        f[self.static] = 0.0
        return f

    def copy(self):
        return Cluster(self.calc, self.positions.copy(), self.static.copy())

    def draw(self, ax, size=400, color="C0", static_color="C1", labels=False,
             title=True, alpha=1.0):
        colors = [static_color if s else color for s in self.static]
        ax.scatter(*self.positions.T, s=size, c=colors, edgecolor="black", alpha=alpha, zorder=2)
        if labels:
            for i, (x, y) in enumerate(self.positions):
                ax.text(x, y, str(i), color="white", ha="center", va="center", fontsize=8, zorder=3)
        if title:
            ax.set_title(f"E = {self.energy:.3f}")
        ax.set_aspect("equal")
        return ax


# --------------------------------------------------------------------------
# Local optimisation
# --------------------------------------------------------------------------
def mc_quench(cluster, n_sweeps, step, kT=0.0, rng=None):
    """Metropolis Monte Carlo on the dynamic atoms, one atom at a time.

    With ``kT = 0`` only downhill moves are accepted (a zero-temperature
    quench). Returns the list of accepted configurations (the path).
    """
    rng = np.random.default_rng(rng)
    pos = cluster.positions.copy()
    energy = cluster.calc.energy(pos)
    path = [pos.copy()]
    for _ in range(n_sweeps):
        for i in cluster.dynamic:
            trial = pos.copy()
            trial[i] += step * rng.normal(size=2)
            e_trial = cluster.calc.energy(trial)
            dE = e_trial - energy
            if dE <= 0 or (kT > 0 and rng.random() < np.exp(-dE / kT)):
                pos, energy = trial, e_trial
                path.append(pos.copy())
    cluster.positions = pos
    return path


def line_search_relax(cluster, max_steps=1000, fmax=0.05, per_atom=False, max_step=0.5):
    """Steepest descent with a line search along the force.

    Each step moves along the (normalised) force direction to the minimum of
    the energy on that line, looking at most ``max_step`` ahead (so that an
    atom cannot jump through the repulsive core of another). With
    ``per_atom=True`` the dynamic atoms are moved one at a time. Returns the
    path of configurations.
    """
    path = [cluster.positions.copy()]
    for _ in range(max_steps):
        forces = cluster.forces()
        if np.max(np.linalg.norm(forces, axis=1)) < fmax:
            break
        groups = [[i] for i in cluster.dynamic] if per_atom else [cluster.dynamic]
        for atoms in groups:
            f = cluster.forces()[atoms]
            norm = np.linalg.norm(f)
            if norm < 1e-12:
                continue
            direction = np.zeros_like(cluster.positions)
            direction[atoms] = f / norm
            start = cluster.positions.copy()
            res = minimize_scalar(lambda a: cluster.calc.energy(start + a * direction),
                                  bounds=(0, max_step), method="bounded")
            cluster.positions = start + res.x * direction
        path.append(cluster.positions.copy())
    return path


def random_positions(n, box_half_width=4.0, min_distance=0.5, rng=None):
    """``n`` random 2D points in a square box, no two closer than ``min_distance``."""
    rng = np.random.default_rng(rng)
    pos = []
    while len(pos) < n:
        p = rng.uniform(-box_half_width, box_half_width, 2)
        if all(np.linalg.norm(p - q) > min_distance for q in pos):
            pos.append(p)
    return np.array(pos)


def energy_landscape(cluster, atom, xs, ys):
    """Energy as a function of the position of one atom (others fixed)."""
    pos = cluster.positions.copy()
    Z = np.empty((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            pos[atom] = (x, y)
            Z[i, j] = cluster.calc.energy(pos)
    return Z
