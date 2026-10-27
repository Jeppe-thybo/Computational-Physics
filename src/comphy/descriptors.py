"""Descriptors of atomic clusters.

* Global (whole-cluster) descriptors used for unsupervised learning in week 4:
  :class:`DistanceMoments`, :class:`ExtremeNeighborCount`,
  :class:`PairDistances`, :class:`CoordinationNumbers`,
  :class:`ConnectivityGraphSpectrum`, :class:`CoulombMatrixSpectrum`.
  Each is callable on an ``(N, 2)`` array and has a ``draw`` method.
* The per-atom :func:`radial_descriptor` used for classification in week 11.
"""

import math

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class Descriptor:
    """Base class: ``descriptor(pos)`` returns a feature vector."""

    color = "C0"
    ylim = None
    xticklabels = None

    def __call__(self, pos):
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    def draw(self, ax, pos):
        vector = self(pos)
        x = np.arange(len(vector)) if self.xticklabels is None else range(len(self.xticklabels))
        ax.bar(x, vector, width=0.8, color=self.color)
        if self.xticklabels is not None:
            ax.set_xticks(list(x), self.xticklabels)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)
        ax.set_title(self.name)
        return ax


class DistanceMoments(Descriptor):
    """Mean and standard deviation of all interatomic distances."""
    color, ylim, xticklabels = "C4", (0, 2.3), [r"$\mu$", r"$\sigma$"]

    def __call__(self, pos):
        d = pdist(pos)
        return np.array([d.mean(), d.std()])


class ExtremeNeighborCount(Descriptor):
    """Smallest and largest number of neighbours (within ``cutoff``) of any atom."""
    color, ylim, xticklabels = "C5", (0, 7), [r"$N_\mathrm{lowest}$", r"$N_\mathrm{highest}$"]

    def __init__(self, cutoff=1.5):
        self.cutoff = cutoff

    def __call__(self, pos):
        n = (squareform(pdist(pos)) < self.cutoff).sum(axis=1) - 1
        return np.array([n.min(), n.max()])


class PairDistances(Descriptor):
    """Histogram of all interatomic distances."""
    color, ylim = "C1", (0, 20.1)

    def __init__(self, bin_width=0.5, r_max=7.0):
        self.edges = np.arange(0, r_max + 1e-9, bin_width)

    def __call__(self, pos):
        return np.histogram(pdist(pos), self.edges)[0]

    def draw(self, ax, pos):
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        ax.bar(centres, self(pos), width=0.8 * np.diff(self.edges), color=self.color)
        ax.set(ylim=self.ylim, title=self.name, xlabel="r")
        return ax


class CoordinationNumbers(Descriptor):
    """Histogram of how many neighbours (within ``cutoff``) each atom has."""
    color, ylim = "C2", (0, 6.2)

    def __init__(self, cutoff=1.2, max_neighbours=8):
        self.cutoff, self.max_neighbours = cutoff, max_neighbours

    def __call__(self, pos):
        n = (squareform(pdist(pos)) < self.cutoff).sum(axis=1) - 1
        return np.bincount(n, minlength=self.max_neighbours + 1)[: self.max_neighbours + 1]


class ConnectivityGraphSpectrum(Descriptor):
    """Sorted eigenvalues of minus the connectivity (adjacency) matrix."""
    color, ylim = "C3", (-4.1, 2.3)

    def __init__(self, cutoff=1.2):
        self.cutoff = cutoff

    def __call__(self, pos):
        A = (squareform(pdist(pos)) < self.cutoff).astype(float)
        np.fill_diagonal(A, 0)
        return np.sort(np.linalg.eigvalsh(-A))


class CoulombMatrixSpectrum(Descriptor):
    """Sorted eigenvalues of the matrix with entries ``1/r_ij`` (and 1 on the diagonal)."""
    color, ylim = "C4", (-2, 8)

    def __call__(self, pos):
        r = squareform(pdist(pos))
        np.fill_diagonal(r, 1.0)
        return np.sort(np.linalg.eigvalsh(1 / r))


ALL_DESCRIPTORS = [DistanceMoments, ExtremeNeighborCount, PairDistances, CoordinationNumbers,
                   ConnectivityGraphSpectrum, CoulombMatrixSpectrum]


def radial_descriptor(positions, r_cut=7.0, sigma=0.3, n_features=32):
    r"""Gaussian-smeared radial distribution around every atom.

    For atom *i* and grid point :math:`r_k \in [0, r_{cut}]`

    .. math::
        G_i(r_k) = \cos\!\left(\frac{\pi r_k}{2 r_{cut}}\right)
                   \sum_{j:\, r_{ij} < r_{cut}} e^{-(r_{ij} - r_k)^2 / 2\sigma^2}

    (vectorised version of ``descriptor_optimized`` from exercise 58).

    Parameters
    ----------
    positions : array or tensor of shape ``(N, d)`` or ``(B, N, d)``

    Returns
    -------
    tensor of shape ``(N, n_features)`` or ``(B, N, n_features)``
    """
    pos = torch.as_tensor(positions, dtype=torch.float32)
    distances = torch.cdist(pos, pos)                       # (..., N, N)
    grid = torch.linspace(0, r_cut, n_features)             # (F,)
    inside = (distances < r_cut).float().unsqueeze(-1)      # (..., N, N, 1)
    gauss = torch.exp(-(distances.unsqueeze(-1) - grid) ** 2 / (2 * sigma**2))
    weight = torch.cos(math.pi * grid / (2 * r_cut))
    return (gauss * inside).sum(dim=-2) * weight


def descriptor_grid(r_cut=7.0, n_features=32):
    """The radial grid ``r_k`` used by :func:`radial_descriptor`."""
    return torch.linspace(0, r_cut, n_features)
