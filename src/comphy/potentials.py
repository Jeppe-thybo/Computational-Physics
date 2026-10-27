"""Pair potentials and energy models for atomic clusters."""

import torch
from torch import nn

from .models import MLP


def pair_distances(positions):
    """All unique interatomic distances.

    Parameters
    ----------
    positions : tensor of shape ``(N, d)`` or ``(B, N, d)``

    Returns
    -------
    tensor of shape ``(N(N-1)/2,)`` or ``(B, N(N-1)/2)``
    """
    n = positions.shape[-2]
    i, j = torch.triu_indices(n, n, offset=1)
    return (positions[..., i, :] - positions[..., j, :]).norm(dim=-1)


class PairPotential(nn.Module):
    """Base class: total energy = sum of ``V(r_ij)`` over all atom pairs."""

    def V(self, r):
        raise NotImplementedError

    def forward(self, positions):
        """Energy of one cluster ``(N, d)`` or a batch of clusters ``(B, N, d)``."""
        return self.V(pair_distances(positions)).sum(dim=-1)


class LennardJonesGauss(PairPotential):
    r"""Lennard-Jones-Gauss pair potential.

    .. math::
        V(r) = A\left[\frac{1}{r^{12}} - \frac{2}{r^{6}}
               - \varepsilon\, e^{-(r-r_0)^2 / 2\sigma^2}\right] + E_\infty

    With the defaults ``A = 1`` and ``E_inf = 0`` this is the potential used in
    weeks 8-10. With ``trainable=True`` all five constants become parameters
    that can be fitted to data (exercise 54).
    """

    def __init__(self, eps=1.0, sigma_squared=0.02, r0=1.7, A=1.0, e_inf=0.0,
                 trainable=False):
        super().__init__()
        values = dict(eps=eps, sigma_squared=sigma_squared, r0=r0, A=A, e_inf=e_inf)
        for name, value in values.items():
            tensor = torch.tensor(float(value))
            if trainable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def V(self, r):
        lj = 1 / r**12 - 2 / r**6
        gauss = self.eps * torch.exp(-(r - self.r0) ** 2 / (2 * self.sigma_squared))
        return self.A * (lj - gauss) + self.e_inf

    def extra_repr(self):
        return ", ".join(f"{k}={getattr(self, k).item():.3f}"
                         for k in ("eps", "sigma_squared", "r0", "A", "e_inf"))


class NeuralPairPotential(PairPotential):
    """Pair potential ``V(r)`` represented by a small neural network."""

    def __init__(self, hidden=(8, 8, 8), activation="silu"):
        super().__init__()
        self.net = MLP(1, 1, hidden=hidden, activation=activation)

    def V(self, r):
        return self.net(r.reshape(-1)).reshape(r.shape)
