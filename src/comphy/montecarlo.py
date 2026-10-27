"""Sampling the Boltzmann distribution of a 1D potential (weeks 2-3)."""

import numpy as np
from scipy import integrate


def boltzmann_weight(V, x, kT):
    """Unnormalised Boltzmann factor ``exp(-V(x)/kT)``."""
    return np.exp(-V(x) / kT)


def boltzmann_pdf(V, kT, lims):
    """Normalised Boltzmann distribution on the interval ``lims`` as a function."""
    Z = integrate.quad(lambda x: boltzmann_weight(V, x, kT), *lims, limit=200)[0]
    return lambda x: boltzmann_weight(V, np.asarray(x, dtype=float), kT) / Z


def thermal_average(V, kT, lims, observable=None):
    r"""Exact thermal average :math:`\langle A \rangle = \int A e^{-V/kT} / \int e^{-V/kT}`.

    ``observable`` defaults to the potential energy itself.
    """
    A = V if observable is None else observable
    num = integrate.quad(lambda x: A(x) * boltzmann_weight(V, x, kT), *lims, limit=200)[0]
    den = integrate.quad(lambda x: boltzmann_weight(V, x, kT), *lims, limit=200)[0]
    return num / den


def rejection_sample(V, kT, lims, n_trials, rng=None):
    """Direct sampling by rejection: draw x uniformly in ``lims`` and keep it
    with probability ``exp(-(V(x) - V_min)/kT)``.

    Returns only the accepted points (fewer than ``n_trials``).
    """
    rng = np.random.default_rng(rng)
    grid = np.linspace(*lims, 2000)
    v_min = np.min(V(grid))
    x = rng.uniform(*lims, n_trials)
    keep = rng.random(n_trials) < np.exp(-(V(x) - v_min) / kT)
    return x[keep]


def metropolis(V, kT, x0, n_steps, step=None, lims=None, rng=None):
    """Metropolis Monte Carlo chain.

    Parameters
    ----------
    step : float, optional
        Random-walk proposals ``x' = x + step * N(0, 1)``.
    lims : (float, float), optional
        If ``step`` is None, proposals are drawn uniformly in ``lims``
        (independent of the current position).

    Returns
    -------
    samples : array of length ``n_steps``
    acceptance : float
        Fraction of accepted proposals.
    """
    if step is None and lims is None:
        raise ValueError("give either a random-walk step or proposal limits")
    rng = np.random.default_rng(rng)
    x, v = x0, V(x0)
    samples = np.empty(n_steps)
    accepted = 0
    for i in range(n_steps):
        samples[i] = x
        x_new = x + step * rng.normal() if step is not None else rng.uniform(*lims)
        v_new = V(x_new)
        if v_new <= v or rng.random() < np.exp(-(v_new - v) / kT):
            x, v = x_new, v_new
            accepted += 1
    return samples, accepted / n_steps
