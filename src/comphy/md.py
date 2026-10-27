"""Molecular dynamics: velocity Verlet and a simple stochastic thermostat (week 5)."""

import numpy as np


def velocity_verlet(x, v, force, n_steps, dt=0.01, mass=1.0, fixed=None):
    """Integrate Newton's equations with the velocity-Verlet scheme.

    Parameters
    ----------
    x, v : float or array
        Initial positions and velocities (any shape, e.g. ``(N, 2)``).
    force : callable
        ``force(x)`` returning an array shaped like ``x``.
    fixed : boolean array, optional
        Atoms (rows) that do not move.

    Returns
    -------
    xs, vs : arrays of shape ``(n_steps + 1, *x.shape)``
    """
    x = np.array(x, dtype=float)
    v = np.array(v, dtype=float)
    mobile = 1.0 if fixed is None else (~np.asarray(fixed))[:, None].astype(float)
    a = force(x) / mass * mobile
    xs, vs = [x.copy()], [v.copy()]
    for _ in range(n_steps):
        x = x + v * dt + 0.5 * a * dt**2
        a_new = force(x) / mass * mobile
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
        xs.append(x.copy())
        vs.append(v.copy())
    return np.array(xs), np.array(vs)


def thermostatted_md(x, force, kT, n_segments, steps_per_segment=50, dt=0.01,
                     mass=1.0, fixed=None, rng=None):
    """NVT sampling by repeatedly redrawing all velocities from the
    Maxwell-Boltzmann distribution and running a short NVE trajectory
    (an Andersen-type thermostat).

    ``kT`` may be a number or a sequence of length ``n_segments`` (a cooling
    schedule for simulated annealing).

    Returns
    -------
    xs, vs : positions and velocities at the end of every segment
    """
    rng = np.random.default_rng(rng)
    kTs = np.broadcast_to(np.asarray(kT, dtype=float), (n_segments,))
    x = np.array(x, dtype=float)
    xs, vs = [], []
    for t in kTs:
        v = rng.normal(0.0, np.sqrt(t / mass), size=x.shape)
        if fixed is not None:
            v[np.asarray(fixed)] = 0.0
        traj_x, traj_v = velocity_verlet(x, v, force, steps_per_segment, dt, mass, fixed)
        x = traj_x[-1]
        xs.append(x)
        vs.append(traj_v[-1])
    return np.array(xs), np.array(vs)


def kinetic_energy(v, mass=1.0):
    """Total kinetic energy of one configuration of velocities."""
    return 0.5 * mass * np.sum(np.asarray(v) ** 2)
