"""Quick sanity checks for the shared helper package (run with ``pytest``)."""

import numpy as np
import pytest
import torch

from comphy.descriptors import radial_descriptor
from comphy.models import MLP
from comphy.pinn import PINN
from comphy.potentials import LennardJonesGauss, NeuralPairPotential, pair_distances
from comphy.training import EarlyStopping, fit


def test_pair_distances_single_and_batch():
    pos = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    d = pair_distances(pos)
    assert torch.allclose(d, torch.tensor([3.0, 4.0, 5.0]))
    batch = torch.stack([pos, 2 * pos])
    assert torch.allclose(pair_distances(batch), torch.stack([d, 2 * d]))


def test_lennard_jones_minimum_without_gaussian():
    calc = LennardJonesGauss(eps=0.0)
    r = torch.linspace(0.9, 1.5, 6001)
    r_min = r[torch.argmin(calc.V(r))]
    assert abs(r_min.item() - 1.0) < 1e-3
    assert calc.V(torch.tensor(1.0)).item() == pytest.approx(-1.0)


def test_trainable_potential_has_parameters():
    assert len(list(LennardJonesGauss(trainable=True).parameters())) == 5
    assert len(list(LennardJonesGauss(trainable=False).parameters())) == 0


def test_cluster_energy_is_sum_over_pairs():
    pos = torch.rand(5, 2) * 3
    calc = LennardJonesGauss()
    assert torch.allclose(calc(pos), calc.V(torch.pdist(pos)).sum())
    assert NeuralPairPotential()(torch.rand(4, 5, 2)).shape == (4,)


def test_mlp_shapes():
    assert MLP(1, 1, hidden=(8, 8))(torch.rand(10)).shape == (10,)
    assert MLP(2, 3, hidden=(4,))(torch.rand(10, 2)).shape == (10, 3)
    assert MLP(2, 3, hidden=())(torch.rand(10, 2)).shape == (10, 3)


def test_fit_recovers_a_line():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 20)
    model = MLP(1, 1, hidden=())
    fit(model, x, 2 * x + 1, steps=2000, lr=0.05)
    w, b = model.layers[0].weight.item(), model.layers[0].bias.item()
    assert w == pytest.approx(2, abs=1e-2) and b == pytest.approx(1, abs=1e-2)


def test_early_stopping():
    stopper = EarlyStopping(patience=2)
    assert not any(stopper(v) for v in [3, 2, 1, 1.5])
    assert stopper(1.2)


def test_radial_descriptor_shapes_and_permutation():
    pos = np.random.default_rng(0).random((6, 2)) * 3
    d = radial_descriptor(pos)
    assert d.shape == (6, 32)
    assert radial_descriptor(np.stack([pos, pos])).shape == (2, 6, 32)
    perm = [3, 1, 0, 5, 4, 2]
    assert torch.allclose(radial_descriptor(pos[perm]), d[perm], atol=1e-5)


def test_pinn_derivatives():
    torch.manual_seed(0)
    model = PINN(depth=2, width=8)
    x = torch.linspace(-1, 1, 5).view(-1, 1).requires_grad_(True)
    y, dy, d2y = model.derivatives(x, order=2)
    h = 1e-2
    with torch.no_grad():
        fd = (model(x + h) - model(x - h)) / (2 * h)
        fd2 = (model(x + h) - 2 * model(x) + model(x - h)) / h**2
    assert torch.allclose(dy, fd, atol=1e-3)
    assert torch.allclose(d2y, fd2, atol=1e-2)


@pytest.mark.parametrize("calc_name", ["lj", "ljg", "periodic"])
def test_numpy_forces_match_finite_differences(calc_name):
    from comphy.atoms import LennardJones, LennardJonesGauss

    calc = {"lj": LennardJones(), "ljg": LennardJonesGauss(),
            "periodic": LennardJones(box=[4.0, 4.0])}[calc_name]
    pos = np.random.default_rng(1).random((6, 2)) * 3
    h = 1e-6
    for i, d in [(0, 0), (3, 1)]:
        shifted = pos.copy()
        shifted[i, d] += h
        fd = -(calc.energy(shifted) - calc.energy(pos)) / h
        assert calc.forces(pos)[i, d] == pytest.approx(fd, rel=1e-3, abs=1e-3)


def test_periodic_energy_is_translation_invariant():
    from comphy.atoms import LennardJones

    calc = LennardJones(box=[5.0, 5.0])
    pos = np.random.default_rng(2).random((5, 2)) * 5
    assert calc.energy(pos) == pytest.approx(calc.energy((pos + [3.1, -1.7]) % 5.0))


def test_metropolis_reproduces_equipartition():
    from comphy.montecarlo import metropolis, thermal_average

    V = lambda x: 0.5 * np.asarray(x) ** 2
    sample, _ = metropolis(V, kT=0.3, x0=0.0, n_steps=50_000, step=0.5, rng=0)
    assert V(sample).mean() == pytest.approx(0.15, rel=0.1)
    assert thermal_average(V, 0.3, (-10, 10)) == pytest.approx(0.15, rel=1e-6)


def test_velocity_verlet_conserves_energy():
    from comphy.md import velocity_verlet

    xs, vs = velocity_verlet(1.0, 0.0, lambda x: -x, n_steps=5000, dt=0.01)
    energy = 0.5 * xs**2 + 0.5 * vs**2
    assert np.ptp(energy) < 1e-4


def test_regression():
    from comphy.regression import KernelRidge, PolynomialRegression, gaussian_kernel

    x = np.linspace(-1, 1, 10)
    y = 1 + 2 * x - 3 * x**2
    assert np.allclose(PolynomialRegression([0, 1, 2]).fit(x, y).beta, [1, 2, -3])
    krr = KernelRidge(gaussian_kernel(0.5), lam=1e-8).fit(x, y)
    assert np.allclose(krr.predict(x), y, atol=1e-4)


def test_graph_helpers():
    pytest.importorskip("torch_geometric")
    from comphy.graphs import edges_within_cutoff, random_graph, same_spectrum

    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    edges = edges_within_cutoff(pos, cutoff=2.0)
    assert edges.tolist() == [[0, 1], [1, 0]]
    assert edges_within_cutoff(pos, 2.0, self_loops=True).shape[1] == 5

    torch.manual_seed(0)
    g = random_graph(5)
    relabelled = g.clone()
    perm = torch.tensor([4, 2, 0, 1, 3])
    relabelled.edge_index = torch.argsort(perm)[g.edge_index]
    assert g.num_nodes == 5
    assert same_spectrum(g, relabelled)
