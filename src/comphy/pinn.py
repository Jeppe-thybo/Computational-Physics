"""Physics-informed neural networks for 1D differential equations (week 14).

A :class:`PINN` is a small network ``y(x)``. :func:`train_pinn` minimises

* the squared residual of the differential equation at random collocation
  points in ``x_range``,
* the squared error at prescribed values ``y(x_k) = y_k``,
* the squared error at prescribed slopes ``y'(x_k) = s_k``,
* and optionally any extra loss term (e.g. a normalisation condition).
"""

import torch
from torch import nn
from tqdm.auto import tqdm


class PINN(nn.Module):
    """Fully connected network ``R -> R`` with SiLU activations.

    Parameters
    ----------
    depth : int
        Number of hidden layers.
    width : int
        Neurons per hidden layer.
    """

    def __init__(self, depth=2, width=8):
        super().__init__()
        layers = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        self.depth, self.width = depth, width
        self.epochs_trained = 0

    def forward(self, x):
        return self.layers(x)

    def derivatives(self, x, order=1):
        """Return ``[y, y', ..., y^(order)]`` evaluated at ``x`` (shape ``(n, 1)``).

        ``x`` must have ``requires_grad=True``.
        """
        out = [self.forward(x)]
        for _ in range(order):
            out.append(torch.autograd.grad(out[-1], x, torch.ones_like(out[-1]),
                                           create_graph=True)[0])
        return out

    def describe(self):
        return f"depth = {self.depth}, width = {self.width}"


def _as_column(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1)


def train_pinn(model, residual, x_range, epochs, order=1, values=(), slopes=(),
               extra_loss=None, lr=1e-3, n_collocation=30, optimizer=None,
               progress=False):
    """Train a :class:`PINN` on a differential equation.

    Parameters
    ----------
    residual : callable
        ``residual(x, y, y', ...)`` returning a tensor that is zero when the
        equation is satisfied. Receives ``order + 1`` derivatives.
    x_range : (float, float)
        Interval the collocation points are drawn from.
    values, slopes : sequence of (x, value) pairs
        Boundary / initial conditions on ``y`` and ``y'``.
    extra_loss : callable, optional
        ``extra_loss(model)`` added to the loss every step.
    progress : bool
        Show a progress bar.

    Returns
    -------
    list of float
        Loss per epoch.
    """
    mse = nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    x_min, x_max = x_range

    if values:
        x_val = _as_column([p[0] for p in values]).requires_grad_(True)
        y_val = _as_column([p[1] for p in values])
    if slopes:
        x_slope = _as_column([p[0] for p in slopes]).requires_grad_(True)
        s_val = _as_column([p[1] for p in slopes])

    losses = []
    for _ in tqdm(range(epochs), disable=not progress, leave=False):
        optimizer.zero_grad()
        x = (torch.rand(n_collocation, 1) * (x_max - x_min) + x_min).requires_grad_(True)
        r = residual(x, *model.derivatives(x, order))
        loss = mse(r, torch.zeros_like(r))
        if values:
            loss = loss + mse(model(x_val), y_val)
        if slopes:
            loss = loss + mse(model.derivatives(x_slope, 1)[1], s_val)
        if extra_loss is not None:
            loss = loss + extra_loss(model)
        loss.backward()
        optimizer.step()
        model.epochs_trained += 1
        losses.append(loss.item())
    return losses
