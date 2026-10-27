"""Generic neural-network building blocks."""

import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Fully connected feed-forward network.

    Replaces the many hand-written ``ReLuModel``, ``SiLuModel4``, ... classes
    from the original notebook: ``MLP(hidden=(8, 8), activation="relu")`` is a
    1 -> 8 -> 8 -> 1 network with ReLU activations.

    Parameters
    ----------
    n_in, n_out : int
        Input and output dimension.
    hidden : sequence of int
        Width of each hidden layer (empty for a purely linear model).
    activation : str
        One of :data:`ACTIVATIONS`.
    """

    def __init__(self, n_in=1, n_out=1, hidden=(8, 8), activation="silu"):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.activation = activation
        widths = [n_in, *hidden, n_out]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(ACTIVATIONS[activation]())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.n_in == 1:
            x = x.reshape(-1, 1)
        y = self.layers(x)
        return y.flatten() if self.n_out == 1 else y

    def predict_label(self, x):
        """Index of the largest output (for classification networks)."""
        return torch.argmax(self.forward(x), dim=1)

    def __repr__(self):
        hidden = [m.out_features for m in self.layers if isinstance(m, nn.Linear)][:-1]
        return f"MLP({self.n_in} -> {hidden} -> {self.n_out}, {self.activation})"
