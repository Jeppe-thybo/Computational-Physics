"""Graph utilities and message-passing layers (weeks 12-13).

Requires PyTorch Geometric.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj


# --------------------------------------------------------------------------
# Building graphs
# --------------------------------------------------------------------------
def edges_within_cutoff(positions, cutoff, self_loops=False):
    """Directed edge list connecting all node pairs closer than ``cutoff``.

    Returns a ``(2, E)`` long tensor ordered like a double loop over (i, j).
    """
    pos = torch.as_tensor(positions, dtype=torch.float32)
    close = torch.cdist(pos, pos) < cutoff
    if not self_loops:
        close.fill_diagonal_(False)
    return close.nonzero().T.contiguous()


def random_connected_positions(num_nodes, cutoff=2.5, box_size=10.0, min_frac=0.75):
    """Random node positions that form one connected graph.

    Every new node is placed at least ``min_frac * cutoff`` from all existing
    nodes and within ``cutoff`` of at least one of them (exercise 59).
    """
    positions = [torch.rand(1, 2) * box_size]
    while len(positions) < num_nodes:
        candidate = torch.rand(1, 2) * box_size
        d = torch.cdist(torch.vstack(positions), candidate)
        if torch.all(d >= min_frac * cutoff) and torch.any(d <= cutoff):
            positions.append(candidate)
    return torch.vstack(positions)


def random_graph(num_nodes, cutoff=2.5, box_size=10.0):
    """Random connected graph with node attribute ``x = node index``."""
    pos = random_connected_positions(num_nodes, cutoff, box_size)
    edge_index = edges_within_cutoff(pos, cutoff)
    x = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, pos=pos)


# --------------------------------------------------------------------------
# Comparing graphs
# --------------------------------------------------------------------------
def adjacency_spectrum(graph, decimals=4):
    """Sorted eigenvalues of the adjacency matrix."""
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0]
    return np.sort(np.linalg.eigvalsh(adj.numpy()).round(decimals))


def same_spectrum(graph_a, graph_b):
    """True if two graphs have identical adjacency spectra.

    Isomorphic graphs always have the same spectrum, so this is a cheap way to
    detect duplicates (a few non-isomorphic "cospectral" graphs exist, but not
    for the small graphs used here).
    """
    a, b = adjacency_spectrum(graph_a), adjacency_spectrum(graph_b)
    return a.shape == b.shape and np.allclose(a, b)


def unique_graphs(graphs):
    """Keep one representative of every adjacency spectrum."""
    unique = []
    for g in graphs:
        if not any(same_spectrum(g, u) for u in unique):
            unique.append(g)
    return unique


# --------------------------------------------------------------------------
# Message passing
# --------------------------------------------------------------------------
class NeighbourAggregation(MessagePassing):
    """Parameter-free message passing: every node receives the aggregate
    (``"add"``, ``"mean"``, ``"max"``, ...) of its neighbours' attributes."""

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class PhiPsiConv(MessagePassing):
    r"""Message-passing layer with learnable message and update functions.

    .. math::
        x_i' = \phi\Big(x_i \,\Vert\, \textstyle\sum_{j \in N(i)} \psi(x_j)\, e_{ij}\Big)

    where :math:`\psi` and :math:`\phi` are linear layers and the optional edge
    attribute :math:`e_{ij}` scales each message.
    """

    def __init__(self, d_in, d_out, d_message=None, aggr="add"):
        super().__init__(aggr=aggr)
        d_message = d_in if d_message is None else d_message
        self.psi = torch.nn.Linear(d_in, d_message)
        self.phi = torch.nn.Linear(d_in + d_message, d_out)

    def forward(self, x, edge_index, edge_attr=None):
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.phi(torch.cat([x, messages], dim=1))

    def message(self, x_j, edge_attr):
        m = self.psi(x_j)
        return m if edge_attr is None else m * edge_attr


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def plot_graph(ax, graph, values=None, size=500, cmap="viridis", vmin=None,
               vmax=None, show_index=True, fmt="{:.1f}", fontsize=9,
               node_color="C0", title=None):
    """Draw a 2D graph.

    Parameters
    ----------
    values : array-like, optional
        One number per node, written inside the node and used for its colour.
        Pass ``graph.x`` / ``graph.y`` to show node attributes.
    show_index : bool
        Write the node index in a small box next to each node.
    """
    pos = np.asarray(graph.pos, dtype=float)
    for i, j in graph.edge_index.T.tolist():
        if i != j:
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="black", lw=1, zorder=0)

    if values is not None:
        values = np.asarray(torch.as_tensor(values).detach(), dtype=float).reshape(-1)
        norm = Normalize(vmin=values.min() if vmin is None else vmin,
                         vmax=values.max() if vmax is None else vmax)
        colors = plt.get_cmap(cmap)(norm(values))
    else:
        colors = node_color
    ax.scatter(pos[:, 0], pos[:, 1], s=size, c=colors, edgecolor="black", zorder=1)

    for i, (x, y) in enumerate(pos):
        if values is not None:
            ax.text(x, y, fmt.format(values[i]), color="white", ha="center",
                    va="center", fontsize=fontsize, fontweight="bold", zorder=2)
        elif show_index:
            ax.text(x, y, str(i), color="white", ha="center", va="center",
                    fontsize=fontsize, zorder=2)
        if show_index and values is not None:
            ax.annotate(str(i), (x, y), xytext=(9, 9), textcoords="offset points",
                        fontsize=fontsize - 2, ha="center", va="center", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))

    pad = 0.15 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), 1.0)
    ax.set(xlim=(pos[:, 0].min() - pad, pos[:, 0].max() + pad),
           ylim=(pos[:, 1].min() - pad, pos[:, 1].max() + pad),
           xticks=[], yticks=[], aspect="equal", title=title)
    return ax
