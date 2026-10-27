"""Creating and drawing 2D atomic clusters."""

import numpy as np


def random_positions(n_atoms, box_width=8.0, rng=None):
    """Uniformly random positions in a square box centred at the origin."""
    rng = np.random.default_rng(rng)
    return box_width * rng.random((n_atoms, 2)) - box_width / 2


def plot_cluster(ax, positions, chain=False, size=300, color="C0", labels=True,
                 alpha=1.0, edgecolor="black"):
    """Draw atoms as numbered circles.

    Parameters
    ----------
    positions : array of shape ``(N, 2)``
    chain : bool
        Connect consecutive atoms with lines (spring / bead systems).
    color : str or list
        One colour for all atoms, or one per atom.
    """
    positions = np.asarray(positions, dtype=float)
    if chain:
        ax.plot(positions[:, 0], positions[:, 1], color="C0", lw=2, zorder=1)
    ax.scatter(positions[:, 0], positions[:, 1], s=size, c=color, alpha=alpha,
               edgecolor=edgecolor, zorder=2)
    if labels:
        for i, (x, y) in enumerate(positions):
            ax.text(x, y, str(i), color="white", fontsize=8, ha="center",
                    va="center", zorder=3)
    ax.set_aspect("equal")
    return ax


def draw_gallery(ax, clusters, colors="C0", ncols=7, spacing=9.0, labels=None, size=40):
    """Draw many clusters side by side in one axes, on a grid.

    Parameters
    ----------
    clusters : sequence of ``(N, 2)`` arrays
    colors : one colour, or one colour per cluster
    labels : optional text written under each cluster
    """
    if isinstance(colors, str):
        colors = [colors] * len(clusters)
    for k, (pos, c) in enumerate(zip(clusters, colors)):
        pos = np.asarray(pos) - np.mean(pos, axis=0)
        offset = spacing * np.array([k % ncols, -(k // ncols)])
        ax.scatter(*(pos + offset).T, s=size, color=c, edgecolor="black", lw=0.5)
        if labels is not None:
            ax.text(offset[0], offset[1] - 0.45 * spacing, labels[k], ha="center",
                    va="center", fontsize=8)
    ax.set(aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax
