"""Small plotting helpers used across the notebooks."""

import matplotlib.pyplot as plt
import numpy as np


def setup_axes(ax, xlim=None, ylim=None, xlabel="x", ylabel="y", title=None,
               zero_lines=True, grid=True, equal=False):
    """Apply the axis style used throughout the exercises.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xlim, ylim : tuple, optional
        Axis limits.
    xlabel, ylabel, title : str, optional
    zero_lines : bool
        Draw the x- and y-axis (``y = 0`` and ``x = 0``) as thin black lines.
    grid : bool
    equal : bool
        Use an equal aspect ratio.
    """
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if zero_lines:
        ax.axhline(0, color="black", lw=1, zorder=0)
        ax.axvline(0, color="black", lw=1, zorder=0)
    if grid:
        ax.grid(True, alpha=0.4)
    if equal:
        ax.set_aspect("equal")
    return ax


def annotate(ax, text, loc=(0.05, 0.95), fontsize=9, **kwargs):
    """Put a small text box in axes coordinates (default: upper left)."""
    style = dict(transform=ax.transAxes, fontsize=fontsize, va="top", ha="left",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    style.update(kwargs)
    ax.text(*loc, text, **style)


def parity_plot(ax, y_true, y_pred, label=None, lims=None, **scatter_kwargs):
    """Scatter predicted against true values with the ``y = x`` line."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ax.scatter(y_true, y_pred, label=label, **scatter_kwargs)
    if lims is None:
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
    ax.plot(lims, lims, color="black", lw=1, zorder=0)
    ax.set(xlim=lims, ylim=lims, xlabel="True energy", ylabel="Predicted energy")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    return ax


def plot_loss_history(ax, history, test_loss=None, title=None, logy=False):
    """Plot training / validation loss curves from :func:`comphy.training.train`.

    ``history`` is a dict with ``"train"`` and optionally ``"val"`` lists.
    If ``test_loss`` is given it is drawn as a short red bar at the end.
    """
    ax.plot(history["train"], label="Training loss", color="C0")
    if history.get("val"):
        ax.plot(history["val"], label="Validation loss", color="C1")
    n = len(history["train"])
    if test_loss is not None:
        ax.plot([0.8 * n, n], [test_loss, test_loss], color="red", lw=4,
                label=f"Test loss = {test_loss:.3g}")
    if history.get("stopped_epoch") is not None:
        ax.axvline(history["stopped_epoch"], color="gray", ls="--", lw=1,
                   label=f"Early stop (epoch {history['stopped_epoch']})")
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.4)
    return ax


def iteration_colors(n, cmap="RdYlGn"):
    """``n`` colours sampled from a colormap (used for iterated maps)."""
    return plt.get_cmap(cmap)(np.linspace(0, 1, n))
