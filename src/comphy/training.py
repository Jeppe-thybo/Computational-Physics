"""Training loops.

The original notebook repeated the same ~40-line training loop for almost every
exercise. These two functions cover all of those cases:

* :func:`fit` - full-batch gradient descent on a fixed set of points.
* :func:`train` - mini-batch training from ``DataLoader`` objects, with an
  optional validation set and early stopping. Works for ordinary
  ``TensorDataset`` loaders *and* for PyTorch Geometric graph loaders.
"""

import numpy as np
import torch


class EarlyStopping:
    """Stop when the validation loss has not improved for ``patience`` epochs."""

    def __init__(self, patience=100):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def fit(model, x, y, steps=1000, lr=0.01, loss_fn=None, params=None,
        weight_decay=0.0):
    """Full-batch training with Adam.

    Parameters
    ----------
    model : callable
        ``model(x)`` must return a tensor with the same shape as ``y``.
    params : iterable, optional
        Parameters to optimise (default: all of ``model.parameters()``).
        Used e.g. to fit only the amplitude of a Gaussian.

    Returns
    -------
    list of float
        Loss after every step.
    """
    loss_fn = loss_fn or torch.nn.MSELoss()
    params = list(model.parameters()) if params is None else list(params)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _forward(model, batch):
    """Return ``(prediction, target, batch_size)`` for a tensor or graph batch."""
    if isinstance(batch, (list, tuple)):
        x, y = batch
        return model(x), y, len(y)
    # PyTorch Geometric batch: the model takes the whole batch, target is .y
    return model(batch), batch.y, batch.num_graphs


def evaluate(model, loader, loss_fn=None):
    """Mean loss over a loader (weighted by batch size)."""
    loss_fn = loss_fn or torch.nn.MSELoss()
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pred, target, size = _forward(model, batch)
            total += loss_fn(pred, target).item() * size
            n += size
    return total / n


def predict(model, loader):
    """Collect ``(targets, predictions)`` over a loader as numpy arrays."""
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            pred, target, _ = _forward(model, batch)
            targets.append(target.reshape(-1).numpy())
            preds.append(pred.reshape(-1).numpy())
    return np.concatenate(targets), np.concatenate(preds)


def train(model, train_loader, val_loader=None, epochs=1000, lr=0.01,
          loss_fn=None, optimizer=None, patience=None, print_every=None,
          weight_decay=0.0, restore_best=False):
    """Mini-batch training with optional validation and early stopping.

    Parameters
    ----------
    patience : int, optional
        Enable early stopping on the validation loss (or on the training
        loss if no ``val_loader`` is given).
    restore_best : bool
        Reload the weights from the epoch with the lowest monitored loss.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "stopped_epoch": int | None}``
    """
    loss_fn = loss_fn or torch.nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr,
                                              weight_decay=weight_decay)
    stopper = EarlyStopping(patience) if patience else None
    history = {"train": [], "val": [], "stopped_epoch": None}
    best_state, best_loss = None, float("inf")

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred, target, size = _forward(model, batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total += loss.item() * size
            n += size
        history["train"].append(total / n)

        if val_loader is not None:
            history["val"].append(evaluate(model, val_loader, loss_fn))
        monitored = history["val"][-1] if val_loader is not None else history["train"][-1]

        if restore_best and monitored < best_loss:
            best_loss = monitored
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if print_every and epoch % print_every == 0:
            msg = f"epoch {epoch:5d}  train loss {history['train'][-1]:.4g}"
            if val_loader is not None:
                msg += f"  val loss {history['val'][-1]:.4g}"
            print(msg)

        if stopper is not None and stopper(monitored):
            history["stopped_epoch"] = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return history
