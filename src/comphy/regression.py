"""Linear, ridge and kernel ridge regression written with NumPy (week 6)."""

import numpy as np


def polynomial_features(x, powers):
    """Design matrix with columns ``x**p`` for every ``p`` in ``powers``.

    Negative powers give expansions in ``1/x`` (useful for pair potentials).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.stack([x**p for p in powers], axis=1)


def fit_linear(X, y, lam=0.0):
    r"""Least-squares coefficients :math:`\beta = (X^TX + \lambda I)^{-1}X^Ty`.

    ``lam = 0`` is ordinary least squares (OLS), ``lam > 0`` ridge regression.
    """
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ np.asarray(y, dtype=float))


class PolynomialRegression:
    """OLS / ridge regression on the basis ``{x^p : p in powers}``."""

    def __init__(self, powers, lam=0.0):
        self.powers, self.lam = list(powers), lam

    def fit(self, x, y):
        self.beta = fit_linear(polynomial_features(x, self.powers), y, self.lam)
        return self

    def predict(self, x):
        return polynomial_features(x, self.powers) @ self.beta

    def label(self):
        terms = " ".join("1" if p == 0 else f"$x^{{{p}}}$" for p in self.powers)
        kind = "OLS" if self.lam == 0 else f"ridge, λ={self.lam:g}"
        return f"[{terms}] ({kind})"


# --------------------------------------------------------------------------
# Kernels: functions K(X1, X2) of two 2D arrays (n1, d), (n2, d) -> (n1, n2)
# --------------------------------------------------------------------------
def _2d(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def polynomial_kernel(order):
    """:math:`K(x, x') = (1 + x\\cdot x')^{order}`."""
    def K(X1, X2):
        return (1 + _2d(X1) @ _2d(X2).T) ** order
    K.label = f"polynomial kernel, order {order}"
    return K


def gaussian_kernel(sigma):
    """:math:`K(x, x') = \\exp(-|x - x'|^2 / 2\\sigma^2)`."""
    def K(X1, X2):
        X1, X2 = _2d(X1), _2d(X2)
        d2 = (X1**2).sum(1)[:, None] + (X2**2).sum(1)[None, :] - 2 * X1 @ X2.T
        return np.exp(-np.maximum(d2, 0) / (2 * sigma**2))
    K.label = f"Gaussian kernel, σ = {sigma:g}"
    return K


class KernelRidge:
    r"""Kernel ridge regression: :math:`f(x) = \sum_i \alpha_i K(x, x_i)` with
    :math:`\alpha = (K + \lambda I)^{-1} y`."""

    def __init__(self, kernel, lam=1e-3):
        self.kernel, self.lam = kernel, lam

    def fit(self, X, y):
        self.X = _2d(X)
        K = self.kernel(self.X, self.X)
        self.alpha = np.linalg.solve(K + self.lam * np.eye(len(K)), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.kernel(_2d(X), self.X) @ self.alpha

    def label(self):
        return f"{self.kernel.label}, λ = {self.lam:g}"
