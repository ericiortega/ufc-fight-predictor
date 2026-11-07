"""Probability calibrators that preserve fight symmetry: f(1 - p) = 1 - f(p).

All are fit on time-ordered validation predictions inside the training window,
mirrored (p, y) and (1 - p, 1 - y), so neither corner is favored.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ufc_model.models.data import mirror

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class Identity:
    name = "none"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Identity:
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


class Platt:
    """Logistic regression on the logit, without intercept (symmetry forces it to 0)."""

    name = "platt"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Platt:
        pm, ym = mirror(p, y)
        self.model = LogisticRegression(fit_intercept=False, C=1e6).fit(_logit(pm)[:, None], ym)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(np.asarray(p))[:, None])[:, 1]


class Isotonic:
    """Isotonic regression on mirrored data, re-symmetrized on output."""

    name = "isotonic"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Isotonic:
        pm, ym = mirror(p, y)
        self.model = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip").fit(
            pm, ym
        )
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (self.model.predict(p) + 1 - self.model.predict(1 - p)) / 2


class Temperature:
    """Single temperature T on the logit: p' = sigmoid(logit(p) / T)."""

    name = "temperature"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Temperature:
        z, y = _logit(np.asarray(p)), np.asarray(y, dtype=float)

        def nll(t: float) -> float:
            q = np.clip(_sigmoid(z / t), _EPS, 1 - _EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.t = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(p)) / self.t)


CALIBRATORS = {c.name: c for c in (Identity, Platt, Isotonic, Temperature)}


def make_calibrator(name: str):
    return CALIBRATORS[name]()
