"""Core DP primitives: marginals, Gaussian mechanism, exponential mechanism, projection."""

from __future__ import annotations

import numpy as np


def marginal(codes: np.ndarray, shape: tuple[int, ...], attrs: tuple[int, ...]) -> np.ndarray:
    """Exact contingency table over `attrs` (a tuple of column indices).

    Under add/remove neighbours a single record changes exactly one cell by 1,
    so the L1 and L2 sensitivity of a marginal are both 1.
    """
    sub_shape = tuple(shape[a] for a in attrs)
    flat = np.ravel_multi_index(tuple(codes[:, a] for a in attrs), sub_shape)
    return np.bincount(flat, minlength=int(np.prod(sub_shape))).reshape(sub_shape).astype(float)


def gaussian_mechanism(values: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return values + rng.normal(0.0, sigma, size=values.shape)


def exponential_mechanism(
    scores: np.ndarray, epsilon: float, sensitivity: float, rng: np.random.Generator
) -> int:
    """Pick index i with probability proportional to exp(eps * score_i / (2 * sensitivity)).

    Implemented with the Gumbel-max trick, which is numerically stable for large scores.
    """
    scores = np.asarray(scores, dtype=float)
    logits = epsilon * scores / (2.0 * sensitivity)
    return int(np.argmax(logits + rng.gumbel(size=logits.shape)))


def project_to_simplex(x: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection of x onto {y >= 0, sum(y) = total}.

    Post-processing of noisy counts — costs no privacy. Uses the sort-based
    algorithm of Duchi et al. (2008).
    """
    shape = x.shape
    v = x.ravel().astype(float)
    total = max(float(total), 0.0)
    if total == 0.0:
        return np.zeros(shape)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    ind = np.arange(1, len(u) + 1)
    cond = u - css / ind > 0
    r = ind[cond][-1]
    theta = css[cond][-1] / r
    return np.maximum(v - theta, 0.0).reshape(shape)
