"""Utility and privacy-sanity metrics for synthetic data."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .domain import Domain
from .mechanisms import marginal


def marginal_tvd(real: pd.DataFrame, synth: pd.DataFrame, domain: Domain, k: int = 2) -> float:
    """Average total-variation distance over all k-way marginals (0 = identical, 1 = disjoint)."""
    rc, sc = domain.encode(real), domain.encode(synth)
    errs = []
    for attrs in combinations(range(len(domain)), k):
        p = marginal(rc, domain.shape, attrs)
        q = marginal(sc, domain.shape, attrs)
        errs.append(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())
    return float(np.mean(errs))


def _features(df: pd.DataFrame, domain: Domain, target: str) -> np.ndarray:
    # Use discretized codes so train/test always share the same feature space.
    codes = domain.encode(df)
    keep = [i for i, n in enumerate(domain.names) if n != target]
    return codes[:, keep]


def ml_efficacy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    domain: Domain,
    target: str,
    positive=None,
    seed: int = 0,
) -> dict[str, float]:
    """Train on `train` (real or synthetic), evaluate on real held-out `test`.

    Compare the score from synthetic training data against real training data
    ("train-synthetic-test-real", TSTR) to measure downstream usefulness.
    """
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()
    if positive is not None:
        y_train, y_test = (y_train == positive).astype(int), (y_test == positive).astype(int)
    if len(np.unique(y_train)) < 2:
        return {"accuracy": float("nan"), "auc": float("nan")}
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(_features(train, domain, target), y_train)
    x_test = _features(test, domain, target)
    out = {"accuracy": float(accuracy_score(y_test, clf.predict(x_test)))}
    if len(np.unique(y_test)) == 2:
        out["auc"] = float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
    return out


def exact_match_rate(real: pd.DataFrame, synth: pd.DataFrame, domain: Domain) -> float:
    """Fraction of synthetic rows whose discretized record appears verbatim in the real data.

    This is a *sanity check*, not a privacy guarantee — the guarantee comes from the
    DP accounting. Compare against the same rate for an independent real holdout:
    common records (e.g. a popular category combination) will legitimately repeat.
    """
    real_keys = {tuple(r) for r in domain.encode(real)}
    synth_codes = domain.encode(synth)
    return float(np.mean([tuple(r) in real_keys for r in synth_codes]))


def evaluate(real_train, real_test, synth, domain, target, positive=None) -> dict[str, float]:
    tstr = ml_efficacy(synth, real_test, domain, target, positive)
    return {
        "tvd_1way": marginal_tvd(real_train, synth, domain, k=1),
        "tvd_2way": marginal_tvd(real_train, synth, domain, k=2),
        "tstr_accuracy": tstr["accuracy"],
        "tstr_auc": tstr.get("auc", float("nan")),
        "exact_match_rate": exact_match_rate(real_train, synth, domain),
    }
