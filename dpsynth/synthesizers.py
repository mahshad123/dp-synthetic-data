"""Differentially private synthetic data generators.

* ``IndependentSynthesizer`` — baseline. Measures every 1-way marginal and samples
  columns independently. Preserves per-column distributions, destroys correlations.
* ``MSTSynthesizer`` — a from-scratch, dependency-free take on the MST algorithm
  (McKenna, Miklau & Sheldon, 2021; winner of the NIST 2018 DP synthetic data
  challenge). It privately learns a maximum-spanning tree of pairwise dependencies,
  measures the selected 2-way marginals, and samples from the resulting tree-shaped
  graphical model.

  Differences from the paper: post-processing uses simplex projection plus
  tree-structured sampling instead of Private-PGM inference. This keeps the code
  short and readable at a small cost in utility.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd

from .accounting import (
    PrivacyLedger,
    eps_delta_to_rho,
    exponential_mech_eps,
    gaussian_sigma,
    rho_to_eps_delta,
)
from .domain import Domain
from .mechanisms import exponential_mechanism, gaussian_mechanism, marginal, project_to_simplex


class _BaseSynthesizer:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-6, seed: int | None = None):
        self.epsilon = epsilon
        self.delta = delta
        self.rho = eps_delta_to_rho(epsilon, delta)
        self.rng = np.random.default_rng(seed)
        self.ledger = PrivacyLedger(self.rho)
        self.domain: Domain | None = None
        self._fitted = False

    # ---- shared helpers -------------------------------------------------
    def _measure_one_way(self, codes: np.ndarray, rho_total: float) -> list[np.ndarray]:
        d = len(self.domain)
        rho_each = rho_total / d
        sigma = gaussian_sigma(rho_each)
        noisy = []
        for i in range(d):
            m = marginal(codes, self.domain.shape, (i,))
            self.ledger.spend(rho_each, f"1-way marginal [{self.domain.names[i]}]")
            noisy.append(gaussian_mechanism(m, sigma, self.rng))
        # Every noisy 1-way sum is an unbiased estimate of N; average them (post-processing).
        self.n_hat = max(float(np.mean([m.sum() for m in noisy])), 1.0)
        return [project_to_simplex(m, self.n_hat) for m in noisy]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit() before sample()")

    def privacy_report(self) -> str:
        return self.ledger.summary(self.delta)

    @property
    def epsilon_spent(self) -> float:
        return rho_to_eps_delta(self.ledger.spent, self.delta)


class IndependentSynthesizer(_BaseSynthesizer):
    """Baseline: all budget on 1-way marginals, columns sampled independently."""

    def fit(self, df: pd.DataFrame, domain: Domain) -> "IndependentSynthesizer":
        self.domain = domain
        codes = domain.encode(df)
        self.one_way = self._measure_one_way(codes, self.rho)
        self._fitted = True
        return self

    def sample(self, n: int | None = None) -> pd.DataFrame:
        self._check_fitted()
        n = int(round(self.n_hat)) if n is None else n
        cols = []
        for m in self.one_way:
            p = m / m.sum()
            cols.append(self.rng.choice(len(p), size=n, p=p))
        return self.domain.decode(np.column_stack(cols), self.rng)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


class MSTSynthesizer(_BaseSynthesizer):
    """Private maximum-spanning-tree synthesizer.

    Budget split (zCDP, equal thirds as in the original paper):
      1. measure all 1-way marginals            (Gaussian mechanism)
      2. select d-1 tree edges                   (exponential mechanism)
      3. measure the selected 2-way marginals    (Gaussian mechanism)
    """

    def fit(self, df: pd.DataFrame, domain: Domain) -> "MSTSynthesizer":
        self.domain = domain
        codes = domain.encode(df)
        d = len(domain)

        if d == 1:
            self.one_way = self._measure_one_way(codes, self.rho)
            self.tree_edges, self.two_way = [], {}
            self._fitted = True
            return self

        rho_third = self.rho / 3.0
        self.one_way = self._measure_one_way(codes, rho_third)
        self.tree_edges = self._select_tree(codes, rho_third)
        self.two_way = self._measure_two_way(codes, self.tree_edges, rho_third)
        self._fitted = True
        return self

    # ---- step 2: private structure learning ----------------------------
    def _select_tree(self, codes: np.ndarray, rho_total: float) -> list[tuple[int, int]]:
        d = len(self.domain)
        shape = self.domain.shape
        rho_each = rho_total / (d - 1)
        eps_each = exponential_mech_eps(rho_each)

        # Score = L1 distance between the true pairwise marginal and what the
        # (already private) independent estimate predicts. A large score means the
        # pair is strongly dependent and worth spending budget on.
        # Sensitivity is 1: adding/removing a record moves one true cell by 1 and the
        # estimate is fixed post-processed output.
        candidates, scores = [], []
        for a, b in combinations(range(d), 2):
            true = marginal(codes, shape, (a, b))
            est = np.outer(self.one_way[a], self.one_way[b]) / self.n_hat
            candidates.append((a, b))
            scores.append(np.abs(true - est).sum())
        scores = np.array(scores)

        uf = _UnionFind(d)
        edges: list[tuple[int, int]] = []
        while len(edges) < d - 1:
            valid = [i for i, (a, b) in enumerate(candidates) if uf.find(a) != uf.find(b)]
            pick = valid[exponential_mechanism(scores[valid], eps_each, 1.0, self.rng)]
            a, b = candidates[pick]
            self.ledger.spend(
                rho_each, f"select edge #{len(edges) + 1}"
            )
            uf.union(a, b)
            edges.append((a, b))
        return edges

    # ---- step 3: measure chosen pairs -----------------------------------
    def _measure_two_way(self, codes, edges, rho_total) -> dict[tuple[int, int], np.ndarray]:
        rho_each = rho_total / len(edges)
        sigma = gaussian_sigma(rho_each)
        out = {}
        for a, b in edges:
            m = marginal(codes, self.domain.shape, (a, b))
            names = self.domain.names
            self.ledger.spend(rho_each, f"2-way marginal [{names[a]} x {names[b]}]")
            out[(a, b)] = project_to_simplex(gaussian_mechanism(m, sigma, self.rng), self.n_hat)
        return out

    # ---- sampling (pure post-processing) --------------------------------
    def _pair_table(self, parent: int, child: int) -> np.ndarray:
        if (parent, child) in self.two_way:
            return self.two_way[(parent, child)]
        return self.two_way[(child, parent)].T

    def sample(self, n: int | None = None) -> pd.DataFrame:
        self._check_fitted()
        n = int(round(self.n_hat)) if n is None else n
        d = len(self.domain)
        out = np.zeros((n, d), dtype=int)

        adj = defaultdict(list)
        for a, b in self.tree_edges:
            adj[a].append(b)
            adj[b].append(a)

        # Root at the highest-degree node; walk the tree breadth-first.
        root = max(range(d), key=lambda i: len(adj[i]))
        p_root = self.one_way[root] / self.one_way[root].sum()
        out[:, root] = self.rng.choice(len(p_root), size=n, p=p_root)

        seen, queue = {root}, deque([root])
        while queue:
            parent = queue.popleft()
            for child in adj[parent]:
                if child in seen:
                    continue
                table = self._pair_table(parent, child)  # shape (|parent|, |child|)
                fallback = self.one_way[child] / self.one_way[child].sum()
                for v in range(table.shape[0]):
                    rows = np.where(out[:, parent] == v)[0]
                    if len(rows) == 0:
                        continue
                    row = table[v]
                    p = row / row.sum() if row.sum() > 0 else fallback
                    out[rows, child] = self.rng.choice(len(p), size=len(rows), p=p)
                seen.add(child)
                queue.append(child)
        return self.domain.decode(out, self.rng)

    def describe_tree(self) -> list[str]:
        names = self.domain.names
        return [f"{names[a]} -- {names[b]}" for a, b in self.tree_edges]
