"""Schema / domain handling.

Differential privacy requires that the *domain* (bin edges, category lists) is
public knowledge. Deriving bin edges from the private data (e.g. quantiles)
leaks information, so this module only accepts bounds supplied in a schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Column:
    name: str
    kind: str  # "categorical" or "numeric"
    categories: list | None = None  # categorical
    lower: float | None = None  # numeric
    upper: float | None = None  # numeric
    bins: int = 10  # numeric

    @property
    def size(self) -> int:
        return len(self.categories) if self.kind == "categorical" else self.bins

    def encode(self, values: pd.Series) -> np.ndarray:
        if self.kind == "categorical":
            lookup = {c: i for i, c in enumerate(self.categories)}
            # Unknown categories are mapped to the last category ("other" by convention);
            # this is a data-independent rule so it does not affect privacy.
            return values.map(lambda v: lookup.get(v, len(self.categories) - 1)).to_numpy(int)
        clipped = np.clip(values.to_numpy(float), self.lower, self.upper)
        width = (self.upper - self.lower) / self.bins
        idx = np.floor((clipped - self.lower) / width).astype(int)
        return np.clip(idx, 0, self.bins - 1)

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "categorical":
            return np.asarray(self.categories, dtype=object)[codes]
        width = (self.upper - self.lower) / self.bins
        # Sample uniformly inside the bin so numeric columns look continuous.
        return self.lower + (codes + rng.random(len(codes))) * width


class Domain:
    def __init__(self, columns: list[Column]):
        if not columns:
            raise ValueError("schema must contain at least one column")
        self.columns = columns

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(c.size for c in self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        missing = set(self.names) - set(df.columns)
        if missing:
            raise ValueError(f"data is missing columns: {sorted(missing)}")
        return np.column_stack([c.encode(df[c.name]) for c in self.columns])

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        return pd.DataFrame(
            {c.name: c.decode(codes[:, i], rng) for i, c in enumerate(self.columns)}
        )

    @classmethod
    def from_dict(cls, spec: dict) -> "Domain":
        cols = []
        for name, s in spec.items():
            kind = s["type"]
            if kind == "categorical":
                cols.append(Column(name, kind, categories=list(s["categories"])))
            elif kind == "numeric":
                cols.append(
                    Column(name, kind, lower=float(s["lower"]), upper=float(s["upper"]),
                           bins=int(s.get("bins", 10)))
                )
            else:
                raise ValueError(f"unknown column type {kind!r} for {name}")
        return cls(cols)

    @classmethod
    def from_json(cls, path: str | Path) -> "Domain":
        return cls.from_dict(json.loads(Path(path).read_text()))
