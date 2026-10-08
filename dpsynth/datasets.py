"""Demo datasets.

``make_census_like`` generates an offline, fully synthetic census-style table with
known dependencies (education -> occupation -> income, age -> hours, ...). It lets
the tests and benchmarks run without downloading anything, and because the ground
truth structure is known you can check that the private structure learner finds it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EDUCATION = ["HS", "Some-college", "Bachelors", "Masters", "Doctorate"]
OCCUPATION = ["Service", "Sales", "Admin", "Craft", "Tech", "Professional"]
REGION = ["Northeast", "Midwest", "South", "West"]

CENSUS_SCHEMA = {
    "age": {"type": "numeric", "lower": 17, "upper": 90, "bins": 15},
    "education": {"type": "categorical", "categories": EDUCATION},
    "occupation": {"type": "categorical", "categories": OCCUPATION},
    "hours_per_week": {"type": "numeric", "lower": 1, "upper": 99, "bins": 10},
    "region": {"type": "categorical", "categories": REGION},
    "sex": {"type": "categorical", "categories": ["F", "M"]},
    "income": {"type": "categorical", "categories": ["<=50K", ">50K"]},
}


def make_census_like(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(6.0, 6.5, n) + 17, 17, 90)
    edu_idx = np.clip(
        np.round(rng.normal(1.3 + 0.012 * (age - 17), 1.1, n)), 0, len(EDUCATION) - 1
    ).astype(int)

    # Occupation depends strongly on education.
    occ_logits = np.outer(edu_idx, np.linspace(-1.2, 1.6, len(OCCUPATION)))
    occ_p = np.exp(occ_logits) / np.exp(occ_logits).sum(1, keepdims=True)
    occ_idx = (occ_p.cumsum(1) > rng.random((n, 1))).argmax(1)

    hours = np.clip(rng.normal(40 + 0.15 * (age - 40) - 0.004 * (age - 40) ** 2, 9, n), 1, 99)
    region = rng.choice(len(REGION), n, p=[0.18, 0.21, 0.38, 0.23])
    sex = rng.choice(2, n)

    z = -5.3 + 0.55 * edu_idx + 0.35 * occ_idx + 0.035 * (hours - 40) + 0.03 * np.minimum(age, 55)
    income = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame(
        {
            "age": np.round(age),
            "education": np.array(EDUCATION, dtype=object)[edu_idx],
            "occupation": np.array(OCCUPATION, dtype=object)[occ_idx],
            "hours_per_week": np.round(hours),
            "region": np.array(REGION, dtype=object)[region],
            "sex": np.array(["F", "M"], dtype=object)[sex],
            "income": np.array(["<=50K", ">50K"], dtype=object)[income],
        }
    )
