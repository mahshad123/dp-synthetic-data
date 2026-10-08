"""Privacy-utility benchmark: MST vs. independent baseline across epsilon.

    python benchmarks/privacy_utility.py            # writes results/ table + plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpsynth import Domain, IndependentSynthesizer, MSTSynthesizer
from dpsynth.datasets import CENSUS_SCHEMA, make_census_like
from dpsynth.evaluate import evaluate, exact_match_rate, ml_efficacy

EPSILONS = [0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = [0, 1, 2]
OUT = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train, test = make_census_like(20_000, seed=0), make_census_like(5_000, seed=1)
    dom = Domain.from_dict(CENSUS_SCHEMA)

    real_auc = ml_efficacy(train, test, dom, "income", ">50K")["auc"]
    holdout_match = exact_match_rate(train, test, dom)

    rows = []
    for eps in EPSILONS:
        for name, cls in [("Independent", IndependentSynthesizer), ("MST", MSTSynthesizer)]:
            for seed in SEEDS:
                synth = cls(epsilon=eps, delta=1e-6, seed=seed).fit(train, dom).sample()
                rows.append({"method": name, "epsilon": eps, "seed": seed,
                             **evaluate(train, test, synth, dom, "income", ">50K")})
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "epsilon"]).mean(numeric_only=True).drop(columns="seed").round(3)
    agg.to_csv(OUT / "privacy_utility.csv")

    md = agg.reset_index().to_markdown(index=False)
    md += f"\n\nReference: real-data AUC = {real_auc:.3f}; " \
          f"real holdout exact-match rate = {holdout_match:.3f}\n"
    (OUT / "privacy_utility.md").write_text(md)
    print(md)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, g in agg.reset_index().groupby("method"):
        axes[0].plot(g.epsilon, g.tvd_2way, marker="o", label=name)
        axes[1].plot(g.epsilon, g.tstr_auc, marker="o", label=name)
    axes[1].axhline(real_auc, ls="--", c="gray", label="Train on real data")
    axes[0].set(xscale="log", xlabel="epsilon", ylabel="avg 2-way TVD (lower = better)",
                title="Pairwise marginal error")
    axes[1].set(xscale="log", xlabel="epsilon", ylabel="AUC on real test set",
                title="Train-synthetic, test-real")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "privacy_utility.png", dpi=150)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
