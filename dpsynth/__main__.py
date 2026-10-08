"""Command line interface.

    python -m dpsynth --input data.csv --schema schema.json --epsilon 1.0 --output synth.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .domain import Domain
from .synthesizers import IndependentSynthesizer, MSTSynthesizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dpsynth", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV with the private data")
    p.add_argument("--schema", required=True, help="JSON schema with PUBLIC bounds/categories")
    p.add_argument("--output", required=True, help="where to write the synthetic CSV")
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--rows", type=int, default=None, help="rows to generate (default: noisy N)")
    p.add_argument("--method", choices=["mst", "independent"], default="mst")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    domain = Domain.from_json(args.schema)
    df = pd.read_csv(args.input)
    cls = MSTSynthesizer if args.method == "mst" else IndependentSynthesizer
    synth = cls(epsilon=args.epsilon, delta=args.delta, seed=args.seed).fit(df, domain)
    synth.sample(args.rows).to_csv(args.output, index=False)

    print(synth.privacy_report(), file=sys.stderr)
    if isinstance(synth, MSTSynthesizer):
        print("Learned dependency tree:", *synth.describe_tree(), sep="\n  ", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
