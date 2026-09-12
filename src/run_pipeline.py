"""
run_pipeline.py
===============
One-command end-to-end driver:

  1. generate the 5-year synthetic data (collection_long, issuance_long, national_daily)
  2. train & evaluate the XGBoost engineered-features baseline (7/30/90-day)

Run from the project root:
    python -m src.run_pipeline

Optional flags are passed through to the baseline step:
    python -m src.run_pipeline --target issuance --region national
"""

from __future__ import annotations

import argparse

from .synthetic_data import generate
from .train_baseline import run_baseline


def main():
    ap = argparse.ArgumentParser(description="Run the full blood forecast pipeline")
    ap.add_argument("--skip-generate", action="store_true",
                    help="Reuse existing CSVs in data/ instead of regenerating")
    ap.add_argument("--target", default="collection", choices=["collection", "issuance"])
    ap.add_argument("--region", default="national")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the RNG seed for the synthetic generator")
    args = ap.parse_args()

    if not args.skip_generate:
        print("=" * 70)
        print("STEP 1/2  Generating calibrated synthetic blood supply data")
        print("=" * 70)
        out = generate(seed=args.seed) if args.seed is not None else generate()
        nd = out["national_daily"]
        print(f"  collection_long : {out['collection_long'].shape[0]:,} rows")
        print(f"  issuance_long   : {out['issuance_long'].shape[0]:,} rows")
        print(f"  national_daily  : {out['national_daily'].shape[0]:,} rows "
              f"({nd.index.min().date()} -> {nd.index.max().date()})")
        print(f"  collection/yr   : {nd['collection'].sum()/5:,.0f} units")
        print(f"  issuance/yr     : {nd['issuance'].sum()/5:,.0f} units")

    print()
    print("=" * 70)
    print("STEP 2/2  Training XGBoost engineered-features baseline")
    print("=" * 70)
    run_baseline(target=args.target, region=args.region)


if __name__ == "__main__":
    main()
