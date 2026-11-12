"""
Draw the fixed 90-sentence sample (10 per domain, seed 42) used by the
component analysis, the model comparison and the human check.

  python evaluation/make_sample90.py            # writes results/rerun/sample90_ids.json
  python evaluation/make_sample90.py --force    # re-draw (same seed -> same ids)
"""

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, load_all_cases, DOMAINS, write_json, configure_stdout,
)

SEED = 42
PER_DOMAIN = 10


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RERUN_DIR, "sample90_ids.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"exists, not overwriting: {args.out}")
        return

    cases = load_all_cases()
    rng = random.Random(SEED)
    ids_by_domain = {}
    for d in DOMAINS:
        pool = sorted(c["id"] for c in cases if c["domain"] == d)
        ids_by_domain[d] = sorted(rng.sample(pool, PER_DOMAIN))
    ids = [i for d in DOMAINS for i in ids_by_domain[d]]
    write_json(args.out, {"seed": SEED, "per_domain": PER_DOMAIN,
                          "ids_by_domain": ids_by_domain, "ids": ids})
    print(f"written {len(ids)} ids -> {args.out}")
    for d in DOMAINS:
        print(f"  {d:<12} {ids_by_domain[d]}")


if __name__ == "__main__":
    main()
