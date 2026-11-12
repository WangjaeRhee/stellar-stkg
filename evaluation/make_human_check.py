"""
Build the human-verification sheet (no LLM verdicts included).

  python evaluation/make_human_check.py --extract results/rerun/extract_qwen2.5-7b.jsonl

-> results/rerun/human_check.csv with columns
   id, domain, text, extracted,
   gt_entity1, ext_entity1, human_e1,
   gt_relation, ext_relation, human_rel,
   gt_entity2, ext_entity2, human_e2,
   gt_time, ext_time, human_time,
   gt_location, ext_location, human_loc, notes
Fill human_* with PASS / FAIL (also accepted: P/F, 1/0, O/X, Y/N).
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, read_jsonl, load_ids, FIELDS, configure_stdout,
)

HUMAN_COL = {"entity1": "human_e1", "relation": "human_rel",
             "entity2": "human_e2", "time": "human_time",
             "location": "human_loc"}


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--ids", default=os.path.join(RERUN_DIR, "sample90_ids.json"))
    ap.add_argument("--out", default=os.path.join(RERUN_DIR, "human_check.csv"))
    args = ap.parse_args()

    ids = load_ids(args.ids)
    by_id = {r["id"]: r for r in read_jsonl(args.extract)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} sample ids not in extract file: {missing[:5]}...")

    cols = ["id", "domain", "text", "extracted"]
    for f in FIELDS:
        cols += [f"gt_{f}", f"ext_{f}", HUMAN_COL[f]]
    cols.append("notes")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in ids:
            r = by_id.get(i)
            if r is None:
                continue
            proc = (r.get("processed_quintuples") or [None])[0]
            row = {"id": r["id"], "domain": r["domain"], "text": r["text"],
                   "extracted": "yes" if proc else "no", "notes": ""}
            for f in FIELDS:
                row[f"gt_{f}"] = r["ground_truth"].get(f, "")
                row[f"ext_{f}"] = proc.get(f, "") if proc else ""
                row[HUMAN_COL[f]] = ""
            w.writerow(row)
    print(f"written: {args.out} ({len(ids) - len(missing)} rows)")


if __name__ == "__main__":
    main()
