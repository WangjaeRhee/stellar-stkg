"""
Human vs LLM-judge agreement on the 90-sentence sample.

  python evaluation/agreement.py --human results/rerun/human_check.csv \
         --judge results/rerun/judge_qwen2.5-7b.jsonl

Per element: n compared, raw agreement %, Cohen's kappa; plus the
quintuple-level (all five PASS) agreement.  Rows whose human cell is blank
are skipped for that element.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, read_jsonl, write_json, FIELDS, configure_stdout,
)

HUMAN_COL = {"entity1": "human_e1", "relation": "human_rel",
             "entity2": "human_e2", "time": "human_time",
             "location": "human_loc"}
TRUE_SET = {"pass", "p", "1", "o", "y", "yes", "true", "t", "ok"}
FALSE_SET = {"fail", "f", "0", "x", "n", "no", "false"}


def parse_human(v):
    v = (v or "").strip().lower()
    if v in TRUE_SET:
        return True
    if v in FALSE_SET:
        return False
    return None


def kappa(pairs):
    """pairs: list of (human_bool, judge_bool)"""
    n = len(pairs)
    if n == 0:
        return None, None
    agree = sum(1 for h, j in pairs if h == j)
    po = agree / n
    ph = sum(1 for h, _ in pairs if h) / n
    pj = sum(1 for _, j in pairs if j) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    k = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    return round(100 * po, 1), round(k, 3)


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default=os.path.join(RERUN_DIR, "human_check.csv"))
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", default=os.path.join(RERUN_DIR, "agreement.json"))
    args = ap.parse_args()

    judge = {r["id"]: r for r in read_jsonl(args.judge)}
    with open(args.human, "r", encoding="utf-8-sig", newline="") as fh:
        human_rows = list(csv.DictReader(fh))

    result = {}
    pairs_all = []
    confusion = {}
    for f in FIELDS:
        pairs = []
        conf = {"both_pass": 0, "human_pass_judge_fail": 0,
                "human_fail_judge_pass": 0, "both_fail": 0}
        for row in human_rows:
            jr = judge.get(row["id"])
            h = parse_human(row.get(HUMAN_COL[f]))
            if jr is None or h is None:
                continue
            j = jr["elements"][f]["verdict"] == "PASS"
            pairs.append((h, j))
            if h and j:
                conf["both_pass"] += 1
            elif h and not j:
                conf["human_pass_judge_fail"] += 1
            elif not h and j:
                conf["human_fail_judge_pass"] += 1
            else:
                conf["both_fail"] += 1
        agree, k = kappa(pairs)
        result[f] = {"n": len(pairs), "agreement_pct": agree, "kappa": k}
        confusion[f] = conf

    # quintuple level: only rows where all five human cells are filled
    for row in human_rows:
        jr = judge.get(row["id"])
        if jr is None:
            continue
        hs = [parse_human(row.get(HUMAN_COL[f])) for f in FIELDS]
        if any(h is None for h in hs):
            continue
        pairs_all.append((all(hs), bool(jr["all_pass"])))
    agree, k = kappa(pairs_all)
    result["all_pass"] = {"n": len(pairs_all), "agreement_pct": agree, "kappa": k}

    print("| element | n | agreement % | kappa |")
    print("|---|---|---|---|")
    for key in FIELDS + ["all_pass"]:
        r = result[key]
        print(f"| {key} | {r['n']} | {r['agreement_pct']} | {r['kappa']} |")
    print("\nconfusion (human vs judge):")
    for f in FIELDS:
        print(f"  {f}: {confusion[f]}")
    write_json(args.out, {"human": args.human, "judge": args.judge,
                          "result": result, "confusion": confusion})
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
