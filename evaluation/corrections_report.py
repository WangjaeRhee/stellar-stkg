"""
How often each ENTITY2_CORRECTIONS rule fired in an extraction file.

  python evaluation/corrections_report.py --extract results/rerun/extract_qwen2.5-7b.jsonl

Counts are over the FIRST processed quintuple of each sentence (the one that
is judged).  "changed" = the rule actually altered the value; "noop" = the
rule matched but mapped the value to itself (e.g. "employees" -> "Employees").
"""

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import read_jsonl, write_json, configure_stdout  # noqa: E402


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.extract)
    per_rule = defaultdict(lambda: {"matched": 0, "changed": 0, "noop": 0,
                                    "examples": []})
    total_first = 0
    matched_first = 0
    changed_first = 0
    by_domain = defaultdict(lambda: {"matched": 0, "changed": 0})

    for r in records:
        proc = r.get("processed_quintuples") or []
        if not proc:
            continue
        total_first += 1
        q = proc[0]
        key = q.get("entity2_rule")
        if not key:
            continue
        before = q.get("entity2_before_rules")
        after = q.get("entity2")
        changed = (str(before).strip().lower() != str(after).strip().lower())
        matched_first += 1
        changed_first += int(changed)
        d = per_rule[key]
        d["matched"] += 1
        d["changed" if changed else "noop"] += 1
        if len(d["examples"]) < 5:
            d["examples"].append({"id": r["id"], "domain": r["domain"],
                                  "before": before, "after": after,
                                  "gt_entity2": r["ground_truth"].get("entity2")})
        by_domain[r["domain"]]["matched"] += 1
        by_domain[r["domain"]]["changed"] += int(changed)

    print(f"sentences with an extraction: {total_first}")
    print(f"rule matched: {matched_first}   value changed: {changed_first}\n")
    print("| rule key | matched | changed | noop | example (before -> after, gt) |")
    print("|---|---|---|---|---|")
    for key, d in sorted(per_rule.items(), key=lambda kv: -kv[1]["matched"]):
        ex = d["examples"][0]
        print(f"| {key} | {d['matched']} | {d['changed']} | {d['noop']} | "
              f"{ex['before']} -> {ex['after']} (gt: {ex['gt_entity2']}, {ex['id']}) |")
    print("\nby domain:")
    for dom, d in sorted(by_domain.items()):
        print(f"  {dom:<12} matched={d['matched']} changed={d['changed']}")

    out = args.out or args.extract.replace("extract_", "corrections_", 1).replace(
        ".jsonl", ".json")
    write_json(out, {"extract": args.extract, "sentences_with_extraction": total_first,
                     "matched": matched_first, "changed": changed_first,
                     "per_rule": per_rule, "by_domain": by_domain})
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
