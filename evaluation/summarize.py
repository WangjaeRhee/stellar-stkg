"""
Compute every paper table from judge files.

Metric names: element ACCURACY (= PASS / n, n includes extraction failures)
and quintuple EXACT-MATCH rate (= all five PASS / n). No F1.

  python evaluation/summarize.py --judge results/rerun/judge_qwen2.5-7b.jsonl
  python evaluation/summarize.py --judge a.jsonl=Direct --judge b.jsonl=+Atomic ... \
         --out results/rerun/summary_ablation

Writes <out>.json and <out>.md (defaults to summary_<judge basename>).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    read_jsonl, write_json, load_ids, FIELDS, DOMAINS, configure_stdout,
)


def stats_for(records):
    n = len(records)
    if n == 0:
        return None
    s = {"n": n,
         "extracted": sum(1 for r in records if r.get("extracted"))}
    for f in FIELDS:
        s[f] = round(100.0 * sum(
            1 for r in records if r["elements"][f]["verdict"] == "PASS") / n, 1)
        s[f + "_exact_str"] = round(100.0 * sum(
            1 for r in records if r["elements"][f].get("exact_match")) / n, 1)
    s["elem_avg"] = round(sum(s[f] for f in FIELDS) / len(FIELDS), 1)
    s["exact"] = round(100.0 * sum(1 for r in records if r["all_pass"]) / n, 1)
    s["exact_count"] = sum(1 for r in records if r["all_pass"])
    return s


ID_FILTER = {"keep": None, "drop": None}


def summarize_file(path):
    records = read_jsonl(path)
    if ID_FILTER["keep"] is not None:
        records = [r for r in records if r["id"] in ID_FILTER["keep"]]
    if ID_FILTER["drop"] is not None:
        records = [r for r in records if r["id"] not in ID_FILTER["drop"]]
    by_domain = {}
    for d in DOMAINS:
        recs = [r for r in records if r["domain"] == d]
        st = stats_for(recs)
        if st:
            by_domain[d] = st
    micro = stats_for(records)
    macro = {}
    if by_domain:
        keys = FIELDS + ["elem_avg", "exact"]
        for k in keys:
            macro[k] = round(sum(v[k] for v in by_domain.values()) / len(by_domain), 1)
        macro["n_domains"] = len(by_domain)
    return {"file": path, "n": len(records), "domains": by_domain,
            "micro": micro, "macro": macro}


def domain_table(label, summ):
    cols = ["n", "extracted"] + FIELDS + ["elem_avg", "exact"]
    head = "| domain | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"### {label}  (n={summ['n']})", "", head, sep]
    for d, st in summ["domains"].items():
        lines.append("| " + d + " | " + " | ".join(str(st[c]) for c in cols) + " |")
    if summ["macro"]:
        m = summ["macro"]
        lines.append("| **macro avg** |  |  | " + " | ".join(
            str(m[c]) for c in FIELDS + ["elem_avg", "exact"]) + " |")
    if summ["micro"]:
        m = summ["micro"]
        lines.append("| **micro (all)** | " + " | ".join(
            str(m[c]) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def comparison_table(labelled):
    labels = [l for l, _ in labelled]
    rows = FIELDS + ["elem_avg", "exact", "extracted", "n"]
    lines = ["### Comparison (micro over all records in each file)", "",
             "| metric | " + " | ".join(labels) + " |",
             "|" + "---|" * (len(labels) + 1)]
    for r in rows:
        vals = []
        for _, s in labelled:
            m = s["micro"] or {}
            vals.append(str(m.get(r, "")))
        lines.append(f"| {r} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True,
                    help="path or path=label (repeatable)")
    ap.add_argument("--out", default=None, help="output stem (no extension)")
    ap.add_argument("--title", default="Summary")
    ap.add_argument("--ids", default=None, help="only these ids (json list)")
    ap.add_argument("--exclude-ids", default=None, help="drop these ids")
    ap.add_argument("--no-domains", action="store_true",
                    help="comparison table only")
    args = ap.parse_args()
    if args.ids:
        ID_FILTER["keep"] = set(load_ids(args.ids))
    if args.exclude_ids:
        ID_FILTER["drop"] = set(load_ids(args.exclude_ids))

    labelled = []
    for item in args.judge:
        if "=" in item:
            path, label = item.split("=", 1)
        else:
            path = item
            label = os.path.basename(path).replace(".jsonl", "")
        labelled.append((label, summarize_file(path)))

    if args.out:
        stem = args.out
    else:
        first = labelled[0][1]["file"]
        d, base = os.path.split(first)
        stem = os.path.join(d, base.replace("judge_", "summary_", 1)
                            .replace(".jsonl", ""))

    md = [f"# {args.title}", "",
          "Metrics: element accuracy = PASS / n (n includes extraction "
          "failures); exact = all five elements PASS / n.", ""]
    if len(labelled) > 1:
        md.append(comparison_table(labelled))
    if not args.no_domains:
        for label, summ in labelled:
            md.append(domain_table(label, summ))
    text = "\n".join(md)
    print(text)

    write_json(stem + ".json", {label: summ for label, summ in labelled})
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nwritten: {stem}.json / {stem}.md")


if __name__ == "__main__":
    main()
