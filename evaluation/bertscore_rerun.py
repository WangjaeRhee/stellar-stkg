"""
G-BERTScore on a saved extraction file (same method as bertscore_eval.py:
quintuple -> "e1 | rel | e2 | time | loc", bert_score lang="en",
model_type="roberta-large", default layer, no idf, no baseline rescaling).

  python evaluation/bertscore_rerun.py --extract results/rerun/extract_qwen2.5-7b_final-loconly.jsonl
-> results/rerun/bertscore_<extract stem>.json / .md
"""

import argparse
import datetime
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    read_jsonl, write_json, DOMAINS, configure_stdout,
)


def quintuple_to_text(q):
    return " | ".join(str(q.get(k, "")).strip()
                      for k in ("entity1", "relation", "entity2", "time", "location"))


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-type", default="roberta-large")
    args = ap.parse_args()

    records = read_jsonl(args.extract)
    gt_texts, ext_texts, meta = [], [], []
    skipped = 0
    for r in records:
        proc = (r.get("processed_quintuples") or [None])[0]
        if not proc:
            skipped += 1
            continue
        gt_texts.append(quintuple_to_text(r["ground_truth"]))
        ext_texts.append(quintuple_to_text(proc))
        meta.append({"id": r["id"], "domain": r["domain"]})
    print(f"records={len(records)} scored={len(gt_texts)} no_extraction={skipped}",
          flush=True)

    from bert_score import score as bert_score  # noqa: E402
    P, R, F1 = bert_score(ext_texts, gt_texts, lang="en",
                          model_type=args.model_type, verbose=True)

    per_domain = defaultdict(list)
    rows = []
    for i, m in enumerate(meta):
        p, rc, f = P[i].item(), R[i].item(), F1[i].item()
        per_domain[m["domain"]].append((p, rc, f))
        rows.append({**m, "P": round(p * 100, 2), "R": round(rc * 100, 2),
                     "F1": round(f * 100, 2), "gt": gt_texts[i], "ext": ext_texts[i]})

    def agg(lst):
        n = len(lst)
        return {"precision": round(sum(x[0] for x in lst) / n * 100, 1),
                "recall": round(sum(x[1] for x in lst) / n * 100, 1),
                "f1": round(sum(x[2] for x in lst) / n * 100, 1), "n": n}

    result = {
        "extract_file": os.path.basename(args.extract),
        "model_type": args.model_type,
        "method": "bertscore_eval.py (lang=en, roberta-large, default layer, no idf, no baseline)",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "overall": agg([x for d in per_domain.values() for x in d]),
        "domains": {d: agg(per_domain[d]) for d in DOMAINS if d in per_domain},
        "per_case": rows,
    }
    stem = args.out or os.path.join(
        os.path.dirname(args.extract),
        "bertscore_" + os.path.basename(args.extract).replace("extract_", "", 1)
        .replace(".jsonl", ""))
    write_json(stem + ".json", result)
    md = [f"# G-BERTScore ({args.model_type}) - {os.path.basename(args.extract)}", "",
          "| domain | n | P | R | F1 |", "|---|---|---|---|---|"]
    for d, s in result["domains"].items():
        md.append(f"| {d} | {s['n']} | {s['precision']} | {s['recall']} | {s['f1']} |")
    o = result["overall"]
    md.append(f"| **overall** | {o['n']} | {o['precision']} | {o['recall']} | {o['f1']} |")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"written: {stem}.json / {stem}.md")


if __name__ == "__main__":
    main()
