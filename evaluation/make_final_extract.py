"""
Derive the FINAL-configuration extraction file from the main run.

Final configuration (chosen on sample90, 2026-10-14):
  Location normalization ONLY; no Entity / Relation LLM normalization and
  no rule-based entity2 correction.

postprocess_quintuples(normalize_relation=False, normalize_entity=False)
leaves entity1 / relation / entity2 untouched and normalizes location, so
the final-configuration quintuple is exactly
  {entity1_original, relation_original, entity2_original, time, location}
of the already-saved processed quintuple.  No LLM call is needed and the
extraction itself is unchanged (same raw quintuples, same timestamps).

  python evaluation/make_final_extract.py
    -> results/rerun/extract_qwen2.5-7b_final-loconly.jsonl
Also verifies that the values equal the ones judged in
results/rerun/judge_qwen2.5-7b_norm-loc.jsonl.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, read_jsonl, append_jsonl, FIELDS, configure_stdout,
)

FINAL_CONFIG = {
    "name": "final-loconly",
    "normalize_location": True,
    "normalize_relation": False,
    "normalize_entity": False,
    "apply_corrections": False,
    "chosen_on": "sample90 (exact 60.0 none/loc-only vs 43.3 all normalizers)",
    "chosen_date": "2026-10-14",
}


def to_final(proc):
    """processed quintuple (all normalizers on) -> final-config quintuple."""
    return {
        "entity1": proc.get("entity1_original", proc.get("entity1")),
        "relation": proc.get("relation_original", proc.get("relation")),
        "entity2": proc.get("entity2_original", proc.get("entity2")),
        "time": proc.get("time"),
        "location": proc.get("location"),
        "location_original": proc.get("location_original"),
    }


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract",
                    default=os.path.join(RERUN_DIR, "extract_qwen2.5-7b.jsonl"))
    ap.add_argument("--out",
                    default=os.path.join(RERUN_DIR,
                                         "extract_qwen2.5-7b_final-loconly.jsonl"))
    ap.add_argument("--verify-judge",
                    default=os.path.join(RERUN_DIR, "judge_qwen2.5-7b_norm-loc.jsonl"))
    args = ap.parse_args()

    if os.path.exists(args.out):
        os.remove(args.out)
    records = read_jsonl(args.extract)
    for r in records:
        rec = dict(r)
        rec["processed_quintuples"] = [to_final(p) for p in
                                       (r.get("processed_quintuples") or [])]
        rec["postprocess_config"] = FINAL_CONFIG
        rec["derived_from"] = os.path.basename(args.extract)
        append_jsonl(args.out, rec)
    print(f"written: {args.out} ({len(records)} records)")

    if args.verify_judge and os.path.exists(args.verify_judge):
        judged = {j["id"]: j for j in read_jsonl(args.verify_judge)}
        mism = []
        for r in read_jsonl(args.out):
            j = judged.get(r["id"])
            if j is None:
                mism.append((r["id"], "not judged"))
                continue
            proc = (r.get("processed_quintuples") or [None])[0]
            for f in FIELDS:
                ev = str(proc.get(f, "") if proc else "")
                jv = str(j["elements"][f]["ext"])
                if ev != jv:
                    mism.append((r["id"], f, ev, jv))
        tot = len(records) * len(FIELDS)
        print(f"verify vs {os.path.basename(args.verify_judge)}: "
              f"{tot - len(mism)}/{tot} element values identical; "
              f"mismatches={len(mism)}")
        for m in mism[:10]:
            print("  ", m)


if __name__ == "__main__":
    main()
