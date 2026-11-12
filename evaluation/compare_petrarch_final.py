"""
PETRARCH3 vs STELLAR (final configuration) on the 155 sentences PETRARCH coded,
4 elements (entity1, relation, entity2, time; location excluded because
PETRARCH3 does not output it), judged by the single shared judge
(evaluation/judge.py) with the shared verdict cache.

PETRARCH side (as in evaluation/reeval_petrarch4.py, the only local script
implementing the 155-subset comparison; eval_155.py was not found):
  - events read from results/petrarch/evts.petr_out.txt (copy of
    C:/Users/Rhee/Desktop/KMOU/petrarch3/evts.petr_out.txt, 2026-09-05)
  - CASE<idx> maps to the idx-th sentence in dataset order (raw 0-49, dom 50-109,
    ext 110-449), the same order compare_petrarch.py used to build input.xml
  - entity1/entity2 = source/target actor TEXT if present else CAMEO actor code,
    relation = event text (tags stripped) else CAMEO code,
    time = YYYYMMDD reformatted to YYYY-MM-DD (format only)
  - when a sentence has several events, the one whose actor texts contain the
    ground-truth entity1/entity2 most often is judged (reeval_petrarch4 rule)
  NOTE: the PETRARCH input date was taken from the ground-truth time when the
  XML was built, so PETRARCH time accuracy is trivially 100 %.

STELLAR side: first quintuple of --stellar (final configuration file); its
verdicts are taken from the existing judge file for the same values.

  python evaluation/compare_petrarch_final.py [--dry-run]
-> results/rerun/petrarch_vs_final_4elem.json / .md,
   results/rerun/judge_petrarch_4elem.jsonl (PETRARCH verdicts)
"""

import argparse
import datetime
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, load_all_cases, read_jsonl, write_json, configure_stdout,
)
from evaluation.judge import BinaryJudge, JudgeError, _norm, MISSING_VALUES  # noqa: E402
from config import LLM_MODEL, LLM_HOST  # noqa: E402

FIELDS4 = ["entity1", "relation", "entity2", "time"]
PETR_OUT = os.path.join(ROOT, "results", "petrarch", "evts.petr_out.txt")


def normalize_date(d):
    d = str(d).strip()
    if re.fullmatch(r"\d{8}", d):
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def clean_event_text(t):
    t = re.sub(r"<[^>]*>", " ", str(t).strip()).replace("...", " ")
    return " ".join(t.split())


def load_petrarch(path):
    events = defaultdict(list)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            m = re.search(r"CASE(\d+)_", parts[5])
            if not m:
                continue
            idx = int(m.group(1))
            src_t = parts[7] if len(parts) > 7 else ""
            tgt_t = parts[8] if len(parts) > 8 else ""
            evt_t = parts[9] if len(parts) > 9 else ""
            rel = clean_event_text(evt_t)
            events[idx].append({
                "entity1": src_t if src_t not in ("", "---") else parts[1],
                "entity2": tgt_t if tgt_t not in ("", "---") else parts[2],
                "relation": rel if rel and rel != "---" else parts[3],
                "time": normalize_date(parts[0]),
                "cameo": {"source": parts[1], "target": parts[2], "code": parts[3]},
            })
    return events


def best_event(gt, evs):
    best, bs = None, -1
    for e in evs:
        sc = sum(1 for f in ("entity1", "entity2")
                 if str(gt.get(f, "")).lower() in str(e.get(f, "")).lower())
        if sc > bs:
            bs, best = sc, e
    return best


def pass4(rec):
    return all(rec["elements"][f]["verdict"] == "PASS" for f in FIELDS4)


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--petrarch", default=PETR_OUT)
    ap.add_argument("--stellar", default=os.path.join(
        RERUN_DIR, "extract_qwen2.5-7b_final-loconly.jsonl"))
    ap.add_argument("--stellar-judge", default=os.path.join(
        RERUN_DIR, "judge_qwen2.5-7b_norm-loc.jsonl"),
        help="existing judge file for the STELLAR side (same values)")
    ap.add_argument("--cache", default=os.path.join(RERUN_DIR, "judge_cache.json"))
    ap.add_argument("--judge-model", default=LLM_MODEL)
    ap.add_argument("--host", default=LLM_HOST)
    ap.add_argument("--out", default=os.path.join(RERUN_DIR, "petrarch_vs_final_4elem"))
    ap.add_argument("--dry-run", action="store_true",
                    help="only count PETRARCH pairs that still need an LLM verdict")
    args = ap.parse_args()

    cases = load_all_cases()
    petr = load_petrarch(args.petrarch)
    idxs = [i for i in sorted(petr) if i < len(cases)]
    ids155 = [cases[i]["id"] for i in idxs]
    print(f"PETRARCH coded sentences: {len(idxs)}; events: "
          f"{sum(len(petr[i]) for i in idxs)}", flush=True)

    stellar = {r["id"]: r for r in read_jsonl(args.stellar)}
    sjudge = {r["id"]: r for r in read_jsonl(args.stellar_judge)}
    for i in idxs:
        c = cases[i]
        sj = sjudge.get(c["id"])
        sp = (stellar[c["id"]].get("processed_quintuples") or [None])[0]
        if sj and sp:
            for f in FIELDS4:
                assert str(sj["elements"][f]["ext"]) == str(sp.get(f, "")), (c["id"], f)

    judge = BinaryJudge(model=args.judge_model, host=args.host)
    judge.load_cache(args.cache, [])
    need = []
    for i in idxs:
        gt = cases[i]["ground_truth"]
        e = best_event(gt, petr[i])
        for f in FIELDS4:
            g, x = _norm(gt.get(f, "")), _norm(e.get(f, ""))
            if g == x or x in MISSING_VALUES:
                continue
            if f"{f}|{g}|{x}" not in judge.cache:
                need.append((cases[i]["id"], f, gt.get(f), e.get(f)))
    print(f"PETRARCH element pairs needing an LLM verdict: {len(need)} "
          f"(cache entries: {len(judge.cache)})", flush=True)
    if args.dry_run:
        for item in need[:8]:
            print("  ", item)
        return

    out_judge = os.path.join(RERUN_DIR, "judge_petrarch_4elem.jsonl")
    done = {r["id"]: r for r in read_jsonl(out_judge)}
    rows = []
    for i in idxs:
        c = cases[i]
        gt = c["ground_truth"]
        if c["id"] in done:
            rows.append(done[c["id"]])
            continue
        e = best_event(gt, petr[i])
        elements = {}
        try:
            for f in FIELDS4:
                r = judge.judge_element(gt.get(f, ""), e.get(f, ""), f)
                elements[f] = {"gt": gt.get(f, ""), "ext": e.get(f, ""), **r}
        except JudgeError as ex:
            judge.save_cache(args.cache)
            print(f"\nABORT at {c['id']}: {ex}\n{len(rows)} PETRARCH records done; "
                  f"re-run to resume.", flush=True)
            sys.exit(1)
        rec = {"id": c["id"], "domain": c["domain"], "system": "petrarch3",
               "n_events": len(petr[i]), "cameo": e["cameo"], "elements": elements,
               "all_pass4": all(elements[f]["verdict"] == "PASS" for f in FIELDS4),
               "judge_model": args.judge_model}
        with open(out_judge, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rows.append(rec)
    judge.save_cache(args.cache)
    print(f"judge llm calls={judge.llm_calls}", flush=True)

    n = len(rows)
    st_rows = [sjudge[cases[i]["id"]] for i in idxs]
    petr_stats = {f: sum(1 for r in rows if r["elements"][f]["verdict"] == "PASS")
                  for f in FIELDS4}
    petr_stats["overall4"] = sum(1 for r in rows if r["all_pass4"])
    st_stats = {f: sum(1 for r in st_rows if r["elements"][f]["verdict"] == "PASS")
                for f in FIELDS4}
    st_stats["overall4"] = sum(1 for r in st_rows if pass4(r))
    st_extracted = sum(1 for r in st_rows if r["extracted"])

    def pct(v):
        return round(v / n * 100, 1)

    by_domain = defaultdict(lambda: {"n": 0, "petr4": 0, "st4": 0})
    for r, s in zip(rows, st_rows):
        d = by_domain[r["domain"]]
        d["n"] += 1
        d["petr4"] += int(r["all_pass4"])
        d["st4"] += int(pass4(s))

    result = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "basis": "PETRARCH3-coded subset (sentences with >=1 event)",
        "n": n, "ids": ids155,
        "criteria": "4 elements (entity1, relation, entity2, time); location excluded",
        "judge": {"model": args.judge_model, "prompt": "evaluation/judge.py (unchanged)",
                  "cache": os.path.basename(args.cache), "new_llm_calls": judge.llm_calls},
        "petrarch3": {**{f: pct(petr_stats[f]) for f in FIELDS4},
                      "overall4": pct(petr_stats["overall4"]), "extracted": n,
                      "counts": petr_stats,
                      "note": "time trivially 100 %: input date was the ground-truth date"},
        "stellar_final": {**{f: pct(st_stats[f]) for f in FIELDS4},
                          "overall4": pct(st_stats["overall4"]),
                          "extracted": st_extracted, "counts": st_stats,
                          "extract_file": os.path.basename(args.stellar)},
        "by_domain": {d: {"n": v["n"],
                          "petrarch3_overall4": round(v["petr4"] / v["n"] * 100, 1),
                          "stellar_overall4": round(v["st4"] / v["n"] * 100, 1)}
                      for d, v in by_domain.items()},
    }
    write_json(args.out + ".json", result)
    md = [f"# PETRARCH3 vs STELLAR (final config) on the {n} PETRARCH-coded sentences, 4 elements",
          "",
          f"STELLAR extracted {st_extracted}/{n}; judge = {args.judge_model}, "
          f"evaluation/judge.py, shared cache; new LLM verdicts this run: {judge.llm_calls}",
          "", "| element | PETRARCH3 | STELLAR (final) |", "|---|---|---|"]
    for f in FIELDS4 + ["overall4"]:
        md.append(f"| {f} | {result['petrarch3'][f]} | {result['stellar_final'][f]} |")
    md += ["", "| domain | n | PETRARCH3 exact(4) | STELLAR exact(4) |", "|---|---|---|---|"]
    for d, v in result["by_domain"].items():
        md.append(f"| {d} | {v['n']} | {v['petrarch3_overall4']} | {v['stellar_overall4']} |")
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"written: {args.out}.json / .md")


if __name__ == "__main__":
    main()
