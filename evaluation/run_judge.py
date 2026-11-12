"""
Judge a saved extraction file -> results/rerun/judge_<...>.jsonl

One line per sentence:
  {id, domain, extracted, elements: {field: {gt, ext, verdict, exact_match,
   judge_raw, method}}, all_pass, judge_model, corrections_applied}

Only the FIRST processed quintuple is judged (as in the paper).
Extraction failures are recorded with extracted=false and every element FAIL.

Usage
  python evaluation/run_judge.py --extract results/rerun/extract_qwen2.5-7b.jsonl
  python evaluation/run_judge.py --extract ... --no-corrections   # entity2 before rules
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402
from evaluation.dataset import (  # noqa: E402
    read_jsonl, existing_ids, append_jsonl, load_ids, configure_stdout,
)
from evaluation.judge import BinaryJudge, JudgeError  # noqa: E402
from config import LLM_MODEL, LLM_HOST  # noqa: E402


def default_out(extract_path, no_corrections):
    d, base = os.path.split(extract_path)
    base = base.replace("extract_", "judge_", 1)
    if no_corrections:
        base = base.replace(".jsonl", "_norules.jsonl")
    return os.path.join(d, base)


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge-model", default=LLM_MODEL)
    ap.add_argument("--host", default=LLM_HOST)
    ap.add_argument("--no-corrections", action="store_true",
                    help="judge entity2 as it was before ENTITY2_CORRECTIONS")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use-original", nargs="*", default=[],
                    choices=["entity1", "entity2", "relation", "location"],
                    help="judge these fields on their *_original value "
                         "(before LLM normalization; entity2 also skips rules)")
    ap.add_argument("--cache", default=None,
                    help="json file of verdicts keyed by field|gt|ext; loaded "
                         "before and saved after the run so identical pairs "
                         "get identical verdicts across configurations")
    ap.add_argument("--seed-cache-from", nargs="*", default=[],
                    help="judge jsonl files whose LLM verdicts pre-fill the cache")
    args = ap.parse_args()

    out = args.out or default_out(args.extract, args.no_corrections)
    if args.use_original and not args.out:
        tag = "orig-" + "-".join(args.use_original)
        out = out.replace(".jsonl", f"_{tag}.jsonl")
    records = read_jsonl(args.extract)
    if args.ids:
        keep = set(load_ids(args.ids))
        records = [r for r in records if r["id"] in keep]
    if args.limit:
        records = records[: args.limit]
    done = existing_ids(out)
    todo = [r for r in records if r["id"] not in done]
    print(f"extract={args.extract}\noutput={out}", flush=True)
    print(f"records={len(records)} already_done={len(done)} todo={len(todo)}",
          flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    judge = BinaryJudge(model=args.judge_model, host=args.host)
    if args.cache or args.seed_cache_from:
        judge.load_cache(args.cache, args.seed_cache_from)
        print(f"judge cache entries: {len(judge.cache)}", flush=True)
    written = 0
    for r in tqdm(todo, desc="judge"):
        processed = r.get("processed_quintuples") or []
        if processed:
            ext = dict(processed[0])
            if args.no_corrections and "entity2_before_rules" in ext:
                ext["entity2"] = ext["entity2_before_rules"]
            for f in args.use_original:
                if f + "_original" in ext:
                    ext[f] = ext[f + "_original"]
        else:
            ext = None
        try:
            elements, all_pass = judge.judge_quintuple(r["ground_truth"], ext)
        except JudgeError as e:
            print(f"\nABORT at {r['id']}: {e}\n"
                  f"{written} records written this run; re-run to resume.",
                  flush=True)
            sys.exit(1)
        written += 1
        append_jsonl(out, {
            "id": r["id"],
            "domain": r["domain"],
            "extracted": ext is not None,
            "elements": elements,
            "all_pass": all_pass,
            "judge_model": args.judge_model,
            "corrections_applied": (not args.no_corrections)
                                   and "entity2" not in args.use_original,
            "use_original": args.use_original,
            "extract_file": os.path.basename(args.extract),
        })

    print(f"\njudge llm calls={judge.llm_calls} errors={judge.errors}",
          flush=True)
    if args.cache:
        judge.save_cache(args.cache)
        print(f"judge cache saved: {args.cache} ({len(judge.cache)} entries)",
              flush=True)


if __name__ == "__main__":
    main()
