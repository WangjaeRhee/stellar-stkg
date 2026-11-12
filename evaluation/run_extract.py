"""
Extraction only (no judging) -> results/rerun/extract_<model>.jsonl

One line per sentence:
  {id, domain, text, ground_truth, atomic_facts, raw_quintuples,
   processed_quintuples, elapsed_sec, error, normalizer_fallbacks,
   model, provider, normalizer_model, prompt_version, timestamp}

Resumable: ids already present in the output file are skipped.

Usage
  python evaluation/run_extract.py --limit 10
  python evaluation/run_extract.py                      # all 450
  python evaluation/run_extract.py --ids results/rerun/sample90_ids.json \
         --provider openai --model gpt-4o-mini
"""

import argparse
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402
from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, load_all_cases, load_ids, model_slug, existing_ids,
    append_jsonl, write_json, configure_stdout,
)
from config import LLM_MODEL, LLM_HOST  # noqa: E402
from modules.extractor import STKGExtractor  # noqa: E402
from modules.postprocessor import (  # noqa: E402
    postprocess_quintuples, drain_fallback_log, drain_error_log,
    get_normalizer_stats, configure_normalizers,
)
from modules.llm_common import LLM_OPTIONS  # noqa: E402


class OpenAIExtractor(STKGExtractor):
    """Same prompts / pipeline; only _call_llm goes to the OpenAI API."""

    def __init__(self, api_model, api_key):
        import openai
        self.model = api_model
        self.client = openai.OpenAI(api_key=api_key)
        self.prompt_version = STKGExtractor().prompt_version
        self.last_error = None
        self.error_count = 0

    def _call_llm(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_OPTIONS["temperature"],
                seed=LLM_OPTIONS["seed"],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            print(f"  API error: {e}", flush=True)
            return ""


def build_extractor(provider, model, host):
    if provider == "ollama":
        return STKGExtractor(model=model, host=host)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("OPENAI_API_KEY is not set in the environment -> "
                  "skipping this step.", flush=True)
            sys.exit(3)
        return OpenAIExtractor(model, key)
    raise ValueError(provider)


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="json list of ids to run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="suffix for the output name")
    ap.add_argument("--no-show", action="store_true",
                    help="do not print the per-sentence preview at the end")
    ap.add_argument("--max-consecutive-errors", type=int, default=3)
    ap.add_argument("--host", default=LLM_HOST,
                    help="Ollama host for extraction AND normalizers "
                         "(default: config.LLM_HOST)")
    args = ap.parse_args()
    if args.host != LLM_HOST:
        configure_normalizers(host=args.host)
        print(f"NOTE: using Ollama host {args.host} (config: {LLM_HOST})",
              flush=True)

    slug = model_slug(args.model)
    out = args.out or os.path.join(
        RERUN_DIR, f"extract_{slug}{'_' + args.tag if args.tag else ''}.jsonl")
    stats_path = out.replace(".jsonl", "_stats.json")

    cases = load_all_cases()
    if args.ids:
        keep = set(load_ids(args.ids))
        cases = [c for c in cases if c["id"] in keep]
    if args.limit:
        cases = cases[: args.limit]

    done = existing_ids(out)
    todo = [c for c in cases if c["id"] not in done]
    print(f"model={args.model} provider={args.provider}", flush=True)
    print(f"output={out}", flush=True)
    print(f"cases={len(cases)} already_done={len(done)} todo={len(todo)}",
          flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    extractor = build_extractor(args.provider, args.model, args.host)
    normalizer_model = LLM_MODEL  # postprocessor always uses the config model
    new_records = []
    consecutive_errors = 0

    for case in tqdm(todo, desc="extract"):
        t0 = time.time()
        extractor.last_error = None
        drain_error_log()
        atomic_facts, raw, processed, error = [], [], [], None
        try:
            trace = extractor.extract_with_trace(case["text"])
            atomic_facts = trace["atomic_facts"]
            raw = trace["quintuples"]
            if raw:
                processed = postprocess_quintuples(
                    raw, normalize_relation=True, normalize_entity=True,
                    apply_corrections=True)
            error = extractor.last_error
        except Exception as e:  # keep going, record the failure
            error = f"{type(e).__name__}: {e}"
        norm_errors = drain_error_log()

        # An LLM transport error is NOT an extraction result: do not persist
        # it (so a re-run retries the sentence) and stop after 3 in a row.
        if error or norm_errors:
            consecutive_errors += 1
            print(f"\n[{case['id']}] LLM call error -> record not written "
                  f"(extractor: {error}; normalizer errors: {len(norm_errors)})",
                  flush=True)
            if consecutive_errors >= args.max_consecutive_errors:
                print(f"ABORT: {consecutive_errors} consecutive LLM errors. "
                      f"Is the Ollama server reachable? Re-run to resume.",
                      flush=True)
                break
            continue
        consecutive_errors = 0
        record = {
            "id": case["id"],
            "domain": case["domain"],
            "text": case["text"],
            "ground_truth": case["ground_truth"],
            "atomic_facts": atomic_facts,
            "raw_quintuples": raw,
            "processed_quintuples": processed,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": error,
            "normalizer_fallbacks": drain_fallback_log(),
            "model": args.model,
            "provider": args.provider,
            "normalizer_model": normalizer_model,
            "host": args.host if args.provider == "ollama" else "openai-api",
            "prompt_version": extractor.prompt_version,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(out, record)
        new_records.append(record)

    stats = {
        "model": args.model,
        "provider": args.provider,
        "records_written_this_run": len(new_records),
        "skipped_due_to_llm_errors": len(todo) - len(new_records),
        "extraction_failures_this_run": sum(
            1 for r in new_records if not r["processed_quintuples"]),
        "normalizer_stats_this_process": get_normalizer_stats(),
        "fallback_records_this_run": sum(
            1 for r in new_records if r["normalizer_fallbacks"]),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(stats_path, stats)
    print(f"\nstats: {stats}", flush=True)

    if not args.no_show and len(new_records) <= 20:
        print("\n=== preview ===")
        for r in new_records:
            print(f"\n[{r['id']}] ({r['domain']}) {r['text']}")
            print(f"  facts     : {r['atomic_facts']}")
            print(f"  gt        : {r['ground_truth']}")
            if r["raw_quintuples"]:
                q = r["raw_quintuples"][0]
                print(f"  raw       : {[q.get(k) for k in ('entity1','relation','entity2','time','location')]}")
            if r["processed_quintuples"]:
                p = r["processed_quintuples"][0]
                print(f"  processed : {[p.get(k) for k in ('entity1','relation','entity2','time','location')]}")
                if p.get("entity2_rule"):
                    print(f"  e2 rule   : {p['entity2_before_rules']!r} -> {p['entity2']!r} (key={p['entity2_rule']})")
            else:
                print("  processed : <extraction failed>")
            if r["normalizer_fallbacks"]:
                for fb in r["normalizer_fallbacks"]:
                    print(f"  fallback  : {fb['normalizer']} {fb['original']!r} ({fb['reason']}) raw={fb['raw'][:80]!r}")
            if r["error"]:
                print(f"  error     : {r['error']}")
            print(f"  elapsed   : {r['elapsed_sec']}s")


if __name__ == "__main__":
    main()
