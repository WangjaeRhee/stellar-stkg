"""
Component contribution analysis (extraction only; judge with run_judge.py).

Configurations (same definitions as the old ablation_eval.py, so the ladder
is comparable with the paper's Table 3):
  1 direct            one-shot quintuple prompt on the original text
  2 atomic_direct     minimal decomposition -> one-shot prompt on fact[0]
  3 multistage_nocot  minimal decomposition -> entity / relation / time-loc
                      prompts WITHOUT CoT, time-loc from fact[0]
  4 multistage_cot    STKGExtractor.extract_quintuples (v12 prompts, no
                      normalization)
  5 full              4 + postprocess (location / relation / entity
                      normalization + entity2 rules)
                      NOTE: old ablation_eval used normalize_relation=False
                      here; the full pipeline in the paper uses True, so True
                      is used now.

Each config writes results/rerun/extract_ablation_<config>_<model>.jsonl in
the same record format as run_extract.py.  `diag` records whether the year
of the sentence survived into fact[0] (to explain the time drop in 2/3).

  python evaluation/run_ablation.py                       # sample90, all configs
  python evaluation/run_ablation.py --configs direct full --limit 5
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ollama  # noqa: E402
from tqdm import tqdm  # noqa: E402
from evaluation.dataset import (  # noqa: E402
    RERUN_DIR, load_all_cases, load_ids, model_slug, existing_ids,
    append_jsonl, configure_stdout,
)
from config import LLM_MODEL, LLM_HOST  # noqa: E402
from modules.llm_common import LLM_OPTIONS  # noqa: E402
from modules.extractor import STKGExtractor  # noqa: E402
from modules.postprocessor import (  # noqa: E402
    postprocess_quintuples, drain_fallback_log, drain_error_log,
    configure_normalizers,
)

CONFIGS = ["direct", "atomic_direct", "multistage_nocot", "multistage_cot", "full",
           "fair_direct", "fair_atomic_direct", "fair_multistage_nocot"]
# "fair_*" ladder: framework decomposition prompt (STKGExtractor.decompose_to_atomic_facts)
# and time/location from the ORIGINAL text with the framework prompt
# (STKGExtractor.extract_time_location) in every step, so that
#   A  direct -> A' fair_direct        changes only: time/location prompt (framework prompt
#                                      on the original text instead of the single-query values)
#   A' fair_direct -> B fair_atomic_direct   changes only: decomposition + single-query on fact[0]
#   B -> C fair_multistage_nocot       changes only: single query -> entity / relation prompts
#   C -> D multistage_cot              changes only: entity/relation prompts gain CoT + examples
#   D -> E full                        changes only: normalization
_client = None
_model = LLM_MODEL
_last_error = None


def call_llm(prompt):
    global _last_error
    try:
        r = _client.chat(model=_model,
                         messages=[{"role": "user", "content": prompt}],
                         options=LLM_OPTIONS)
        return r["message"]["content"]
    except Exception as e:
        _last_error = str(e)
        print(f"LLM error: {e}", flush=True)
        return ""


def _json_obj(content):
    s, e = content.find("{"), content.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(content[s:e])
    except Exception:
        return None


def _json_arr(content):
    s, e = content.find("["), content.rfind("]") + 1
    if s == -1 or e == 0:
        return None
    try:
        arr = json.loads(content[s:e])
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


# ---------------- config 1 ----------------
def extract_direct(text):
    prompt = f"""Extract a spatio-temporal quintuple from this text.

A quintuple has 5 elements:
- entity1: the actor who performs the action (person, country, org)
- relation: the action performed (2-5 words)
- entity2: the actor who is affected (person, country, org)
- time: date in YYYY-MM-DD format
- location: country where the event occurred

Return ONLY a JSON object.

Text: "{text}"
Output:"""
    q = _json_obj(call_llm(prompt))
    return [], ([q] if isinstance(q, dict) else [])


# ---------------- minimal decomposition (configs 2, 3) ----------------
def decompose_minimal(text):
    prompt = f"""Decompose into atomic facts. Return ONLY a JSON array.
Text: {text}
Output:"""
    facts = _json_arr(call_llm(prompt))
    if not facts:
        facts = [text]
    return [str(f) for f in facts]


# ---------------- config 2 ----------------
def extract_atomic_direct(text):
    facts = decompose_minimal(text)
    _, q = extract_direct(facts[0] if facts else text)
    return facts, q


# ---------------- config 3 ----------------
def extract_multistage_nocot(text):
    facts = decompose_minimal(text)
    fact = facts[0] if facts else text

    prompt2 = f"""Extract the two main actors from this sentence.
entity1 = who performs the action
entity2 = who is affected by the action
Both must be a person, country, or organization.
Return ONLY a JSON object.
Sentence: "{fact}"
Output:"""
    entities = _json_obj(call_llm(prompt2))
    if not isinstance(entities, dict):
        return facts, []
    e1 = entities.get("entity1", "unknown")
    e2 = entities.get("entity2", "unknown")

    prompt3 = f"""What action did "{e1}" perform toward "{e2}"?
Sentence: "{fact}"
Return ONLY a JSON object with "relation" key.
Output:"""
    rel_obj = _json_obj(call_llm(prompt3))
    rel = rel_obj.get("relation", "unknown") if isinstance(rel_obj, dict) else "unknown"

    prompt4 = f"""Extract time (YYYY-MM-DD) and location (country) from:
"{fact}"
Return ONLY a JSON object.
Output:"""
    tl = _json_obj(call_llm(prompt4))
    if not isinstance(tl, dict):
        tl = {"time": "unknown", "location": "unknown"}

    return facts, [{"entity1": e1, "relation": rel, "entity2": e2,
                    "time": tl.get("time", "unknown"),
                    "location": tl.get("location", "unknown")}]


# ---------------- fair ladder (B, C) ----------------
def _timeloc_from_text(text):
    tl = _extractor.extract_time_location(text)
    return {"time": tl.get("time", "unknown"), "location": tl.get("location", "unknown")}


def extract_fair_direct(text):
    """A': single-query prompt on the ORIGINAL text (as `direct`), but time /
    location replaced by framework extraction from the original text, so
    that A -> A' isolates the time/location prompt change."""
    _, q = extract_direct(text)
    if q:
        q[0].update(_timeloc_from_text(text))
    return [], q


def extract_fair_atomic_direct(text):
    """B: framework decomposition -> single-query prompt on fact[0];
    time/location replaced by framework extraction from the original text."""
    facts = [str(f) for f in _extractor.decompose_to_atomic_facts(text)] or [text]
    _, q = extract_direct(facts[0])
    if q:
        q[0].update(_timeloc_from_text(text))
    return facts, q


def extract_fair_multistage_nocot(text):
    """C: framework decomposition -> plain entity prompt -> plain relation prompt
    (no CoT, no examples); time/location from the original text (framework prompt)."""
    facts = [str(f) for f in _extractor.decompose_to_atomic_facts(text)] or [text]
    fact = facts[0]
    prompt2 = f"""Extract the two main actors from this sentence.
entity1 = who performs the action
entity2 = who is affected by the action
Both must be a person, country, or organization.
Return ONLY a JSON object.
Sentence: "{fact}"
Output:"""
    entities = _json_obj(call_llm(prompt2))
    if not isinstance(entities, dict):
        return facts, []
    e1 = entities.get("entity1", "unknown")
    e2 = entities.get("entity2", "unknown")
    prompt3 = f"""What action did "{e1}" perform toward "{e2}"?
Sentence: "{fact}"
Return ONLY a JSON object with "relation" key.
Output:"""
    rel_obj = _json_obj(call_llm(prompt3))
    rel = rel_obj.get("relation", "unknown") if isinstance(rel_obj, dict) else "unknown"
    q = {"entity1": e1, "relation": rel, "entity2": e2}
    q.update(_timeloc_from_text(text))
    return facts, [q]


# ---------------- configs 4, 5 ----------------
_extractor = None


def extract_multistage_cot(text):
    trace = _extractor.extract_with_trace(text)
    return trace["atomic_facts"], trace["quintuples"]


def year_diag(text, facts):
    years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", text))
    first = facts[0] if facts else ""
    return {"years_in_text": sorted(years),
            "year_in_first_fact": any(y in first for y in years) if years else None,
            "n_facts": len(facts)}


def main():
    global _client, _model, _extractor, _last_error
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--ids", default=os.path.join(RERUN_DIR, "sample90_ids.json"))
    ap.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--host", default=LLM_HOST)
    args = ap.parse_args()

    _model = args.model
    _client = ollama.Client(host=args.host)
    _extractor = STKGExtractor(model=args.model, host=args.host)
    if args.host != LLM_HOST:
        configure_normalizers(host=args.host)
        print(f"NOTE: using Ollama host {args.host}", flush=True)
    slug = model_slug(args.model)

    cases = load_all_cases()
    keep = set(load_ids(args.ids))
    cases = [c for c in cases if c["id"] in keep]
    if args.limit:
        cases = cases[: args.limit]

    for cfg in args.configs:
        out = os.path.join(RERUN_DIR, f"extract_ablation_{cfg}_{slug}.jsonl")
        done = existing_ids(out)
        todo = [c for c in cases if c["id"] not in done]
        print(f"\n== {cfg}: cases={len(cases)} done={len(done)} todo={len(todo)}"
              f" -> {out}", flush=True)
        consecutive_errors = 0
        for case in tqdm(todo, desc=cfg):
            t0 = time.time()
            error = None
            facts, raw, processed = [], [], []
            _last_error = None
            _extractor.last_error = None
            drain_error_log()
            try:
                if cfg == "direct":
                    facts, raw = extract_direct(case["text"])
                elif cfg == "atomic_direct":
                    facts, raw = extract_atomic_direct(case["text"])
                elif cfg == "multistage_nocot":
                    facts, raw = extract_multistage_nocot(case["text"])
                elif cfg == "multistage_cot":
                    facts, raw = extract_multistage_cot(case["text"])
                elif cfg == "full":
                    facts, raw = extract_multistage_cot(case["text"])
                elif cfg == "fair_direct":
                    facts, raw = extract_fair_direct(case["text"])
                elif cfg == "fair_atomic_direct":
                    facts, raw = extract_fair_atomic_direct(case["text"])
                elif cfg == "fair_multistage_nocot":
                    facts, raw = extract_fair_multistage_nocot(case["text"])
                raw = [q for q in raw if isinstance(q, dict)]
                if cfg == "full":
                    processed = postprocess_quintuples(
                        raw, normalize_relation=True, normalize_entity=True,
                        apply_corrections=True) if raw else []
                else:
                    processed = [dict(q) for q in raw]
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            llm_err = error or _last_error or _extractor.last_error
            norm_errors = drain_error_log()
            if llm_err or norm_errors:
                consecutive_errors += 1
                print(f"\n[{case['id']}] LLM call error -> record not written "
                      f"({llm_err}; normalizer errors: {len(norm_errors)})",
                      flush=True)
                if consecutive_errors >= 3:
                    print("ABORT: 3 consecutive LLM errors. Re-run to resume.",
                          flush=True)
                    sys.exit(1)
                continue
            consecutive_errors = 0
            append_jsonl(out, {
                "id": case["id"], "domain": case["domain"],
                "text": case["text"], "ground_truth": case["ground_truth"],
                "atomic_facts": facts, "raw_quintuples": raw,
                "processed_quintuples": processed,
                "elapsed_sec": round(time.time() - t0, 2), "error": error,
                "normalizer_fallbacks": drain_fallback_log(),
                "config": cfg, "model": args.model, "provider": "ollama",
                "prompt_version": _extractor.prompt_version,
                "diag": year_diag(case["text"], facts),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            })


if __name__ == "__main__":
    main()
