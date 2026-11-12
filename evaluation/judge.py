"""
Single shared LLM-as-Judge (Binary PASS/FAIL).

The prompt is the one from evaluation/llm_judge_eval.py (unchanged).
Every evaluation script must import BinaryJudge from here so that all
tables are produced by exactly one judge.

Deterministic short-cuts (no LLM call):
  - exact match (case/space-insensitive)        -> PASS  (as in llm_judge_eval)
  - extracted value empty / "unknown" / "none"  -> FAIL  (the prompt's own
    "Missing value" rule; also used by full_eval / domain_extra_eval)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ollama
from config import LLM_MODEL, LLM_HOST
from modules.llm_common import LLM_OPTIONS

FIELDS = ["entity1", "relation", "entity2", "time", "location"]

JUDGE_PROMPT = """You are a fair judge for knowledge graph evaluation.

Task: Determine if the extracted value is semantically equivalent to the ground truth.

Element type: {element_type}
Ground truth: "{gt_val}"
Extracted: "{ext_val}"

Rules for PASS:
- Same person (e.g. "Putin" = "Vladimir Putin")
- Same country (e.g. "US" = "United States")
- Same organization (e.g. "EU" = "European Union")
- Same action with different wording (e.g. "Ban" = "Impose restrictions")
- Abbreviation or nickname (e.g. "UN" = "United Nations")
- Sub-entity of the same parent (e.g. "Kyiv" PASS for "Ukraine")
- Same meaning expressed differently (e.g. "Condemn" = "Criticize")

Rules for FAIL:
- Completely different entity (e.g. "Japan" ≠ "United States")
- Missing value (e.g. "unknown" ≠ "Belgium")
- Wrong entity type (e.g. "Missiles" ≠ "Ukraine")
- Opposite meaning
- No meaningful connection

Think step by step:
1. What does the ground truth refer to?
2. What does the extracted value refer to?
3. Do they refer to the same thing?

Answer with ONLY "PASS" or "FAIL":"""

MISSING_VALUES = {"", "unknown", "none", "null", "n/a"}


def _norm(v):
    return str(v if v is not None else "").lower().strip()


def parse_verdict(content):
    """Same parsing rule as llm_judge_eval.py."""
    c = (content or "").strip().upper()
    is_pass = "PASS" in c and "FAIL" not in c.split("PASS")[0]
    return "PASS" if is_pass else "FAIL"


class JudgeError(RuntimeError):
    """Raised when the judge LLM cannot be reached / fails."""


class BinaryJudge:
    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}
        self.llm_calls = 0
        self.errors = 0

    def load_cache(self, path=None, seed_files=()):
        """Pre-fill the verdict cache from a cache json and/or judge jsonl
        files (only LLM-made verdicts are taken from jsonl files)."""
        import json
        import os
        for sf in seed_files:
            if not os.path.exists(sf):
                continue
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    for field, e in r["elements"].items():
                        if e.get("method") != "llm":
                            continue
                        key = f"{field}|{_norm(e['gt'])}|{_norm(e['ext'])}"
                        self.cache.setdefault(key, {
                            "verdict": e["verdict"], "exact_match": False,
                            "judge_raw": e["judge_raw"], "method": "llm"})
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.cache.update(json.load(f))

    def save_cache(self, path):
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)

    def judge_element(self, gt_val, ext_val, element_type):
        """
        Returns {"verdict": "PASS"|"FAIL", "exact_match": bool,
                 "judge_raw": str, "method": "exact"|"rule_missing"|"llm"|"error"}
        """
        gt_n, ext_n = _norm(gt_val), _norm(ext_val)
        if gt_n == ext_n:
            return {"verdict": "PASS", "exact_match": True,
                    "judge_raw": "EXACT MATCH", "method": "exact"}
        if ext_n in MISSING_VALUES:
            return {"verdict": "FAIL", "exact_match": False,
                    "judge_raw": "RULE: missing value", "method": "rule_missing"}

        key = f"{element_type}|{gt_n}|{ext_n}"
        if key in self.cache:
            return dict(self.cache[key])

        prompt = JUDGE_PROMPT.format(element_type=element_type,
                                     gt_val=gt_val, ext_val=ext_val)
        try:
            self.llm_calls += 1
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=LLM_OPTIONS,
            )
            content = response["message"]["content"]
            result = {"verdict": parse_verdict(content), "exact_match": False,
                      "judge_raw": content.strip()[:300], "method": "llm"}
        except Exception as e:
            # Never turn a transport error into a FAIL verdict: let the
            # caller stop and resume later.
            self.errors += 1
            raise JudgeError(f"judge LLM call failed for {element_type} "
                             f"(gt={gt_val!r}, ext={ext_val!r}): {e}") from e

        self.cache[key] = result
        return dict(result)

    def judge_quintuple(self, gt, ext):
        """ext may be None (extraction failure) -> every element FAIL."""
        elements = {}
        for field in FIELDS:
            gt_val = gt.get(field, "")
            if ext is None:
                elements[field] = {
                    "gt": gt_val, "ext": "", "verdict": "FAIL",
                    "exact_match": False,
                    "judge_raw": "RULE: no extraction", "method": "rule_missing",
                }
                continue
            ext_val = ext.get(field, "")
            r = self.judge_element(gt_val, ext_val, field)
            elements[field] = {"gt": gt_val, "ext": ext_val, **r}
        all_pass = all(elements[f]["verdict"] == "PASS" for f in FIELDS)
        return elements, all_pass
