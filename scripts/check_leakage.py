"""
Prompt-example leakage check.

Collects every few-shot example string in
    modules/extractor.py, modules/postprocessor.py, evaluation/judge.py
and computes, for each one, the maximum word-overlap ratio with the 450
test sentences:

    ratio = |content tokens of example ∩ content tokens of test sentence|
            / |content tokens of example|

Tokens are lower-cased, punctuation-split, stop-words removed and lightly
stemmed (plural / -ed / -ing), so "launched" matches "launch".

Policy
  - example with >= 3 content tokens and ratio >= THRESHOLD  -> FAIL
  - example with <= 2 content tokens (single entity names, two-word
    canonical labels such as "Freeze assets")                -> listed as
    EXEMPT with its best match, never a failure
  - everything else                                          -> OK

Exit code 1 when any FAIL exists.

Usage:  python scripts/check_leakage.py [--threshold 0.6] [--json out.json]
"""

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation.dataset import load_all_cases, configure_stdout  # noqa: E402

SOURCE_FILES = [
    os.path.join("modules", "extractor.py"),
    os.path.join("modules", "postprocessor.py"),
    os.path.join("evaluation", "judge.py"),
]

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by",
    "with", "and", "or", "as", "is", "was", "were", "be", "been", "its",
    "it", "this", "that", "into", "over", "near", "after", "before",
    "during", "about", "against", "between", "out", "up", "new", "s",
}

# strings that are prompt scaffolding, not examples
SKIP_EXACT = {
    "entity1", "entity2", "relation", "time", "location", "unknown",
    "pass", "fail", "yyyy-mm-dd", "provide", "provided", "normalized",
    "country", "output", "answer", "...", "denied", "deny", "rejected",
    "reject",
}


def tokenize(text):
    text = text.lower().replace("'s", " ")
    raw = re.findall(r"[a-z0-9]+", text)
    out = []
    for t in raw:
        if t in STOPWORDS:
            continue
        out.append(stem(t))
    return out


def stem(t):
    if len(t) <= 3 or t.isdigit():
        return t
    for suf in ("ies", "ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            base = t[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return t


_QUOTED = re.compile(r'"([^"\n]+)"')


def collect_examples(path):
    """Return list of (string, line_no) found on example-like lines."""
    found = []
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as f:
        lines = f.readlines()
    for no, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith('"""') or s.startswith("#") or s.startswith("'''"):
            continue  # docstrings / comments are not prompt examples
        is_example_line = (
            "→" in s or "e.g." in s or "≠" in s
            or (s.startswith('"') and s.endswith('"') and len(s) > 4)
            or s.startswith('"') and "→" in s
        )
        m = re.match(r"-\s*NOT\s.*\((.+)\)", s)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                if part:
                    found.append((part, no))
            continue
        if not is_example_line:
            continue
        for q in _QUOTED.findall(line):
            q = q.strip()
            if not q or "{" in q or "}" in q:
                continue
            if q.lower() in SKIP_EXACT:
                continue
            if re.fullmatch(r"\.\.\.\s*", q):
                continue
            found.append((q, no))
    return found


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--min-tokens", type=int, default=3,
                    help="examples with fewer content tokens are exempt")
    ap.add_argument("--json", default=None, help="write full report here")
    args = ap.parse_args()

    cases = load_all_cases()
    test_tokens = [(c["id"], set(tokenize(c["text"])), c["text"]) for c in cases]
    gt_values = {}
    for c in cases:
        for k, v in c["ground_truth"].items():
            gt_values.setdefault(str(v).lower().strip(), set()).add(k)

    rows = []
    seen = set()
    for src in SOURCE_FILES:
        for ex, line_no in collect_examples(src):
            key = (src, ex)
            if key in seen:
                continue
            seen.add(key)
            toks = set(tokenize(ex))
            if not toks:
                continue
            best = (0.0, None, None)
            for tid, tt, ttext in test_tokens:
                r = len(toks & tt) / len(toks)
                if r > best[0]:
                    best = (r, tid, ttext)
            ratio, tid, ttext = best
            if len(toks) < args.min_tokens:
                status = "EXEMPT"
            elif ratio >= args.threshold:
                status = "FAIL"
            else:
                status = "OK"
            rows.append({
                "file": src.replace("\\", "/"), "line": line_no, "example": ex,
                "n_tokens": len(toks), "max_overlap": round(ratio, 3),
                "best_test_id": tid, "best_test_text": ttext, "status": status,
                "equals_gt_label": sorted(gt_values.get(ex.lower().strip(), [])),
            })

    fails = [r for r in rows if r["status"] == "FAIL"]
    exempt = [r for r in rows if r["status"] == "EXEMPT"]
    oks = [r for r in rows if r["status"] == "OK"]

    print(f"Prompt example leakage check  (threshold={args.threshold}, "
          f"tests={len(cases)}, examples={len(rows)})\n")

    def table(title, items, show_text=True):
        print(f"== {title} ({len(items)}) ==")
        if not items:
            print("  (none)\n")
            return
        for r in sorted(items, key=lambda x: -x["max_overlap"]):
            print(f"  {r['max_overlap']:.2f}  [{r['n_tokens']:>2} tok] "
                  f"{r['file']}:{r['line']}  \"{r['example']}\""
                  f"  -> {r['best_test_id']}")
            if show_text and r["max_overlap"] >= 0.4:
                print(f"         best match: {r['best_test_text'][:110]}")
        print()

    table("FAIL  (overlap >= threshold, >= 3 tokens)", fails)
    table("EXEMPT (<= 2 content tokens: single names / short labels)", exempt,
          show_text=False)
    table("OK", oks, show_text=False)

    coincide = [r for r in rows if r["equals_gt_label"]]
    print(f"== Examples identical to some ground-truth label ({len(coincide)}) ==")
    for r in sorted(coincide, key=lambda x: x["example"].lower()):
        print(f"  \"{r['example']}\"  as {','.join(r['equals_gt_label'])}"
              f"  ({r['file']}:{r['line']})")
    print()

    print(f"SUMMARY: fail={len(fails)} exempt={len(exempt)} ok={len(oks)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"threshold": args.threshold, "rows": rows,
                       "summary": {"fail": len(fails), "exempt": len(exempt),
                                   "ok": len(oks)}},
                      f, ensure_ascii=False, indent=2)
        print(f"report written: {args.json}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
