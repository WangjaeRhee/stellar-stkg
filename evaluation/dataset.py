"""
Shared helpers for the clean re-run (results/rerun).

- load_all_cases(): the 450 test sentences with stable ids
    raw000..raw049  data/raw_text_test.json      (politics)
    dom000..dom059  data/domain_test.json        (business / disaster)
    ext000..ext339  data/domain_extra_test.json  (7 domains + extra biz/disaster)
- jsonl helpers with resume support
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RERUN_DIR = os.path.join(ROOT, "results", "rerun")

FIELDS = ["entity1", "relation", "entity2", "time", "location"]
DOMAINS = ["politics", "business", "disaster", "sports", "science",
           "health", "culture", "crime", "environment"]

DATA_FILES = [
    ("raw", os.path.join("data", "raw_text_test.json"), "politics"),
    ("dom", os.path.join("data", "domain_test.json"), None),
    ("ext", os.path.join("data", "domain_extra_test.json"), None),
]


def configure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_all_cases():
    cases = []
    for prefix, rel_path, default_domain in DATA_FILES:
        path = os.path.join(ROOT, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for i, item in enumerate(items):
            cases.append({
                "id": f"{prefix}{i:03d}",
                "domain": item.get("domain", default_domain) or default_domain,
                "text": item["text"],
                "ground_truth": item["ground_truth"],
                "source_file": rel_path.replace("\\", "/"),
                "source_index": i,
            })
    return cases


def load_ids(path):
    """Accepts a json file containing either a list or {"ids": [...]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["ids"]
    return list(data)


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", model).strip("-")


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def existing_ids(path):
    return {r["id"] for r in read_jsonl(path)}


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
