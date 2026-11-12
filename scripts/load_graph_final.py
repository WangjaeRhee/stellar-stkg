"""
Load the final-configuration quintuples into Neo4j:
  extraction file -> modules/fusion.py (KnowledgeFusion.fuse)
                  -> modules/graph.py  (STKGBuilder: clear_db, create_indexes, insert_batch)

Same order as main.py.  The database is CLEARED first (--no-clear to keep).

  python scripts/load_graph_final.py \
      --extract results/rerun/extract_qwen2.5-7b_final-loconly.jsonl
-> results/rerun/graph_load_final.json  (fusion stats, Neo4j stats, before/after counts)
   results/rerun/fused_qwen2.5-7b_final-loconly.json (fused quintuples as inserted)
"""

import argparse
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import RERUN_DIR, read_jsonl, write_json, configure_stdout  # noqa: E402
from modules.fusion import KnowledgeFusion  # noqa: E402
from modules.graph import STKGBuilder  # noqa: E402
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, FUSION_SIMILARITY_THRESHOLD  # noqa: E402


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default=os.path.join(
        RERUN_DIR, "extract_qwen2.5-7b_final-loconly.jsonl"))
    ap.add_argument("--no-clear", action="store_true")
    args = ap.parse_args()

    records = read_jsonl(args.extract)
    quintuples = []
    for r in records:
        proc = (r.get("processed_quintuples") or [None])[0]
        if proc:
            q = {k: proc.get(k, "") for k in
                 ("entity1", "relation", "entity2", "time", "location")}
            q["source_id"] = r["id"]
            q["domain"] = r["domain"]
            quintuples.append(q)
    print(f"input: {len(quintuples)} quintuples from {len(records)} records", flush=True)

    # Step 1: knowledge fusion (modules/fusion.py)
    fusion = KnowledgeFusion()
    fused = fusion.fuse(quintuples)
    fstats = fusion.get_stats(quintuples, fused)
    print(f"fusion: {fstats}", flush=True)
    fused_path = os.path.join(RERUN_DIR, "fused_qwen2.5-7b_final-loconly.json")
    write_json(fused_path, fused)

    # Step 2: graph construction (modules/graph.py)
    builder = STKGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    before = builder.get_stats()
    print(f"neo4j before: {before}", flush=True)
    if not args.no_clear:
        builder.clear_db()
    builder.create_indexes()
    ok, fail = builder.insert_batch(fused)
    after = builder.get_stats()
    builder.close()
    print(f"inserted ok={ok} fail={fail}", flush=True)
    print(f"neo4j after: {after}", flush=True)

    out = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "extract_file": os.path.basename(args.extract),
        "neo4j_uri": NEO4J_URI,
        "cleared_first": not args.no_clear,
        "fusion": {"model": "all-MiniLM-L6-v2", "canonicalize_threshold": 0.9,
                   "dedup_threshold": FUSION_SIMILARITY_THRESHOLD, **fstats},
        "insert": {"ok": ok, "fail": fail},
        "neo4j_before_clear": before,
        "neo4j_after": after,
        "fused_file": os.path.basename(fused_path),
    }
    write_json(os.path.join(RERUN_DIR, "graph_load_final.json"), out)
    print("written: results/rerun/graph_load_final.json", flush=True)


if __name__ == "__main__":
    main()
