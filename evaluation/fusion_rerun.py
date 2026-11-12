"""
Run modules/fusion.py (KnowledgeFusion) on a saved extraction file and count
what the graph builder (modules/graph.py) would create, without a Neo4j DB.

Input quintuples: the first processed quintuple of every sentence (the one
that is evaluated), i.e. 450 quintuples for the full test set.

Graph counting mirrors STKGBuilder.insert_quintuple:
  MERGE (e1:Entity {name}) MERGE (e2:Entity {name}) CREATE (e1)-[:RELATION]->(e2)
  -> nodes = distinct non-empty entity names (exact string match),
     edges = quintuples with both entity1 and entity2 non-empty.

  python evaluation/fusion_rerun.py --extract results/rerun/extract_qwen2.5-7b_final-loconly.jsonl
-> results/rerun/fusion_<extract stem>.json / .md
"""

import argparse
import datetime
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.dataset import read_jsonl, write_json, configure_stdout  # noqa: E402
from modules.fusion import KnowledgeFusion  # noqa: E402
from config import FUSION_SIMILARITY_THRESHOLD  # noqa: E402


def graph_counts(quintuples):
    nodes, edges = set(), 0
    for q in quintuples:
        e1, e2 = q.get("entity1", ""), q.get("entity2", "")
        if not e1 or not e2:
            continue
        nodes.add(e1)
        nodes.add(e2)
        edges += 1
    locs = {str(q.get("location", "unknown")) for q in quintuples
            if q.get("entity1") and q.get("entity2")}
    times = {str(q.get("time", "unknown")) for q in quintuples
             if q.get("entity1") and q.get("entity2")}
    return {"nodes": len(nodes), "edges": edges,
            "unique_locations": len(locs), "unique_times": len(times)}


def main():
    configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = read_jsonl(args.extract)
    quintuples = []
    for r in records:
        proc = (r.get("processed_quintuples") or [None])[0]
        if proc:
            q = {k: proc.get(k, "") for k in
                 ("entity1", "relation", "entity2", "time", "location")}
            q["_id"] = r["id"]
            q["_domain"] = r["domain"]
            quintuples.append(q)
    print(f"records={len(records)} input quintuples={len(quintuples)}", flush=True)

    fusion = KnowledgeFusion()  # threshold from config
    # Step 1 (canonicalize) and Step 2 (deduplicate) separately so both can be reported
    canon = fusion.canonicalize_entities(quintuples)
    fused = fusion.deduplicate(canon)

    # --- canonicalization details
    merged_pairs = []
    groups = defaultdict(set)
    for q in canon:
        for f in ("entity1", "entity2"):
            if f + "_original" in q and q[f + "_original"] != q[f]:
                groups[q[f]].add(q[f + "_original"])
    for canonical, members in groups.items():
        for m in sorted(members):
            merged_pairs.append({"from": m, "to": canonical})
    n_entities_before = len({q[f] for q in quintuples for f in ("entity1", "entity2") if q[f]})
    n_entities_after = len({q[f] for q in canon for f in ("entity1", "entity2") if q[f]})

    # --- dedup details
    dup_groups = [q for q in fused if q.get("merge_count", 1) > 1]
    removed = len(canon) - len(fused)
    dup_examples = []
    for q in dup_groups:
        dup_examples.append({"kept": {k: q[k] for k in ("entity1", "relation", "entity2", "time", "location")},
                             "merge_count": q["merge_count"], "kept_id": q.get("_id")})

    g_before = graph_counts(quintuples)
    g_after = graph_counts(fused)

    result = {
        "extract_file": os.path.basename(args.extract),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "fusion_model": "all-MiniLM-L6-v2",
        "canonicalize_threshold": 0.9,
        "dedup_threshold": FUSION_SIMILARITY_THRESHOLD,
        "quintuples_before": len(quintuples),
        "quintuples_after_canonicalize": len(canon),
        "quintuples_after": len(fused),
        "duplicates_removed": removed,
        "duplicate_groups": len(dup_groups),
        "entities_before": n_entities_before,
        "entities_after_canonicalize": n_entities_after,
        "entity_merge_groups": len(groups),
        "entity_pairs_merged": len(merged_pairs),
        "merged_pairs": merged_pairs,
        "merge_groups": {k: sorted(v) for k, v in groups.items()},
        "duplicate_examples": dup_examples,
        "graph_before_fusion": g_before,
        "graph_after_fusion": g_after,
        "graph_count_method": "code count mirroring STKGBuilder.insert_quintuple "
                              "(MERGE Entity by exact name, CREATE one RELATION edge per quintuple with both entities non-empty); Neo4j not executed",
    }
    stem = args.out or os.path.join(
        os.path.dirname(args.extract),
        "fusion_" + os.path.basename(args.extract).replace("extract_", "", 1).replace(".jsonl", ""))
    write_json(stem + ".json", result)

    md = [f"# Knowledge fusion - {os.path.basename(args.extract)}", "",
          f"- input quintuples: {len(quintuples)} (first quintuple per sentence)",
          f"- entity canonicalization (cos >= 0.9): {len(groups)} groups, "
          f"{len(merged_pairs)} entity strings merged into another; distinct entities {n_entities_before} -> {n_entities_after}",
          f"- deduplication (cos >= {FUSION_SIMILARITY_THRESHOLD} on 'e1 rel e2 time'): "
          f"{len(dup_groups)} groups, {removed} quintuples removed; {len(canon)} -> {len(fused)}",
          "", "## Merged entity pairs (first 10)", "", "| from | to (representative = longest) |", "|---|---|"]
    for p in merged_pairs[:10]:
        md.append(f"| {p['from']} | {p['to']} |")
    md += ["", "## Duplicate groups", "", "| kept quintuple | merge_count |", "|---|---|"]
    for d in dup_examples:
        k = d["kept"]
        md.append(f"| {k['entity1']} / {k['relation']} / {k['entity2']} / {k['time']} / {k['location']} | {d['merge_count']} |")
    md += ["", "## Graph size (code count, Neo4j not executed)", "",
           "| | nodes | edges | unique locations | unique times |", "|---|---|---|---|---|",
           f"| before fusion | {g_before['nodes']} | {g_before['edges']} | {g_before['unique_locations']} | {g_before['unique_times']} |",
           f"| after fusion | {g_after['nodes']} | {g_after['edges']} | {g_after['unique_locations']} | {g_after['unique_times']} |"]
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"written: {stem}.json / {stem}.md")


if __name__ == "__main__":
    main()
