"""
STKG Automatic Construction Pipeline
Spatio-Temporal Knowledge Graph Construction Framework
Leveraging the Complementary Relationship between LLM and STKG

Usage:
    python main.py
    python main.py --sample 1000 --files 10
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from modules.fusion import KnowledgeFusion
from modules.graph import STKGBuilder
from config import (
    DATA_DIR, OUTPUT_DIR, SAMPLE_SIZE, MAX_FILES,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    LLM_MODEL, LLM_HOST
)

def load_icews_files(data_dir, max_files=None):
    """
    Load ICEWS .tab files
    Args:
        data_dir: Data directory path
        max_files: Maximum number of files (None for all)
    Returns:
        DataFrame: Loaded data
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.tab')
    ])
    if max_files:
        files = files[:max_files]

    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            df = pd.read_csv(
                path, sep='\t',
                encoding='utf-8',
                on_bad_lines='skip'
            )
            dfs.append(df)
            print(f"  Loaded: {f} ({len(df)} rows)")
        except Exception as e:
            print(f"  Failed to load: {f} → {e}")

    if not dfs:
        raise ValueError("No files were loaded.")

    return pd.concat(dfs, ignore_index=True)

def run_pipeline(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    sample_size=SAMPLE_SIZE,
    max_files=MAX_FILES,
    model=LLM_MODEL,
    host=LLM_HOST,
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    clear_db=True
):
    """
    Execute the complete STKG construction pipeline
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  STKG Automatic Construction Pipeline")
    print(f"  Model: {model}")
    print(f"  Samples: {sample_size} / Files: {max_files}")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading data...")
    df = load_icews_files(data_dir, max_files=max_files)
    df = df.head(sample_size)
    print(f"Loaded {len(df)} rows total")

    # Module 1: Knowledge Extraction
    print(f"\n[2/4] Module 1 — Knowledge Extraction...")
    extractor = STKGExtractor(model=model, host=host)
    raw_quintuples = []
    results = []
    success = 0
    fail = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            extracted = extractor.extract_from_icews_row(row)

            if isinstance(extracted, list) and len(extracted) > 0:
                processed = postprocess_quintuples(
                    extracted,
                    normalize_relation=False,
                    normalize_entity=False
                )
            else:
                processed = []

            record = {
                "row_id": int(i),
                "ground_truth": {
                    "entity1": str(row.get('Source Name', '')),
                    "relation": str(row.get('Event Text', '')),
                    "entity2": str(row.get('Target Name', '')),
                    "time": str(row.get('Event Date', '')),
                    "location": str(row.get('Country', 'unknown'))
                },
                "extracted": processed
            }
            results.append(record)

            if processed:
                raw_quintuples.extend(processed)
                success += 1
            else:
                fail += 1

        except Exception as e:
            print(f"\nError processing row {i}: {e}")
            fail += 1

    print(f"Extraction success: {success}/{len(df)} ({success/len(df)*100:.1f}%)")
    print(f"Extraction failure: {fail}/{len(df)} ({fail/len(df)*100:.1f}%)")

    # Save results
    raw_path = os.path.join(output_dir, 'pipeline_raw.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved: {raw_path}")

    # Module 2: Knowledge Fusion
    print(f"\n[3/4] Module 2 — Knowledge Fusion...")
    fusion = KnowledgeFusion()
    fused = fusion.fuse(raw_quintuples)
    stats = fusion.get_stats(raw_quintuples, fused)

    print(f"Before fusion: {stats['before']} items")
    print(f"After fusion: {stats['after']} items")
    print(f"Duplicates removed: {stats['removed']} items ({stats['reduction_rate']:.1f}%)")

    fused_path = os.path.join(output_dir, 'pipeline_fused.json')
    with open(fused_path, 'w', encoding='utf-8') as f:
        json.dump(fused, f, ensure_ascii=False, indent=2)
    print(f"Fused results saved: {fused_path}")

    # Module 3: STKG Construction
    print(f"\n[4/4] Module 3 — STKG Construction...")
    try:
        builder = STKGBuilder(neo4j_uri, neo4j_user, neo4j_password)
        if clear_db:
            builder.clear_db()
        builder.create_indexes()
        s, f_count = builder.insert_batch(fused)
        neo4j_stats = builder.get_stats()
        builder.close()

        print(f"Saved successfully: {s} / Failed: {f_count}")
        print(f"Number of nodes: {neo4j_stats['nodes']}")
        print(f"Number of edges: {neo4j_stats['edges']}")
    except ConnectionError as e:
        print(f"Neo4j connection failed: {e}")
        print("Please check if Neo4j is running.")
        neo4j_stats = {"nodes": 0, "edges": 0}

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print(f"  Total extracted:    {len(raw_quintuples)} items")
    print(f"  After fusion:       {len(fused)} items")
    print(f"  STKG nodes:         {neo4j_stats['nodes']}")
    print(f"  STKG edges:         {neo4j_stats['edges']}")
    print("=" * 60)

    return results, fused, neo4j_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='STKG Automatic Construction Pipeline'
    )
    parser.add_argument(
        '--sample', type=int, default=SAMPLE_SIZE,
        help=f'Number of samples (default: {SAMPLE_SIZE})'
    )
    parser.add_argument(
        '--files', type=int, default=MAX_FILES,
        help=f'Number of files (default: {MAX_FILES})'
    )
    parser.add_argument(
        '--no-clear', action='store_true',
        help='Skip database initialization'
    )
    args = parser.parse_args()

    run_pipeline(
        sample_size=args.sample,
        max_files=args.files,
        clear_db=not args.no_clear
    )