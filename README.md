# STELLAR

**Spatio-Temporal Extraction via LLM-Automated Reasoning**

A framework for automatically extracting spatio-temporal quintuples from unstructured text using large language models.

---

## Overview

STELLAR extracts **spatio-temporal quintuples (Entity1, Relation, Entity2, Time, Location)** from unstructured text and builds a spatio-temporal knowledge graph (STKG). Existing LLM-based knowledge graph construction focuses on triples or temporal quadruples; STELLAR extends this to include explicit spatial information.

### Key Ideas

- **Multi-stage CoT pipeline** — extraction is decomposed into four stages (atomic fact decomposition, entity extraction, relation extraction, time/location extraction) with Chain-of-Thought reasoning at each stage, so that the subject and object are identified from the meaning of the sentence rather than its surface structure.
- **No dictionary required** — rule-based coders such as PETRARCH depend on hand-built CAMEO dictionaries and only process text within their coverage. STELLAR works without any dictionary.
- **Location included** — the spatial element is extracted together with the other four, so no separate geocoder is needed to form a quintuple.
- **Domain independent** — evaluated on 9 domains without prompt modification.

---

## Results

All numbers below are from the submitted manuscript: Qwen2.5-7B (Q4_K_M, Ollama), temperature 0, fixed seed, 450 texts.

### vs. Rule-based Method (PETRARCH)

Evaluated on the 155 texts for which PETRARCH produced an event (STELLAR produced a quintuple for all 155). Location is excluded because PETRARCH does not output it, and Time is excluded because PETRARCH takes the date as input.

| Element | PETRARCH | STELLAR | Diff. |
|---------|----------|---------|-------|
| Entity1 | 60.6 | 93.5 | +32.9 |
| Relation | 75.5 | 80.6 | +5.1 |
| Entity2 | 44.5 | 89.7 | +45.2 |
| Exact match (3 elem.) | 29.0 | 69.7 | +40.7 |

The largest gap is in Entity2: PETRARCH takes the grammatical object from the parse tree, while STELLAR identifies the semantic target through CoT reasoning.

3-element exact match by domain (%):

| Domain | n | PETRARCH | STELLAR |
|--------|---|----------|---------|
| Politics | 30 | 20.0 | 40.0 |
| Business | 14 | 71.4 | 50.0 |
| Disaster | 9 | 11.1 | 33.3 |
| Sports | 12 | 66.7 | 100.0 |
| Science | 9 | 44.4 | 77.8 |
| Health | 23 | 4.3 | 69.6 |
| Culture | 6 | 0.0 | 83.3 |
| Crime | 33 | 9.1 | 84.8 |
| Environment | 19 | 63.2 | 94.7 |

The comparison uses data that was not written in the CAMEO schema, so PETRARCH's numbers may differ from its performance under the conditions it was designed for.

### Domain Generalization (9 domains, n=50 each, no prompt modification)

| Element | Pol. | Bus. | Dis. | Spo. | Sci. | Hea. | Cul. | Cri. | Env. | All |
|---------|------|------|------|------|------|------|------|------|------|-----|
| Entity1 | 80 | 92 | 80 | 96 | 90 | 98 | 98 | 100 | 100 | 92.7 |
| Relation | 70 | 72 | 68 | 94 | 74 | 80 | 88 | 92 | 72 | 78.9 |
| Entity2 | 60 | 84 | 74 | 96 | 88 | 90 | 88 | 96 | 94 | 85.6 |
| Time | 100 | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99.8 |
| Location | 90 | 96 | 98 | 96 | 98 | 96 | 96 | 100 | 98 | 96.4 |
| Exact match | 32 | 60 | 44 | 90 | 66 | 68 | 82 | 88 | 68 | 66.4 |

**Element average accuracy 90.7% | Quintuple exact match 66.4% | Quintuple text similarity (BERTScore F1) 98.0%**

Politics is the lowest because its ground truth writes actors at country level ("Ukraine") while the extraction keeps the wording of the text ("Ukrainian troops"). BERTScore is inflated by the identically formatted time and location elements and should be read as an auxiliary metric.

### Knowledge Fusion and Graph Construction

Applied to the 450 extracted quintuples:

| Step | Result |
|------|--------|
| Entity canonicalization (cosine sim >= 0.9) | 17 entity strings merged, 527 -> 510 distinct entities |
| Duplicate removal (weighted cosine sim >= 0.85) | 44 quintuples removed, 450 -> 406 |
| Graph | 497 nodes, 406 edges |

---

## Architecture

```
[Input] Unstructured Text
    |
[Module 1] Knowledge Extraction
    Step 1: Atomic Fact Decomposition
    Step 2: Entity Extraction (CoT reasoning)
    Step 3: Relation Extraction
    Step 4: Time / Location Extraction (from original text)
    + Location normalization to country level (LLM)
    |
    Spatio-Temporal Quintuples
    |
[Module 2] Knowledge Fusion
    Entity Canonicalization (cosine sim >= 0.9)
    Duplicate Removal (weighted cosine sim >= 0.85)
    |
[Output] Spatio-Temporal Knowledge Graph (Neo4j)
```

Location is defined as the country where the event physically occurred or was initiated. For example, in "France provided military aid to Ukraine", Location is France, not Ukraine.

Entity1, Relation and Entity2 keep the wording of the source text; unifying different surface forms of the same entity is left to the fusion module.

---

## Installation

### Requirements

- Python 3.10+
- Ollama with Qwen2.5-7B (or any LLM)
- Neo4j Desktop 2.2.0+ (optional, for graph storage)
- GPU: 12GB+ VRAM recommended

### Setup

```bash
git clone https://github.com/WangjaeRhee/stellar-stkg.git
cd stellar-stkg

conda create -n stellar python=3.10 -y
conda activate stellar
pip install -r requirements.txt

ollama pull qwen2.5:7b

cp config_example.py config.py
# Edit config.py: Ollama host, Neo4j credentials
```

---

## Usage

### Quintuple Extraction

```python
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples

extractor = STKGExtractor()
text = "The United States imposed sanctions on Iran on March 15, 2023."

quintuples = extractor.extract_quintuples(text)

# Configuration used in the paper: location normalization only
result = postprocess_quintuples(
    quintuples,
    normalize_relation=False,
    normalize_entity=False,
    apply_corrections=False,
)
print(result[0])   # entity1, relation, entity2, time, location
```

### Knowledge Fusion

```python
from modules.fusion import KnowledgeFusion

fusion = KnowledgeFusion()
fused = fusion.fuse(all_quintuples)
```

### Graph Storage and Query

```bash
# fuse the extracted quintuples and load them into Neo4j (clears the DB first)
python scripts/load_graph_final.py --extract results/rerun/extract_qwen2.5-7b_final-loconly.jsonl

python cli.py    # natural language query over the STKG
```

Because time and location are stored as edge attributes, queries such as "what did entity X do, when and where" are supported, which triple-based knowledge graphs cannot express.

---

## Reproducing the Paper Results

Result files are not tracked; every script writes to `results/rerun/` and is resumable (ids already present in an output file are skipped). Run from the project root with Ollama serving `qwen2.5:7b`.

```bash
R=results/rerun

# 0. prompt examples must not overlap the 450 test texts
python scripts/check_leakage.py --json $R/leakage_report.json

# 1. extraction (450 texts)
python evaluation/run_extract.py --model qwen2.5:7b --no-show

# 2. final configuration (location normalization only) + judge + tables
python evaluation/run_judge.py --extract $R/extract_qwen2.5-7b.jsonl \
    --use-original entity1 relation entity2 \
    --out $R/judge_qwen2.5-7b_norm-loc.jsonl --cache $R/judge_cache.json
python evaluation/make_final_extract.py
python evaluation/summarize.py --judge "$R/judge_qwen2.5-7b_norm-loc.jsonl=Final" \
    --out $R/summary_final --title "Final configuration (450)"      # Table 4

# 3. quintuple text similarity
python evaluation/bertscore_rerun.py --extract $R/extract_qwen2.5-7b_final-loconly.jsonl

# 4. PETRARCH comparison on the 155 PETRARCH-coded texts                # Tables 2, 3
python evaluation/compare_petrarch_final.py

# 5. knowledge fusion counts, then Neo4j load
python evaluation/fusion_rerun.py --extract $R/extract_qwen2.5-7b_final-loconly.jsonl
python scripts/load_graph_final.py --extract $R/extract_qwen2.5-7b_final-loconly.jsonl
```

`scripts/rerun_all.ps1` chains the extraction and judging steps together with additional analyses that are not reported in the paper (component analysis on a fixed 90-text sample, model comparison, normalizer and rule on/off comparison, human-check sheet).

### PETRARCH output

`results/petrarch/evts.petr_out.txt` is the PETRARCH3 output used in the paper and is kept in the repository, because regenerating it needs extra setup:

```bash
# 1. Stanford CoreNLP server
cd stanford-corenlp-4.5.7
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

# 2. PETRARCH3 (Python 3 port)
git clone https://github.com/oudalab/petrarch3
cd petrarch3 && pip install -e .
```

PETRARCH3 contains several Python 2 to 3 conversion bugs that must be patched before running. `evaluation/compare_petrarch.py` builds the PETRARCH input from the 450 texts and runs it (set `PETRARCH_DIR` in the script to your checkout).

---

## Evaluation Details

### Metrics

| Metric | Description |
|--------|-------------|
| Element accuracy | Each of the five elements judged independently |
| Quintuple exact match | All five elements must pass simultaneously |
| Quintuple text similarity | BERTScore between the serialized quintuple and ground truth |

For each text, the first extracted quintuple is compared with the ground truth.

### LLM-as-Judge Binary

- **PASS** — semantically equivalent (a different spelling of the same person, country or organization; a different wording of the same action; a sub-entity of the same parent entity)
- **FAIL** — semantically different, missing, or wrong type

All tables are produced by a single shared judge (`evaluation/judge.py`) with a verdict cache, so identical (ground truth, extracted) pairs receive identical verdicts across experiments.

### Dataset

450 English news-style texts, 50 per domain across 9 domains: politics, business, disaster, sports, science, health, culture, crime, environment. Texts and ground-truth quintuples were generated with an LLM from real events between 2018 and 2024, as single sentences of 20-40 words, with time in `YYYY-MM-DD` and location at country level. The few-shot examples in the prompts do not overlap the dataset (`scripts/check_leakage.py`).

### Known Limitations

- The extractor and the judge use the same LLM
- The dataset is LLM-generated, consists of short single sentences, and was not separately reviewed
- Location is limited to country level
- The fusion module sometimes merges distinct entities or events, because differences in time are not weighted strongly enough at the current thresholds
- Accuracy drops where the ground truth follows a different naming convention from the source text (e.g. politics)

---

## Project Structure

```
stellar-stkg/
├── config_example.py
├── main.py                        # ICEWS-style structured data -> STKG in Neo4j
├── cli.py                         # natural language query
├── requirements.txt
├── modules/                       # STELLAR framework
│   ├── extractor.py               # multi-stage CoT extraction
│   ├── postprocessor.py           # location / entity / relation normalization
│   ├── fusion.py                  # canonicalization + deduplication
│   ├── graph.py                   # Neo4j storage
│   ├── query.py                   # natural language query
│   └── llm_common.py              # deterministic LLM options
├── evaluation/
│   ├── dataset.py                 # 450 test cases with stable ids, jsonl helpers
│   ├── run_extract.py             # extraction only
│   ├── judge.py / run_judge.py    # shared binary judge
│   ├── summarize.py               # tables from judge files
│   ├── make_final_extract.py      # final-configuration extraction file
│   ├── bertscore_rerun.py         # quintuple text similarity
│   ├── compare_petrarch.py        # build PETRARCH input and run it
│   ├── compare_petrarch_final.py  # PETRARCH vs STELLAR on the 155-subset
│   ├── fusion_rerun.py            # fusion and graph counts
│   └── ...                        # additional analyses (ablation, sample90, human check)
├── scripts/
│   ├── check_leakage.py           # prompt example vs test set overlap
│   ├── load_graph_final.py        # fusion -> Neo4j
│   └── rerun_all.ps1              # chained pipeline (PowerShell)
├── data/                          # 450 test samples, 9 domains
│   ├── raw_text_test.json
│   ├── domain_test.json
│   └── domain_extra_test.json
└── results/petrarch/evts.petr_out.txt
```

---

## References

- Wei et al. (NeurIPS 2022) — Chain-of-Thought
- Lairgi et al. (EACL 2026) — Atomic Fact Decomposition
- Zhang & Soh (EMNLP 2024) — Entity Canonicalization
- Norris et al. (JOSS 2017) — PETRARCH2
- Schrodt (2012) — CAMEO codebook
- Liu et al. (EMNLP 2023) — G-Eval
- Zhang et al. (ICLR 2020) — BERTScore

## Citation

```bibtex
@article{rhee2026stellar,
  title={Spatio-Temporal Quintuple Extraction Framework for
         Knowledge Graph Construction based on LLM-Automated Reasoning},
  author={Rhee, Wang-Jae},
  year={2026}
}
```

## License

MIT License

## Authors

Wang-Jae Rhee

Division of Artificial Intelligence Engineering, Korea Maritime and Ocean University
