# STELLAR

**Spatio-Temporal Extraction via LLM-Automated Reasoning**

A framework for automatically extracting spatio-temporal quintuples from unstructured text using LLMs.

---

## Overview

STELLAR extracts **spatio-temporal quintuples (Entity1, Relation, Entity2, Time, Location)** from unstructured text. Existing LLM-based knowledge graph construction focuses on triples or temporal quadruples; STELLAR extends this to include explicit spatial information.

### Key Ideas

- **Multi-stage CoT pipeline**: single-prompt extraction of five elements at once reaches only 2.0% exact match. Decomposing the task and adding Chain-of-Thought reasoning at each step raises it to 22.0%.
- **No dictionary required**: rule-based coders such as PETRARCH depend on hand-built CAMEO dictionaries and only process text within their coverage. STELLAR works without any dictionary.
- **Domain independent**: verified across 9 domains without prompt modification.
- **Model independent**: swapping only the LLM raises exact match from 51.0% to 60.0%.

---

## Results

### vs. Rule-based Method (PETRARCH)

Evaluated on the 155 samples where both systems produced output. Location is excluded because PETRARCH does not output spatial information.

| Element | PETRARCH | STELLAR | Diff |
|---------|----------|---------|------|
| Entity1 | 59.4% | 78.7% | +19.4%p |
| Relation | 76.1% | 72.9% | -3.2%p |
| Entity2 | 43.2% | 89.0% | +45.8%p |
| Time | 100.0% | 100.0% | 0.0%p |
| Quadruple exact match | 27.7% | 52.3% | +24.5%p |

**Extraction success rate (450 samples): PETRARCH 34.4%, STELLAR 99.8%**

PETRARCH fails on 295 of 450 texts because the actors and verbs are not in the CAMEO dictionary. The largest accuracy gap appears in Entity2: PETRARCH takes the grammatical object from the parse tree, while STELLAR identifies the semantic target through CoT reasoning.

### Component Contribution (n=50)

| Configuration | Exact match | Contribution |
|---------------|-------------|--------------|
| Direct extraction | 2.0% | - |
| + Atomic decomposition | 0.0% | -2.0%p |
| + Multi-stage | 0.0% | 0.0%p |
| + CoT reasoning | 22.0% | **+22.0%p** |
| + Entity normalization | 38.0% | **+16.0%p** |

Decomposition alone does not help. The gain comes from generating intermediate reasoning at each step.

### Domain Independence (9 domains, 450 samples, no prompt modification)

| Domain | Exact match |
|--------|-------------|
| Sports | 88% |
| Culture | 72% |
| Health | 62% |
| Politics | 56% |
| Business | 56% |
| Environment | 54% |
| Science | 50% |
| Disaster | 34% |
| Crime | 20% |

**Element average F1 87.0% | Quintuple text similarity F1 97.0% | Quintuple exact match 50.7%**

Time (99.8%) and Location (96.2%) are stable across all domains. Crime is limited by Entity1 accuracy (40%), as the LLM normalizes law-enforcement organizations to their parent countries.

### Model Comparison (same framework, only `_call_llm` replaced)

| Element | Qwen2.5-7B | GPT-4o-mini | GPT-4o |
|---------|------------|-------------|--------|
| Entity1 | 94.7% | 94.0% | 90.0% |
| Relation | 72.7% | 76.0% | 80.0% |
| Entity2 | 79.7% | 78.0% | 84.0% |
| Time | 97.7% | 100.0% | 100.0% |
| Location | 89.7% | 94.0% | 86.0% |
| Exact match | 51.0% ±0.8% | 60.0% | 60.0% |

Qwen2.5-7B result is the mean of 3 runs.

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
    + LLM-based normalization (Entity / Relation / Location)
    + Rule-based Entity2 correction
    |
    Spatio-Temporal Quintuples
    |
[Module 2] Knowledge Fusion
    Entity Canonicalization (cosine sim >= 0.9)
    Duplicate Removal (weighted cosine sim >= 0.85)
    |
[Output] Spatio-Temporal Knowledge Graph
```

Location is defined as the country where the event physically occurred or was initiated. For example, in "France provided military aid to Ukraine", Location is France, not Ukraine.

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
result = postprocess_quintuples(
    quintuples,
    normalize_relation=True,
    normalize_entity=True
)

print(result[0])
# {'entity1': 'United States', 'relation': 'Impose sanctions',
#  'entity2': 'Iran', 'time': '2023-03-15',
#  'location': 'United States'}
```

### Using a Different LLM

```python
from modules.extractor import STKGExtractor
import openai

class GPTExtractor(STKGExtractor):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.prompt_version = "v11_timeloc_fix"

    def _call_llm(self, prompt):
        r = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return r.choices[0].message.content

extractor = GPTExtractor(api_key="your-key")
```

### Knowledge Fusion

```python
from modules.fusion import KnowledgeFusion

fusion = KnowledgeFusion()
fused = fusion.fuse(all_quintuples)
```

### Graph Storage and Query

```bash
python main.py --sample 100 --files 5   # build STKG in Neo4j
python cli.py                            # natural language query
```

---

## Evaluation

```bash
# Core evaluation
python evaluation/llm_judge_eval.py       # binary judge
python evaluation/full_eval.py            # P/R/F1 + exact match
python evaluation/repeat_eval.py          # 3-run reproducibility

# Component analysis
python evaluation/ablation_eval.py        # component contribution
python evaluation/sc_eval.py              # self-consistency

# Generalization
python evaluation/domain_extra_eval.py    # 9 domains
python evaluation/api_compare_eval.py     # model comparison (needs OpenAI key)
python evaluation/bertscore_eval.py       # quintuple text similarity

# Rule-based comparison
python evaluation/compare_petrarch.py     # run PETRARCH (needs CoreNLP + PETRARCH3)
python evaluation/reeval_petrarch4.py     # 4-element comparison

# Figures
python generate_figures.py
```

### Reproducing the PETRARCH comparison

PETRARCH requires additional setup:

```bash
# 1. Stanford CoreNLP server
cd stanford-corenlp-4.5.7
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

# 2. PETRARCH3 (Python 3 port)
git clone https://github.com/oudalab/petrarch3
cd petrarch3 && pip install -e .
```

Note: PETRARCH3 contains several Python 2 to 3 conversion bugs that must be patched before running. See `evaluation/compare_petrarch.py` for details.

---

## Evaluation Details

### Metrics

| Metric | Description |
|--------|-------------|
| Element accuracy | Each of the five elements evaluated independently |
| Quintuple exact match | All five elements must pass simultaneously |
| Element average F1 | Mean of the five element-level scores |
| Quintuple text similarity | BERTScore between the serialized quintuple and ground truth |

### LLM-as-Judge Binary

- **PASS**: semantically equivalent (e.g. "Putin" = "Vladimir Putin")
- **FAIL**: semantically different, missing, or wrong type

Exact string matching would reject correct extractions that differ only in surface form, so a binary LLM judge is used instead.

### Known Limitations

- Judge and extractor use the same model in local experiments, introducing potential self-evaluation bias
- Ground truth was semi-automatically generated and then manually reviewed
- 450 samples across 9 domains; larger-scale validation is future work
- Quintuple exact match of 50.7% is not yet sufficient for fully automated deployment
- Entity normalization granularity is not domain-adaptive (see Crime domain results)

---

## Project Structure

```
stellar-stkg/
├── config_example.py
├── main.py                    # build STKG in Neo4j
├── cli.py                     # natural language query
├── generate_figures.py
├── requirements.txt
├── modules/                   # STELLAR framework
│   ├── extractor.py           # multi-stage CoT extraction
│   ├── postprocessor.py       # normalization + rule correction
│   ├── fusion.py              # canonicalization + deduplication
│   ├── graph.py               # Neo4j storage
│   └── query.py               # natural language query
├── evaluation/                # reproducible evaluation scripts
└── data/                      # 450 test samples, 9 domains
    ├── raw_text_test.json
    ├── domain_test.json
    └── domain_extra_test.json
```

---

## References

- Wei et al. (NeurIPS 2022) — Chain-of-Thought
- Wang et al. (ICLR 2023) — Self-Consistency
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
  author={Rhee, Wang-Jae and Oh, Seo-Jin and Lee, Kwang-Il},
  year={2026}
}
```

## License

MIT License

## Authors

Wang-Jae Rhee, Seo-Jin Oh, Kwang-Il Lee
Division of Artificial Intelligence Engineering, Korea Maritime and Ocean University
Cybermarine System LAB
