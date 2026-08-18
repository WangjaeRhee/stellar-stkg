# STELLAR

**Spatio-Temporal Extraction via LLM-Automated Reasoning for Knowledge Graph Construction**

LLM-based Spatio-Temporal Quintuple Automatic Extraction Framework

---

## Overview

A framework that automatically extracts **spatio-temporal quintuples (Entity1, Relation, Entity2, Time, Location)** from unstructured text to build STKG.

Previous research has been limited to triples (Entity, Relation, Entity) or quadruples (+Time), and **this is the first automatic extraction of quintuples including Location**.

### Key Features

- **Quintuple Extraction**: Spatio-temporal knowledge representation including Location information
- **Multi-stage CoT Extraction**: Atomic Decomposition → Entity(CoT) → Relation → Time/Location
- **Domain Independent**: Verified to work across 8 domains without prompt modification
- **Model Independent**: Performance improvement by simply switching LLMs (Qwen 7B → GPT-4o)
- **LLM-based Normalization**: Automatic normalization of Entity/Relation/Location

### Performance

| Model | entity1 | relation | entity2 | time | location | overall |
|-------|---------|----------|---------|------|----------|---------|
| Qwen2.5-7B | 94.7% | 72.7% | 79.7% | 97.7% | 89.7% | 51.0% |
| GPT-4o-mini | 94.0% | 76.0% | 78.0% | 100% | 94.0% | 60.0% |
| GPT-4o | 90.0% | 80.0% | 84.0% | 100% | 86.0% | 60.0% |

*Evaluation: LLM-as-Judge Binary, 3-run average for Qwen 7B (51.0% ± 0.8%)*

---

## Architecture

```
[Input] Unstructured Text
    ↓
[Module 1] Knowledge Extraction
    Step 1: Atomic Fact Decomposition (ATOM)
    Step 2: Entity Extraction (CoT reasoning)
    Step 3: Relation Extraction
    Step 4: Time/Location Extraction (from original text)
    + LLM Normalization + Rule-based Correction
    ↓
[Module 2] Knowledge Fusion
    Entity Canonicalize (EDC) + Cosine Dedup (ATOM)
    ↓
[Module 3] STKG Construction
    Neo4j Graph Database
    ↓
[Module 4] CLI Query
    Natural Language → STKG Search → LLM Answer
```

---

## Installation

### Requirements

- Python 3.10+
- Ollama with Qwen2.5-7B (or any LLM)
- Neo4j Desktop 2.2.0+
- GPU: 12GB+ VRAM recommended

### Setup

```bash
git clone https://github.com/WangjaeRhee/stellar-stkg.git
cd stellar-stkg

conda create -n stkg_project python=3.10 -y
conda activate stkg_project
pip install -r requirements.txt

# Install and run Ollama (https://ollama.com)
ollama pull qwen2.5:7b

# Configure
cp config_example.py config.py
# Edit config.py with your Neo4j password and Ollama host
```

---

## Usage

### Full Pipeline (Structured Data)

```bash
python main.py --sample 100 --files 5
```

### CLI Query

```bash
python cli.py

# Example queries:
# > What did Russia do?
# > Who was involved in events in Ukraine?
# > 2022년 10월에 어떤 일이 있었나요?
```

### Unstructured Text Extraction

```python
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples

extractor = STKGExtractor()
text = "Russia launched missiles at Ukraine on October 8, 2022."

quintuples = extractor.extract_quintuples(text)
processed = postprocess_quintuples(
    quintuples,
    normalize_relation=True,
    normalize_entity=True
)

print(processed[0])
# {'entity1': 'Russia', 'relation': 'Launch missiles',
#  'entity2': 'Ukraine', 'time': '2022-10-08',
#  'location': 'Ukraine'}
```

### Using Different LLM (e.g., GPT-4o)

```python
from modules.extractor import STKGExtractor
import openai

class GPTExtractor(STKGExtractor):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.prompt_version = "v11_timeloc_fix"

    def _call_llm(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

extractor = GPTExtractor(api_key="your-key")
quintuples = extractor.extract_quintuples(text)
```

---

## Evaluation

```bash
# Main evaluation (LLM-as-Judge Binary)
python evaluation/llm_judge_eval.py

# 3-run reproducibility
python evaluation/repeat_eval.py

# Ablation study
python evaluation/ablation_eval.py

# ATOM comparison (requires itext2kg)
python evaluation/compare_atom.py

# API model comparison (requires OpenAI API key)
python evaluation/api_compare_eval.py

# Domain tests
python evaluation/domain_eval.py        # 3 domains
python evaluation/domain_extra_eval.py  # 8 domains

# Self-Consistency
python evaluation/sc_eval.py
```

---

## Experimental Results

### ATOM Comparison (Same model: Qwen 7B, n=50)

| Metric | ATOM | STELLAR | Diff |
|--------|------|---------|------|
| entity1 | 64.0% | 98.0% | +34.0% |
| relation | 40.0% | 72.0% | +32.0% |
| entity2 | 18.0% | 76.0% | +58.0% |
| time | 82.0% | 100.0% | +18.0% |
| location | N/A | 84.0% | - |
| **overall** | **8.0%** | **50.0%** | **+42.0%** |

### Ablation Study (n=50, Binary)

| Configuration | Overall |
|--------------|---------|
| Direct extraction | 2.0% |
| + Atomic Decomposition | 0.0% |
| + Multi-stage | 0.0% |
| + CoT reasoning | 22.0% (+22%p) |
| + Entity normalization | 38.0% (+16%p) |

### Domain Independence (8 domains, no prompt modification)

| Domain | Overall | # Samples |
|--------|---------|-----------|
| Politics/Military | 51.0% | 100 |
| Sports | 50.0% | 10 |
| Business | 43.3% | 30 |
| Health | 40.0% | 10 |
| Crime | 40.0% | 10 |
| Disaster | 36.7% | 30 |
| Culture | 20.0% | 10 |
| Science | 10.0% | 10 |

---

## Project Structure

```
stellar-stkg/
├── config_example.py       # Configuration template
├── main.py                 # Full pipeline (structured data)
├── cli.py                  # CLI query interface
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
│   ├── extractor.py        # Multi-stage CoT extraction (v11)
│   ├── postprocessor.py    # LLM normalization + rule correction
│   ├── fusion.py           # Entity canonicalize + dedup
│   ├── graph.py            # Neo4j STKG storage
│   └── query.py            # Natural language query
├── evaluation/
│   ├── llm_judge_eval.py   # LLM-as-Judge Binary
│   ├── repeat_eval.py      # 3-run reproducibility
│   ├── ablation_eval.py    # Ablation study
│   ├── compare_atom.py     # ATOM comparison
│   ├── api_compare_eval.py # API model comparison
│   ├── domain_eval.py      # Domain test (3 domains)
│   ├── domain_extra_eval.py# Domain test (8 domains)
│   └── sc_eval.py          # Self-Consistency
└── data/
    ├── raw_text_test.json  # 100 unstructured text samples
    ├── domain_test.json    # 60 domain samples (business/disaster)
    └── domain_extra_test.json # 50 domain samples (5 new domains)
```

---

## References

- ATOM (Lairgi et al., EACL 2026) - Atomic Fact Decomposition
- EDC (Zhang & Soh, EMNLP 2024) - Entity Canonicalize
- Wei et al. (NeurIPS 2022) - Chain-of-Thought
- Wang et al. (ICLR 2023) - Self-Consistency
- G-Eval (Liu et al., EMNLP 2023) - LLM-as-Judge

## Citation

```bibtex
@article{rhee2026stellar,
  title={STELLAR: Spatio-Temporal Quintuple Extraction Framework via LLM-Automated Reasoning},
  author={Wangjae Rhee and Kwangil Lee},
  year={2026}
}
```

## License

MIT License

## Author

Wangjae Rhee
School of Artificial Intelligence, Korea Maritime & Ocean University
Cybermarine System LAB
