
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
