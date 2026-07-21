"""
LLM-as-Judge Evaluation (Binary Method)
Best Practice: Binary(Pass/Fail) has highest reliability
- EvidentlyAI (2024)
- G-Eval (EMNLP 2023)
- MT-Bench (2023)
"""

import json
import sys
import ollama
from tqdm import tqdm
sys.path.append('.')
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from config import LLM_MODEL, LLM_HOST

class LLMJudgeBinary:
    """Binary (Pass/Fail) LLM Judge"""

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}

    def judge_element(self, gt_val, ext_val, element_type):
        """Binary judgment: PASS or FAIL"""
        cache_key = f"{element_type}:{gt_val}:{ext_val}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # If exact match, immediately PASS
        if str(gt_val).lower().strip() == str(ext_val).lower().strip():
            result = {'pass': True, 'reason': 'Exact match'}
            self.cache[cache_key] = result
            return result

        prompt = f"""You are a fair judge for knowledge graph evaluation.

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

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['message']['content'].strip().upper()

            is_pass = 'PASS' in content and 'FAIL' not in content.split('PASS')[0]

            result = {'pass': is_pass, 'reason': content[:100]}
            self.cache[cache_key] = result
            return result
        except:
            return {'pass': False, 'reason': 'Error'}

    def judge_quintuple(self, gt, ext):
        """Binary evaluation of entire quintuple"""
        fields = ['entity1', 'relation', 'entity2', 'time', 'location']
        results = {}

        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = ext.get(field, '')
            results[field] = self.judge_element(
                gt_val, ext_val, field
            )

        results['overall'] = {
            'pass': all(results[f]['pass'] for f in fields)
        }

        return results


def run_evaluation(data_file, normalize_relation=True,
                   normalize_entity=True):
    with open(data_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total = len(test_cases)
    judge = LLMJudgeBinary()
    extractor = STKGExtractor()

    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    passed = {f: 0 for f in fields + ['overall']}
    success = 0
    errors = []

    print(f"=== LLM-as-Judge Binary Evaluation ({total} items) ===", flush=True)
    print(f"Entity normalization: {normalize_entity}", flush=True)
    print(f"Relation normalization: {normalize_relation}\n", flush=True)

    for i, case in enumerate(tqdm(test_cases, desc="Evaluating")):
        quintuples = extractor.extract_quintuples(case['text'])

        if not quintuples:
            errors.append({
                'idx': i + 1,
                'text': case['text'][:60],
                'error': 'Extraction failed'
            })
            continue

        processed = postprocess_quintuples(
            quintuples,
            normalize_relation=normalize_relation,
            normalize_entity=normalize_entity
        )

        if not processed:
            continue

        ext = processed[0]
        gt = case['ground_truth']
        success += 1

        result = judge.judge_quintuple(gt, ext)

        case_errors = {}
        for field in fields:
            if result[field]['pass']:
                passed[field] += 1
            else:
                case_errors[field] = {
                    'gt': gt.get(field, ''),
                    'ext': ext.get(field, ''),
                    'reason': result[field]['reason'][:50]
                }

        if result['overall']['pass']:
            passed['overall'] += 1

        if case_errors:
            errors.append({
                'idx': i + 1,
                'text': case['text'][:60],
                'errors': case_errors
            })

    # Output results
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== LLM-as-Judge Binary Evaluation Results ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    print(f"Extraction success: {success}/{total} "
          f"({success/total*100:.1f}%)\n", flush=True)

    print(f"{'Element':<12} {'PASS':>8} {'FAIL':>8} {'PASS Rate':>10}",
          flush=True)
    print("-" * 42, flush=True)

    result_scores = {}
    for field in fields:
        p = passed[field]
        f_count = total - p
        rate = p / total * 100
        print(f"{field:<12} {p:>8} {f_count:>8} {rate:>9.1f}%",
              flush=True)
        result_scores[field] = rate

    p = passed['overall']
    f_count = total - p
    rate = p / total * 100
    print(f"{'overall':<12} {p:>8} {f_count:>8} {rate:>9.1f}%",
          flush=True)
    result_scores['overall'] = rate

    # Error details
    print(f"\n=== FAIL Details (Top 10) ===\n", flush=True)
    count = 0
    for err in errors:
        if 'errors' in err:
            count += 1
            if count > 10:
                break
            print(f"[{err['idx']}] {err['text']}...", flush=True)
            for field, detail in err['errors'].items():
                print(f"  {field}: FAIL", flush=True)
                print(f"    Ground truth: {detail['gt']}", flush=True)
                print(f"    Extracted: {detail['ext']}", flush=True)
            print(flush=True)

    # Save results
    output = {
        "method": "LLM-as-Judge Binary (Best Practice)",
        "total": total,
        "success": success,
        "scores": result_scores,
        "errors_count": len([e for e in errors if 'errors' in e])
    }

    with open(r'results\extracted\llm_judge_result.json',
              'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved successfully", flush=True)

if __name__ == "__main__":
    run_evaluation(r'data\raw_text_test.json')