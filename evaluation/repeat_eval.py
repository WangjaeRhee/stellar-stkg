"""
3회 반복 실험 - 재현성 검증
llm_judge_eval.py와 동일한 Judge 사용
"""

import json
import sys
import numpy as np
import ollama
from tqdm import tqdm
sys.path.append('.')
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from config import LLM_MODEL, LLM_HOST

client = ollama.Client(host=LLM_HOST)

def binary_judge(gt_val, ext_val, element_type):
    """llm_judge_eval.py와 동일한 Judge"""
    if str(gt_val).lower().strip() == str(ext_val).lower().strip():
        return True

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

Answer with ONLY "PASS" or "FAIL":"""

    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip().upper()
        return 'PASS' in content and \
               'FAIL' not in content.split('PASS')[0]
    except:
        return False

def run_single(test_cases, run_num):
    """1회 실험"""
    total = len(test_cases)
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    extractor = STKGExtractor()

    passed = {f: 0 for f in fields + ['overall']}

    print(f"\n--- Run {run_num}/3 ---", flush=True)

    for i, case in enumerate(tqdm(test_cases,
                                   desc=f"Run {run_num}")):
        quintuples = extractor.extract_quintuples(case['text'])

        if not quintuples:
            continue

        processed = postprocess_quintuples(
            quintuples,
            normalize_relation=True,
            normalize_entity=True
        )

        if not processed:
            continue

        ext = processed[0]
        gt = case['ground_truth']

        overall_pass = True
        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = str(ext.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                passed[field] += 1
            else:
                overall_pass = False

        if overall_pass:
            passed['overall'] += 1

    scores = {}
    for field in fields + ['overall']:
        scores[field] = passed[field] / total * 100

    print(f"  Run {run_num} overall: {scores['overall']:.1f}%",
          flush=True)

    return scores

def run_repeat(data_file, num_runs=3):
    with open(data_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total = len(test_cases)
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']

    print(f"=== 3회 반복 실험 ({total}개) ===", flush=True)

    all_scores = []
    for run in range(1, num_runs + 1):
        scores = run_single(test_cases, run)
        all_scores.append(scores)

    # 평균 ± 표준편차 계산
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== 3회 반복 실험 결과 ({total}개) ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for i, scores in enumerate(all_scores):
        print(f"Run {i+1}: overall {scores['overall']:.1f}%",
              flush=True)

    print(f"\n{'요소':<12} {'Run1':>8} {'Run2':>8} {'Run3':>8} "
          f"{'평균':>8} {'±표준편차':>10}", flush=True)
    print("-" * 58, flush=True)

    result = {}
    for field in fields + ['overall']:
        vals = [s[field] for s in all_scores]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"{field:<12} {vals[0]:>7.1f}% {vals[1]:>7.1f}% "
              f"{vals[2]:>7.1f}% {mean:>7.1f}% {std:>8.1f}%",
              flush=True)
        result[field] = {
            'runs': vals,
            'mean': round(float(mean), 1),
            'std': round(float(std), 1)
        }

    # 결과 저장
    output = {
        "method": "3-run repeat (Binary, same judge as llm_judge_eval)",
        "total": total,
        "num_runs": num_runs,
        "results": result
    }

    with open(r'results\extracted\repeat_eval_result.json',
              'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    run_repeat(r'data\raw_text_test.json', num_runs=3)