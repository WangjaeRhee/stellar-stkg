"""
ATOM vs STELLAR 4요소 기준 overall 재계산
Location 제외하고 동일 기준으로 비교
"""

import json
import sys
import ollama
from tqdm import tqdm
sys.path.append('.')
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from config import LLM_MODEL, LLM_HOST

client = ollama.Client(host=LLM_HOST)

def binary_judge(gt_val, ext_val, element_type):
    if str(gt_val).lower().strip() == str(ext_val).lower().strip():
        return True
    if not ext_val or str(ext_val).lower().strip() in ['unknown', 'none', '']:
        return False

    prompt = f"""You are a fair judge for knowledge graph evaluation.

Element type: {element_type}
Ground truth: "{gt_val}"
Extracted: "{ext_val}"

Rules for PASS:
- Same person (e.g. "Putin" = "Vladimir Putin")
- Same country (e.g. "US" = "United States")
- Same organization (e.g. "EU" = "European Union")
- Same action with different wording
- Sub-entity of the same parent
- Same meaning expressed differently

Rules for FAIL:
- Completely different entity
- Missing value
- Wrong entity type

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

def main():
    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    test_cases = test_cases[:50]
    total = len(test_cases)
    extractor = STKGExtractor()

    fields_5 = ['entity1', 'relation', 'entity2', 'time', 'location']
    fields_4 = ['entity1', 'relation', 'entity2', 'time']

    passed_5 = {f: 0 for f in fields_5 + ['overall_5']}
    passed_4 = {f: 0 for f in fields_4 + ['overall_4']}

    print(f"=== STELLAR 4요소 vs 5요소 overall 비교 ({total}개) ===\n",
          flush=True)

    for case in tqdm(test_cases, desc="STELLAR 추출"):
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

        # 5요소 평가
        all_5_pass = True
        for field in fields_5:
            gt_val = gt.get(field, '')
            ext_val = str(ext.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                passed_5[field] += 1
            else:
                all_5_pass = False
        if all_5_pass:
            passed_5['overall_5'] += 1

        # 4요소 평가 (Location 제외)
        all_4_pass = True
        for field in fields_4:
            gt_val = gt.get(field, '')
            ext_val = str(ext.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                passed_4[field] += 1
            else:
                all_4_pass = False
        if all_4_pass:
            passed_4['overall_4'] += 1

    # 결과
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== 비교 결과 ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    stellar_5 = passed_5['overall_5'] / total * 100
    stellar_4 = passed_4['overall_4'] / total * 100

    print(f"STELLAR 5요소 overall (e1,r,e2,t,l): {stellar_5:.1f}%",
          flush=True)
    print(f"STELLAR 4요소 overall (e1,r,e2,t):   {stellar_4:.1f}%",
          flush=True)
    print(f"ATOM    4요소 overall (e1,r,e2,t):   8.0%", flush=True)
    print(f"\n동일 기준(4요소) 비교:", flush=True)
    print(f"  ATOM:    8.0%", flush=True)
    print(f"  STELLAR: {stellar_4:.1f}% (차이: +{stellar_4-8:.1f}%p)",
          flush=True)

if __name__ == "__main__":
    main()