"""
다양한 도메인 테스트 (프롬프트 수정 없이)
스포츠, 과학, 보건, 문화, 범죄
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
    if not ext_val or str(ext_val).lower().strip() in \
            ['unknown', 'none', '']:
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

def evaluate_domain(test_cases, domain_name):
    cases = [c for c in test_cases if c['domain'] == domain_name]
    total = len(cases)
    if total == 0:
        return None

    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    passed = {f: 0 for f in fields + ['overall']}
    extractor = STKGExtractor()

    for case in tqdm(cases, desc=domain_name):
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

    return scores

def main():
    with open(r'data\domain_extra_test.json', 'r',
              encoding='utf-8') as f:
        test_cases = json.load(f)

    domains = ['sports', 'science', 'health', 'culture', 'crime']

    print(f"=== 다양한 도메인 테스트 (프롬프트 수정 없이) ===",
          flush=True)
    print(f"도메인: {domains}", flush=True)
    print(f"각 도메인 10개씩\n", flush=True)

    all_scores = {}

    for domain in domains:
        scores = evaluate_domain(test_cases, domain)
        if scores:
            all_scores[domain] = scores

    # 최종 비교
    print(f"\n{'=' * 80}", flush=True)
    print(f"=== 8개 도메인 비교 (프롬프트 수정 없이) ===", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    # 기존 3개 도메인 추가
    all_scores['politics'] = {
        'entity1': 94.7, 'relation': 72.7,
        'entity2': 79.7, 'time': 97.7,
        'location': 89.7, 'overall': 51.0
    }
    all_scores['business'] = {
        'entity1': 90.0, 'relation': 73.3,
        'entity2': 73.3, 'time': 96.7,
        'location': 83.3, 'overall': 43.3
    }
    all_scores['disaster'] = {
        'entity1': 86.7, 'relation': 66.7,
        'entity2': 66.7, 'time': 96.7,
        'location': 93.3, 'overall': 36.7
    }

    fields = ['entity1', 'relation', 'entity2', 'time',
              'location', 'overall']

    domain_order = ['politics', 'business', 'disaster',
                    'sports', 'science', 'health',
                    'culture', 'crime']

    header = f"{'요소':<12}"
    for d in domain_order:
        header += f" {d:>10}"
    print(header, flush=True)
    print("-" * (12 + 11 * len(domain_order)), flush=True)

    for field in fields:
        row = f"{field:<12}"
        for d in domain_order:
            val = all_scores.get(d, {}).get(field, 0)
            row += f" {val:>9.1f}%"
        print(row, flush=True)

    # 평균 계산
    print(f"\n--- 전체 평균 ---", flush=True)
    for field in fields:
        vals = [all_scores[d][field] for d in domain_order
                if d in all_scores]
        avg = sum(vals) / len(vals)
        print(f"  {field}: {avg:.1f}%", flush=True)

    # 결과 저장
    with open(r'results\extracted\domain_extra_result.json',
              'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    main()