"""
9개 도메인 테스트 (프롬프트 수정 없이)
politics(100) + business(50) + disaster(50) +
sports(50) + science(50) + health(50) +
culture(50) + crime(50) + environment(50)
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

def evaluate_domain(all_cases, domain_name, extractor):
    cases = [c for c in all_cases if c['domain'] == domain_name]
    total = len(cases)
    if total == 0:
        return None

    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    passed = {f: 0 for f in fields + ['overall']}

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
    # 모든 데이터 로드
    all_cases = []

    # politics (raw_text_test.json)
    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            case['domain'] = 'politics'
            all_cases.append(case)

    # business + disaster (domain_test.json)
    with open(r'data\domain_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            all_cases.append(case)

    # 나머지 (domain_extra_test.json)
    with open(r'data\domain_extra_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            all_cases.append(case)

    # 도메인별 집계
    from collections import Counter
    domain_counts = Counter(c['domain'] for c in all_cases)

    domains = ['politics', 'business', 'disaster', 'sports',
               'science', 'health', 'culture', 'crime', 'environment']

    print(f"=== 9개 도메인 테스트 (프롬프트 수정 없이) ===",
          flush=True)
    print(f"총 {len(all_cases)}개\n", flush=True)
    for d in domains:
        print(f"  {d}: {domain_counts.get(d, 0)}개", flush=True)
    print(flush=True)

    extractor = STKGExtractor()
    all_scores = {}

    for domain in domains:
        count = domain_counts.get(domain, 0)
        if count == 0:
            continue
        scores = evaluate_domain(all_cases, domain, extractor)
        if scores:
            all_scores[domain] = scores
            all_scores[domain]['count'] = count

    # 최종 비교
    print(f"\n{'=' * 100}", flush=True)
    print(f"=== 9개 도메인 비교 (프롬프트 수정 없이) ===", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    fields = ['entity1', 'relation', 'entity2', 'time',
              'location', 'overall']

    header = f"{'요소':<12}"
    for d in domains:
        if d in all_scores:
            header += f" {d:>12}"
    print(header, flush=True)
    print("-" * (12 + 13 * len(all_scores)), flush=True)

    for field in fields:
        row = f"{field:<12}"
        for d in domains:
            if d in all_scores:
                val = all_scores[d].get(field, 0)
                row += f" {val:>11.1f}%"
        print(row, flush=True)

    # 샘플 수
    row = f"{'(n)':<12}"
    for d in domains:
        if d in all_scores:
            row += f" {all_scores[d]['count']:>11}"
    print(row, flush=True)

    # 평균 계산
    print(f"\n--- 전체 평균 ---", flush=True)
    for field in fields:
        vals = [all_scores[d][field] for d in domains
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