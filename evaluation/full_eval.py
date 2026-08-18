"""
전체 평가: Element-level P/R/F1 + Quintuple Strict Match + Element Average
9개 도메인, 500개 텍스트
"""

import json
import sys
import ollama
from tqdm import tqdm
from collections import Counter
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
    passed = {f: 0 for f in fields}
    strict_pass = 0
    extract_success = 0

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
        extract_success += 1

        all_pass = True
        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = str(ext.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                passed[field] += 1
            else:
                all_pass = False

        if all_pass:
            strict_pass += 1

    # 계산
    result = {'count': total, 'extracted': extract_success}

    for field in fields:
        p = passed[field] / extract_success * 100 if extract_success > 0 else 0
        r = passed[field] / total * 100
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        result[f'{field}_P'] = round(p, 1)
        result[f'{field}_R'] = round(r, 1)
        result[f'{field}_F1'] = round(f1, 1)

    # Element Average (Recall 기준)
    avg_r = sum(passed[f] / total * 100 for f in fields) / len(fields)
    result['elem_avg'] = round(avg_r, 1)

    # Strict Match
    result['strict'] = round(strict_pass / total * 100, 1)

    return result

def main():
    all_cases = []

    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            case['domain'] = 'politics'
            all_cases.append(case)

    with open(r'data\domain_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            all_cases.append(case)

    with open(r'data\domain_extra_test.json', 'r', encoding='utf-8') as f:
        for case in json.load(f):
            all_cases.append(case)

    domain_counts = Counter(c['domain'] for c in all_cases)
    domains = ['politics', 'business', 'disaster', 'sports',
               'science', 'health', 'culture', 'crime', 'environment']

    print(f"=== Full Evaluation (P/R/F1 + Strict + Avg) ===", flush=True)
    print(f"Total: {len(all_cases)}\n", flush=True)
    for d in domains:
        print(f"  {d}: {domain_counts.get(d, 0)}", flush=True)
    print(flush=True)

    extractor = STKGExtractor()
    all_scores = {}

    for domain in domains:
        if domain_counts.get(domain, 0) == 0:
            continue
        scores = evaluate_domain(all_cases, domain, extractor)
        if scores:
            all_scores[domain] = scores

    # 결과 출력
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']

    print(f"\n{'=' * 120}", flush=True)
    print(f"=== Element-level Precision / Recall / F1 ===", flush=True)
    print(f"{'=' * 120}\n", flush=True)

    for field in fields:
        print(f"--- {field} ---", flush=True)
        header = f"  {'Domain':<14} {'P':>8} {'R':>8} {'F1':>8}"
        print(header, flush=True)
        print(f"  {'-' * 40}", flush=True)
        for d in domains:
            if d not in all_scores:
                continue
            s = all_scores[d]
            print(f"  {d:<14} {s[f'{field}_P']:>7.1f}% {s[f'{field}_R']:>7.1f}% {s[f'{field}_F1']:>7.1f}%", flush=True)

        # 전체 평균
        ps = [all_scores[d][f'{field}_P'] for d in domains if d in all_scores]
        rs = [all_scores[d][f'{field}_R'] for d in domains if d in all_scores]
        f1s = [all_scores[d][f'{field}_F1'] for d in domains if d in all_scores]
        print(f"  {'AVG':<14} {sum(ps)/len(ps):>7.1f}% {sum(rs)/len(rs):>7.1f}% {sum(f1s)/len(f1s):>7.1f}%", flush=True)
        print(flush=True)

    # Strict + Avg 요약
    print(f"\n{'=' * 100}", flush=True)
    print(f"=== Summary: Element Average vs Strict Match ===", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    header = f"  {'Domain':<14} {'n':>5} {'ext':>5} {'Elem.Avg':>10} {'Strict':>10}"
    print(header, flush=True)
    print(f"  {'-' * 50}", flush=True)
    for d in domains:
        if d not in all_scores:
            continue
        s = all_scores[d]
        print(f"  {d:<14} {s['count']:>5} {s['extracted']:>5} {s['elem_avg']:>9.1f}% {s['strict']:>9.1f}%", flush=True)

    # 전체 평균
    avgs = [all_scores[d]['elem_avg'] for d in domains if d in all_scores]
    stricts = [all_scores[d]['strict'] for d in domains if d in all_scores]
    print(f"  {'-' * 50}", flush=True)
    print(f"  {'AVERAGE':<14} {'':>5} {'':>5} {sum(avgs)/len(avgs):>9.1f}% {sum(stricts)/len(stricts):>9.1f}%", flush=True)

    # 결과 저장
    with open(r'results\extracted\full_eval_result.json',
              'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    main()