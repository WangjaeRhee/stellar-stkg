"""
도메인 교체 실험
비즈니스/기술 vs 자연재해 도메인 비교
프레임워크 범용성 검증
"""

import json
import sys
import ollama
from tqdm import tqdm
sys.path.append('.')
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from config import LLM_MODEL, LLM_HOST

class LLMJudge:
    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.client = ollama.Client(host=host)
        self.model = model
        self.cache = {}

    def judge(self, gt_val, ext_val, element_type):
        cache_key = f"{element_type}:{gt_val}:{ext_val}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if str(gt_val).lower().strip() == str(ext_val).lower().strip():
            result = {'score': 4, 'matched': True}
            self.cache[cache_key] = result
            return result

        prompt = f"""Compare ground truth and extracted value for "{element_type}".

Ground truth: "{gt_val}"
Extracted: "{ext_val}"

Score 1-4:
1: Completely wrong
2: Partially related
3: Similar meaning
4: Same meaning or exact match

Score:"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['message']['content']
            score = 1
            for c in content:
                if c.isdigit() and int(c) in [1, 2, 3, 4]:
                    score = int(c)
                    break
            result = {'score': score, 'matched': score >= 3}
            self.cache[cache_key] = result
            return result
        except:
            return {'score': 1, 'matched': False}

def run_domain_eval(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    extractor = STKGExtractor()
    judge = LLMJudge()

    domains = {}
    for case in test_cases:
        d = case['domain']
        if d not in domains:
            domains[d] = []
        domains[d].append(case)

    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    all_results = {}

    print("=" * 60, flush=True)
    print("  도메인 교체 실험 (LLM-as-Judge)", flush=True)
    print("=" * 60, flush=True)

    for domain, cases in domains.items():
        print(f"\n--- 도메인: {domain} ({len(cases)}개) ---",
              flush=True)

        matched = {f: 0 for f in fields + ['overall']}
        scores = {f: [] for f in fields}
        total = len(cases)
        success = 0

        for case in tqdm(cases, desc=f"{domain}"):
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
            success += 1

            overall_match = True
            for field in fields:
                gt_val = gt.get(field, '')
                ext_val = str(ext.get(field, ''))
                result = judge.judge(gt_val, ext_val, field)
                scores[field].append(result['score'])
                if result['matched']:
                    matched[field] += 1
                else:
                    overall_match = False

            if overall_match:
                matched['overall'] += 1

        # 도메인별 결과
        domain_scores = {}
        print(f"\n  추출 성공: {success}/{total}", flush=True)
        print(f"  {'요소':<12} {'매칭률':>10} {'평균점수':>10}",
              flush=True)
        print(f"  {'-' * 36}", flush=True)

        for field in fields:
            m = matched[field] / total * 100
            avg = sum(scores[field]) / len(scores[field]) \
                if scores[field] else 0
            print(f"  {field:<12} {m:>9.1f}% {avg:>8.2f}/4",
                  flush=True)
            domain_scores[field] = m

        m = matched['overall'] / total * 100
        print(f"  {'overall':<12} {m:>9.1f}%", flush=True)
        domain_scores['overall'] = m

        all_results[domain] = domain_scores

    # 도메인 간 비교
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== 도메인 간 비교 ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    domain_names = list(all_results.keys())
    header = f"{'요소':<12}"
    for d in domain_names:
        header += f" {d:>12}"
    print(header, flush=True)
    print("-" * (12 + 13 * len(domain_names)), flush=True)

    for field in fields + ['overall']:
        row = f"{field:<12}"
        for d in domain_names:
            val = all_results[d].get(field, 0)
            row += f" {val:>11.1f}%"
        print(row, flush=True)

    # 결과 저장
    with open(r'results\extracted\domain_eval_result.json',
              'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    run_domain_eval(r'data\domain_test.json')