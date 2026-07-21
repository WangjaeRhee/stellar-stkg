"""
Self-Consistency 평가
같은 텍스트 3번 추출 → 다수결 → 성능 비교
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
- Same action with different wording (e.g. "Ban" = "Impose restrictions")
- Sub-entity of the same parent (e.g. "Kyiv" PASS for "Ukraine")
- Same meaning expressed differently (e.g. "Condemn" = "Criticize")

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

def evaluate(test_cases, extractor, use_sc=False, label=""):
    """평가 실행"""
    total = len(test_cases)
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    passed = {f: 0 for f in fields + ['overall']}

    print(f"\n--- {label} ---", flush=True)

    for case in tqdm(test_cases, desc=label):
        if use_sc:
            quintuples = extractor.extract_quintuples_sc(
                case['text'], n_samples=3)
        else:
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

    # 결과 출력
    scores = {}
    print(f"\n  {'요소':<12} {'PASS':>8} {'PASS율':>10}", flush=True)
    print(f"  {'-' * 34}", flush=True)

    for field in fields + ['overall']:
        rate = passed[field] / total * 100
        scores[field] = rate
        print(f"  {field:<12} {passed[field]:>8} {rate:>9.1f}%",
              flush=True)

    return scores

def main():
    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # 50개만 사용 (시간 절약)
    test_cases = test_cases[:50]
    total = len(test_cases)

    print(f"=== Self-Consistency 비교 실험 ({total}개) ===",
          flush=True)

    extractor = STKGExtractor()

    # 1회 추출
    scores_single = evaluate(
        test_cases, extractor,
        use_sc=False, label="1회 추출"
    )

    # Self-Consistency (3회 다수결)
    scores_sc = evaluate(
        test_cases, extractor,
        use_sc=True, label="Self-Consistency (3회)"
    )

    # 비교
    print(f"\n{'=' * 50}", flush=True)
    print(f"=== 1회 추출 vs Self-Consistency 비교 ===",
          flush=True)
    print(f"{'=' * 50}\n", flush=True)

    fields = ['entity1', 'relation', 'entity2', 'time',
              'location', 'overall']
    print(f"{'요소':<12} {'1회':>10} {'SC(3회)':>10} {'차이':>10}",
          flush=True)
    print("-" * 46, flush=True)

    for field in fields:
        s = scores_single.get(field, 0)
        sc = scores_sc.get(field, 0)
        diff = sc - s
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"{field:<12} {s:>9.1f}% {sc:>9.1f}% "
              f"{arrow}{abs(diff):>7.1f}%", flush=True)

    # 결과 저장
    result = {
        "single": scores_single,
        "self_consistency": scores_sc,
        "sample_size": total
    }

    with open(r'results\extracted\sc_eval_result.json',
              'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # 100개 전체
        with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        total = len(test_cases)
        print(f"=== Self-Consistency 100개 전체 ===", flush=True)
        extractor = STKGExtractor()
        scores = evaluate(
            test_cases, extractor,
            use_sc=True, label=f"SC 100개"
        )
        with open(r'results\extracted\sc_eval_full_result.json',
                  'w', encoding='utf-8') as f:
            json.dump({"scores": scores, "total": total},
                      f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장 완료", flush=True)
    else:
        main()