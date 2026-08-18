"""
G-BERTScore 평가: 5중항 텍스트 유사도
+ 정성적 사례 분석
"""

import json
import sys
from tqdm import tqdm
from bert_score import score as bert_score
sys.path.append('.')
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples

def quintuple_to_text(q):
    """5중항을 텍스트로 변환"""
    e1 = str(q.get('entity1', '')).strip()
    r = str(q.get('relation', '')).strip()
    e2 = str(q.get('entity2', '')).strip()
    t = str(q.get('time', '')).strip()
    l = str(q.get('location', '')).strip()
    return f"{e1} | {r} | {e2} | {t} | {l}"

def main():
    # 데이터 로드
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

    print(f"=== G-BERTScore Evaluation ===", flush=True)
    print(f"Total: {len(all_cases)}\n", flush=True)

    extractor = STKGExtractor()

    gt_texts = []
    ext_texts = []
    case_details = []

    for case in tqdm(all_cases, desc="Extracting"):
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

        gt_text = quintuple_to_text(gt)
        ext_text = quintuple_to_text(ext)

        gt_texts.append(gt_text)
        ext_texts.append(ext_text)
        case_details.append({
            'original': case['text'][:100],
            'domain': case.get('domain', 'unknown'),
            'gt': gt_text,
            'ext': ext_text
        })

    print(f"\nExtracted: {len(gt_texts)} / {len(all_cases)}", flush=True)

    # BERTScore 계산
    print("\nCalculating BERTScore...", flush=True)
    P, R, F1 = bert_score(
        ext_texts, gt_texts,
        lang="en",
        model_type="roberta-large",
        verbose=True
    )

    # 도메인별 집계
    from collections import defaultdict
    domain_scores = defaultdict(list)

    for i, detail in enumerate(case_details):
        domain_scores[detail['domain']].append({
            'P': P[i].item(),
            'R': R[i].item(),
            'F1': F1[i].item(),
            'gt': detail['gt'],
            'ext': detail['ext'],
            'original': detail['original']
        })

    # 전체 결과
    avg_p = P.mean().item() * 100
    avg_r = R.mean().item() * 100
    avg_f1 = F1.mean().item() * 100

    print(f"\n{'=' * 60}", flush=True)
    print(f"=== G-BERTScore Results ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    print(f"Overall G-BERTScore:", flush=True)
    print(f"  Precision: {avg_p:.1f}%", flush=True)
    print(f"  Recall:    {avg_r:.1f}%", flush=True)
    print(f"  F1:        {avg_f1:.1f}%\n", flush=True)

    # 도메인별
    print(f"{'Domain':<15} {'P':>8} {'R':>8} {'F1':>8} {'n':>5}",
          flush=True)
    print(f"{'-' * 45}", flush=True)

    domains = ['politics', 'business', 'disaster', 'sports',
               'science', 'health', 'culture', 'crime', 'environment']

    for d in domains:
        if d not in domain_scores:
            continue
        scores = domain_scores[d]
        dp = sum(s['P'] for s in scores) / len(scores) * 100
        dr = sum(s['R'] for s in scores) / len(scores) * 100
        df1 = sum(s['F1'] for s in scores) / len(scores) * 100
        print(f"{d:<15} {dp:>7.1f}% {dr:>7.1f}% {df1:>7.1f}% {len(scores):>5}",
              flush=True)

    # 정성적 분석: 최고/최저 사례
    all_scored = []
    for i, detail in enumerate(case_details):
        all_scored.append({
            **detail,
            'f1': F1[i].item()
        })

    all_scored.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"=== Top 5 Best Cases ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for i, case in enumerate(all_scored[:5]):
        print(f"[{i+1}] F1: {case['f1']*100:.1f}% ({case['domain']})",
              flush=True)
        print(f"  Text: {case['original']}...", flush=True)
        print(f"  GT:   {case['gt']}", flush=True)
        print(f"  Ext:  {case['ext']}\n", flush=True)

    print(f"{'=' * 60}", flush=True)
    print(f"=== Top 5 Worst Cases ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for i, case in enumerate(all_scored[-5:]):
        print(f"[{i+1}] F1: {case['f1']*100:.1f}% ({case['domain']})",
              flush=True)
        print(f"  Text: {case['original']}...", flush=True)
        print(f"  GT:   {case['gt']}", flush=True)
        print(f"  Ext:  {case['ext']}\n", flush=True)

    # 저장
    result = {
        "overall": {
            "precision": round(avg_p, 1),
            "recall": round(avg_r, 1),
            "f1": round(avg_f1, 1),
            "n": len(gt_texts)
        },
        "domains": {},
        "best_cases": [
            {"f1": round(c['f1']*100, 1), "domain": c['domain'],
             "gt": c['gt'], "ext": c['ext']}
            for c in all_scored[:5]
        ],
        "worst_cases": [
            {"f1": round(c['f1']*100, 1), "domain": c['domain'],
             "gt": c['gt'], "ext": c['ext']}
            for c in all_scored[-5:]
        ]
    }

    for d in domains:
        if d in domain_scores:
            scores = domain_scores[d]
            result["domains"][d] = {
                "precision": round(sum(s['P'] for s in scores)/len(scores)*100, 1),
                "recall": round(sum(s['R'] for s in scores)/len(scores)*100, 1),
                "f1": round(sum(s['F1'] for s in scores)/len(scores)*100, 1),
                "n": len(scores)
            }

    with open(r'results\extracted\bertscore_eval_result.json',
              'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    main()