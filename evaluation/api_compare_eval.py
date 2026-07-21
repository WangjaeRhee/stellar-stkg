"""
API 모델 비교 실험
modules/extractor.py + postprocessor.py 그대로 사용
LLM 호출만 OpenAI로 교체
→ 프레임워크 완전 동일, 모델만 다름
"""

import json
import sys
import openai
import ollama
from tqdm import tqdm
sys.path.append('.')
from config import OPENAI_API_KEY, LLM_MODEL, LLM_HOST
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples

oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
ollama_client = ollama.Client(host=LLM_HOST)


class APIExtractor(STKGExtractor):
    """
    STKGExtractor를 상속받아
    _call_llm만 OpenAI로 교체
    → 프롬프트, 파이프라인, 후처리 모두 동일
    """

    def __init__(self, api_model="gpt-4o-mini"):
        # 부모 초기화 안 함 (Ollama 연결 불필요)
        self.api_model = api_model
        self.prompt_version = "v10_domain_balanced"

    def _call_llm(self, prompt):
        """OpenAI API로 교체"""
        try:
            response = oai_client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  API 오류: {e}", flush=True)
            return ""


def binary_judge(gt_val, ext_val, element_type):
    """Binary Judge (Qwen 7B 고정)"""
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
- Same action with different wording (e.g. "Ban" = "Impose restrictions")
- Sub-entity of the same parent (e.g. "Kyiv" PASS for "Ukraine")
- Same meaning expressed differently

Rules for FAIL:
- Completely different entity
- Missing value
- Wrong entity type
- No meaningful connection

Answer with ONLY "PASS" or "FAIL":"""

    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['message']['content'].strip().upper()
        return 'PASS' in content and \
               'FAIL' not in content.split('PASS')[0]
    except:
        return False


def evaluate_model(test_cases, extractor, model_name):
    """모델별 평가 (동일 파이프라인)"""
    total = len(test_cases)
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    passed = {f: 0 for f in fields + ['overall']}
    success = 0

    print(f"\n{'=' * 50}", flush=True)
    print(f"  모델: {model_name} ({total}개)", flush=True)
    print(f"{'=' * 50}", flush=True)

    for i, case in enumerate(tqdm(test_cases, desc=model_name)):
        # 동일한 extract_quintuples 메서드 사용
        quintuples = extractor.extract_quintuples(case['text'])

        if not quintuples:
            continue

        # 동일한 postprocessor 사용
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
    print(f"\n  추출 성공: {success}/{total}", flush=True)
    print(f"  {'요소':<12} {'PASS':>8} {'PASS율':>10}", flush=True)
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

    test_cases = test_cases[:50]
    total = len(test_cases)

    print(f"=== API 모델 비교 실험 ({total}개) ===", flush=True)
    print(f"프레임워크: modules/extractor.py (동일)", flush=True)
    print(f"후처리: modules/postprocessor.py (동일)", flush=True)
    print(f"Judge: Qwen 7B (고정)", flush=True)
    print(f"차이점: _call_llm만 교체\n", flush=True)

    results = {}

    # Qwen 7B (기존 3회 평균)
    results['Qwen2.5-7B'] = {
        'entity1': 94.7, 'relation': 72.7,
        'entity2': 79.7, 'time': 97.7,
        'location': 89.7, 'overall': 51.0
    }
    print("Qwen2.5-7B: 기존 3회 평균 결과 사용\n", flush=True)

    # GPT-4o-mini
    mini_extractor = APIExtractor(api_model="gpt-4o-mini")
    results['GPT-4o-mini'] = evaluate_model(
        test_cases, mini_extractor, "GPT-4o-mini")

    # GPT-4o
    gpt4o_extractor = APIExtractor(api_model="gpt-4o")
    results['GPT-4o'] = evaluate_model(
        test_cases, gpt4o_extractor, "GPT-4o")

    # 최종 비교
    print(f"\n{'=' * 70}", flush=True)
    print(f"=== 모델 비교 최종 결과 (동일 프레임워크) ===", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    models = list(results.keys())
    fields = ['entity1', 'relation', 'entity2', 'time',
              'location', 'overall']

    header = f"{'요소':<12}"
    for m in models:
        header += f" {m:>14}"
    print(header, flush=True)
    print("-" * (12 + 15 * len(models)), flush=True)

    for field in fields:
        row = f"{field:<12}"
        for m in models:
            val = results[m].get(field, 0)
            row += f" {val:>13.1f}%"
        print(row, flush=True)

    # 결과 저장
    with open(r'results\extracted\api_compare_result.json',
              'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)


if __name__ == "__main__":
    main()