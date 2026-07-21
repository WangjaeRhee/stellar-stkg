"""
ATOM vs 우리 프레임워크 공정 비교
비정형 텍스트 + 동일 모델 (Qwen 7B)
최적화: 임베딩으로 후보 필터 → Judge는 최종만
"""

import json
import sys
import asyncio
import numpy as np
import ollama
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer

sys.path.append('.')
sys.path.append(r'C:\Users\Rhee\Desktop\KMOU\itext2kg')

from langchain_ollama import ChatOllama, OllamaEmbeddings
from itext2kg.atom import Atom
from modules.extractor import STKGExtractor
from modules.postprocessor import postprocess_quintuples
from config import LLM_HOST, LLM_MODEL

client = ollama.Client(host=LLM_HOST)

def normalize_timestamp(time_str):
    time_str = str(time_str).strip().strip('[]')
    try:
        ts = float(time_str)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    except (ValueError, OSError):
        return time_str

def binary_judge(gt_val, ext_val, element_type):
    if str(gt_val).lower().strip() == str(ext_val).lower().strip():
        return True
    if not gt_val or not ext_val:
        return False
    if str(ext_val).lower().strip() in ['unknown', 'n/a', 'none', '']:
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

async def run_atom(texts, sample_size=50):
    """ATOM으로 추출"""
    print("ATOM 초기화 중...", flush=True)

    llm = ChatOllama(
        base_url=LLM_HOST,
        model="qwen2.5:7b",
        temperature=0
    )
    embeddings = OllamaEmbeddings(
        base_url=LLM_HOST,
        model="nomic-embed-text"
    )

    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    facts_dict = {"2022-10-01": texts[:sample_size]}

    print(f"ATOM 실행 중 ({sample_size}개)...", flush=True)
    kg = await atom.build_graph_from_different_obs_times(
        atomic_facts_with_obs_timestamps=facts_dict
    )

    atom_quintuples = []
    for rel in kg.relationships:
        t_start = getattr(rel.properties, 't_start', [])
        time_str = str(t_start[0]) if t_start else 'unknown'

        q = {
            "entity1": getattr(rel.startEntity, 'name', 'unknown'),
            "relation": getattr(rel, 'name', 'unknown'),
            "entity2": getattr(rel.endEntity, 'name', 'unknown'),
            "time": normalize_timestamp(time_str),
            "location": "N/A"
        }
        atom_quintuples.append(q)

    return atom_quintuples

def run_ours(test_cases, sample_size=50):
    """우리 프레임워크로 추출"""
    print(f"우리 프레임워크 실행 중 ({sample_size}개)...",
          flush=True)

    extractor = STKGExtractor()
    our_quintuples = []

    for case in tqdm(test_cases[:sample_size], desc="우리"):
        quintuples = extractor.extract_quintuples(case['text'])
        if quintuples:
            processed = postprocess_quintuples(
                quintuples,
                normalize_relation=False,
                normalize_entity=True
            )
            if processed:
                our_quintuples.append(processed[0])
            else:
                our_quintuples.append(None)
        else:
            our_quintuples.append(None)

    return our_quintuples

def find_best_match_embedding(gt, kg_quintuples, embed_model):
    """임베딩으로 가장 유사한 KG 관계 찾기"""
    gt_text = f"{gt.get('entity1','')} {gt.get('relation','')} {gt.get('entity2','')}"

    kg_texts = []
    for q in kg_quintuples:
        text = f"{q.get('entity1','')} {q.get('relation','')} {q.get('entity2','')}"
        kg_texts.append(text)

    if not kg_texts:
        return None, 0

    gt_emb = embed_model.encode([gt_text])
    kg_embs = embed_model.encode(kg_texts)

    sims = np.dot(kg_embs, gt_emb.T).flatten()
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]

    return kg_quintuples[best_idx], best_sim

def evaluate_atom_optimized(ground_truths, atom_quintuples,
                             embed_model):
    """
    ATOM 평가 (최적화)
    1. 임베딩으로 최적 후보 찾기
    2. 최적 후보만 Binary Judge
    """
    fields = ['entity1', 'relation', 'entity2', 'time']
    field_pass = {f: 0 for f in fields + ['overall']}
    total = len(ground_truths)

    print(f"\nATOM 평가 중 ({total}개)...", flush=True)

    for gt in tqdm(ground_truths, desc="ATOM 평가"):
        best_match, sim = find_best_match_embedding(
            gt, atom_quintuples, embed_model
        )

        if best_match is None:
            continue

        all_pass = True
        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = str(best_match.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                field_pass[field] += 1
            else:
                all_pass = False

        if all_pass:
            field_pass['overall'] += 1

    scores = {}
    for field in fields + ['overall']:
        scores[field] = field_pass[field] / total * 100

    return scores

def evaluate_ours(ground_truths, our_quintuples):
    """우리 평가 (1:1 인덱스 매칭)"""
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']
    field_pass = {f: 0 for f in fields + ['overall']}
    total = len(ground_truths)

    print(f"\n우리 평가 중 ({total}개)...", flush=True)

    for i, gt in enumerate(tqdm(ground_truths, desc="우리 평가")):
        if i >= len(our_quintuples) or our_quintuples[i] is None:
            continue

        ext = our_quintuples[i]
        all_pass = True

        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = str(ext.get(field, ''))
            if binary_judge(gt_val, ext_val, field):
                field_pass[field] += 1
            else:
                all_pass = False

        if all_pass:
            field_pass['overall'] += 1

    scores = {}
    for field in fields + ['overall']:
        scores[field] = field_pass[field] / total * 100

    return scores

async def main():
    sample_size = 50

    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    test_cases = test_cases[:sample_size]
    texts = [case['text'] for case in test_cases]
    ground_truths = [case['ground_truth'] for case in test_cases]

    print("=" * 60, flush=True)
    print("  ATOM vs 우리 프레임워크 공정 비교", flush=True)
    print("  비정형 텍스트 + Qwen 7B + Binary Judge", flush=True)
    print("=" * 60, flush=True)

    # 임베딩 모델 로드
    print("\n임베딩 모델 로드 중...", flush=True)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ATOM 실행
    print("\n[1/4] ATOM 실행...", flush=True)
    atom_quintuples = await run_atom(texts, sample_size)
    print(f"ATOM KG 크기: {len(atom_quintuples)}개\n", flush=True)

    for i, q in enumerate(atom_quintuples[:3]):
        print(f"  [{i+1}] {q['entity1']} → {q['relation']} → "
              f"{q['entity2']} (time: {q['time']})", flush=True)

    # 우리 실행
    print("\n[2/4] 우리 프레임워크 실행...", flush=True)
    our_quintuples = run_ours(test_cases, sample_size)
    our_valid = [q for q in our_quintuples if q is not None]
    print(f"우리 추출: {len(our_valid)}개\n", flush=True)

    for i, q in enumerate(our_quintuples[:3]):
        if q:
            print(f"  [{i+1}] {q['entity1']} → {q['relation']} → "
                  f"{q['entity2']} (time: {q['time']}, "
                  f"loc: {q['location']})", flush=True)

    # ATOM 평가
    print("\n[3/4] ATOM 평가 (임베딩 후보 + Binary Judge)...",
          flush=True)
    atom_scores = evaluate_atom_optimized(
        ground_truths, atom_quintuples, embed_model
    )

    # 우리 평가
    print("\n[4/4] 우리 평가 (1:1 Binary Judge)...", flush=True)
    our_scores = evaluate_ours(ground_truths, our_quintuples)

    # 결과 출력
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== 공정 비교 결과 ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    print(f"ATOM KG: {len(atom_quintuples)}개 관계", flush=True)
    print(f"우리 추출: {len(our_valid)}개\n", flush=True)

    print(f"{'요소':<12} {'ATOM':>10} {'우리':>10} {'차이':>10}",
          flush=True)
    print("-" * 46, flush=True)

    for field in ['entity1', 'relation', 'entity2', 'time']:
        a = atom_scores.get(field, 0)
        o = our_scores.get(field, 0)
        diff = o - a
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"{field:<12} {a:>9.1f}% {o:>9.1f}% "
              f"{arrow}{abs(diff):>7.1f}%", flush=True)

    print(f"{'location':<12} {'N/A':>10} "
          f"{our_scores.get('location', 0):>9.1f}%", flush=True)

    a_ov = atom_scores.get('overall', 0)
    o_ov = our_scores.get('overall', 0)
    diff = o_ov - a_ov
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"{'overall':<12} {a_ov:>9.1f}% {o_ov:>9.1f}% "
          f"{arrow}{abs(diff):>7.1f}%", flush=True)

    print(f"\n핵심:", flush=True)
    print(f"  둘 다 Qwen 7B, 같은 50개 비정형 텍스트", flush=True)
    print(f"  ATOM → Location 없음 (TKG)", flush=True)
    print(f"  우리 → Location 포함 (STKG)", flush=True)

    # 결과 저장
    result = {
        "method": "Fair comparison (unstructured text, Binary)",
        "sample_size": sample_size,
        "atom_kg_size": len(atom_quintuples),
        "our_count": len(our_valid),
        "atom_scores": atom_scores,
        "our_scores": our_scores
    }

    with open(r'results\extracted\compare_atom_fair.json',
              'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)

if __name__ == "__main__":
    asyncio.run(main())