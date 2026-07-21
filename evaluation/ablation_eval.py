"""
Ablation Study: Measuring contribution of each module
Same data + Same model + Only settings changed
LLM-as-Judge Binary evaluation
"""

import json
import sys
import ollama
from tqdm import tqdm
sys.path.append('.')
from modules.postprocessor import postprocess_quintuples
from config import LLM_MODEL, LLM_HOST

client = ollama.Client(host=LLM_HOST)

def call_llm(prompt):
    try:
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except:
        return ""

# ============================================================
# Configuration 1: Direct extraction (without framework)
# ============================================================
def extract_direct(text):
    prompt = f"""Extract a spatio-temporal quintuple from this text.

A quintuple has 5 elements:
- entity1: the actor who performs the action (person, country, org)
- relation: the action performed (2-5 words)
- entity2: the actor who is affected (person, country, org)
- time: date in YYYY-MM-DD format
- location: country where the event occurred

Return ONLY a JSON object.

Text: "{text}"
Output:"""

    try:
        content = call_llm(prompt)
        start = content.find('{')
        end = content.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        return json.loads(content[start:end])
    except:
        return None

# ============================================================
# Configuration 2: Atomic Decomposition + Direct
# ============================================================
def extract_atomic_direct(text):
    # Decomposition
    prompt1 = f"""Decompose into atomic facts. Return ONLY a JSON array.
Text: {text}
Output:"""
    try:
        content = call_llm(prompt1)
        start = content.find('[')
        end = content.rfind(']') + 1
        facts = json.loads(content[start:end]) if start != -1 else [text]
    except:
        facts = [text]

    # Direct extraction
    fact = facts[0] if facts else text
    return extract_direct(fact)

# ============================================================
# Configuration 3: Multi-stage extraction (without CoT)
# ============================================================
def extract_multistage_no_cot(text):
    # Decomposition
    prompt1 = f"""Decompose into atomic facts. Return ONLY a JSON array.
Text: {text}
Output:"""
    try:
        content = call_llm(prompt1)
        start = content.find('[')
        end = content.rfind(']') + 1
        facts = json.loads(content[start:end]) if start != -1 else [text]
    except:
        facts = [text]

    fact = facts[0] if facts else text

    # Entity 추출 (CoT 없이)
    prompt2 = f"""Extract the two main actors from this sentence.
entity1 = who performs the action
entity2 = who is affected by the action
Both must be a person, country, or organization.
Return ONLY a JSON object.
Sentence: "{fact}"
Output:"""

    try:
        content = call_llm(prompt2)
        start = content.find('{')
        end = content.rfind('}') + 1
        entities = json.loads(content[start:end])
    except:
        return None

    e1 = entities.get('entity1', 'unknown')
    e2 = entities.get('entity2', 'unknown')

    # Relation
    prompt3 = f"""What action did "{e1}" perform toward "{e2}"?
Sentence: "{fact}"
Return ONLY a JSON object with "relation" key.
Output:"""
    try:
        content = call_llm(prompt3)
        start = content.find('{')
        end = content.rfind('}') + 1
        rel = json.loads(content[start:end]).get('relation', 'unknown')
    except:
        rel = 'unknown'

    # Time/Location
    prompt4 = f"""Extract time (YYYY-MM-DD) and location (country) from:
"{fact}"
Return ONLY a JSON object.
Output:"""
    try:
        content = call_llm(prompt4)
        start = content.find('{')
        end = content.rfind('}') + 1
        tl = json.loads(content[start:end])
    except:
        tl = {"time": "unknown", "location": "unknown"}

    return {
        "entity1": e1, "relation": rel, "entity2": e2,
        "time": tl.get('time', 'unknown'),
        "location": tl.get('location', 'unknown')
    }

# ============================================================
# Configuration 4: Multi-stage + CoT (without normalization)
# ============================================================
def extract_multistage_cot(text):
    from modules.extractor import STKGExtractor
    extractor = STKGExtractor()
    quintuples = extractor.extract_quintuples(text)
    if quintuples:
        return quintuples[0]
    return None

# ============================================================
# Configuration 5: Full framework (Multi-stage + CoT + Normalization)
# ============================================================
def extract_full_pipeline(text):
    from modules.extractor import STKGExtractor
    extractor = STKGExtractor()
    quintuples = extractor.extract_quintuples(text)
    if quintuples:
        processed = postprocess_quintuples(
            quintuples,
            normalize_relation=False,
            normalize_entity=True
        )
        if processed:
            return processed[0]
    return None

# ============================================================
# Binary Judge
# ============================================================
def binary_judge(gt_val, ext_val, element_type):
    if str(gt_val).lower().strip() == str(ext_val).lower().strip():
        return True

    prompt = f"""Are these semantically equivalent for "{element_type}"?
Ground truth: "{gt_val}"
Extracted: "{ext_val}"
Rules: PASS if same person/country/org/meaning. FAIL if different.
Answer ONLY "PASS" or "FAIL":"""

    try:
        content = call_llm(prompt)
        return 'PASS' in content.upper() and \
               'FAIL' not in content.upper().split('PASS')[0]
    except:
        return False

# ============================================================
# 메인 실행
# ============================================================
def run_ablation(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total = len(test_cases)
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']

    configs = [
        ("1. Direct", extract_direct),
        ("2. Atomic+Direct", extract_atomic_direct),
        ("3. Multi-stage", extract_multistage_no_cot),
        ("4. Multi+CoT", extract_multistage_cot),
        ("5. Full pipeline", extract_full_pipeline),
    ]

    all_results = {}

    for config_name, extract_fn in configs:
        print(f"\n{'=' * 50}", flush=True)
        print(f"  {config_name}", flush=True)
        print(f"{'=' * 50}", flush=True)

        passed = {f: 0 for f in fields + ['overall']}
        success = 0

        for i, case in enumerate(tqdm(test_cases, desc=config_name)):
            ext = extract_fn(case['text'])

            if not ext:
                continue

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

        # Results
        scores = {}
        print(f"\n  Extraction success: {success}/{total}", flush=True)
        for field in fields + ['overall']:
            rate = passed[field] / total * 100
            scores[field] = rate

        print(f"  {'Element':<12} {'PASS Rate':>10}", flush=True)
        print(f"  {'-' * 26}", flush=True)
        for field in fields + ['overall']:
            print(f"  {field:<12} {scores[field]:>9.1f}%", flush=True)

        all_results[config_name] = scores

    # Final comparison
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"=== Ablation Study Final Comparison (Binary) ===", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    header = f"{'Element':<12}"
    for name, _ in configs:
        short = name.split('. ')[1][:10]
        header += f" {short:>12}"
    print(header, flush=True)
    print("-" * (12 + 13 * len(configs)), flush=True)

    for field in fields + ['overall']:
        row = f"{field:<12}"
        for name, _ in configs:
            val = all_results[name].get(field, 0)
            row += f" {val:>11.1f}%"
        print(row, flush=True)

    # Save results
    with open(r'results\extracted\ablation_result.json',
              'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved successfully", flush=True)

if __name__ == "__main__":
    run_ablation(r'data\raw_text_test.json')