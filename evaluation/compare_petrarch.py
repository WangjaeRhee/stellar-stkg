"""
PETRARCH3 (규칙 기반) vs STELLAR (LLM 기반) 비교
동일 데이터, 9개 도메인 450개
"""

import json
import sys
import os
import re
import subprocess
import requests
from collections import defaultdict
from tqdm import tqdm

sys.path.append('.')
from config import LLM_MODEL, LLM_HOST
import ollama

client = ollama.Client(host=LLM_HOST)

CORENLP_URL = "http://localhost:9000"
PETRARCH_DIR = r"C:\Users\Rhee\Desktop\KMOU\petrarch3"


def corenlp_parse(text):
    """CoreNLP로 구문 트리 생성"""
    try:
        params = {
            'annotators': 'tokenize,ssplit,pos,parse',
            'outputFormat': 'json'
        }
        resp = requests.post(
            CORENLP_URL,
            params={'properties': json.dumps(params)},
            data=text.encode('utf-8'),
            timeout=30
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        sentences = data.get('sentences', [])
        if not sentences:
            return None

        # 첫 문장의 파스 트리
        parse = sentences[0].get('parse', '')
        # 개행 제거 및 공백 정리
        parse = ' '.join(parse.split())
        return parse
    except Exception as e:
        return None


def build_petrarch_xml(cases, output_path):
    """PETRARCH3 입력 XML 생성 (샘플과 동일 형식)"""
    lines = ['<Sentences>']

    valid_count = 0
    for i, case in enumerate(cases):
        text = case['text']
        gt_time = case.get('ground_truth', {}).get('time', '')
        date = gt_time.replace('-', '') if gt_time else '20220101'
        if len(date) != 8 or not date.isdigit():
            date = '20220101'

        parse = corenlp_parse(text)
        if not parse:
            continue

        safe_text = (text.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;'))
        safe_parse = parse.replace('&', '&amp;')

        # 샘플 형식: = 주변 공백, id는 STORY_문장번호 형태
        lines.append(
            f'<Sentence date = "{date}" id ="CASE{i:04d}_1" '
            f'source = "AFP" sentence = "True">'
        )
        lines.append('<Text>')
        lines.append(safe_text)
        lines.append('</Text>')
        lines.append('<Parse>')
        lines.append(safe_parse)
        lines.append('</Parse>')
        lines.append('</Sentence>')
        valid_count += 1

    lines.append('</Sentences>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return valid_count


def run_petrarch(input_xml, output_name='petr_out.txt'):
    """PETRARCH3 실행 (출력은 PETRARCH 폴더에 생성됨)"""
    cmd = [
        sys.executable, '-m', 'petrarch3.petrarch3', 'parse',
        '-i', input_xml,
        '-P',
        '-o', output_name
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=PETRARCH_DIR,
            capture_output=True,
            text=True,
            timeout=3600
        )
        out_path = os.path.join(PETRARCH_DIR, 'evts.' + output_name)
        return result.returncode == 0, out_path, result.stderr
    except Exception as e:
        return False, '', str(e)


def parse_petrarch_output(output_txt):
    """PETRARCH3 출력 파싱 → 오중항 변환"""
    events = defaultdict(list)

    if not os.path.exists(output_txt):
        return events

    with open(output_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 6:
                continue

            date = parts[0]
            source = parts[1]
            target = parts[2]
            cameo = parts[3]
            sent_id = parts[5]

            m = re.search(r'CASE(\d+)_', sent_id)
            if not m:
                continue
            idx = int(m.group(1))

            # actor text가 있으면 활용 (원문 표현)
            src_text = parts[7] if len(parts) > 7 else ''
            tgt_text = parts[8] if len(parts) > 8 else ''
            evt_text = parts[9] if len(parts) > 9 else ''

            events[idx].append({
                'entity1': src_text if src_text not in ('', '---') else source,
                'relation': evt_text if evt_text not in ('', '---') else cameo,
                'entity2': tgt_text if tgt_text not in ('', '---') else target,
                'time': date,
                'location': '',
                'cameo_source': source,
                'cameo_target': target,
                'cameo_code': cameo
            })

    return events


def binary_judge(gt_val, ext_val, element_type):
    """LLM-as-Judge Binary (STELLAR 평가와 동일)"""
    if not ext_val or str(ext_val).strip() == '':
        return False
    if str(gt_val).lower().strip() == str(ext_val).lower().strip():
        return True
    if str(ext_val).lower().strip() in ['unknown', 'none', '---']:
        return False

    prompt = f"""You are a fair judge for knowledge graph evaluation.

Element type: {element_type}
Ground truth: "{gt_val}"
Extracted: "{ext_val}"

Rules for PASS:
- Same person (e.g. "Putin" = "Vladimir Putin")
- Same country (e.g. "US" = "United States", "RUS" = "Russia")
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
        return 'PASS' in content and 'FAIL' not in content.split('PASS')[0]
    except:
        return False


def main():
    print("=== PETRARCH3 (Rule-based) vs STELLAR (LLM) ===\n", flush=True)

    # 데이터 로드
    all_cases = []
    with open(r'data\raw_text_test.json', 'r', encoding='utf-8') as f:
        for c in json.load(f):
            c['domain'] = 'politics'
            all_cases.append(c)
    with open(r'data\domain_test.json', 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_cases.append(c)
    with open(r'data\domain_extra_test.json', 'r', encoding='utf-8') as f:
        for c in json.load(f):
            all_cases.append(c)

    print(f"Total cases: {len(all_cases)}", flush=True)

    # CoreNLP 연결 확인
    print("\nChecking CoreNLP server...", flush=True)
    test_parse = corenlp_parse("Russia attacked Ukraine.")
    if not test_parse:
        print("ERROR: CoreNLP server not responding.", flush=True)
        print("Start it with:", flush=True)
        print('  cd stanford-corenlp-4.5.7', flush=True)
        print('  java -mx4g -cp "*" '
              'edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000',
              flush=True)
        return
    print(f"  OK: {test_parse[:60]}...", flush=True)

    # XML 생성
    os.makedirs('results/petrarch', exist_ok=True)
    input_xml = os.path.abspath(r'results\petrarch\input.xml')
    output_txt = os.path.abspath(r'results\petrarch\output.txt')

    print("\nParsing with CoreNLP (this takes a while)...", flush=True)
    valid = build_petrarch_xml(all_cases, input_xml)
    print(f"  Parsed: {valid} / {len(all_cases)}", flush=True)

    # PETRARCH3 실행
    print("\nRunning PETRARCH3...", flush=True)
    ok, output_txt, stderr = run_petrarch(input_xml, 'petr_out.txt')
    print(f"  Return: {'OK' if ok else 'FAILED'}", flush=True)
    if stderr:
        print(f"  stderr: {stderr[:500]}", flush=True)
    print(f"  Output: {output_txt}", flush=True)

    # 결과 파싱
    events = parse_petrarch_output(output_txt)
    print(f"  Coded events: {sum(len(v) for v in events.values())}",
          flush=True)
    print(f"  Cases with events: {len(events)} / {len(all_cases)}",
          flush=True)

    # 도메인별 평가
    domains = ['politics', 'business', 'disaster', 'sports',
               'science', 'health', 'culture', 'crime', 'environment']
    fields = ['entity1', 'relation', 'entity2', 'time', 'location']

    domain_stats = {d: {
        'total': 0, 'extracted': 0,
        **{f: 0 for f in fields}, 'overall': 0
    } for d in domains}

    print("\nEvaluating PETRARCH3 output...", flush=True)

    for i, case in enumerate(tqdm(all_cases, desc="Judging")):
        domain = case.get('domain', 'unknown')
        if domain not in domain_stats:
            continue

        domain_stats[domain]['total'] += 1

        ext_list = events.get(i, [])
        if not ext_list:
            continue

        domain_stats[domain]['extracted'] += 1
        ext = ext_list[0]
        gt = case['ground_truth']

        all_pass = True
        for field in fields:
            gt_val = gt.get(field, '')
            ext_val = ext.get(field, '')
            if binary_judge(gt_val, ext_val, field):
                domain_stats[domain][field] += 1
            else:
                all_pass = False

        if all_pass:
            domain_stats[domain]['overall'] += 1

    # 결과 출력
    print(f"\n{'=' * 100}", flush=True)
    print(f"=== PETRARCH3 Results (Rule-based) ===", flush=True)
    print(f"{'=' * 100}\n", flush=True)

    header = f"  {'Element':<12}"
    for d in domains:
        header += f" {d[:8]:>10}"
    print(header, flush=True)
    print(f"  {'-' * (12 + 11 * len(domains))}", flush=True)

    for field in fields + ['overall']:
        row = f"  {field:<12}"
        for d in domains:
            t = domain_stats[d]['total']
            v = domain_stats[d][field] / t * 100 if t else 0
            row += f" {v:>9.1f}%"
        print(row, flush=True)

    # 추출률
    row = f"  {'ext.rate':<12}"
    for d in domains:
        t = domain_stats[d]['total']
        v = domain_stats[d]['extracted'] / t * 100 if t else 0
        row += f" {v:>9.1f}%"
    print(row, flush=True)

    row = f"  {'(n)':<12}"
    for d in domains:
        row += f" {domain_stats[d]['total']:>10}"
    print(row, flush=True)

    # 전체 평균
    print(f"\n--- Overall Average ---", flush=True)
    for field in fields + ['overall']:
        tot = sum(domain_stats[d]['total'] for d in domains)
        val = sum(domain_stats[d][field] for d in domains)
        print(f"  {field}: {val/tot*100:.1f}%" if tot else f"  {field}: 0%",
              flush=True)

    tot = sum(domain_stats[d]['total'] for d in domains)
    ext = sum(domain_stats[d]['extracted'] for d in domains)
    print(f"  extraction rate: {ext/tot*100:.1f}%" if tot else "",
          flush=True)

    # 저장
    with open(r'results\extracted\petrarch_result.json',
              'w', encoding='utf-8') as f:
        json.dump(domain_stats, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료", flush=True)


if __name__ == "__main__":
    main()