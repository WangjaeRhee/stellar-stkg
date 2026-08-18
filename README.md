# STELLAR: LLM 자동 추론 기반 시공간 지식그래프 오중항 추출 프레임워크

## STELLAR: Spatio-Temporal Extraction via LLM-Automated Reasoning for Knowledge Graph Construction

### 이왕재, 오서진, 이광일†
한국해양대학교 AI공학부, Cybermarine System LAB

---

## 요 약

지식그래프(Knowledge Graph)는 비정형 텍스트에서 구조화된 지식을 추출하여 기계가 이해할 수 있는 형태로 표현하는 핵심 기술이다. 기존 대규모 언어모델(Large Language Model, LLM) 기반 지식그래프 구축 연구는 삼중항(Triple) 또는 시간 정보를 포함한 사중항(Quadruple) 추출에 집중하고 있으며, 공간 정보까지 포함한 오중항(Quintuple) 추출에 대한 연구는 부족한 실정이다. 본 연구는 이러한 한계를 보완하기 위해 비정형 텍스트에서 시공간 오중항을 자동으로 추출하는 프레임워크 STELLAR(Spatio-Temporal Extraction via LLM-Automated Reasoning)를 제안한다. STELLAR는 비정형 텍스트에서 개체, 관계, 시간, 공간 정보를 동시에 추출하는 문제를 해결하기 위해 원자적 사실 분해(Atomic Fact Decomposition)와 단계적 사고 추론(Chain-of-Thought, CoT)을 결합한 다단계 지식 추출 모듈과, 추출된 오중항의 중복 및 유사 개체 통합 문제를 해결하기 위해 개체 정규화(Entity Canonicalize) 기반 지식 융합 모듈로 구성된다. 9개 도메인, 450개 텍스트 대상 실험에서 요소 평균 F1 87.0%, 오중항 텍스트 유사도 F1 97.0%, 오중항 완전 일치율 50.7%를 달성한다. 동일 조건에서 ATOM과 4요소 기준으로 비교한 결과, STELLAR 72.0%, ATOM 8.0%로 64.0%p의 차이를 보인다. 동일 프레임워크에서 LLM만 교체한 결과, Qwen2.5-7B 51.0%에서 GPT-4o-mini 및 GPT-4o 60.0%로 9%p 향상된다.

키워드 : 대규모 언어모델, 시공간 지식그래프, 오중항 추출, 단계적 사고 추론, 지식 융합

---

## Abstract

This study proposes STELLAR, a framework that extends existing triple and quadruple extraction by automatically extracting spatio-temporal quintuples from unstructured text using Large Language Models (LLMs). STELLAR consists of two core modules: a Knowledge Extraction module combining Atomic Fact Decomposition with multi-stage Chain-of-Thought (CoT) reasoning, and a Knowledge Fusion module applying Entity Canonicalize and cosine similarity-based deduplication. In experiments on 450 unstructured text samples across 9 domains, STELLAR achieves Element Average F1 of 87.0%, G-BERTScore F1 of 97.0%, and Quintuple Strict Match of 50.7%. Under identical conditions (Qwen2.5-7B), STELLAR achieves 72.0% compared to ATOM's 8.0% on 4-element criteria, noting that ATOM was designed for GPT-4. Replacing only the LLM component improves overall from 51.0% to 60.0% with GPT-4o-mini and GPT-4o.

Keywords : Chain-of-Thought, Knowledge Fusion, Large Language Model, Quintuple Extraction, Spatio-Temporal Knowledge Graph

---

## 1. 서 론

최근 대규모 언어모델(Large Language Model, LLM)의 발전과 함께 지식그래프(Knowledge Graph, KG)를 자동으로 구축하는 연구가 활발히 진행되고 있다[3, 6]. 지식그래프는 현실 세계의 지식을 개체(Entity)와 관계(Relation)로 구조화하여 표현하는 기술로, 검색, 추천, 질의응답 등에서 핵심 인프라로 활용되고 있다[16].

그러나 기존 연구는 대부분 삼중항(Triple) 수준에 머물러 있으며, 사중항(Quadruple)까지 확장한 연구도 제한적이다. 현실 세계의 지식은 시간과 공간의 맥락 속에서 존재하며, 동일한 사건이라도 시점과 장소에 따라 의미가 달라진다. 시간 지식그래프(Temporal Knowledge Graph, TKG)에 대한 연구는 진행되어 왔으나[4], 공간 정보를 포함한 시공간 지식그래프(Spatio-Temporal Knowledge Graph, STKG)의 자동 구축 연구는 부족한 실정이다[1, 2]. STKG에서 오중항(Quintuple)은 행위 주체, 관계, 행위 대상, 시간, 공간의 5가지 요소로 구성되며, "누가, 무엇을, 누구에게, 언제, 어디서"의 이벤트 정보를 구조화하여 표현한다.

본 논문에서는 이러한 한계를 보완하기 위해 LLM을 활용하여 비정형 텍스트에서 시공간 오중항을 자동으로 추출하는 프레임워크 STELLAR(Spatio-Temporal Extraction via LLM-Automated Reasoning)를 제안하고 있다. STELLAR는 지식 추출과 지식 융합의 2개 핵심 모듈로 구성된다. 원자적 사실 분해(Atomic Fact Decomposition)와 단계적 사고 추론(Chain-of-Thought, CoT)[8]을 결합한 다단계 파이프라인을 통해 개체, 관계, 시간, 공간 정보를 체계적으로 추출한다.

본 논문의 구성은 다음과 같다. 제 2장에서는 STKG 및 LLM 기반 지식그래프 구축의 관련 연구를 다루고 있으며, 제 3장에서는 제안 프레임워크의 구조와 각 모듈에 대해서 설명하고 있다. 제 4장에서는 실험 환경과 결과에 대해서 기술하고 있다. 마지막으로 제 5장에서는 결론과 향후 연구과제에 대해서 다루고 있다.

---

## 2. 본 론

### 2.1 시공간 지식그래프

STKG는 시간과 공간을 동시에 명시적으로 표현하는 지식그래프이다. Hao 등[1]은 STKG 구축 및 임베딩 방법에 대한 종합 서베이를 수행하고 있으며, 데이터 융합, 의미 모델링, 표현 학습, 추론의 통합 프레임워크를 제안하고 있다. Plamper 등[2]은 STKG 모델에 대한 체계적 리뷰를 통해 모델링 가이드라인과 미해결 과제를 제시하고 있다. Liu 등[15]은 STKG에서 LLM을 활용한 멀티홉 추론을 수행하고 있으나, STKG 자동 구축이 아닌 추론에 초점을 맞추고 있다. 기존 연구들은 STKG의 오중항을 임베딩하거나 추론하는 데 집중하고 있으며, 비정형 텍스트에서 오중항을 자동으로 추출하여 STKG를 구축하는 방향의 연구는 부족한 상황이다.

### 2.2 LLM 기반 지식그래프 구축

LLM 기반 지식그래프 구축은 최근 활발히 연구되고 있다[3, 6]. Pan 등[3]은 LLM과 지식그래프의 통합 로드맵을 제시하고 있으며, Anuyah 등[7]은 전문가 주석 삼중항에서 F1 92.4%를 달성하고 있다.

ATOM[4]은 원자적 사실 분해를 통해 텍스트를 원자적 사실로 분해하고 TKG를 구축하는 프레임워크를 제안하고 있다. ATOM은 GPT-4.1 환경에서 사실적 F1(Factual F1) 0.638을 보고하고 있으나[4], 공간 정보를 포함하지 않는 한계가 있다.

EDC[5]는 추출-정의-정규화(Extract-Define-Canonicalize)의 3단계 프레임워크를 제안하여 LLM 기반 지식그래프의 스키마 정규화 문제를 해결하고 있다. 본 연구는 EDC의 3단계 중 정규화(Canonicalize)만을 선택적으로 적용하고 있다. 추출(Extract) 단계는 삼중항만 지원하여 오중항에 적합하지 않으며, 정의(Define) 단계는 사전 정의된 스키마에 의존하므로 도메인 독립성을 저해할 수 있기 때문이다.

### 2.3 단계적 사고 추론 기반 정보 추출

Wei 등[8]이 제안한 CoT 추론은 LLM의 복잡한 추론 능력을 향상시키는 기법이다. Ma 등[9]은 CoT를 활용한 관계 추출(CoT-ER)을 통해 소수 샘플 환경에서 지도 학습에 필적하는 성능을 달성하고 있다. Li 등[10]은 개체와 관계 추출을 분리하는 다단계 추론(ERA-CoT)을 제안하고 있으며, Wang 등[17]은 자기 일관성(Self-Consistency)을 제안하여 다수결 기반 추론으로 성능을 향상시키고 있다.

### 2.4 LLM 기반 평가

LLM을 활용한 자동 평가는 인간 평가와 높은 상관관계를 보이는 것으로 알려져 있다. Liu 등[11]은 GPT-4 기반 자연어 생성 평가(G-Eval)에서 인간 판단과 0.843의 상관관계를 달성하고 있다. EvidentlyAI[12]는 이진 판정(Binary) 방식이 가장 신뢰도 높은 LLM 기반 평가임을 권장하고 있다. Zhang 등[18]이 제안한 BERTScore는 텍스트 간 의미적 유사도를 측정하는 지표로, 지식그래프 평가에서 삼중항 수준의 텍스트 유사도 측정에 활용되고 있다.

---

## 3. 제안 프레임워크

### 3.1 프레임워크 개요

제안 프레임워크 STELLAR는 Fig. 1과 같이 지식 추출과 지식 융합의 2개 핵심 모듈로 구성된다. 지식 추출 모듈은 비정형 텍스트에서 개체, 관계, 시간, 공간을 동시에 추출하는 문제를 다단계 파이프라인으로 해결하며, 지식 융합 모듈은 추출된 오중항의 중복 및 유사 개체 통합 문제를 코사인 유사도 기반으로 해결한다. 추출된 오중항은 그래프 데이터베이스 저장 및 자연어 질의 등의 응용 계층에서 활용할 수 있다.

### 3.2 모듈 1: 지식 추출

지식 추출 모듈은 ATOM[4]의 원자적 사실 분해와 ERA-CoT[10]의 다단계 추론을 결합한 4단계 파이프라인으로 구성된다.

#### 3.2.1 1단계: 원자적 사실 분해

ATOM[4]의 방법론을 적용하여 입력 텍스트를 원자적 사실 집합으로 분해한다. 각 원자적 사실은 하나의 이벤트만 포함하는 독립적 문장이다.

#### 3.2.2 2단계: 개체 추출

CoT 추론[8]을 활용하여 각 원자적 사실에서 행위 주체(entity1)와 행위 대상(entity2)을 추출한다. CoT 추론의 핵심은 행위 대상 추출 시 문법적 목적어가 아닌 의미적 대상을 식별하는 것이다. 예를 들어, "The United States imposed sanctions on Iran"에서 문법적 목적어는 "sanctions"이지만, 의미적 행위 대상은 "Iran"이다.

#### 3.2.3 3단계: 관계 추출

추출된 행위 주체와 행위 대상 사이의 관계를 식별한다. 관계는 짧은 동사구 형태로 추출되며, LLM 기반 관계 정규화를 통해 표준 형식으로 변환한다.

#### 3.2.4 4단계: 시간 및 공간 추출

본 연구에서 공간 정보는 이벤트가 물리적으로 발생하거나 개시된 국가로 정의한다. 시간과 공간 정보는 원자적 사실 분해 과정에서 손실될 수 있으므로, 분해된 원자적 사실이 아닌 원본 텍스트에서 추출한다.

#### 3.2.5 정규화 및 후보정

추출된 오중항에 대해 3종의 LLM 기반 정규화를 수행한다. 공간 정규화는 도시 및 지역을 국가 단위로 변환하고, 개체 정규화는 하위 조직 및 도시를 상위 수준 행위자로 매핑하며, 관계 정규화는 자연어 표현을 표준 형식으로 변환한다.

### 3.3 모듈 2: 지식 융합

EDC[5]의 정규화를 적용하여 의미적으로 동일한 개체를 통합한다. 코사인 유사도 0.9 이상인 개체 쌍을 동일 개체로 판단하여 대표 표현으로 통합한다. 이후 ATOM[4]의 가중 코사인 유사도 기반 중복 제거를 적용하며, 가중 유사도가 임계값 0.85 이상이면 중복으로 판단하여 제거한다.

### 3.4 평가 방법

본 연구에서는 3가지 평가 지표를 사용한다. 첫째, 요소별 정밀도/재현율/F1은 오중항의 각 요소를 독립적으로 평가한다. 둘째, 오중항 텍스트 유사도(G-BERTScore)는 오중항 전체를 하나의 텍스트로 변환하여 정답과의 의미적 유사도를 BERTScore[18]로 측정한다. 셋째, 오중항 완전 일치율(Quintuple Strict Match)은 5개 요소가 모두 PASS인 경우만 PASS로 판정하는 엄격한 기준이다.

---

## 4. 실험 및 결과

### 4.1 실험 환경

GPU: NVIDIA RTX 2060 12GB, LLM: Qwen2.5-7B(Ollama)/GPT-4o-mini/GPT-4o(API), 임베딩: all-MiniLM-L6-v2, 그래프 DB: Neo4j Desktop 2.2.0

### 4.2 데이터셋

비정형 텍스트 450개를 9개 도메인(정치, 비즈니스, 자연재해, 스포츠, 과학, 보건, 문화, 범죄, 환경 각 50개)에서 수집하였다. 정답 데이터는 반자동 방식(GPT 생성 및 수작업 검수)으로 작성하였다.

### 4.3 ATOM 대비 동일 조건 비교

본 실험은 ATOM[4]과의 프레임워크 구조적 차이가 성능에 미치는 영향을 분석하기 위한 것이다. ATOM은 GPT-4 기반으로 설계된 TKG 구축 프레임워크로, GPT-4.1 환경에서 사실적 F1 0.638을 보고하고 있다[4]. 소규모 모델(Qwen2.5-7B) 환경에서 동일 조건 비교 결과, 4요소 기준 STELLAR 72.0%, ATOM 8.0%로 64.0%p의 차이를 보이고 있다. 5요소 기준에서도 62.0%를 달성하고 있다.

### 4.4 요소 기여도 분석

CoT 추론이 +22.0%p로 가장 큰 기여를 보이고 있으며, 개체 정규화가 +16.0%p로 두 번째 기여를 보이고 있다.

### 4.5 자기 일관성 횟수별 비교

3회 추론이 52.0%로 가장 높은 성능을 보이고 있으며, 5회 이상에서는 수확체감 현상을 확인하고 있다.

### 4.6 모델 비교

모델 교체만으로 전체 일치율이 51.0%에서 60.0%로 9.0%p 향상되고 있다.

### 4.7 도메인 독립성 검증

9개 도메인, 450개 텍스트에서 전체 평균 요소 평균 F1 87.0%, 오중항 완전 일치율 50.7%를 달성하고 있다. 시간(99.8%)과 공간(96.2%)은 모든 도메인에서 안정적이다. 스포츠(88%), 문화(72%), 보건(62%)에서 높은 성능을 보이고 있다. 범죄(20%)에서 낮은 성능은 행위 주체(40%)에 기인하며, LLM이 조직명을 상위 국가로 정규화하는 경향 때문이다.

### 4.8 오중항 텍스트 유사도 분석

전체 평균 G-BERTScore F1 97.0%를 달성하고 있으며, 모든 도메인에서 95.9% 이상을 보이고 있다. 이는 오중항 완전 일치율(50.7%)과 대조적인 결과로, 추출된 오중항이 의미적으로는 정답과 매우 유사함을 나타내고 있다.

### 4.9 STKG 시각화

비정형 텍스트에서 추출한 오중항으로 STKG를 구축하였다. Fig. 6은 United States를 중심으로 한 네트워크 구조를 보이고 있으며, Fig. 7은 실제 저장된 오중항 데이터를 보이고 있다.

---

## 5. 결 론

본 논문에서는 LLM을 활용한 시공간 오중항 자동 추출 프레임워크 STELLAR를 제안하고 있다. 9개 도메인, 450개 텍스트에서 요소 평균 F1 87.0%, G-BERTScore F1 97.0%, 오중항 완전 일치율 50.7%를 달성하고 있다. 동일 조건에서 ATOM과 4요소 기준 비교 시 64.0%p의 차이를 보이고 있으며, 모델 교체만으로 9.0%p의 성능 향상을 확인하고 있다.

향후 연구과제로는 대형 모델을 활용한 비교 실험, 도메인 특화 파인튜닝을 통한 성능 향상, 개체 정의 확장을 통한 도메인 커버리지 확대 등이 있다. 또한, 실시간 데이터를 처리하여 STKG를 동적으로 업데이트하는 시스템으로 발전시킬 수 있으며, 이는 상황 인식 및 재난 대응 등의 실시간 의사결정 지원에 활용될 수 있다.

---

## REFERENCES

[1] H. Hao, G. Si, L. Lu, Q. Liu, and F. Zhou, "From data fusion to dynamic reasoning: A survey on spatio-temporal knowledge graph construction and embedding methods," Neurocomputing, vol. 658, p. 131590, Jan. 2025. DOI: 10.1016/j.neucom.2024.131590.

[2] P. Plamper, M. Ley, and H. Paulheim, "A survey on spatio-temporal knowledge graph models," arXiv preprint arXiv: 2512.16487, Dec. 2025. DOI: 10.48550/arXiv.2512.16487.

[3] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, "Unifying large language models and knowledge graphs: A roadmap," IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3580-3599, Jul. 2024. DOI: 10.1109/TKDE.2024.3352100.

[4] Y. Lairgi, N. Labroche, A. Maurer, and A. Mazeika, "ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs," in Proceeding of the 2026 Conference of the European Chapter of the Association for Computational Linguistics, Zagreb, Croatia, pp. 1-15, 2026.

[5] B. Zhang and H. Soh, "Extract, define, canonicalize: An LLM-based framework for knowledge graph construction," in Proceeding of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, FL, pp. 1-12, 2024.

[6] Y. Jiang et al., "LLM-empowered knowledge graph construction: A survey," arXiv preprint arXiv: 2510.20345, Oct. 2025. DOI: 10.48550/arXiv.2510.20345.

[7] O. Anuyah, I. Akujuobi, and X. L. Dong, "CoDe-KG: A benchmark for evaluating LLM-based knowledge graph construction," in Proceeding of the 2025 Conference on Empirical Methods in Natural Language Processing, Suzhou, China, pp. 1-15, 2025.

[8] J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," in Advances in Neural Information Processing Systems, vol. 35, pp. 24824-24837, 2022.

[9] X. Ma, J. Li, and M. Zhang, "Chain of thought with explicit evidence reasoning for few-shot relation extraction," in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, pp. 2334-2352, 2023.

[10] Y. Li, F. Li, and B. Chen, "ERA-CoT: Improving chain-of-thought through entity relationship analysis," in Proceeding of the 62nd Annual Meeting of the Association for Computational Linguistics, Bangkok, Thailand, pp. 1-12, 2024.

[11] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, "G-Eval: NLG evaluation using GPT-4 with better human alignment," in Proceeding of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, pp. 2511-2522, 2023.

[12] EvidentlyAI. LLM-as-Judge best practices [Internet]. Available: https://www.evidentlyai.com/blog/llm-as-a-judge.

[13] HuggingFace. LLM Judge Cookbook [Internet]. Available: https://huggingface.co/learn/cookbook/en/llm_judge.

[14] ICEWS. Integrated Crisis Early Warning System [Internet]. Available: https://dataverse.harvard.edu/dataverse/icews.

[15] Z. Liu et al., "Spatiotemporal knowledge graph multi-hop reasoning based on large language models," Expert Systems with Applications, vol. 267, p. 126211, Apr. 2025. DOI: 10.1016/j.eswa.2025.126211.

[16] M. Hofer, D. Obraczka, A. Saeedi, H. Köpcke, and E. Rahm, "Construction of knowledge graphs: Current state and challenges," Information, vol. 15, no. 8, p. 509, Aug. 2024. DOI: 10.3390/info15080509.

[17] X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," in Proceeding of the 11th International Conference on Learning Representations, Kigali, Rwanda, pp. 1-25, 2023.

[18] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in Proceeding of the 8th International Conference on Learning Representations, Addis Ababa, Ethiopia, pp. 1-43, 2020.
