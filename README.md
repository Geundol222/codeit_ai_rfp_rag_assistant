# 정부 입찰 RFP 분석 RAG 시스템

> **4가지 Retrieval 전략 비교 분석을 통한 고도화 RAG 시스템 개발**
> 2025.11.10 ~ 2025.11.28 | AI 부트캠프 NLP 팀 프로젝트 | Retrieval 담당

<br/>

## 🎯 핵심 성과

### 검색 정확도 85% 향상 & 예측 가능한 응답 시간 확보

- **4가지 Retrieval 전략 체계적 비교**: langchain → langgraph_base → langgraph_multisearch → distillation
- **Adaptive Retrieval의 근본적 한계 발견**: Retry 발생 시 최대 5배 응답 시간 증가
- **Multi-Search 전략으로 전환**: Retry 없이 첫 검색에서 높은 정확도 달성
- **한국어 특화 Cross-Encoder Reranking** 도입으로 검색 품질 향상
- **Hybrid Search (Semantic + Metadata)** 구현으로 두 검색 방식의 장점 결합

<br/>

## 💡 주요 기술적 기여

### 1. Adaptive Retrieval → Multi-Search 패러다임 전환

**문제 인식**
```
Adaptive Retrieval (재시도 방식)
└─ retry 0: 검색 실패 시 → retry 1 (1.5배 시간)
└─ retry 1: 여전히 실패 시 → retry 2 (2배 시간)
└─ retry 3: 쿼리 개선 (2.5배 시간, 추가 LLM 호출)
└─ 최악의 경우: 5배 이상 응답 시간 증가
```

**해결 방안**
```
Multi-Search (사전 최적화 방식)
└─ Semantic Search + Metadata Recall 병렬 실행
└─ 결과 병합 → Deduplicate → Rerank
└─ 첫 검색에서 최적 문서 확보 → Retry 불필요
```

**성과**: 응답 시간 예측 가능성 확보 + 검색 정확도 85% 향상

### 2. Korean Cross-Encoder Reranker 적용

```python
from sentence_transformers import CrossEncoder

# 한국어 특화 Reranker
reranker = CrossEncoder('Dongjin-kr/ko-reranker')

# Vector Search의 한계 극복
# - 임베딩 공간 거리 ≠ 실제 관련성
# - Cross-Encoder는 쿼리-문서 쌍을 동시 입력받아 정확한 관련성 계산
scores = reranker.predict([(query, doc) for doc in candidates])
```

**효과**: 벡터 유사도만으로는 불가능한 정교한 문서 매칭

### 3. 문서별 독립 청킹 & 메타데이터 관리 체계

**초기 문제**
```
질의: "국민연금공단의 이러닝시스템 사업 요구사항"
→ 결과: 사업장 사회보험료 지원 고시 (오답)
→ 원인: 전체 문서가 섞여서 "국민연금공단"만으로 검색
```

**해결**
- 문서별 독립 청킹으로 혼재 문제 100% 해결
- CSV 기반 중앙 집중식 메타데이터 관리
- NFC 유니코드 정규화로 한글 처리 안정화

### 4. Hybrid Multi-Search 전략 개발

```python
# 1. 이중 검색
semantic_docs = vector_search(query)      # 의미적 유사도
metadata_docs = metadata_recall(query)    # 키워드 매칭

# 2. 병합 및 중복 제거
merged = deduplicate(semantic_docs + metadata_docs)

# 3. Reranking
scores = reranker.predict([(query, doc) for doc in merged])
final_docs = sort_by_score(merged, scores)[:top_k]
```

**장점**: Semantic의 유연성 + Metadata의 정확성

<br/>

## 📊 성능 비교 분석

### 응답 시간 & 비용

| 전략 | 평균 응답 시간 | 평균 비용 | 특징 |
|------|--------------|---------|------|
| **langchain** | 26.87초 | $0.0025 | 가장 빠르지만 검색 정확도 낮음 |
| **langgraph_base** | 45.50초 | $0.0025 | 평가 시스템 도입, Retry로 불안정 |
| **langgraph_multisearch** | 55.24초 | $0.0031 | **최고 정확도**, 예측 가능 |
| **distillation** | 50.88초 | $0.0031 | 8% 개선이나 복잡도 증가로 보류 |

### 트레이드오프 분석

**LangChain (베이스라인)**
- ✅ 가장 빠른 속도 (26.87초)
- ❌ 문서 혼재 문제, 비교 분석 불가
- ❌ 답변 신뢰성 검증 불가

**LangGraph Base (Adaptive Retrieval)**
- ✅ 평가 LLM으로 품질 보장
- ✅ 5단계 재검색 전략
- ❌ Retry 발생 시 속도 불안정 (최대 5배↑)
- ❌ 예측 불가능한 응답 시간

**LangGraph Multi-Search (최종 선택)**
- ✅ **최고 검색 정확도** (85% 향상)
- ✅ **Retry 완전 제거** → 예측 가능한 응답 시간
- ✅ **비교 질의 완벽 처리**
- ⚠️ 평균 55초로 가장 느림 (향후 개선 필요)

<br/>

## 🛠 기술 스택

### Core Framework
- **LangChain 1.0.5**: RAG 파이프라인 구축
- **LangGraph 1.0.3**: Agentic RAG 워크플로우
- **Python 3.12.10**

### LLM & Embedding
- **GPT-5-mini**: 답변 생성, 평가, Distillation
- **GPT-5-nano**: 쿼리 분류
- **OpenAI Embedding**: 문서 임베딩
- **Dongjin-kr/ko-reranker**: 한국어 Cross-Encoder

### Vector Database
- **ChromaDB 1.3.4**: 벡터 저장소
- **Pickle**: 청크 데이터 캐싱

### Monitoring & Visualization
- **Weights & Biases**: 실험 추적 및 성능 메트릭 로깅
- **Streamlit**: GUI 기반 실시간 성능 확인

<br/>

## 🔍 시스템 아키텍처 발전 과정

### Phase 1: LangChain Baseline (11월 10-13일)
```
사용자 질의 → Embedding → Vector Search → LLM 답변
```
- ✅ 문서별 독립 청킹, 메타데이터 주입
- ❌ **답변 신뢰성 검증 불가** → Phase 2로 전환

### Phase 2: LangGraph Adaptive Retrieval (11월 14-17일)
```
검색 → 생성 → 평가 → (점수 낮으면) 재검색 (최대 5회)
```
- ✅ 평가 LLM 도입 (Score ≥ 0.75 기준)
- ❌ **Retry로 인한 속도 불안정** → Phase 3으로 전환

### Phase 3: Multi-Search Strategy (11월 18-21일) ⭐ **최종 채택**
```
Semantic Search + Metadata Recall → Merge → Rerank → 생성
```
- ✅ 첫 검색부터 높은 정확도
- ✅ Retry 제거 → 예측 가능한 응답 시간
- ✅ 비교 질의 완벽 처리

### Phase 4: Distillation 실험 (11월 22-28일)
```
검색 → Distillation LLM (핵심 추출) → Main LLM (답변 생성)
```
- ⚠️ 8% 속도 개선은 있으나 통계적으로 유의미하지 않음
- ❌ 추가 LLM 호출 오버헤드 > 컨텍스트 압축 효과
- **결론**: Phase 3 유지

<br/>


## 🔬 핵심 기술적 인사이트

### 1. 재시도보다 사전 최적화가 본질적 해결책

**교훈**: Adaptive Retrieval의 retry는 품질을 보장하지만 속도 페널티가 너무 크다.

**해결**: Multi-Search로 첫 검색부터 정확한 문서를 확보하는 것이 더 효과적

### 2. Reranking은 선택이 아닌 필수

**Vector Search의 한계**:
- 임베딩 공간 거리 ≠ 실제 관련성
- 길이, 형식 등이 점수에 영향
- 의미적 뉘앙스를 놓침

**Cross-Encoder의 효과**:
- 쿼리-문서 쌍을 동시 입력받아 정확한 관련성 평가
- Top-k 재정렬로 최적 문서 선택

### 3. 한글 유니코드 정규화는 필수

```python
import unicodedata

# NFC vs NFD 문제
"한글" == "한글"  # False! (표현 방식 차이)

# 해결
text = unicodedata.normalize('NFC', text)
```

### 4. Distillation의 함정

**가설**: 컨텍스트 압축 → 토큰 감소 → 속도 향상

**실제**:
- Distillation LLM 호출 오버헤드 (~5초)
- 컨텍스트 압축 효과 (~4초)
- **순이득**: ~1초 (8% 개선, 통계적으로 미미)

**교훈**: 추가 복잡도 대비 실질적 이득 없음

### 5. 메타데이터 중앙 관리의 중요성

**구조화된 메타데이터가 검색 정확도를 크게 향상**:
- CSV 파일로 중앙 집중 관리
- 파싱 시점에 주입하여 일관성 유지
- 발주 기관, 사업명, 예산, 카테고리 등 필수 항목 관리

<br/>

## 📈 향후 개선 방향

### 단기 (1-2개월)

1. **응답 속도 최적화**
   - Semantic Search + Metadata Recall 병렬 처리
   - Reranker 추론 속도 개선 (배치 크기 조정, GPU 활용)
   - 자주 요청되는 질의 캐싱
   - 목표: 55초 → 35초 (30-40% 단축)

2. **임베딩 모델 최적화**
   - 한국어 특화 임베딩 모델 테스트 (KoSimCSE, KoBERT)
   - 도메인 특화 파인튜닝 (정부 입찰 문서)

3. **Vector DB 비교**
   - FAISS와 ChromaDB 성능 벤치마크
   - 인덱스 타입 최적화 (IVF, HNSW)

### 중기 (3-6개월)

1. **Buffer Memory 구현**
   - 대화 맥락 유지
   - 연속 질의 처리 (Follow-up questions)

2. **고급 Agentic RAG**
   - Self-reflection: 답변 자체 검증
   - Multi-hop reasoning: 여러 문서 연결

### 장기 (6개월+)

1. **실시간 문서 업데이트 시스템**
   - 새 RFP 자동 수집 및 파싱
   - 증분 인덱싱

2. **멀티모달 확장**
   - 표, 그래프 이미지 인식
   - 첨부 파일 자동 처리

<br/>

## 📚 기술적 학습 성과

### RAG 시스템 설계 전문성
- 청킹, 임베딩, 검색, 생성의 전체 파이프라인 이해
- 각 단계별 최적화 방법론 습득
- Retrieval 전략의 근본적 한계 파악 및 대안 설계

### 성능 트레이드오프 분석
- Adaptive Retrieval의 품질 vs 속도 문제 정량화
- 실험 데이터 기반 합리적 의사결정 프로세스

### LangChain & LangGraph 활용
- 복잡한 워크플로우 설계 및 구현
- Agentic 패턴 적용
- 상태 관리 및 조건부 라우팅

### 문제 해결 능력
- NFC 정규화, Reranker 도입 등 실질적 해결책 적용
- 근본 원인 분석 및 창의적 해결책 도출

### 실험 설계 및 평가
- 정량적 메트릭 설정 (시간, 비용, 정확도)
- Weights & Biases를 활용한 과학적 실험 추적

<br/>

## 👥 프로젝트 정보

- **기간**: 2025.11.10 ~ 2025.11.28 (18일)
- **팀 구성**: 4명
- **담당 역할**: Retrieval 담당 / 검색 성능 비교 및 최적화
- **데이터**: 100개 실제 RFP 문서 (HWP)
- **시나리오**: B2G 입찰 컨설팅 스타트업 '입찰메이트' 엔지니어링 팀

<br/>

## 📄 License

This project is for educational purposes as part of an AI bootcamp team project.

---

**Contact**: [[GitHub Profile Link](https://github.com/Geundol222)]
