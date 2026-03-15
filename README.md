# Secure Hybrid RAG Enterprise Assistant (PoC)

보안을 고려한 하이브리드 RAG 기반 기업용 AI 어시스턴트 Proof of Concept입니다.

**포폴 강조**: 질문 분석만 외부 API, **문서 검색·답변 생성은 로컬 AI(Ollama)** 로 동작합니다.  
→ 상세 구성·로컬 AI 연동 포인트: [docs/PORTFOLIO.md](docs/PORTFOLIO.md)

## 핵심 보안 원칙

**민감한 내부 문서는 절대 외부 LLM API로 전송되지 않습니다.**

- ✅ 외부 API로 전송되는 것: 사용자 질문만
- ❌ 외부 API로 전송되지 않는 것: 내부 문서, 정책, 매뉴얼

## 아키텍처

```
사용자 질문
    ↓
외부 LLM (Intent + Retrieval Planning) ← 질문만 전송
    ↓
구조화된 검색 계획 (JSON)
    ↓
로컬 벡터 데이터베이스 검색 ← 내부에서만 처리
    ↓
로컬 LLM 답변 생성 ← 내부에서만 처리
    ↓
최종 응답
```

## 기술 스택

- **Python**: 핵심 언어
- **Streamlit**: 웹 UI
- **FAISS**: 벡터 데이터베이스
- **SentenceTransformers**: 임베딩 생성
- **OpenAI API**: 의도 파악 및 쿼리 계획 (외부, 질문만 전송)
- **로컬 LLM (Ollama)**: 답변 생성 (내부 전담) — qwen2.5:3b/7b, 추론 시간 표시

## 설치 방법

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
```bash
cp .env.example .env
# .env: OPENAI_API_KEY (필수), OLLAMA_ENABLED, OLLAMA_MODEL 등 (선택)
```

3. (선택) 로컬 LLM 사용 시 Ollama 설치 및 모델 다운로드:
```bash
ollama pull qwen2.5:3b
```

4. 벡터 데이터베이스 초기화:
```bash
python init_vectordb.py
```

5. 애플리케이션 실행:
```bash
streamlit run app.py
```

## 프로젝트 구조

```
RAGforCompany/
├── app.py                    # Streamlit 메인 앱
├── requirements.txt          # Python 의존성
├── .env.example             # 환경 변수 예시
├── init_vectordb.py         # 벡터 DB 초기화 스크립트
├── services/
│   ├── __init__.py
│   ├── query_planner.py     # 외부 LLM 쿼리 플래너
│   ├── vector_search.py     # 로컬 벡터 검색
│   └── rag_generator.py     # 로컬 RAG 답변 생성
├── prompts/
│   ├── query_planner_prompt.txt
│   └── rag_answer_prompt.txt
└── data/
    ├── internal_docs/        # 내부 문서 (샘플)
    └── vectordb/            # FAISS 인덱스 저장소
```

## 모듈 설명

### query_planner.py
- 사용자 질문을 외부 OpenAI API로 전송
- 의도 파악 및 검색 계획 생성
- JSON 형식의 구조화된 검색 계획 반환

### vector_search.py
- FAISS 기반 로컬 벡터 검색
- 내부 문서는 로컬에서만 처리됨
- 관련 문서 청크 반환

### rag_generator.py
- **로컬 LLM(Ollama)** 를 사용한 답변 생성 — 내부 문서는 외부로 나가지 않음
- 검색된 문서(관련도 순) + 사용자 질문 → Ollama `/api/generate` 호출
- temperature·프롬프트로 환각·돈 얘기 억제; 추론 시간 노출

## 데모 시나리오

1. 사용자가 질문 입력: "출장비 정산은 어떻게 하나요?"
2. 외부 LLM이 질문만 분석하여 검색 계획 생성
3. 로컬 벡터 DB에서 관련 문서 검색
4. 로컬 LLM이 검색된 문서 기반으로 답변 생성
5. 투명성을 위해 의도, 카테고리, 소스 문서 표시

## 주의사항

- 이 프로젝트는 PoC이며, 프로덕션 환경에서 사용하려면 추가 보안 검토가 필요합니다.
- 로컬 LLM이 없는 경우 시뮬레이션 모드로 동작합니다.
