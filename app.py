"""
Secure Hybrid RAG Enterprise Assistant
Streamlit 메인 애플리케이션

금융사 엔터프라이즈 내규 시스템 PoC
- 계층적 문서 구조 (정책 > 규정 > 지침 > 매뉴얼)
- 외부 LLM: 질문 분석 및 검색 계획
- 로컬 LLM: 문서 검색 및 답변 생성

보안 원칙:
- 외부 API로는 사용자 질문만 전송
- 내부 문서는 로컬에서만 처리
"""

import os
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv

from services.query_planner import generate_retrieval_plan, get_categories_for_search, get_doc_levels_for_search
from services.vector_search import search_documents, get_vector_store
from services.rag_generator import generate_answer, check_ollama_available

load_dotenv()


st.set_page_config(
    page_title="금융사 내규 AI Assistant",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .security-badge {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
    }
    .complexity-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .complexity-simple { background-color: #D1FAE5; color: #065F46; }
    .complexity-moderate { background-color: #FEF3C7; color: #92400E; }
    .complexity-complex { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)


def check_vectordb_initialized() -> bool:
    """벡터 데이터베이스 초기화 여부를 확인합니다."""
    vectordb_path = Path(__file__).parent / "data" / "vectordb" / "documents.pkl"
    return vectordb_path.exists()


def main():
    st.markdown('<p class="main-header">🏦 금융사 내규 AI Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secure Hybrid RAG - 기업 내규 지능형 검색 시스템</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="security-badge">
            🛡️ 보안 모드: 내부 문서는 외부 API로 전송되지 않습니다
        </div>
        """, unsafe_allow_html=True)
    
    if not check_vectordb_initialized():
        st.warning("""
        ⚠️ **벡터 데이터베이스가 초기화되지 않았습니다.**
        
        터미널에서 다음 명령을 실행해 주세요:
        ```
        python init_vectordb.py
        ```
        """)
        return
    
    st.markdown('<p class="section-header">💬 질문 입력</p>', unsafe_allow_html=True)
    
    example_questions = [
        "해외 출장 가서 현지 고객사 접대비 150만원 썼는데 어떻게 정산해요?",
        "연차휴가 신청은 어떻게 하나요?",
        "협력업체 직원에게 시스템 접근권한 부여하려면?",
        "보안사고 발생 시 어떻게 대응해야 하나요?",
        "법인카드 한도 증액 신청 방법은?",
        "재택근무 시 보안 준수사항이 뭐예요?",
        "신입사원 입사 첫 날 뭐해요?",
    ]
    
    col1, col2 = st.columns([4, 1])
    with col1:
        selected_example = st.selectbox(
            "예시 질문 선택 (선택 사항)",
            ["직접 입력"] + example_questions,
            key="example_select"
        )
    
    if selected_example != "직접 입력":
        question = st.text_area(
            "질문을 입력하세요",
            value=selected_example,
            height=100,
            key="question_input"
        )
    else:
        question = st.text_area(
            "질문을 입력하세요",
            placeholder="예: 해외 출장 중 접대비 150만원 정산 방법이 궁금합니다.",
            height=100,
            key="question_input"
        )
    
    if st.button("🔍 질문하기", type="primary", use_container_width=True):
        if not question.strip():
            st.error("질문을 입력해 주세요.")
            return
            
        process_question(question)


def process_question(question: str):
    """질문을 처리하고 결과를 표시합니다."""
    
    with st.spinner("🤖 AI가 질문을 분석하고 있습니다..."):
        
        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Step 1: AI 질문 분석 (외부 LLM)</p>', unsafe_allow_html=True)
        
        st.info("ℹ️ 이 단계에서는 **사용자 질문만** 외부 API로 전송됩니다.")
        
        with st.spinner("검색 계획 생성 중..."):
            retrieval_plan = generate_retrieval_plan(question)
        
        complexity = retrieval_plan.get('query_complexity', 'simple')
        complexity_kr = {"simple": "단순", "moderate": "보통", "complex": "복잡"}.get(complexity, complexity)
        complexity_class = f"complexity-{complexity}"
        
        st.markdown(f"""
        <span class="complexity-badge {complexity_class}">질문 복잡도: {complexity_kr}</span>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 분석된 의도**")
            st.markdown(f"""
            <div class="info-box">
                {retrieval_plan.get('user_intent', '')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**📁 검색 대상 카테고리**")
            primary_cats = retrieval_plan.get('primary_categories', [])
            related_cats = retrieval_plan.get('related_categories', [])
            
            cats_display = []
            for cat in primary_cats:
                cats_display.append(f"**{cat}**")
            for cat in related_cats:
                cats_display.append(f"{cat} (연관)")
            
            st.markdown(f"""
            <div class="info-box">
                {", ".join(cats_display) if cats_display else "전체 검색"}
            </div>
            """, unsafe_allow_html=True)
            
            doc_levels = retrieval_plan.get('document_levels_needed', [])
            levels_kr = {"POLICY": "정책", "REGULATION": "규정", "GUIDELINE": "지침", "MANUAL": "매뉴얼", "FORM": "양식"}
            levels_display = [levels_kr.get(l, l) for l in doc_levels]
            
            st.markdown("**📑 검색 문서 등급**")
            st.markdown(f"""
            <div class="info-box">
                {" → ".join(levels_display) if levels_display else "전체"}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🔍 생성된 검색 쿼리**")
            for query in retrieval_plan.get('search_queries', []):
                st.markdown(f"- {query}")
            
            st.markdown("**📌 핵심 포커스**")
            for point in retrieval_plan.get('focus_points', []):
                st.markdown(f"- {point}")
            
            special_conds = retrieval_plan.get('special_conditions', [])
            if special_conds:
                cond_kr = {"overseas": "🌏 해외", "urgent": "🚨 긴급", "related_party": "🔗 관계사", 
                          "entertainment": "🍽️ 접대", "external_party": "👥 외부인"}
                conds_display = [cond_kr.get(c, c) for c in special_conds]
                st.markdown("**⚠️ 특수 조건**")
                st.markdown(f"""
                <div class="info-box" style="background-color: #FEF3C7;">
                    {", ".join(conds_display)}
                </div>
                """, unsafe_allow_html=True)
            
            amount = retrieval_plan.get('amount_involved')
            if amount:
                st.markdown("**💰 관련 금액**")
                st.markdown(f"""
                <div class="info-box">
                    {amount:,}원
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">📚 Step 2: 로컬 문서 검색</p>', unsafe_allow_html=True)
        
        st.success("✅ 이 단계는 **로컬 환경**에서만 수행됩니다. 문서 내용은 외부로 전송되지 않습니다.")
        
        with st.spinner("문서 검색 중..."):
            search_categories = get_categories_for_search(retrieval_plan)
            search_levels = get_doc_levels_for_search(retrieval_plan)
            
            retrieved_docs = search_documents(
                search_queries=retrieval_plan.get('search_queries', []),
                categories=search_categories if search_categories else None,
                doc_levels=search_levels if search_levels else None,
                top_k=3,
                use_hierarchy=retrieval_plan.get('cross_reference_check', False),
            )
        
        if retrieved_docs:
            st.markdown(f"**검색된 문서: {len(retrieved_docs)}건**")
            
            for i, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get('metadata', {})
                doc_id = metadata.get('doc_id', '')
                doc_name = metadata.get('doc_name', '')
                doc_level = doc.get('doc_level', 'GENERAL')
                level_kr = {"POLICY": "정책", "REGULATION": "규정", "GUIDELINE": "지침", 
                           "MANUAL": "매뉴얼", "FORM": "양식"}.get(doc_level, doc_level)
                
                title = f"📄 [{level_kr}] {doc_id} {doc_name}" if doc_id else f"📄 {doc['source']}"
                
                with st.expander(f"{title} (관련도: {doc['score']:.2f})"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"**카테고리:** {doc['category']}")
                        st.markdown(f"**문서등급:** {level_kr}")
                    with col2:
                        if metadata.get('department'):
                            st.markdown(f"**담당부서:** {metadata['department']}")
                        if metadata.get('approver'):
                            st.markdown(f"**승인권자:** {metadata['approver']}")
                    
                    st.markdown("**내용:**")
                    st.text(doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content'])
        else:
            st.warning("관련 문서를 찾지 못했습니다.")
        
        st.markdown("---")
        st.markdown('<p class="section-header">💡 Step 3: AI 답변 생성 (로컬 LLM)</p>', unsafe_allow_html=True)
        
        if not retrieved_docs:
            st.success("✅ 답변 생성도 **로컬 환경**에서만 수행됩니다.")
            answer = """**답변**

제공된 내규 문서에서 질문과 관련된 내용을 찾을 수 없습니다.

해당 업무에 대해서는 담당 부서(인사·재무·총무·보안·IT 등)에 직접 문의해 주세요."""
            inference_seconds = 0.0
            st.markdown("### 🤖 AI 답변")
            st.markdown(answer)
        else:
            st.success("✅ 답변 생성도 **로컬 환경**에서만 수행됩니다.")
            with st.spinner("답변 생성 중..."):
                result = generate_answer(
                    question=question,
                    retrieval_plan=retrieval_plan,
                    retrieved_docs=retrieved_docs,
                    use_local_llm=True,
                )
            answer = result["answer"]
            inference_seconds = result.get("inference_seconds", 0)
            st.markdown("### 🤖 AI 답변")
            st.markdown(answer)
        
        st.markdown(f"""
        <div class="info-box" style="margin-top: 0.75rem;">
            ⏱️ <strong>추론 소요 시간:</strong> {inference_seconds:.2f}초
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">📊 처리 요약</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("검색된 문서", f"{len(retrieved_docs)}건")
        
        with col2:
            st.metric("질문 복잡도", complexity_kr)
        
        with col3:
            cross_ref = "예" if retrieval_plan.get('cross_reference_check') else "아니오"
            st.metric("교차 참조", cross_ref)
        
        with col4:
            st.metric("외부 API", "질문 분석")
        
        with col5:
            st.metric("⏱️ 추론 시간", f"{inference_seconds:.1f}초")
        
        with st.expander("🔒 보안 로그"):
            st.markdown("""
            | 단계 | 처리 위치 | 전송 데이터 |
            |------|----------|------------|
            | 질문 분석 | 외부 API | 사용자 질문만 |
            | 문서 검색 | **로컬** | 없음 (내부 처리) |
            | 답변 생성 | **로컬** | 없음 (내부 처리) |
            
            ✅ **내부 문서는 외부로 전송되지 않았습니다.**
            """)


with st.sidebar:
    st.markdown("## ℹ️ 시스템 정보")
    
    ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    ollama_available = check_ollama_available() if ollama_enabled else False
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    
    st.markdown("### 🤖 로컬 LLM 상태")
    if ollama_enabled and ollama_available:
        st.success(f"✅ Ollama 연결됨 ({ollama_model})")
    elif ollama_enabled:
        st.warning(f"⚠️ Ollama 설정됨, 서버 미실행")
        st.code("ollama serve", language="bash")
    else:
        st.info("ℹ️ 시뮬레이션 모드")
    
    st.markdown("### 🏗️ 아키텍처")
    st.markdown("""
    ```
    [질문] → 외부 LLM (분석)
              ↓
    [계층적 검색] ← FAISS
    (정책→규정→지침→매뉴얼)
              ↓
    [로컬 LLM] → 답변
    ```
    """)
    
    st.markdown("### 📑 문서 계층")
    st.markdown("""
    1. **정책** (Policy) - 최상위 원칙
    2. **규정** (Regulation) - 상세 규칙
    3. **지침** (Guideline) - 실행 가이드
    4. **매뉴얼** (Manual) - 상세 절차
    5. **양식** (Form) - 서식/체크리스트
    """)
    
    st.markdown("### 📁 문서 카테고리")
    st.markdown("""
    - **HR**: 인사/복리후생
    - **FINANCE**: 재무/경비
    - **SECURITY**: 정보보안
    - **IT_OPS**: IT운영
    - **COMPLIANCE**: 준법/윤리
    - **VENDOR**: 외주/협력업체
    """)
    
    st.markdown("### 🔐 보안 원칙")
    st.markdown("""
    <div class="warning-box">
    <strong>민감한 내부 문서는 절대 외부 API로 전송되지 않습니다.</strong>
    <br><br>
    ✅ 외부 전송: 사용자 질문<br>
    ❌ 외부 전송 금지: 내부 문서
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ 설정")
    
    if st.button("🔄 벡터 DB 새로고침"):
        st.cache_data.clear()
        st.success("캐시가 초기화되었습니다.")
        st.rerun()


if __name__ == "__main__":
    main()
