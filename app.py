"""
Secure Hybrid RAG Enterprise Assistant
Streamlit 메인 애플리케이션 — 채팅형 내규 검색 UI
"""

import os
import re
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv

from services.query_planner import (
    generate_retrieval_plan,
    get_categories_for_search,
    get_doc_levels_for_search,
)
from services.vector_search import search_documents
from services.rag_generator import generate_answer, check_ollama_available, is_web_demo_mode

load_dotenv()

DOC_LEVEL_KR = {
    "POLICY": "정책",
    "REGULATION": "규정",
    "GUIDELINE": "지침",
    "MANUAL": "매뉴얼",
    "FORM": "양식",
    "GENERAL": "일반",
}

CATEGORY_KR = {
    "HR": "인사",
    "FINANCE": "재무·경비",
    "SECURITY": "정보보안",
    "IT_OPS": "IT운영",
    "COMPLIANCE": "준법·윤리",
    "VENDOR": "외주·협력",
    "ADMIN": "총무",
    "GENERAL": "일반",
}

QUICK_QUESTIONS = {
    "재무·경비": [
        "해외 출장 가서 현지 고객사 접대비 150만원 썼는데 어떻게 정산해요?",
        "법인카드 한도 증액 신청 방법은?",
    ],
    "인사": [
        "연차휴가 신청은 어떻게 하나요?",
        "신입사원 입사 첫 날 뭐해요?",
    ],
    "보안": [
        "협력업체 직원에게 시스템 접근권한 부여하려면?",
        "보안사고 발생 시 어떻게 대응해야 하나요?",
        "재택근무 시 보안 준수사항이 뭐예요?",
    ],
}

st.set_page_config(
    page_title="내규 검색",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin: 0 0 0.25rem 0;
    }
    .app-subtitle {
        font-size: 0.875rem;
        color: #6B7280;
        margin: 0 0 1rem 0;
    }
    .source-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .source-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #3B82F6;
        border-radius: 0.5rem;
        padding: 0.875rem 1rem;
        margin-bottom: 0.625rem;
    }
    .source-card.level-POLICY { border-left-color: #7C3AED; }
    .source-card.level-REGULATION { border-left-color: #2563EB; }
    .source-card.level-GUIDELINE { border-left-color: #0891B2; }
    .source-card.level-MANUAL { border-left-color: #059669; }
    .source-card.level-FORM { border-left-color: #D97706; }
    .source-rank {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 0.25rem;
        margin-right: 0.4rem;
    }
    .source-doc-id {
        font-family: ui-monospace, monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #1E40AF;
    }
    .source-doc-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
        margin: 0.25rem 0 0.35rem 0;
    }
    .source-meta {
        font-size: 0.78rem;
        color: #6B7280;
        margin-bottom: 0.4rem;
    }
    .source-articles {
        font-size: 0.78rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    .article-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        font-size: 0.72rem;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    .source-excerpt {
        font-size: 0.8rem;
        color: #4B5563;
        line-height: 1.55;
        background: #FFFFFF;
        border: 1px solid #F3F4F6;
        border-radius: 0.35rem;
        padding: 0.5rem 0.65rem;
        margin-top: 0.35rem;
    }
    .source-block {
        margin-bottom: 0.35rem;
    }
    div[data-testid="stExpander"]:has(.source-expander-marker) {
        margin-top: -0.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        background: #FAFAFA;
    }
    div[data-testid="stExpander"]:has(.source-expander-marker) details summary {
        font-size: 0.82rem;
        color: #4B5563;
    }
    .answer-meta {
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-top: 0.75rem;
    }
    div[data-testid="stChatMessage"] {
        padding: 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_vectordb_initialized() -> bool:
    vectordb_path = Path(__file__).parent / "data" / "vectordb" / "documents.pkl"
    return vectordb_path.exists()


def extract_articles(content: str, limit: int = 5) -> list[str]:
    """문서 본문에서 조항 제목을 추출합니다."""
    pattern = r"제\d+조\s*\([^)]+\)"
    found = re.findall(pattern, content)
    seen: set[str] = set()
    articles: list[str] = []
    for article in found:
        if article not in seen:
            seen.add(article)
            articles.append(article)
        if len(articles) >= limit:
            break
    return articles


def clean_excerpt(content: str, max_len: int = 180) -> str:
    """인용 발췌문을 정리합니다."""
    text = re.sub(r"\[문서[^\]]*\]|\(문서번호:[^)]+\)|\[(정책|규정|지침|매뉴얼|양식)\]", "", content)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    excerpt = " ".join(lines[:4])
    if len(excerpt) > max_len:
        return excerpt[:max_len].rstrip() + "…"
    return excerpt


def clean_answer_display(text: str) -> str:
    """답변 앞의 모델/모드 라벨을 제거합니다."""
    return re.sub(r"^(🤖|⚙️)\s*\*\*\[.*?\]\*\*\s*\n+", "", text).strip()


def build_source_citations(retrieved_docs: list[dict]) -> list[dict]:
    """검색 결과를 문서 단위 출처 카드 데이터로 변환합니다."""
    by_doc: dict[str, dict] = {}

    for doc in retrieved_docs:
        metadata = doc.get("metadata") or {}
        doc_id = metadata.get("doc_id") or doc.get("source", "")
        if not doc_id:
            continue

        score = doc.get("score", 0)
        articles = extract_articles(doc.get("content", ""))

        chunk_content = doc.get("content", "")
        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "doc_name": metadata.get("doc_name", ""),
                "doc_level": doc.get("doc_level", "GENERAL"),
                "category": doc.get("category", "GENERAL"),
                "department": metadata.get("department", ""),
                "approver": metadata.get("approver", ""),
                "version": metadata.get("version", ""),
                "score": score,
                "articles": articles,
                "excerpt": clean_excerpt(chunk_content),
                "source_file": doc.get("source", ""),
                "chunks": [{"content": chunk_content, "score": score}],
            }
        else:
            existing = by_doc[doc_id]
            merged_articles = list(dict.fromkeys(existing["articles"] + articles))
            existing["articles"] = merged_articles[:6]
            chunk_key = chunk_content[:200]
            known = {c["content"][:200] for c in existing["chunks"]}
            if chunk_key not in known:
                existing["chunks"].append({"content": chunk_content, "score": score})
            if score > existing["score"]:
                existing["score"] = score
                existing["excerpt"] = clean_excerpt(chunk_content)
            existing["chunks"].sort(key=lambda c: c["score"], reverse=True)

    sorted_sources = sorted(by_doc.values(), key=lambda x: x["score"], reverse=True)
    for i, source in enumerate(sorted_sources, 1):
        source["rank"] = i
    return sorted_sources


def render_source_card(source: dict) -> None:
    """검색엔진형 출처 카드를 렌더링합니다."""
    level = source.get("doc_level", "GENERAL")
    level_kr = DOC_LEVEL_KR.get(level, level)
    category_kr = CATEGORY_KR.get(source.get("category", ""), source.get("category", ""))

    meta_parts = [level_kr, category_kr]
    if source.get("department"):
        meta_parts.append(f"담당: {source['department']}")
    if source.get("approver"):
        meta_parts.append(f"승인: {source['approver']}")
    if source.get("version"):
        meta_parts.append(f"v{source['version']}")

    article_tags = "".join(
        f'<span class="article-tag">{a}</span>' for a in source.get("articles", [])
    )
    articles_block = (
        f'<div class="source-articles"><strong>참고 조항</strong> {article_tags}</div>'
        if article_tags
        else ""
    )

    st.markdown(f"""
    <div class="source-card level-{level}">
        <span class="source-rank">출처 {source['rank']}</span>
        <span class="source-doc-id">{source['doc_id']}</span>
        <div class="source-doc-name">{source.get('doc_name') or source.get('source_file', '')}</div>
        <div class="source-meta">{' · '.join(meta_parts)} · 관련도 {source['score']:.0%}</div>
        {articles_block}
        <div class="source-excerpt">{source.get('excerpt', '')}</div>
    </div>
    """, unsafe_allow_html=True)


def get_source_chunks(source: dict, retrieved_docs: list[dict]) -> list[dict]:
    """출처에 해당하는 원문 청크 목록을 반환합니다."""
    if source.get("chunks"):
        return source["chunks"]
    doc_id = source["doc_id"]
    chunks = [
        {"content": d.get("content", ""), "score": d.get("score", 0)}
        for d in retrieved_docs
        if (d.get("metadata") or {}).get("doc_id") == doc_id
        or d.get("source") == source.get("source_file")
    ]
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


def render_source_expander(source: dict, retrieved_docs: list[dict]) -> None:
    """출처별 원문을 개별 expander로 렌더링합니다."""
    doc_name = source.get("doc_name") or source.get("source_file", "")
    label = f"📄 {source['doc_id']} {doc_name} — 원문 보기"
    chunks = get_source_chunks(source, retrieved_docs)

    with st.expander(label, expanded=False):
        st.markdown('<span class="source-expander-marker"></span>', unsafe_allow_html=True)
        if not chunks:
            st.caption("원문을 불러올 수 없습니다.")
            return
        for i, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                st.caption(f"관련 구간 {i}")
            st.text(chunk.get("content", ""))
            if i < len(chunks):
                st.divider()


def render_source_block(source: dict, retrieved_docs: list[dict]) -> None:
    """출처 카드 + 개별 원문 expander를 함께 렌더링합니다."""
    render_source_card(source)
    render_source_expander(source, retrieved_docs)


def render_assistant_message(message: dict) -> None:
    """어시스턴트 메시지(답변 + 출처)를 렌더링합니다."""
    answer = clean_answer_display(message.get("answer", ""))
    st.markdown(answer)

    sources = message.get("sources", [])
    if sources:
        st.markdown('<div class="source-section-title">📎 참고 내규</div>', unsafe_allow_html=True)
        retrieved_docs = message.get("retrieved_docs", [])
        for source in sources:
            render_source_block(source, retrieved_docs)

    inference = message.get("inference_seconds", 0)
    doc_count = len(sources)
    if inference > 0 or doc_count:
        st.markdown(
            f'<div class="answer-meta">참고 문서 {doc_count}건'
            + (f' · 응답 {inference:.1f}초' if inference > 0 else '')
            + '</div>',
            unsafe_allow_html=True,
        )

    if message.get("show_debug"):
        with st.expander("검색 상세 (개발자용)"):
            plan = message.get("retrieval_plan", {})
            st.json(plan)


def run_query_pipeline(question: str) -> dict:
    """질문을 처리하고 채팅 메시지용 결과를 반환합니다."""
    retrieval_plan = generate_retrieval_plan(question)

    search_categories = get_categories_for_search(retrieval_plan)
    search_levels = get_doc_levels_for_search(retrieval_plan)

    retrieved_docs = search_documents(
        search_queries=retrieval_plan.get("search_queries", []),
        categories=search_categories if search_categories else None,
        doc_levels=search_levels if search_levels else None,
        top_k=3,
        use_hierarchy=retrieval_plan.get("cross_reference_check", False),
    )

    sources = build_source_citations(retrieved_docs)

    if not retrieved_docs:
        answer = """**답변**

제공된 내규 문서에서 질문과 관련된 내용을 찾을 수 없습니다.

해당 업무는 담당 부서(인사·재무·총무·보안·IT 등)에 직접 문의해 주세요."""
        inference_seconds = 0.0
    else:
        result = generate_answer(
            question=question,
            retrieval_plan=retrieval_plan,
            retrieved_docs=retrieved_docs,
            use_local_llm=True,
        )
        answer = result["answer"]
        inference_seconds = result.get("inference_seconds", 0)

    return {
        "role": "assistant",
        "answer": answer,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
        "retrieval_plan": retrieval_plan,
        "inference_seconds": inference_seconds,
        "show_debug": False,
    }


def queue_question(question: str) -> None:
    """다음 렌더 사이클에서 처리할 질문을 큐에 넣습니다."""
    question = question.strip()
    if question:
        st.session_state.pending_question = question


def process_pending_question() -> bool:
    """대기 중인 질문을 처리하고 결과를 세션에 저장합니다."""
    question = st.session_state.pop("pending_question", None)
    if not question:
        return False

    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("관련 내규를 검색하고 답변을 작성하는 중…"):
        response = run_query_pipeline(question)
    st.session_state.messages.append(response)
    return True


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📋 내규 검색")
        st.caption("정책 · 규정 · 지침 · 매뉴얼")

        ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        ollama_available = check_ollama_available() if ollama_enabled else False
        web_demo = is_web_demo_mode()

        if web_demo:
            st.warning(
                "웹 데모: Step 1·2는 동일, Step 3만 클라우드 LLM",
                icon="☁️",
            )
        elif ollama_enabled and ollama_available:
            st.success("AI 답변 준비됨 (로컬 Ollama)", icon="✅")
        elif ollama_enabled:
            st.warning("Ollama 미실행", icon="⚠️")
        else:
            st.info("시뮬레이션 모드", icon="ℹ️")

        st.markdown("---")
        st.markdown("**자주 찾는 질문**")
        for category, questions in QUICK_QUESTIONS.items():
            st.caption(category)
            for q in questions:
                if st.button(q, key=f"quick_{category}_{q[:20]}", use_container_width=True):
                    queue_question(q)
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("시스템"):
            if web_demo:
                st.caption(
                    "웹 데모: Step 1(질문 분석)과 Step 3(답변)에 OpenAI를 사용합니다. "
                    "Step 2 벡터 검색은 로컬 FAISS입니다. "
                    "프로덕션에서는 Step 3만 로컬 Ollama로 처리합니다."
                )
            else:
                st.caption("내부 문서는 외부로 전송되지 않습니다.")
            if st.button("벡터 DB 캐시 초기화"):
                st.cache_data.clear()
                st.success("캐시 초기화 완료")


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_sidebar()

    st.markdown('<p class="app-title">내규·매뉴얼 검색</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">질문하시면 관련 내규를 찾아 답변과 출처를 함께 안내합니다.</p>',
        unsafe_allow_html=True,
    )

    if not check_vectordb_initialized():
        st.warning(
            "벡터 데이터베이스가 초기화되지 않았습니다. "
            "터미널에서 `python init_vectordb.py`를 실행해 주세요."
        )
        return

    if not st.session_state.messages and not st.session_state.get("pending_question"):
        st.markdown("**예시 질문**")
        cols = st.columns(2)
        flat_questions = [q for qs in QUICK_QUESTIONS.values() for q in qs]
        for i, q in enumerate(flat_questions[:4]):
            with cols[i % 2]:
                if st.button(q, key=f"welcome_{i}", use_container_width=True):
                    queue_question(q)
                    st.rerun()

    if process_pending_question():
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "📋"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message)

    if prompt := st.chat_input("내규·절차·승인 권한 등 궁금한 내용을 입력하세요"):
        queue_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
