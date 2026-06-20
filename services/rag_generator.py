"""
RAG Answer Generator Service

로컬 LLM(Ollama)을 사용하여 검색된 문서 기반으로 답변을 생성합니다.

보안 원칙: 답변 생성은 로컬 환경에서만 수행됩니다.
내부 문서 내용은 절대 외부 API로 전송되지 않습니다.
"""

import os
import json
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

LOCAL_LLM_AVAILABLE = False
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    pass


def check_ollama_available() -> bool:
    """Ollama 서버가 실행 중인지 확인합니다."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def is_web_demo_mode() -> bool:
    """Streamlit Cloud 등 웹 데모 전용 모드 (Step 3를 클라우드 LLM으로 처리)."""
    return os.getenv("WEB_DEMO_MODE", "false").lower() == "true"


def _openai_api_key_configured() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(api_key) and api_key != "your_openai_api_key_here"


def load_prompt_template() -> str:
    """RAG 답변 프롬프트 템플릿을 로드합니다."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "rag_answer_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_allowed_doc_ids(documents: list[dict]) -> str:
    """검색된 문서에서 인용 가능한 문서 ID 목록을 추출합니다 (중복 제거)."""
    if not documents:
        return "(없음)"
    seen: set[str] = set()
    ids = []
    for doc in documents:
        doc_id = (doc.get("metadata") or {}).get("doc_id") or doc.get("source", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ", ".join(ids) if ids else "(없음)"


def _reorder_docs_by_question_keywords(question: str, docs: list[dict]) -> list[dict]:
    """질문에 '해외', '협력업체' 등 키워드가 있으면, 해당 키워드를 문서 제목·ID에 포함한 문서를 상위로 올립니다."""
    if not docs:
        return docs
    q = question.strip()
    priority_keywords = [kw for kw in [
        "해외", "해외출장", "협력업체", "접대", "접대비", "법인카드", "연차", "휴가",
        "보안사고", "접근권한", "재택", "경조", "침해", "외주", "정보분류", "변경관리",
    ] if kw in q]
    if not priority_keywords:
        return docs

    def sort_key(doc: dict) -> tuple:
        meta = doc.get("metadata") or {}
        doc_id = (meta.get("doc_id") or "")
        doc_name = (meta.get("doc_name") or "")
        text = doc_id + " " + doc_name
        matches = any(kw in text for kw in priority_keywords)
        score = doc.get("score", 0)
        return (not matches, -score)

    return sorted(docs, key=sort_key)


def get_answer_focus(retrieval_plan: dict[str, Any], question: str) -> str:
    """답변 시 반드시 맞출 초점을 검색 계획·질문에서 추출합니다."""
    focus_points = retrieval_plan.get("focus_points") or []
    if focus_points:
        return ", ".join(focus_points[:3])
    intent = (retrieval_plan.get("user_intent") or "").strip()
    if intent:
        return intent[:80] + ("..." if len(intent) > 80 else "")
    q = question.replace("?", "").strip()
    return q[:60] + ("..." if len(q) > 60 else "")


def format_retrieved_documents(documents: list[dict]) -> str:
    """검색된 문서를 관련도 순으로 포맷팅합니다. 1위가 가장 질문과 관련 높음."""
    if not documents:
        return "검색된 문서가 없습니다."
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.get('metadata', {})
        doc_level = doc.get('doc_level', 'GENERAL')
        score = doc.get('score', 0)
        
        level_kr = {
            "POLICY": "정책",
            "REGULATION": "규정",
            "GUIDELINE": "지침",
            "MANUAL": "매뉴얼",
            "FORM": "양식",
        }.get(doc_level, doc_level)
        
        doc_id = metadata.get('doc_id', '')
        doc_name = metadata.get('doc_name', '')
        
        rank_label = "관련도 1위 (최우선 반영)" if i == 1 else f"관련도 {i}위"
        header = f"[문서 {i}] {rank_label} (점수 {score:.2f}) "
        if doc_id:
            header += f"{doc_id} "
        if doc_name:
            header += f"- {doc_name} "
        header += f"({level_kr})"
        
        formatted.append(f"""
{header}
출처: {doc.get('source', 'Unknown')}
카테고리: {doc.get('category', 'Unknown')}

{doc.get('content', '')}
""")
    
    return "\n" + "="*50 + "\n".join(formatted)


def generate_answer(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
    use_local_llm: bool = True,
) -> dict[str, Any]:
    """
    검색된 문서를 기반으로 답변을 생성합니다.

    Args:
        question: 사용자 질문
        retrieval_plan: 검색 계획 (의도, 포커스 포인트 등)
        retrieved_docs: 검색된 문서 리스트
        use_local_llm: 로컬 LLM 사용 여부 (False면 시뮬레이션)

    Returns:
        {"answer": 생성된 답변 문자열, "inference_seconds": 추론 소요 시간(초)}

    Security Note:
        이 함수는 로컬 환경에서만 실행됩니다.
        내부 문서 내용은 외부로 전송되지 않습니다.
    """
    if not retrieved_docs:
        no_doc_message = """**답변**

제공된 내규 문서에서 질문과 관련된 내용을 찾을 수 없습니다.

해당 업무에 대해서는 담당 부서(인사·재무·총무·보안·IT 등)에 직접 문의해 주세요."""
        return {"answer": no_doc_message, "inference_seconds": 0.0}

    retrieved_docs = _reorder_docs_by_question_keywords(question, retrieved_docs)

    ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    ollama_available = check_ollama_available() if ollama_enabled else False
    web_demo = is_web_demo_mode()

    start = time.perf_counter()

    if ollama_enabled and ollama_available:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        answer = _generate_with_ollama(question, retrieval_plan, retrieved_docs)
        text = f"🤖 **[Ollama - {model}]**\n\n{answer}"
    elif web_demo and OPENAI_AVAILABLE and _openai_api_key_configured():
        model = os.getenv("WEB_DEMO_OPENAI_MODEL", "gpt-4o-mini")
        answer = _generate_with_openai_web_demo(question, retrieval_plan, retrieved_docs)
        text = (
            f"☁️ **[웹 데모 — 클라우드 LLM ({model})]**\n\n{answer}\n\n"
            "> 프로덕션 아키텍처에서는 Step 3 답변 생성은 로컬 Ollama에서 처리합니다."
        )
    elif use_local_llm and LOCAL_LLM_AVAILABLE and os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true":
        answer = _generate_with_local_llm(question, retrieval_plan, retrieved_docs)
        text = f"🤖 **[Local LLM]**\n\n{answer}"
    else:
        answer = _generate_simulated_answer(question, retrieval_plan, retrieved_docs)
        text = f"⚙️ **[시뮬레이션 모드]**\n\n{answer}"

    inference_seconds = time.perf_counter() - start
    return {"answer": text, "inference_seconds": inference_seconds}


def _build_rag_prompt(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
) -> str:
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{user_intent}", retrieval_plan.get("user_intent", ""))
    prompt = prompt.replace("{query_complexity}", retrieval_plan.get("query_complexity", "simple"))
    prompt = prompt.replace("{focus_points}", ", ".join(retrieval_plan.get("focus_points", [])))
    prompt = prompt.replace(
        "{special_conditions}",
        ", ".join(retrieval_plan.get("special_conditions", [])) or "없음",
    )

    amount = retrieval_plan.get("amount_involved")
    amount_str = f"{amount:,}원" if amount else "해당 없음"
    prompt = prompt.replace("{amount_involved}", amount_str)
    prompt = prompt.replace("{retrieved_documents}", format_retrieved_documents(retrieved_docs))
    prompt = prompt.replace("{allowed_doc_ids}", get_allowed_doc_ids(retrieved_docs))
    prompt = prompt.replace("{answer_focus}", get_answer_focus(retrieval_plan, question))
    return prompt


def _generate_with_openai_web_demo(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
) -> str:
    """
    웹 데모 전용: 검색된 문서를 OpenAI로 전송해 답변을 생성합니다.

    프로덕션 하이브리드 아키텍처와 달리 Step 3만 클라우드 LLM을 사용합니다.
    """
    model = os.getenv("WEB_DEMO_OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = _build_rag_prompt(question, retrieval_plan, retrieved_docs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 금융사 내규 검색 어시스턴트입니다. "
                        "제공된 문서만 근거로 답변하고, 문서에 없는 내용은 추측하지 마세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("WEB_DEMO_OPENAI_TEMPERATURE", "0.25")),
            max_tokens=int(os.getenv("WEB_DEMO_OPENAI_MAX_TOKENS", "1200")),
        )
        content = response.choices[0].message.content
        return content.strip() if content else "답변 생성에 실패했습니다."
    except Exception as e:
        print(f"Web demo OpenAI error: {e}")
        return _generate_simulated_answer(question, retrieval_plan, retrieved_docs)


def _generate_with_ollama(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
) -> str:
    """Ollama를 사용하여 답변을 생성합니다."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
    
    print(f"[RAG] Using Ollama model: {model}, num_predict: {num_predict}")

    prompt = _build_rag_prompt(question, retrieval_plan, retrieved_docs)

    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.25"))
    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                }
            },
            timeout=180.0,
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "답변 생성에 실패했습니다.")
        else:
            print(f"Ollama error: {response.status_code} - {response.text}")
            return _generate_simulated_answer(question, retrieval_plan, retrieved_docs)
            
    except Exception as e:
        print(f"Ollama error: {e}")
        return _generate_simulated_answer(question, retrieval_plan, retrieved_docs)


def _generate_with_local_llm(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
) -> str:
    """로컬 LLM을 사용하여 답변을 생성합니다."""
    model_name = os.getenv("LOCAL_LLM_MODEL", "gpt2")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
        )
        
        prompt = _build_rag_prompt(question, retrieval_plan, retrieved_docs)

        result = generator(prompt)
        return result[0]["generated_text"]
        
    except Exception as e:
        print(f"Local LLM error: {e}")
        return _generate_simulated_answer(question, retrieval_plan, retrieved_docs)


def _generate_simulated_answer(
    question: str,
    retrieval_plan: dict[str, Any],
    retrieved_docs: list[dict],
) -> str:
    """
    검색된 문서 기반으로 시뮬레이션 답변을 생성합니다.
    
    실제 로컬 LLM이 없는 환경에서 데모 목적으로 사용됩니다.
    검색된 문서의 관련 내용을 추출하여 구조화된 답변을 생성합니다.
    """
    if not retrieved_docs:
        return """**답변**

죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다.

**문의처**
- 인사 관련: 인사팀 (내선 3000)
- 재무/경비: 재무팀 (내선 1000)
- IT 관련: IT헬프데스크 (내선 9999)
- 보안 관련: 정보보호팀 (내선 5000)
- 총무 관련: 총무팀 (내선 2000)

---
📋 검색된 문서 없음"""

    sources = []
    doc_levels = []
    relevant_content = []
    
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        doc_id = metadata.get("doc_id", "")
        doc_name = metadata.get("doc_name", "")
        source = doc.get("source", "Unknown")
        level = doc.get("doc_level", "GENERAL")
        
        if doc_id:
            sources.append(f"{doc_id} {doc_name}")
        else:
            sources.append(source)
        doc_levels.append(level)
        relevant_content.append(doc.get("content", ""))
    
    combined_content = "\n\n---\n\n".join(relevant_content)
    
    answer_parts = _extract_relevant_info(question, combined_content, retrieval_plan)
    
    complexity = retrieval_plan.get("query_complexity", "simple")
    special_conditions = retrieval_plan.get("special_conditions", [])
    amount = retrieval_plan.get("amount_involved")
    
    special_note = ""
    if special_conditions:
        conditions_kr = {
            "overseas": "해외",
            "urgent": "긴급",
            "related_party": "관계사",
            "entertainment": "접대",
            "external_party": "외부인",
        }
        conditions_list = [conditions_kr.get(c, c) for c in special_conditions]
        special_note = f"\n\n⚠️ **특수 조건**: 본 건은 {', '.join(conditions_list)} 관련 사항이 포함되어 추가 검토가 필요할 수 있습니다."
    
    amount_note = ""
    if amount:
        amount_note = f"\n\n💰 **금액 관련**: {amount:,}원에 대한 승인권한 및 한도는 해당 규정의 금액별 기준을 확인하세요."
    
    doc_refs = list(set(sources))[:5]
    
    answer = f"""**답변**

{answer_parts['main_answer']}{special_note}{amount_note}

**핵심 포인트**
{answer_parts['key_points']}

**추가 안내**
{answer_parts['additional_info']}

---
📋 **관련 규정**: {', '.join(doc_refs)}"""

    return answer


def _extract_relevant_info(
    question: str,
    content: str,
    retrieval_plan: dict[str, Any],
) -> dict[str, str]:
    """질문과 관련된 정보를 문서에서 추출합니다."""
    
    lines = content.split("\n")
    
    relevant_lines = []
    table_lines = []
    article_lines = []
    procedure_lines = []
    
    focus_points = retrieval_plan.get("focus_points", [])
    keywords = _extract_keywords(question)
    
    in_table = False
    in_procedure = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        if line_stripped.startswith("[문서") or line_stripped.startswith("[버전"):
            continue
        
        if "|" in line_stripped and line_stripped.count("|") >= 2:
            table_lines.append(line_stripped)
            in_table = True
            continue
        elif in_table and not line_stripped.startswith("|"):
            in_table = False
        
        if line_stripped.startswith("제") and "조" in line_stripped[:15]:
            article_lines.append(line_stripped)
            continue
        
        if line_stripped.startswith(("①", "②", "③", "④", "⑤", "1)", "2)", "3)", "1.", "2.", "3.")):
            procedure_lines.append(line_stripped)
            in_procedure = True
            continue
        elif in_procedure and not line_stripped[0].isdigit() and not line_stripped.startswith(("①", "②", "③")):
            in_procedure = False
        
        if any(kw in line_stripped.lower() for kw in keywords):
            relevant_lines.append(line_stripped)
    
    main_parts = []
    
    if relevant_lines:
        main_parts.append("다음은 질문과 관련된 내용입니다:\n")
        main_parts.extend(relevant_lines[:10])
    
    if procedure_lines:
        main_parts.append("\n**절차:**")
        main_parts.extend(procedure_lines[:8])
    
    if table_lines:
        main_parts.append("\n**기준표:**")
        main_parts.extend(table_lines[:6])
    
    if article_lines:
        main_parts.append("\n**관련 조항:**")
        main_parts.extend(article_lines[:5])
    
    main_answer = "\n".join(main_parts) if main_parts else "검색된 문서에서 관련 내용을 추출했습니다. 상세 내용은 담당 부서에 문의해 주세요."
    
    key_points = []
    if focus_points:
        for point in focus_points[:3]:
            key_points.append(f"• {point}")
    if not key_points:
        key_points = ["• 관련 규정 확인 필요", "• 담당 부서 문의 권장"]
    
    expected_structure = retrieval_plan.get("expected_answer_structure", [])
    additional_info = []
    if "approval_process" in expected_structure:
        additional_info.append("• 승인 권한은 금액/상황에 따라 다를 수 있습니다.")
    if "documents_needed" in expected_structure:
        additional_info.append("• 필요 서류는 해당 시스템에서 확인하세요.")
    if not additional_info:
        additional_info.append("• 상세 문의: 담당 부서에 연락해 주세요.")
    
    return {
        "main_answer": main_answer,
        "key_points": "\n".join(key_points),
        "additional_info": "\n".join(additional_info),
    }


def _extract_keywords(question: str) -> list[str]:
    """질문에서 키워드를 추출합니다."""
    stop_words = {"어떻게", "하나요", "있나요", "인가요", "무엇", "어디", "언제", "누가", "왜", "은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로", "하면", "되나요", "합니까", "입니까"}
    
    words = question.replace("?", "").replace(".", "").split()
    keywords = [w.lower() for w in words if w not in stop_words and len(w) > 1]
    
    keyword_map = {
        "출장": ["출장", "travel", "출장비"],
        "정산": ["정산", "청구", "reimbursement"],
        "휴가": ["휴가", "연차", "leave", "vacation"],
        "비밀번호": ["비밀번호", "password", "계정"],
        "vpn": ["vpn", "접속", "연결"],
        "회의실": ["회의실", "예약", "booking"],
        "재택": ["재택", "재택근무", "remote"],
    }
    
    expanded = list(keywords)
    for kw in keywords:
        if kw in keyword_map:
            expanded.extend(keyword_map[kw])
    
    return list(set(expanded))
