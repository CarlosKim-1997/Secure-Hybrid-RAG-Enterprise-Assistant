"""
Query Planner Service

외부 OpenAI API를 사용하여 사용자 질문을 분석하고
구조화된 검색 계획을 생성합니다.

보안 원칙: 사용자 질문만 외부 API로 전송됩니다.
내부 문서나 민감한 정보는 절대 전송되지 않습니다.

금융사 엔터프라이즈 내규 시스템:
- 문서 계층 인식 (정책 > 규정 > 지침 > 매뉴얼)
- 금액별 승인권한 판단
- 특수 조건 (해외, 관계사, 긴급) 식별
- 규정 간 상호참조 필요 여부 판단
"""

import json
import os
import re
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RetrievalPlan(TypedDict, total=False):
    user_intent: str
    query_complexity: str
    primary_categories: list[str]
    related_categories: list[str]
    document_levels_needed: list[str]
    amount_involved: int | None
    special_conditions: list[str]
    search_queries: list[str]
    cross_reference_check: bool
    focus_points: list[str]
    expected_answer_structure: list[str]


def load_prompt_template() -> str:
    """프롬프트 템플릿을 로드합니다."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "query_planner_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document_catalog() -> list[dict]:
    """문서 목차(규정 목록)를 로드합니다. 내용 없이 ID·이름·카테고리·등급만 포함."""
    catalog_path = Path(__file__).parent.parent / "data" / "vectordb" / "document_catalog.json"
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def format_document_catalog_for_prompt(catalog: list[dict]) -> str:
    """목차를 프롬프트에 넣기 위한 문자열로 포맷합니다. 카테고리별·등급별로 그룹화."""
    if not catalog:
        return "(문서 목차 없음. DOCUMENT CATEGORIES와 DOCUMENT HIERARCHY만 참고하세요.)"
    by_category: dict[str, list[dict]] = {}
    for doc in catalog:
        cat = doc.get("category", "GENERAL")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(doc)
    lines = []
    for cat in sorted(by_category.keys()):
        docs = by_category[cat]
        lines.append(f"[{cat}]")
        for d in docs:
            doc_id = d.get("doc_id", "")
            doc_name = d.get("doc_name", "")
            level = d.get("doc_level", "")
            lines.append(f"  - {doc_id} {doc_name} ({level})")
        lines.append("")
    return "\n".join(lines).strip()


def _validate_plan_against_catalog(plan: dict, catalog: list[dict]) -> dict:
    """플랜의 primary_categories, document_levels_needed를 목차 기준으로 검증·보정합니다."""
    allowed_levels = {"POLICY", "REGULATION", "GUIDELINE", "MANUAL", "FORM"}
    if catalog:
        allowed_categories = {d.get("category", "GENERAL") for d in catalog}
        cats = plan.get("primary_categories", [])
        plan["primary_categories"] = [c for c in cats if c in allowed_categories]
        if not plan["primary_categories"] and cats:
            plan["primary_categories"] = list(allowed_categories)
        rel = plan.get("related_categories", [])
        plan["related_categories"] = [c for c in rel if c in allowed_categories]
    levels = plan.get("document_levels_needed", [])
    plan["document_levels_needed"] = [lev for lev in levels if lev in allowed_levels]
    if not plan["document_levels_needed"]:
        plan["document_levels_needed"] = ["REGULATION", "GUIDELINE", "MANUAL"]
    return plan


def generate_retrieval_plan(question: str, use_mock: bool = False) -> RetrievalPlan:
    """
    사용자 질문을 분석하여 검색 계획을 생성합니다.
    
    Args:
        question: 사용자의 질문
        use_mock: True인 경우 API 호출 없이 모의 응답 반환
        
    Returns:
        RetrievalPlan: 구조화된 검색 계획
        
    Security Note:
        이 함수는 사용자 질문만 외부 API로 전송합니다.
        내부 문서나 민감한 정보는 절대 포함되지 않습니다.
    """
    if use_mock:
        return _generate_mock_plan(question)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("Warning: OPENAI_API_KEY not set. Using mock response.")
        return _generate_mock_plan(question)
    
    client = OpenAI(api_key=api_key)
    catalog = load_document_catalog()
    catalog_text = format_document_catalog_for_prompt(catalog)

    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{document_catalog}", catalog_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an enterprise regulatory query planner for a financial services company. Analyze questions about internal regulations and create retrieval plans. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        plan = json.loads(result_text)
        plan = _validate_plan_against_catalog(plan, catalog)

        return RetrievalPlan(
            user_intent=plan.get("user_intent", ""),
            query_complexity=plan.get("query_complexity", "simple"),
            primary_categories=plan.get("primary_categories", []),
            related_categories=plan.get("related_categories", []),
            document_levels_needed=plan.get("document_levels_needed", ["REGULATION", "GUIDELINE"]),
            amount_involved=plan.get("amount_involved"),
            special_conditions=plan.get("special_conditions", []),
            search_queries=plan.get("search_queries", []),
            cross_reference_check=plan.get("cross_reference_check", False),
            focus_points=plan.get("focus_points", []),
            expected_answer_structure=plan.get("expected_answer_structure", []),
        )
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return _generate_mock_plan(question)
    except Exception as e:
        print(f"API error: {e}")
        return _generate_mock_plan(question)


def _extract_amount(question: str) -> int | None:
    """질문에서 금액을 추출합니다."""
    patterns = [
        r'(\d+)만\s*원',
        r'(\d+)천만\s*원',
        r'(\d+)억\s*원',
        r'(\d{1,3}(?:,\d{3})+)\s*원',
        r'(\d+)\s*원',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            amount_str = match.group(1).replace(',', '')
            amount = int(amount_str)
            
            if '천만' in question[max(0, match.start()-5):match.end()+5]:
                return amount * 10000000
            elif '만' in question[max(0, match.start()-3):match.end()+3]:
                return amount * 10000
            elif '억' in question[max(0, match.start()-3):match.end()+3]:
                return amount * 100000000
            else:
                return amount
    
    return None


def _generate_mock_plan(question: str) -> RetrievalPlan:
    """모의 검색 계획을 생성합니다 (API 키 없을 때 사용)."""
    
    question_lower = question.lower()
    
    categories = []
    related_categories = []
    doc_levels = []
    search_queries = []
    focus_points = []
    special_conditions = []
    cross_ref = False
    complexity = "simple"
    intent = ""
    answer_structure = []
    
    amount = _extract_amount(question)
    
    if "해외" in question:
        special_conditions.append("overseas")
    if "긴급" in question or "급하게" in question:
        special_conditions.append("urgent")
    if "관계사" in question or "계열사" in question:
        special_conditions.append("related_party")
    if "접대" in question:
        special_conditions.append("entertainment")
    
    if any(kw in question_lower for kw in ["출장", "경비", "정산", "비용", "법인카드", "접대"]):
        categories.append("FINANCE")
        doc_levels = ["REGULATION", "GUIDELINE"]
        search_queries.extend([
            "경비 정산 절차 방법",
            "승인 권한 한도",
            "증빙 서류 요건"
        ])
        focus_points.extend(["정산 절차", "승인 권한", "필요 서류", "기한"])
        answer_structure = ["procedures", "approval_process", "documents_needed", "limits"]
        intent = "직원이 경비/출장비 정산 관련 정보를 알고 싶어합니다."
        
        if "접대" in question:
            related_categories.append("COMPLIANCE")
            cross_ref = True
            complexity = "moderate"
        
        if amount and amount >= 1000000:
            complexity = "complex"
            doc_levels.insert(0, "POLICY")
            
    elif any(kw in question_lower for kw in ["휴가", "연차", "병가", "재택", "출산", "육아"]):
        categories.append("HR")
        doc_levels = ["REGULATION", "GUIDELINE"]
        search_queries.extend([
            "휴가 신청 절차",
            "연차 일수 계산",
            "휴가 승인 권한"
        ])
        focus_points.extend(["신청 절차", "휴가 일수", "승인권자"])
        answer_structure = ["procedures", "limits", "approval_process"]
        intent = "직원이 휴가/근무 관련 정보를 알고 싶어합니다."
        
    elif any(kw in question_lower for kw in ["보안", "비밀번호", "접근권한", "계정", "유출"]):
        categories.append("SECURITY")
        doc_levels = ["REGULATION", "GUIDELINE", "MANUAL"]
        search_queries.extend([
            "접근권한 신청 절차",
            "보안 정책 규정",
            "계정 관리 방법"
        ])
        focus_points.extend(["보안 요건", "신청 절차", "승인 체계"])
        answer_structure = ["procedures", "security_requirements", "approval_process"]
        intent = "직원이 보안/접근권한 관련 정보를 알고 싶어합니다."
        
        if "협력업체" in question or "외주" in question:
            related_categories.append("VENDOR")
            cross_ref = True
            complexity = "moderate"
            
    elif any(kw in question_lower for kw in ["vpn", "장비", "소프트웨어", "it", "헬프데스크"]) or ("시스템" in question and "접근" not in question):
        categories.append("IT_OPS")
        doc_levels = ["MANUAL", "GUIDELINE"]
        search_queries.extend([
            "IT 지원 요청 방법",
            "시스템 사용 안내",
            "장비 신청 절차"
        ])
        focus_points.extend(["신청 방법", "연락처", "처리 절차"])
        answer_structure = ["procedures", "contacts"]
        intent = "직원이 IT 관련 지원 정보를 필요로 합니다."

    elif any(kw in question for kw in ["협력업체", "외주"]) and any(kw in question for kw in ["접근권한", "접근 권한", "계정", "시스템 접근"]):
        categories.append("SECURITY")
        related_categories.append("VENDOR")
        doc_levels = ["REGULATION", "GUIDELINE", "MANUAL"]
        search_queries.extend([
            "협력업체 시스템 접근권한",
            "외부인력 계정 발급 절차",
            "접근통제 규정 협력업체"
        ])
        focus_points.extend(["접근권한 신청 절차", "보안 서약서", "권한 범위", "기간 만료 시 회수"])
        answer_structure = ["procedures", "documents_needed", "approval_process", "security_requirements"]
        intent = "협력업체 직원의 시스템 접근권한 부여 절차 안내"
        cross_ref = True
        complexity = "moderate"
        
    elif any(kw in question_lower for kw in ["협력업체", "외주", "용역", "계약"]):
        categories.append("VENDOR")
        related_categories.append("SECURITY")
        doc_levels = ["REGULATION", "GUIDELINE"]
        search_queries.extend([
            "협력업체 관리 규정",
            "외주 계약 절차",
            "협력업체 보안 요건"
        ])
        focus_points.extend(["계약 절차", "보안 요건", "관리 책임"])
        answer_structure = ["procedures", "security_requirements", "approval_process"]
        cross_ref = True
        complexity = "moderate"
        intent = "직원이 협력업체/외주 관련 정보를 알고 싶어합니다."
        
    elif any(kw in question_lower for kw in ["준법", "윤리", "이해충돌", "내부거래", "자금세탁"]):
        categories.append("COMPLIANCE")
        doc_levels = ["POLICY", "REGULATION"]
        search_queries.extend([
            "준법 윤리 정책",
            "이해충돌 방지 규정",
            "신고 절차"
        ])
        focus_points.extend(["금지 사항", "신고 의무", "제재"])
        answer_structure = ["rules", "procedures", "penalties"]
        intent = "직원이 준법/윤리 관련 정보를 알고 싶어합니다."
        
    elif any(kw in question_lower for kw in ["입사", "신입", "온보딩", "교육"]):
        categories.append("HR")
        doc_levels = ["MANUAL", "GUIDELINE"]
        search_queries.extend([
            "신입사원 안내",
            "입사 절차 서류",
            "필수 교육"
        ])
        focus_points.extend(["입사 절차", "필수 교육", "복리후생"])
        answer_structure = ["procedures", "checklist"]
        intent = "직원이 입사/온보딩 관련 정보를 알고 싶어합니다."
        
    else:
        categories = ["HR", "FINANCE", "IT_OPS"]
        doc_levels = ["REGULATION", "GUIDELINE", "MANUAL"]
        search_queries = [
            question,
            question.replace("?", "").replace("요", ""),
            " ".join(question.split()[:3]) if len(question.split()) > 3 else question
        ]
        focus_points = ["관련 규정", "절차", "담당 부서"]
        answer_structure = ["procedures"]
        intent = f"직원이 다음에 대해 문의합니다: {question}"
    
    if special_conditions:
        complexity = "moderate" if complexity == "simple" else "complex"

    plan_dict = {
        "user_intent": intent,
        "query_complexity": complexity,
        "primary_categories": categories,
        "related_categories": related_categories,
        "document_levels_needed": doc_levels,
        "amount_involved": amount,
        "special_conditions": special_conditions,
        "search_queries": search_queries,
        "cross_reference_check": cross_ref,
        "focus_points": focus_points,
        "expected_answer_structure": answer_structure,
    }
    plan_dict = _validate_plan_against_catalog(plan_dict, load_document_catalog())

    return RetrievalPlan(
        user_intent=plan_dict["user_intent"],
        query_complexity=plan_dict["query_complexity"],
        primary_categories=plan_dict["primary_categories"],
        related_categories=plan_dict["related_categories"],
        document_levels_needed=plan_dict["document_levels_needed"],
        amount_involved=plan_dict.get("amount_involved"),
        special_conditions=plan_dict["special_conditions"],
        search_queries=plan_dict["search_queries"],
        cross_reference_check=plan_dict["cross_reference_check"],
        focus_points=plan_dict["focus_points"],
        expected_answer_structure=plan_dict["expected_answer_structure"],
    )


def get_categories_for_search(plan: RetrievalPlan) -> list[str]:
    """검색에 사용할 카테고리 목록을 반환합니다."""
    categories = list(plan.get("primary_categories", []))
    if plan.get("cross_reference_check"):
        categories.extend(plan.get("related_categories", []))
    return list(set(categories))


def get_doc_levels_for_search(plan: RetrievalPlan) -> list[str]:
    """검색에 사용할 문서 등급 목록을 반환합니다."""
    return plan.get("document_levels_needed", ["REGULATION", "GUIDELINE"])
