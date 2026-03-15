"""
벡터 데이터베이스 초기화 스크립트

내부 문서를 로드하고 FAISS 벡터 인덱스를 생성합니다.
이 스크립트는 애플리케이션 실행 전에 한 번 실행해야 합니다.

금융사 엔터프라이즈 내규 시스템:
- 계층 구조: policies > regulations > guidelines > manuals > forms
- 메타데이터 추출 및 인덱싱
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.vector_search import VectorStore


def main():
    print("=" * 70)
    print("🔧 Secure Hybrid RAG - 금융사 내규 벡터 데이터베이스 초기화")
    print("=" * 70)
    print()
    
    store = VectorStore()
    
    print("📂 내부 문서 로드 중...")
    print("   계층 구조: policies > regulations > guidelines > manuals > forms")
    print()
    documents = store.load_documents()
    print(f"   ✅ {len(documents)}개의 문서 청크 로드 완료")
    print()
    
    doc_levels = {}
    for doc in documents:
        level = doc.get('doc_level', 'GENERAL')
        doc_levels[level] = doc_levels.get(level, 0) + 1
    
    print("📊 문서 등급별 청크 수:")
    level_order = ["POLICY", "REGULATION", "GUIDELINE", "MANUAL", "FORM", "GENERAL"]
    for level in level_order:
        if level in doc_levels:
            level_kr = {
                "POLICY": "정책",
                "REGULATION": "규정",
                "GUIDELINE": "지침",
                "MANUAL": "매뉴얼",
                "FORM": "양식",
                "GENERAL": "일반",
            }.get(level, level)
            print(f"   - {level_kr} ({level}): {doc_levels[level]}개")
    print()
    
    categories = {}
    for doc in documents:
        cat = doc.get('category', 'GENERAL')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("📁 카테고리별 청크 수:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count}개")
    print()
    
    sources = set()
    for doc in documents:
        sources.add(doc.get('source', 'Unknown'))
    print(f"📄 총 {len(sources)}개 문서 파일")
    print()
    
    print("🔨 벡터 인덱스 구축 중...")
    print("   (임베딩 모델 다운로드가 필요할 수 있습니다)")
    store.build_index()
    print("   ✅ 벡터 인덱스 구축 완료")
    print()

    catalog = store.get_document_catalog()
    catalog_path = store.vectordb_dir / "document_catalog.json"
    store.vectordb_dir.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"📑 문서 목차 저장: document_catalog.json ({len(catalog)}개 문서)")
    print()

    print("💾 인덱스 저장 중...")
    store.save()
    print(f"   ✅ 저장 완료: {store.vectordb_dir}")
    print()
    
    print("🔍 검색 테스트...")
    test_queries = [
        ("경비 정산 절차", ["FINANCE"]),
        ("해외 출장 접대비", ["FINANCE", "COMPLIANCE"]),
        ("연차휴가 신청 방법", ["HR"]),
        ("협력업체 접근권한", ["SECURITY", "VENDOR"]),
        ("보안사고 대응", ["SECURITY"]),
    ]
    
    for query, expected_cats in test_queries:
        print(f"\n   Query: '{query}'")
        print(f"   Expected: {expected_cats}")
        
        results = store.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            level = result.get('doc_level', 'GENERAL')
            level_kr = {
                "POLICY": "정책",
                "REGULATION": "규정",
                "GUIDELINE": "지침",
                "MANUAL": "매뉴얼",
                "FORM": "양식",
            }.get(level, level)
            
            metadata = result.get('metadata', {})
            doc_id = metadata.get('doc_id', '')
            
            print(f"   {i}. [{result['category']}] [{level_kr}] {doc_id or result['source']} (score: {result['score']:.3f})")
            preview = result['content'][:60].replace('\n', ' ')
            print(f"      {preview}...")
    
    print()
    print("🔍 계층 검색 테스트...")
    query = "경비 정산 승인"
    print(f"   Query: '{query}' (계층 검색)")
    results = store.search_with_hierarchy(query, top_k=5)
    
    levels_found = set()
    for result in results:
        levels_found.add(result.get('doc_level', 'GENERAL'))
    
    print(f"   발견된 문서 등급: {levels_found}")
    for i, result in enumerate(results, 1):
        level = result.get('doc_level', 'GENERAL')
        metadata = result.get('metadata', {})
        doc_id = metadata.get('doc_id', '')
        print(f"   {i}. [{level}] {doc_id or result['source']} (score: {result['score']:.3f})")
    
    print()
    print("=" * 70)
    print("✅ 초기화 완료!")
    print()
    print("다음 명령으로 애플리케이션을 실행하세요:")
    print("   streamlit run app.py --server.headless=true")
    print("=" * 70)


if __name__ == "__main__":
    main()
