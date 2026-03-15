"""
Vector Search Service

로컬 벡터 데이터베이스(FAISS)를 사용하여 문서를 검색합니다.

보안 원칙: 모든 문서 처리와 검색은 로컬 환경에서만 수행됩니다.
문서 내용은 절대 외부로 전송되지 않습니다.

금융사 엔터프라이즈 내규 시스템:
- 계층 구조: 정책(Policy) > 규정(Regulation) > 지침(Guideline) > 매뉴얼(Manual) > 양식(Form)
- 메타데이터: 문서ID, 버전, 시행일, 상위/하위문서, 담당부서, 승인권한
"""

import os
import re
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using simple similarity search.")

from sentence_transformers import SentenceTransformer


class DocumentMetadata(TypedDict, total=False):
    doc_id: str
    doc_name: str
    version: str
    effective_date: str
    last_updated: str
    doc_level: str
    related_laws: str
    parent_doc: str
    child_docs: list[str]
    department: str
    approver: str


class DocumentChunk(TypedDict):
    content: str
    source: str
    category: str
    chunk_id: int
    doc_level: str
    metadata: DocumentMetadata


class SearchResult(TypedDict):
    content: str
    source: str
    category: str
    score: float
    doc_level: str
    metadata: DocumentMetadata


class VectorStore:
    """FAISS 기반 벡터 스토어 클래스"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        벡터 스토어를 초기화합니다.
        
        Args:
            embedding_model: 사용할 임베딩 모델 (다국어 지원)
        """
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents: list[DocumentChunk] = []
        self.data_dir = Path(__file__).parent.parent / "data"
        self.vectordb_dir = self.data_dir / "vectordb"
        
    def load_documents(self, docs_dir: Path | None = None) -> list[DocumentChunk]:
        """
        내부 문서를 로드하고 청킹합니다.
        계층 구조: policies > regulations > guidelines > manuals > forms
        
        Args:
            docs_dir: 문서 디렉토리 경로
            
        Returns:
            청킹된 문서 리스트
        """
        if docs_dir is None:
            docs_dir = self.data_dir / "internal_docs"
            
        documents = []
        
        subdirs = {
            "policies": "POLICY",
            "regulations": "REGULATION",
            "guidelines": "GUIDELINE",
            "manuals": "MANUAL",
            "forms": "FORM",
        }
        
        for subdir, doc_level in subdirs.items():
            subdir_path = docs_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.glob("*.txt"):
                    documents.extend(self._process_file(file_path, doc_level))
        
        for file_path in docs_dir.glob("*.txt"):
            documents.extend(self._process_file(file_path, "GENERAL"))
                
        self.documents = documents
        print(f"Loaded {len(documents)} document chunks from {len(set(d['source'] for d in documents))} files")
        return documents
    
    def _process_file(self, file_path: Path, doc_level: str) -> list[DocumentChunk]:
        """파일을 처리하여 DocumentChunk 리스트를 반환합니다."""
        documents = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        metadata = self._extract_metadata(content)
        category = self._extract_category(content, metadata)
        chunks = self._chunk_document(content, file_path.stem)
        
        for i, chunk in enumerate(chunks):
            chunk_with_context = self._add_context_to_chunk(chunk, metadata, doc_level)
            
            documents.append(DocumentChunk(
                content=chunk_with_context,
                source=file_path.name,
                category=category,
                chunk_id=i,
                doc_level=doc_level,
                metadata=metadata,
            ))
        
        return documents
    
    def _extract_metadata(self, content: str) -> DocumentMetadata:
        """문서에서 메타데이터를 추출합니다."""
        metadata: DocumentMetadata = {}
        
        patterns = {
            "doc_id": r"\[문서ID:\s*([^\]]+)\]",
            "doc_name": r"\[문서명:\s*([^\]]+)\]",
            "version": r"\[버전:\s*([^\]]+)\]",
            "effective_date": r"\[시행일:\s*([^\]]+)\]",
            "last_updated": r"\[최종개정일:\s*([^\]]+)\]",
            "doc_level": r"\[문서등급:\s*([^\]]+)\]",
            "related_laws": r"\[관련법규:\s*([^\]]+)\]",
            "parent_doc": r"\[상위문서:\s*([^\]]+)\]",
            "department": r"\[담당부서:\s*([^\]]+)\]",
            "approver": r"\[승인권자:\s*([^\]]+)\]",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metadata[key] = match.group(1).strip()
        
        child_match = re.search(r"\[하위문서:\s*([^\]]+)\]", content)
        if child_match:
            child_docs = [doc.strip() for doc in child_match.group(1).split(",")]
            metadata["child_docs"] = child_docs
        
        return metadata
    
    def _extract_category(self, content: str, metadata: DocumentMetadata) -> str:
        """문서에서 카테고리를 추출합니다."""
        doc_id = metadata.get("doc_id", "")
        
        if doc_id.startswith("HR-"):
            return "HR"
        elif doc_id.startswith("FIN-"):
            return "FINANCE"
        elif doc_id.startswith("SEC-"):
            return "SECURITY"
        elif doc_id.startswith("IT-"):
            return "IT_OPS"
        elif doc_id.startswith("COM-"):
            return "COMPLIANCE"
        elif doc_id.startswith("VND-"):
            return "VENDOR"
        elif doc_id.startswith("ADM-"):
            return "ADMIN"
        
        source_lower = content.lower()
        if "인사" in source_lower or "휴가" in source_lower or "채용" in source_lower:
            return "HR"
        elif "재무" in source_lower or "경비" in source_lower or "출장" in source_lower:
            return "FINANCE"
        elif "보안" in source_lower or "접근" in source_lower:
            return "SECURITY"
        elif "준법" in source_lower or "윤리" in source_lower or "자금세탁" in source_lower:
            return "COMPLIANCE"
        elif "it" in source_lower or "시스템" in source_lower:
            return "IT_OPS"
        
        return "GENERAL"
    
    def _add_context_to_chunk(self, chunk: str, metadata: DocumentMetadata, doc_level: str) -> str:
        """청크에 컨텍스트 정보를 추가합니다."""
        context_parts = []
        
        if metadata.get("doc_name"):
            context_parts.append(f"[{metadata['doc_name']}]")
        if metadata.get("doc_id"):
            context_parts.append(f"(문서번호: {metadata['doc_id']})")
        if doc_level:
            level_kr = {
                "POLICY": "정책",
                "REGULATION": "규정", 
                "GUIDELINE": "지침",
                "MANUAL": "매뉴얼",
                "FORM": "양식",
            }.get(doc_level, doc_level)
            context_parts.append(f"[{level_kr}]")
        
        if context_parts:
            context = " ".join(context_parts)
            return f"{context}\n\n{chunk}"
        
        return chunk

    def get_document_catalog(self) -> list[dict]:
        """
        문서 목차(목록)를 반환합니다. 문서 내용 없이 ID·이름·카테고리·등급만 포함.
        Query Planner 등에서 실제 문서 구조를 참조할 때 사용합니다.
        """
        if not self.documents:
            return []
        seen: set[str] = set()
        catalog: list[dict] = []
        for doc in self.documents:
            meta = doc.get("metadata", {})
            doc_id = meta.get("doc_id") or doc.get("source", "")
            key = doc_id or doc.get("source", "")
            if key in seen:
                continue
            seen.add(key)
            catalog.append({
                "doc_id": meta.get("doc_id") or doc.get("source", ""),
                "doc_name": meta.get("doc_name", ""),
                "category": doc.get("category", "GENERAL"),
                "doc_level": doc.get("doc_level", "GENERAL"),
            })
        return catalog

    def _chunk_document(self, content: str, source: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
        """
        문서를 청크로 분할합니다.
        
        금융사 내규 문서 특성에 맞게:
        - 조(Article) 단위로 분할 시도
        - 섹션/장 경계 유지
        - 표 데이터 유지
        """
        chunks = []
        
        header_end = content.find("=" * 20)
        header = ""
        body = content
        if header_end > 0:
            metadata_end = content.find("=" * 20, header_end + 20)
            if metadata_end > 0:
                header = content[:metadata_end + 80].strip()
                body = content[metadata_end + 80:].strip()
        
        article_pattern = r"(제\d+조\s*\([^)]+\))"
        parts = re.split(article_pattern, body)
        
        current_chunk = ""
        current_article = ""
        
        for part in parts:
            if re.match(article_pattern, part):
                current_article = part
            else:
                if current_article:
                    article_content = current_article + part
                    current_article = ""
                else:
                    article_content = part
                
                if len(current_chunk) + len(article_content) < chunk_size:
                    current_chunk += article_content + "\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    if len(article_content) > chunk_size:
                        sub_chunks = self._split_long_content(article_content, chunk_size, overlap)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = article_content + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if not chunks:
            chunks = self._split_long_content(body, chunk_size, overlap)
            
        if not chunks and content:
            chunks = [content[:chunk_size]]
            
        return chunks
    
    def _split_long_content(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        """긴 콘텐츠를 분할합니다."""
        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(para) > chunk_size:
                    words = para.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 < chunk_size:
                            temp += word + " "
                        else:
                            if temp:
                                chunks.append(temp.strip())
                            temp = word + " "
                    if temp:
                        current_chunk = temp
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def build_index(self, documents: list[DocumentChunk] | None = None):
        """
        FAISS 인덱스를 구축합니다.
        
        Args:
            documents: 인덱싱할 문서 리스트 (None이면 self.documents 사용)
        """
        if documents is not None:
            self.documents = documents
            
        if not self.documents:
            raise ValueError("No documents to index. Load documents first.")
        
        texts = [doc["content"] for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            self.embeddings = embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.embeddings_normalized = embeddings / norms
            
    def save(self, path: Path | None = None):
        """벡터 인덱스를 저장합니다."""
        if path is None:
            path = self.vectordb_dir
            
        path.mkdir(parents=True, exist_ok=True)
        
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        elif hasattr(self, "embeddings"):
            np.save(str(path / "embeddings.npy"), self.embeddings)
            
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
            
    def load(self, path: Path | None = None) -> bool:
        """저장된 벡터 인덱스를 로드합니다."""
        if path is None:
            path = self.vectordb_dir
            
        docs_path = path / "documents.pkl"
        if not docs_path.exists():
            return False
            
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
            
        if FAISS_AVAILABLE:
            index_path = path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                return True
        else:
            embeddings_path = path / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(str(embeddings_path))
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                self.embeddings_normalized = self.embeddings / norms
                return True
                
        return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
        doc_levels: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            categories: 검색할 카테고리 필터 (None이면 전체 검색)
            doc_levels: 검색할 문서 등급 필터 (POLICY, REGULATION, GUIDELINE, MANUAL, FORM)
            
        Returns:
            검색 결과 리스트
        """
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        
        search_pool_size = min(top_k * 3, len(self.documents))
        
        if FAISS_AVAILABLE and self.index is not None:
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, search_pool_size)
            scores = scores[0]
            indices = indices[0]
        else:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(self.embeddings_normalized, query_norm.T).flatten()
            indices = np.argsort(similarities)[::-1][:search_pool_size]
            scores = similarities[indices]
        
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            
            if categories and doc.get("category") not in categories:
                continue
            
            if doc_levels and doc.get("doc_level") not in doc_levels:
                continue
                
            results.append(SearchResult(
                content=doc["content"],
                source=doc["source"],
                category=doc.get("category", "GENERAL"),
                score=float(score),
                doc_level=doc.get("doc_level", "GENERAL"),
                metadata=doc.get("metadata", {}),
            ))
            
            if len(results) >= top_k:
                break
                
        return results
    
    def search_with_hierarchy(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        계층 구조를 고려하여 검색합니다.
        상위 문서(정책)부터 하위 문서(매뉴얼)까지 균형있게 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            categories: 검색할 카테고리 필터
            
        Returns:
            계층별로 균형 잡힌 검색 결과
        """
        all_results = []
        
        hierarchy = ["POLICY", "REGULATION", "GUIDELINE", "MANUAL", "FORM"]
        per_level = max(1, top_k // len(hierarchy))
        
        for level in hierarchy:
            level_results = self.search(
                query=query,
                top_k=per_level + 1,
                categories=categories,
                doc_levels=[level]
            )
            all_results.extend(level_results)
        
        seen = set()
        unique_results = []
        for r in all_results:
            key = r["content"][:200]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results[:top_k]


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """싱글톤 벡터 스토어 인스턴스를 반환합니다."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        if not _vector_store.load():
            print("Vector index not found. Please run init_vectordb.py first.")
    return _vector_store


def search_documents(
    search_queries: list[str],
    categories: list[str] | None = None,
    doc_levels: list[str] | None = None,
    top_k: int = 3,
    use_hierarchy: bool = True,
) -> list[SearchResult]:
    """
    여러 쿼리로 문서를 검색하고 결과를 병합합니다.
    
    Args:
        search_queries: 검색 쿼리 리스트
        categories: 검색할 카테고리 필터 (HR, FINANCE, SECURITY, IT_OPS, COMPLIANCE, VENDOR)
        doc_levels: 검색할 문서 등급 필터 (POLICY, REGULATION, GUIDELINE, MANUAL, FORM)
        top_k: 쿼리당 반환할 최대 결과 수
        use_hierarchy: 계층 구조를 고려한 균형 검색 사용 여부
        
    Returns:
        중복 제거된 검색 결과 리스트
        
    Security Note:
        모든 검색은 로컬에서 수행됩니다.
        문서 내용은 외부로 전송되지 않습니다.
    """
    store = get_vector_store()
    
    all_results: dict[str, SearchResult] = {}
    
    for query in search_queries:
        if use_hierarchy and not doc_levels:
            results = store.search_with_hierarchy(query, top_k=top_k, categories=categories)
        else:
            results = store.search(query, top_k=top_k, categories=categories, doc_levels=doc_levels)
        
        for result in results:
            key = result["content"][:200]
            if key not in all_results or result["score"] > all_results[key]["score"]:
                all_results[key] = result
    
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: (x["score"], _doc_level_priority(x.get("doc_level", ""))),
        reverse=True,
    )
    
    return sorted_results[:top_k * 2]


def _doc_level_priority(doc_level: str) -> int:
    """문서 등급에 따른 우선순위를 반환합니다. (정책이 가장 높음)"""
    priority = {
        "POLICY": 5,
        "REGULATION": 4,
        "GUIDELINE": 3,
        "MANUAL": 2,
        "FORM": 1,
        "GENERAL": 0,
    }
    return priority.get(doc_level, 0)


def get_document_hierarchy(doc_id: str) -> dict:
    """
    특정 문서의 계층 구조 (상위/하위 문서)를 조회합니다.
    
    Args:
        doc_id: 문서 ID (예: FIN-REG-001)
        
    Returns:
        상위/하위 문서 정보
    """
    store = get_vector_store()
    
    result = {
        "doc_id": doc_id,
        "parent": None,
        "children": [],
    }
    
    for doc in store.documents:
        metadata = doc.get("metadata", {})
        if metadata.get("doc_id") == doc_id:
            result["parent"] = metadata.get("parent_doc")
            result["children"] = metadata.get("child_docs", [])
            break
    
    return result
