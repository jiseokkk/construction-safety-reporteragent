import os
import re
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker


# =====================================
# 🔹 Qwen3 4B Embedding 설정 그대로 유지
# =====================================
def get_qwen_api_embeddings():
    embedder_model_name = "Qwen/Qwen3-Embedding-4B"
    embedder_base_url = "http://211.47.56.71:15653/v1"
    embedder_api_key = "token-abc123"
    
    print(f"🌐 Qwen Embedding API 연결 중: {embedder_base_url}")
    embeddings = OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )
    return embeddings


# =====================================
# 🔹 텍스트 정제 유틸 함수
# =====================================
def _prettify_text(text: str) -> str:
    text = re.sub(r"[\u2027•․·]+", "·", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\.)([가-힣])", r"\1\n\2", text)
    text = re.sub(r"(·\s*)", r"\n- ", text)
    text = re.sub(r"([가-힣])(\s*:\s*)", r"\1\n", text)
    text = text.strip()
    return text


def _clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(표\s*\d+|부록\s*\d+|부록표\s*\d+)", "", text)
    return _prettify_text(text)


def _is_table_heavy(text: str) -> bool:
    return text.count("|") > 5 or text.count("<table") > 0 or len(text.split()) < 30


def _is_noise_section(doc: Document) -> bool:
    noise_keywords = ["부록", "점검", "확인사항", "항타기", "항발기", "점검표"]
    section = doc.metadata.get("section", "")
    text = doc.page_content
    return any(k in section or k in text[:200] for k in noise_keywords)


# =====================================
# 🔹 Retriever 본체
# =====================================
class RerankRetriever:
    EXCLUDED_SECTIONS = {
        '1. 목적', '1. 목 적', '2. 적용범위', '3. 용어의정의', '3. 정의',
        '한국산업안전보건공단', '안전보건기술지침의개요', '지침개정이력',
        '○제정경과', '제정경과', '개정이력',
        '○기술지침의적용및문의', '○관련법규․규칙․고시등'
    }

    EXCLUDED_SECTION_KEYWORDS = [
        '목적', '정의', '적용범위', '총칙', '개요', '일반사항',
        '제정경과', '개정이력', '제정자', '공표일자',
    ]

    EXCLUDED_CONTENT_PATTERNS = [
        '이 지침은', '이 규칙은', '이 기준의 목적', '용어의 뜻은', '용어의 정의는',
        '다음과 같이 정의', '적용범위는', '제정자:', '공표일자:', '개정일자:',
        '안전보건기술지침은', 'www.kosha.or.kr', '한국산업안전보건공단이사장',
        '제정', '개정', '2010년', '2012년', '2020년',
    ]

    def __init__(
        self,
        title_db_path: str,
        content_db_path: str,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        title_top_k: int = 5,
        contents_top_k: int = 8,
        alpha: float = 0.5,  # ✅ dense/sparse 비율
        min_content_length: int = 100,
    ):
        self.title_db_path = title_db_path
        self.content_db_path = content_db_path
        self.reranker_model = reranker_model
        self.title_top_k = title_top_k
        self.contents_top_k = contents_top_k
        self.alpha = alpha
        self.min_content_length = min_content_length

        self.title_db = None
        self.content_db = None

        print(f"🔍 RerankRetriever 초기화 중 (dense:sparse={self.alpha}:{1-self.alpha})")
        self._setup()
        print("✅ RerankRetriever 생성 완료")

    # =====================================
    # 📘 데이터베이스 로드
    # =====================================
    def _setup(self):
        embeddings = get_qwen_api_embeddings()
        if not os.path.exists(self.title_db_path):
            raise FileNotFoundError(f"❌ Title DB 경로를 찾을 수 없습니다: {self.title_db_path}")
        if not os.path.exists(self.content_db_path):
            raise FileNotFoundError(f"❌ Content DB 경로를 찾을 수 없습니다: {self.content_db_path}")

        print(f"📚 Title DB 로드 중: {self.title_db_path}")
        self.title_db = FAISS.load_local(self.title_db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"📖 Content DB 로드 중: {self.content_db_path}")
        self.content_db = FAISS.load_local(self.content_db_path, embeddings, allow_dangerous_deserialization=True)

    # =====================================
    # ⚙️ 문서 필터링
    # =====================================
    def _is_excluded_document(self, doc: Document) -> bool:
        section = doc.metadata.get('section', '').strip()
        content = doc.page_content.strip()
        if section in self.EXCLUDED_SECTIONS:
            return True
        if any(kw in section.lower() for kw in self.EXCLUDED_SECTION_KEYWORDS):
            return True
        if any(p in content[:200] for p in self.EXCLUDED_CONTENT_PATTERNS):
            return True
        if len(content) < self.min_content_length:
            return True
        return False

    # =====================================
    # 🔎 Title DB 필터링
    # =====================================
    def _filter_by_title(self, query: str) -> List[str]:
        print(f"\n🔎 [STAGE 1] Title DB 필터링... (top_k={self.title_top_k})")
        title_docs = self.title_db.similarity_search(query, k=self.title_top_k)
        filtered = list({d.metadata.get("source", "") for d in title_docs if d.metadata.get("source")})
        print(f"✅ 필터링된 파일: {len(filtered)}개")
        for i, f in enumerate(filtered, 1):
            print(f"   [{i}] {f}")
        return filtered

    # =====================================
    # 🧩 Dense/Sparse Hybrid Merge (가중치 적용)
    # =====================================
    def _hybrid_merge(self, dense_results, sparse_results) -> List[Document]:
        dense_dict = {hash(doc.page_content): score for doc, score in dense_results}
        sparse_dict = {hash(doc.page_content): i for i, doc in enumerate(sparse_results)}

        all_docs = []
        for doc, d_score in dense_results:
            h = hash(doc.page_content)
            s_rank = sparse_dict.get(h, len(sparse_results))
            combined_score = self.alpha * d_score + (1 - self.alpha) * (1 - s_rank / len(sparse_results))
            all_docs.append((doc, combined_score))

        # sparse only 추가
        for i, doc in enumerate(sparse_results):
            h = hash(doc.page_content)
            if h not in dense_dict:
                combined_score = (1 - self.alpha) * (1 - i / len(sparse_results))
                all_docs.append((doc, combined_score))

        all_docs = sorted(all_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in all_docs]

    # =====================================
    # 📖 Content DB 검색
    # =====================================
    def _retrieve_from_content_db(self, query: str, files: List[str]) -> List[Document]:
        print(f"\n📄 [STAGE 2] Content DB에서 파일 검색...")

        all_docs = list(self.content_db.docstore._dict.values())
        filtered_docs = [d for d in all_docs if d.metadata.get("source") in files and not self._is_excluded_document(d)]
        if not filtered_docs:
            filtered_docs = all_docs

        dense_results = self.content_db.similarity_search_with_score(query, k=self.contents_top_k * 4)
        sparse_retriever = BM25Retriever.from_documents(filtered_docs)
        sparse_retriever.k = self.contents_top_k * 4
        sparse_results = sparse_retriever.get_relevant_documents(query)

        print(f"📊 Dense/Sparse 결합 중... (alpha={self.alpha})")
        hybrid_docs = self._hybrid_merge(dense_results, sparse_results)

        # reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.contents_top_k * 2)
        reranked = compressor.compress_documents(hybrid_docs, query)

        # ✅ 정제 및 필터링
        cleaned = []
        for d in reranked:
            d.page_content = _clean_html(d.page_content)
            if (
                len(d.page_content) > self.min_content_length
                and not _is_table_heavy(d.page_content)
                and not _is_noise_section(d)
                and not self._is_excluded_document(d)
            ):
                cleaned.append(d)

        print(f"✅ 최종 필터링 후 {len(cleaned)}개 문서 유지")
        return cleaned[: self.contents_top_k]

    # =====================================
    # 🚀 전체 검색 파이프라인
    # =====================================
    def retrieve(self, query: str) -> List[Document]:
        print(f"\n{'='*80}\n📝 입력 쿼리: {query}\n{'='*80}")
        lines = query.splitlines()
        construct = next((l.split(":")[1].strip() for l in lines if "공종" in l), None)
        process = next((l.split(":")[1].strip() for l in lines if "작업프로세스" in l), None)
        core_query = process or construct or query
        print(f"🎯 핵심 검색어: {core_query}")

        files = self._filter_by_title(core_query)
        docs = self._retrieve_from_content_db(core_query, files)

        print(f"\n✅ 최종 검색 결과: {len(docs)}개 문서\n" + "="*80)
        return docs


# =====================================
# 🔹 전역 인스턴스
# =====================================
retriever_instance = RerankRetriever(
    title_db_path="/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB/title_db",
    content_db_path="/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB/content_db",
    reranker_model="BAAI/bge-reranker-v2-m3",
    title_top_k=5,
    contents_top_k=8,
    alpha=0.3,  # ✅ dense:sparse 1:1
    min_content_length=100,
)


# =====================================
# 🔹 (선택적) LangGraph용 Node 함수 - 호환용
# =====================================
def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    예전 LangGraph 구조와의 호환을 위한 node 함수.
    현재는 Orchestrator + RAGAgent 구조를 사용하지만, 남겨둠.
    """
    query = state.get("user_query") or state.get("query", "")
    docs = retriever_instance.retrieve(query)

    docs_text = "\n\n".join(
        f"[{i+1}] ({d.metadata.get('source','?')} - {d.metadata.get('section','?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    sources = [
        {"idx": i + 1, "filename": d.metadata.get("source", ""), "section": d.metadata.get("section", "")}
        for i, d in enumerate(docs)
    ]

    state["retrieved_docs"] = docs
    state["docs_text"] = docs_text
    state["sources"] = sources

    return state
