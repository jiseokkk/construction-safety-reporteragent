import os
import re
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# ---------------------------
# 1) EXACT TITLE MAP
# ---------------------------

EXACT_TITLE_DB_MAP = {
    # 01_bridge (교량)
    "사장교교량공사안전보건작업지침": "01_bridge",
    "교량공사의이동식비계공법(mss)안전작업지침": "01_bridge",
    "강아치교(벤트공법)안전보건작업지침": "01_bridge",
    "pct거더교량공사안전보건작업지침": "01_bridge",
    "현수교교량공사안전보건작업지침": "01_bridge",
    "교량슬래브거푸집해체용작업대차안전작업지침": "01_bridge",
    "교량공사(라멘교)안전보건작업지침": "01_bridge",
    "현수교주탑시공안전보건작업지침": "01_bridge",
    "소규모철근콘크리트교량공사거푸집동바리안전작업지침": "01_bridge",
    "i.l.m교량공사안전보건작업지침": "01_bridge",
    "트러스거더교량공사안전보건작업지침": "01_bridge",
    "해상rcd현장타설말뚝공사(현수교,사장교)안전작업지침": "01_bridge",
    "교량공사(p.s.m공법)안전작업지침": "01_bridge",
    "프리스트레스트콘크리트(psc)교량공사안전작업지침": "01_bridge",
    "f.c.m교량공사안전보건작업지침": "01_bridge",
    # 02_earth (토공사, 흙막이)
    "흙막이공사(soilnailing공법)안전보건작업지침": "02_earth",
    "흙막이공사(지하연속벽)안전보건작업지침": "02_earth",
    "우물통기초안전보건작업지침": "02_earth",
    "시트(sheet)방수안전보건작업지침": "02_earth",
    "흙막이공사(강널말뚝,sheetpile)의안전보건작업지침": "02_earth",
    "흙막이공사(earthanchor공법)안전보건작업지침": "02_earth",
    "흙막이공사(엄지말뚝공법)안전보건작업지침": "02_earth",
    "건설공사굴착면안전기울기준에관한기술지침": "02_earth",
    "흙막이공사(띠장긴장공법,prestressedwalemethod)안전보건작업지침": "02_earth",
    "지하매설물굴착공사안전작업지침": "02_earth",
    "굴착공사계측관리기술지침": "02_earth",
    "옹벽(콘크리트옹벽)공사의안전보건작업지침": "02_earth",
    "중소규모관로공사의안전보건작업지침": "02_earth",
    "가공송전선로철탑심형기초공사안전보건작업지침": "02_earth",
    "흙막이공사(scw공법)안전보건작업지침": "02_earth",
    "굴착기안전보건작업지침": "02_earth",
    "흙막이공사(c.i.p공법)안전보건작업지침": "02_earth",
    "굴착공사안전작업지침": "02_earth",
    "관로매설공사안전보건작업기술지침": "02_earth",
    "블록식보강토옹벽공사안전보건작업지침": "02_earth",
    "관로매설공사(유압식추진공법)안전보건작업지침": "02_earth",
    # 03_tunnel (터널)
    "터널공사(ntr공법)안전보건작업지침": "03_tunnel",
    "터널공사(프론트잭킹)안전보건작업지침": "03_tunnel",
    "터널공사(shield-t.b.m공법)안전보건작업지침": "03_tunnel",
    "발파공사안전보건작업지침": "03_tunnel",
    "터널공사(침매공법)안전보건작업지침": "03_tunnel",
    "탑다운(topdown)공법안전작업지침": "03_tunnel",
    "터널공사(natm공법)안전보건작업지침": "03_tunnel",
    # 04_scaffold (비계, 동바리)
    "철골공사무지보거푸집동바리(데크플레이트공법)안전보건작업지침": "04_scaffold",
    "가설구조물의설계변경요청내용절차등에관한작성지침": "04_scaffold",
    "갱폼(gangform)제작및사용안전지침": "04_scaffold",
    "낙하물방호선반설치지침": "04_scaffold",
    "시스템폼(rcs폼,acs폼중심)안전작업지침": "04_scaffold",
    "강관비계안전작업지침": "04_scaffold",
    "수직보호망설치지침": "04_scaffold",
    "작업발판설치및사용안전지침": "04_scaffold",
    "곤돌라(gondola)안전보건작업지침": "04_scaffold",
    "이동식비계설치및사용안전기술지침": "04_scaffold",
    "슬립폼(slipform)안전작업지침": "04_scaffold",
    "작업의자형달비계안전작업지침": "04_scaffold",
    "수직형추락방망설치기술지침": "04_scaffold",
    "가설계단설치및사용안전보건작업지침": "04_scaffold",
    "파이프서포트동바리안전작업지침": "04_scaffold",
    "시스템비계안전작업지침": "04_scaffold",
    "낙하물방지망설치지침": "04_scaffold",
    "추락방호망설치지침": "04_scaffold",
    "시스템동바리안전작업지침": "04_scaffold",
    # 05_crane (크레인, 장비)
    "건설공사의고소작업대안전보건작업지침": "05_crane",
    "타워크레인설치조립해체작업계획서작성지침": "05_crane",
    "이동식크레인안전보건작업지침": "05_crane",
    "항타기항발기사용작업계획서작성지침": "05_crane",
    "수상바지(barge)선이용건설공사안전작업지침": "05_crane",
    "건설현장의중량물취급작업계획서(이동식크레인)작성지침": "05_crane",
    "덤프트럭및화물자동차안전작업지침": "05_crane",
    "트럭탑재형크레인(cagocrane)안전보건작업지침": "05_crane",
    "건설기계안전보건작업지침": "05_crane",
    # 06_finishing (마감)
    "밀폐공간의방수공사안전보건작업지침": "06_finishing",
    "미장공사안전보건작업지침": "06_finishing",
    "조적공사안전보건작업기술지침": "06_finishing",
    "건축물의석공사(내외장)안전보건작업기술지침": "06_finishing",
    "조경공사(수목식재작업)안전보건작업지침": "06_finishing",
    "냉동냉장물류창고단열공사화재예방안전보건작업지침": "06_finishing",
    "내장공사의안전보건작업지침": "06_finishing",
    "금속커튼월(curtainwall)안전작업지침": "06_finishing",
    "타일(tile)공사안전보건작업지침": "06_finishing",
    "경량철골천장공사안전보건작업지침": "06_finishing",
    # 07_concrete (콘크리트/철골)
    "철탑공사안전보건기술지침": "07_concrete",
    "콘크리트공사의안전보건작업지침": "07_concrete",
    "기성콘크리트파일항타안전보건작업지침": "07_concrete",
    "철골공사안전보건작업지침": "07_concrete",
    "프리캐스트콘크리트건축구조물조립안전보건작업지침": "07_concrete",
    "아스팔트콘크리트포장공사안전보건작업지침": "07_concrete",
    "단순슬래브콘크리트타설안전보건작업지침": "07_concrete",
    # 08_general (공통 안전)
    "해체공사안전보건작업기술지침": "08_general",
    "야간건설공사안전보건작업지침": "08_general",
    "중소규모건설업체본사의안전보건관리에관한지침": "08_general",
    "건설현장용접용단안전보건작업기술지침": "08_general",
    "화학플랜트개보수공사안전보건작업기술지침": "08_general",
    "초고층건축물공사(화재예방)안전보건작업지침": "08_general",
    "건설공사안전보건설계지침": "08_general",
    "안전대사용지침": "08_general",
    "초고층건축물공사(일반사항)안전보건작업지침": "08_general",
    "취약시기건설현장안전작업지침": "08_general",
    "건설공사돌관작업안전보건작업지침": "08_general",
}

# ---------------------------
# 2) EMBEDDING
# ---------------------------

def get_qwen_api_embeddings():
    embedder_model_name = "Qwen/Qwen3-Embedding-4B"
    embedder_base_url = "http://211.47.56.71:15653/v1"
    embedder_api_key = "token-abc123"

    return OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )


# ---------------------------
# 3) TEXT CLEAN
# ---------------------------

def _clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------
# 4) Hybrid Retriever
# ---------------------------

class SingleDBHybridRetriever:
    def __init__(
        self,
        db_dir: str,
        top_k: int = 20,
        alpha: float = 0.3,
        rerank_top_n: int = 5,
        reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ):
        self.db_dir = db_dir
        self.top_k = top_k
        self.alpha = alpha
        self.rerank_top_n = rerank_top_n
        self.reranker_model = reranker_model

        print(f"📂 HybridRetriever 초기화: {db_dir}")

        # 1) load FAISS
        self.embeddings = get_qwen_api_embeddings()
        self.vector_db = FAISS.load_local(
            db_dir, self.embeddings, allow_dangerous_deserialization=True
        )

        # 2) BM25 위한 전체 문서
        self.all_docs = list(self.vector_db.docstore._dict.values())

        # 3) Reranker 사전 로딩
        print(f"🔄 Reranker 모델 로딩: {reranker_model}")
        self.reranker = HuggingFaceCrossEncoder(model_name=reranker_model)
        self.compressor = CrossEncoderReranker(model=self.reranker, top_n=rerank_top_n)
        print(f"✅ Reranker 로딩 완료")

    def _hybrid_merge(self, dense_results, sparse_results):
        dense_dict = {hash(doc.page_content): score for doc, score in dense_results}
        sparse_dict = {hash(doc.page_content): i for i, doc in enumerate(sparse_results)}

        merged = []
        for doc, ds in dense_results:
            h = hash(doc.page_content)
            sr = sparse_dict.get(h, len(sparse_results))
            score = self.alpha * ds + (1 - self.alpha) * (1 - sr / len(sparse_results))
            merged.append((doc, score))

        for i, doc in enumerate(sparse_results):
            h = hash(doc.page_content)
            if h not in dense_dict:
                score = (1 - self.alpha) * (1 - i / len(sparse_results))
                merged.append((doc, score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in merged]

    def retrieve(self, query: str) -> List[Document]:
        print(f"\n🔍 [HybridRetriever] Query: {query}")

        # 1) Dense
        dense = self.vector_db.similarity_search_with_score(query, k=self.top_k)

        # 2) Sparse
        sparse_retriever = BM25Retriever.from_documents(self.all_docs)
        sparse_retriever.k = self.top_k * 4
        sparse = sparse_retriever.get_relevant_documents(query)

        # 3) hybrid merge
        hybrid_docs = self._hybrid_merge(dense, sparse)

        # 4) rerank
        reranked = self.compressor.compress_documents(hybrid_docs, query)

        # 5) clean
        final_docs = []
        for d in reranked[: self.top_k]:
            d.page_content = _clean_text(d.page_content)
            final_docs.append(d)

        print(f"📊 최종 반환 문서: {len(final_docs)}개")
        return final_docs


# ---------------------------
# 5) doctitle → DB 선택 → retriever 검색
# ---------------------------

class DocTitleHybridRouter:
    def __init__(self, db_root_dir: str):
        """
        db_root_dir/
            ├── 01_bridge/
            ├── 02_earth/
            ├── ...
            └── 08_general/
        """
        self.db_root = db_root_dir

    def get_db_from_title(self, doctitle: str) -> str:
        # 완전 일치만 사용
        db = EXACT_TITLE_DB_MAP.get(doctitle)
        if db is None:
            raise ValueError(f"❌ 매핑된 DB 없음: {doctitle}")
        return db

    def retrieve(self, doctitle: str, question: str) -> List[Document]:
        print("=====================================")
        print(f"📘 도큐먼트 제목: {doctitle}")
        print(f"💬 사용자 질문: {question}")
        print("=====================================")

        # 1) doctitle → DB 이름
        db_name = self.get_db_from_title(doctitle)
        print(f"📌 선택된 DB: {db_name}")

        # 2) 절대경로
        db_dir = os.path.join(self.db_root, db_name)

        # 3) hybrid retriever 생성
        retriever = SingleDBHybridRetriever(db_dir=db_dir)

        # 4) 검색
        docs = retriever.retrieve(question)
        return docs
