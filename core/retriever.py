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


def get_qwen_api_embeddings():
    embedder_model_name = "Qwen/Qwen3-Embedding-4B"
    embedder_base_url = "http://211.47.56.71:15653/v1"
    embedder_api_key = "token-abc123"

    return OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )


def _clean_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SingleDBHybridRetriever:
    def __init__(
        self,
        db_dir: str,
        top_k: int = 20,
        alpha: float = 0.7,
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

        # BM25를 위한 전체 문서 로드
        self.all_docs = list(self.vector_db.docstore._dict.values())

        # 🔥 Reranker를 여기서 한 번만 초기화
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

        # 1) Dense(AI semantic)
        dense = self.vector_db.similarity_search_with_score(query, k=self.top_k)

        # 2) Sparse(keyword)
        sparse_retriever = BM25Retriever.from_documents(self.all_docs)
        sparse_retriever.k = self.top_k 
        sparse = sparse_retriever.get_relevant_documents(query)

        # 3) Hybrid merge
        hybrid_docs = self._hybrid_merge(dense, sparse)

        # 4) Rerank - 이미 초기화된 compressor 사용
        reranked = self.compressor.compress_documents(hybrid_docs, query)

        # 5) Clean & return top_k
        final_docs = []
        for d in reranked[: self.top_k]:
            d.page_content = _clean_text(d.page_content)
            final_docs.append(d)

        print(f"📊 최종 반환 문서: {len(final_docs)}개")
        return final_docs