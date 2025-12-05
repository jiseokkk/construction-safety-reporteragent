import pandas as pd
from typing import List, Union
from langchain.schema import Document
import os
import sys

# retriever.py 파일 경로를 가정하고 임포트 (사용자 제공 코드와 일치)
# 실제 환경에서는 이 경로 설정이 필요합니다.
# sys.path.append('/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent') 

# SingleDBHybridRetriever 대신 MultiDBRouter를 임포트
try:
    from core.llm_multidb_retriever import MultiDBRouter, SingleDBHybridRetriever
except ImportError:
    # 예시 환경에서 모듈을 찾지 못할 경우를 대비하여 더미 클래스 사용
    print("⚠️ Warning: Could not import MultiDBRouter/SingleDBHybridRetriever. Using dummy classes for visualization.")
    class DummyRetriever:
        def __init__(self, *args, **kwargs): pass
        def retrieve(self, question): return []
    MultiDBRouter = DummyRetriever
    SingleDBHybridRetriever = DummyRetriever


def evaluate_retrieval(df: pd.DataFrame, router: MultiDBRouter, top_k: int, rerank_top_n: int) -> pd.DataFrame:
    """
    DataFrame의 각 질문에 대해 MultiDBRouter를 사용하여 retrieval을 수행하고 Hit Rate를 측정
    
    Args:
        df: 'question'과 'chunk' 열을 포함한 DataFrame
        router: MultiDBRouter 인스턴스 (라우팅 담당)
        top_k: SingleDBHybridRetriever의 초기 검색 K 값
        rerank_top_n: Reranking 후 최종 반환 K 값
        
    Returns:
        retrieval_result, hit 열이 추가된 DataFrame
    """
    results = []
    hits = []
    
    print(f"총 {len(df)}개의 질문에 대해 Multi-DB Retrieval 수행 중...")
    print(f"설정: Initial K={top_k}, Rerank K={rerank_top_n}\n")
    
    for idx, row in df.iterrows():
        question = row['question']
        ground_truth_chunk = row['chunk']
        
        print(f"[{idx+1}/{len(df)}] 질문: {question[:80]}...")
        
        # Router를 통해 DB 선택 후, 해당 DB에서 Retrieve 수행
        retrieved_docs: List[Document] = router.retrieve(
            question, 
            top_k=top_k, 
            rerank_top_n=rerank_top_n
        )
        
        # Retrieved chunks 추출
        retrieved_chunks = [doc.page_content for doc in retrieved_docs]
        results.append(retrieved_chunks)
        
        # Hit 여부 확인 (ground truth chunk가 retrieved chunks에 포함되어 있는지)
        # Note: 실제 환경에서 ground_truth_chunk와 retrieved_chunks의 텍스트 정규화 필요
        is_hit = any(ground_truth_chunk in chunk or chunk in ground_truth_chunk 
                     for chunk in retrieved_chunks)
        hits.append(is_hit)
        
        print(f"  ✓ Retrieved: {len(retrieved_chunks)}개, Hit: {'✅ True' if is_hit else '❌ False'}\n")
    
    df['retrieval_result'] = results
    df['hit'] = hits
    
    # Hit Rate 계산
    hit_rate = sum(hits) / len(hits) * 100
    print(f"\n{'='*70}")
    print(f"Retrieval 평가 결과")
    print(f"{'='*70}")
    print(f"전체 질문 수: {len(df)}")
    print(f"Hit 수: {sum(hits)}")
    print(f"Miss 수: {len(hits) - sum(hits)}")
    print(f"Hit Rate: {hit_rate:.2f}%")
    print(f"{'='*70}\n")
    
    return df


if __name__ == "__main__":
    # 이 경로 설정은 사용자님의 로컬 환경에 맞게 조정되어야 합니다.
    BASE_DB_DIR = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB2"
    
    try:
        df = pd.read_excel("/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/evaluate_RAG/capstone_retrieval_eval_data.xlsx")
    except FileNotFoundError:
        print("❌ Error: Evaluation data file not found. Creating dummy data.")
        df = pd.DataFrame({
            'question': ["사장교 교량공사 안전보건작업지침의 목적은?", "타워크레인 설치 시 안전 기준은?", "일반적인 안전대 사용지침은?"],
            'chunk': ["안전보건작업 지침의 목적은...", "타워크레인 설치 해체 시...", "안전대는 추락 위험 장소에서..."],
        })
        
    print("데이터 로드 완료")
    print(f"  - 총 {len(df)}개 질문")
    
    # MultiDBRouter 초기화 (DB2 경로 전달)
    router = MultiDBRouter(base_db_dir=BASE_DB_DIR)
    
    # 평가 설정 값
    TOP_K = 20         # 초기 검색 K 값
    RERANK_TOP_N = 5   # Reranking 후 최종 반환 K 값
    
    df_result = evaluate_retrieval(df, router, TOP_K, RERANK_TOP_N)
    
    output_file = "retrieval_evaluation_result_routed.csv"
    df_result.to_csv(output_file, index=False)
    print(f"결과가 '{output_file}'에 저장되었습니다.")