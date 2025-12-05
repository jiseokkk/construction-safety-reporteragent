import pandas as pd
from typing import List
from langchain.schema import Document
import sys
# retriever.py import
sys.path.append('/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent')
from core.retriever import SingleDBHybridRetriever

def evaluate_retrieval(df: pd.DataFrame, retriever: SingleDBHybridRetriever) -> pd.DataFrame:
    """
    DataFrame의 각 질문에 대해 retrieval을 수행하고 Hit Rate를 측정
    
    Args:
        df: 'question'과 'chunk' 열을 포함한 DataFrame
        retriever: SingleDBHybridRetriever 인스턴스
        
    Returns:
        retrieval_result, hit 열이 추가된 DataFrame
    """
    results = []
    hits = []
    
    print(f"총 {len(df)}개의 질문에 대해 Retrieval 수행 중...")
    print(f"설정: top_k={retriever.top_k}, rerank_top_n={retriever.rerank_top_n}\n")
    
    for idx, row in df.iterrows():
        question = row['question']
        ground_truth_chunk = row['chunk']
        
        print(f"[{idx+1}/{len(df)}] 질문: {question[:80]}...")
        
        # Retrieve 수행
        retrieved_docs: List[Document] = retriever.retrieve(question)
        
        # Retrieved chunks 추출
        retrieved_chunks = [doc.page_content for doc in retrieved_docs]
        results.append(retrieved_chunks)
        
        # Hit 여부 확인 (ground truth chunk가 retrieved chunks에 포함되어 있는지)
        is_hit = any(ground_truth_chunk in chunk or chunk in ground_truth_chunk 
                     for chunk in retrieved_chunks)
        hits.append(is_hit)
        
        print(f"   ✓ Retrieved: {len(retrieved_chunks)}개, Hit: {'✅ True' if is_hit else '❌ False'}\n")
    
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

    df = pd.read_excel("/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/evaluate_RAG/capstone_retrieval_eval_data.xlsx")
    
    print("데이터 로드 완료")
    print(f"   - 총 {len(df)}개 질문")
    print(f"   - 컬럼: {list(df.columns)}\n")
    
    # Retriever 초기화
    retriever = SingleDBHybridRetriever(
        db_dir="/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB2",
        top_k=20,           # 1차 검색: 20개
        rerank_top_n=5,     # reranking 후: 5개
        alpha=0.3
    )
    
    df_result = evaluate_retrieval(df, retriever)
    
    output_file = "retrieval_evaluation_result.csv"
    df_result.to_csv(output_file, index=False)
    print(f"결과가 '{output_file}'에 저장되었습니다.")
