import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# retriever.py import
sys.path.append('/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent')

from core.retriever import SingleDBHybridRetriever

def calculate_hit_at_k(retrieved_docs: List, ground_truth_chunk: str, k: int = 5) -> int:
    """
    Retrieved documents 중 ground truth chunk가 있는지 확인
    
    Args:
        retrieved_docs: 검색된 문서 리스트
        ground_truth_chunk: 정답 chunk 텍스트
        k: top-k 문서 확인
    
    Returns:
        1 if hit, 0 if miss
    """
    for doc in retrieved_docs[:k]:
        # 정답 chunk가 검색된 문서에 포함되어 있는지 확인
        if ground_truth_chunk.strip() in doc.page_content:
            return 1
    return 0


def calculate_mrr(retrieved_docs: List, ground_truth_chunk: str) -> float:
    """
    Mean Reciprocal Rank 계산
    
    Args:
        retrieved_docs: 검색된 문서 리스트
        ground_truth_chunk: 정답 chunk 텍스트
    
    Returns:
        reciprocal rank (0 if not found)
    """
    for idx, doc in enumerate(retrieved_docs, 1):
        if ground_truth_chunk.strip() in doc.page_content:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval(
    db_dir: str,
    eval_data_path: str,
    top_k: int = 5,
    alpha: float = 0.3,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    output_path: str = None
):
    """
    RAG Retrieval 성능 평가
    
    Args:
        db_dir: FAISS DB 경로
        eval_data_path: 평가 데이터 엑셀 파일 경로
        top_k: retriever top-k
        alpha: hybrid retrieval alpha (dense weight)
        reranker_model: reranker 모델명
        output_path: 결과 저장 경로 (None이면 저장 안함)
    """
    print("=" * 80)
    print("RAG Retrieval 평가 시작")
    print("=" * 80)
    print(f"📂 DB 경로: {db_dir}")
    print(f"📄 평가 데이터: {eval_data_path}")
    print(f"⚙️  설정: top_k={top_k}, alpha={alpha}, reranker={reranker_model}")
    print("=" * 80)
    
    # 1. Retriever 초기화
    print("\n🔧 Retriever 초기화 중...")
    retriever = SingleDBHybridRetriever(
        db_dir=db_dir,
        top_k=top_k,
        alpha=alpha,
        reranker_model=reranker_model
    )
    
    # 2. 평가 데이터 로드
    print(f"\n📊 평가 데이터 로드 중...")
    eval_df = pd.read_excel(eval_data_path)
    print(f"총 {len(eval_df)}개의 평가 쿼리")
    
    # 3. 평가 수행
    print(f"\n🔍 검색 및 평가 수행 중...")
    results = []
    hit_at_1_sum = 0
    hit_at_3_sum = 0
    hit_at_5_sum = 0
    mrr_sum = 0
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="평가 진행"):
        question = row['question ']  # 공백 주의
        ground_truth_chunk = row['chunk']
        doctitle = row['doctitle']
        
        # 검색 수행
        try:
            retrieved_docs = retriever.retrieve(question)
            
            # Hit@K 계산
            hit_at_1 = calculate_hit_at_k(retrieved_docs, ground_truth_chunk, k=1)
            hit_at_3 = calculate_hit_at_k(retrieved_docs, ground_truth_chunk, k=3)
            hit_at_5 = calculate_hit_at_k(retrieved_docs, ground_truth_chunk, k=5)
            
            # MRR 계산
            mrr = calculate_mrr(retrieved_docs, ground_truth_chunk)
            
            # 누적
            hit_at_1_sum += hit_at_1
            hit_at_3_sum += hit_at_3
            hit_at_5_sum += hit_at_5
            mrr_sum += mrr
            
            # 개별 결과 저장
            results.append({
                'index': idx,
                'doctitle': doctitle,
                'question': question,
                'ground_truth_chunk': ground_truth_chunk,
                'hit@1': hit_at_1,
                'hit@3': hit_at_3,
                'hit@5': hit_at_5,
                'mrr': mrr,
                'retrieved_docs_count': len(retrieved_docs)
            })
            
        except Exception as e:
            print(f"\n⚠️  Query {idx} 처리 중 오류: {str(e)}")
            results.append({
                'index': idx,
                'doctitle': doctitle,
                'question': question,
                'ground_truth_chunk': ground_truth_chunk,
                'hit@1': 0,
                'hit@3': 0,
                'hit@5': 0,
                'mrr': 0,
                'retrieved_docs_count': 0,
                'error': str(e)
            })
    
    # 4. 최종 메트릭 계산
    num_queries = len(eval_df)
    avg_hit_at_1 = hit_at_1_sum / num_queries
    avg_hit_at_3 = hit_at_3_sum / num_queries
    avg_hit_at_5 = hit_at_5_sum / num_queries
    avg_mrr = mrr_sum / num_queries
    
    # 5. 결과 출력
    print("\n" + "=" * 80)
    print("📈 평가 결과")
    print("=" * 80)
    print(f"총 쿼리 수: {num_queries}")
    print(f"\n【 Hit Rate 】")
    print(f"  Hit@1: {avg_hit_at_1:.4f} ({avg_hit_at_1*100:.2f}%)")
    print(f"  Hit@3: {avg_hit_at_3:.4f} ({avg_hit_at_3*100:.2f}%)")
    print(f"  Hit@5: {avg_hit_at_5:.4f} ({avg_hit_at_5*100:.2f}%)")
    print(f"\n【 MRR (Mean Reciprocal Rank) 】")
    print(f"  MRR: {avg_mrr:.4f}")
    print("=" * 80)
    
    # 6. 결과 저장
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)
        print(f"\n💾 상세 결과 저장: {output_path}")
        
        # 요약 저장
        summary_path = output_path.replace('.xlsx', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAG Retrieval 평가 결과 요약\n")
            f.write("=" * 80 + "\n")
            f.write(f"DB 경로: {db_dir}\n")
            f.write(f"평가 데이터: {eval_data_path}\n")
            f.write(f"설정: top_k={top_k}, alpha={alpha}, reranker={reranker_model}\n")
            f.write("=" * 80 + "\n")
            f.write(f"총 쿼리 수: {num_queries}\n\n")
            f.write("【 Hit Rate 】\n")
            f.write(f"  Hit@1: {avg_hit_at_1:.4f} ({avg_hit_at_1*100:.2f}%)\n")
            f.write(f"  Hit@3: {avg_hit_at_3:.4f} ({avg_hit_at_3*100:.2f}%)\n")
            f.write(f"  Hit@5: {avg_hit_at_5:.4f} ({avg_hit_at_5*100:.2f}%)\n\n")
            f.write("【 MRR (Mean Reciprocal Rank) 】\n")
            f.write(f"  MRR: {avg_mrr:.4f}\n")
            f.write("=" * 80 + "\n")
        print(f"📄 요약 결과 저장: {summary_path}")
    
    return {
        'hit@1': avg_hit_at_1,
        'hit@3': avg_hit_at_3,
        'hit@5': avg_hit_at_5,
        'mrr': avg_mrr,
        'results': results
    }


def compare_retrieval_configs(
    db_dir: str,
    eval_data_path: str,
    configs: List[Dict],
    output_dir: str = "/home/claude/eval_results"
):
    """
    여러 retrieval 설정 비교
    
    Args:
        db_dir: FAISS DB 경로
        eval_data_path: 평가 데이터 경로
        configs: 비교할 설정 리스트 (각각 dict with top_k, alpha, reranker_model)
        output_dir: 결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = []
    
    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"설정 {config_idx}/{len(configs)} 평가")
        print(f"{'='*80}")
        
        output_path = os.path.join(
            output_dir, 
            f"config_{config_idx}_k{config['top_k']}_a{config['alpha']}.xlsx"
        )
        
        result = evaluate_retrieval(
            db_dir=db_dir,
            eval_data_path=eval_data_path,
            top_k=config.get('top_k', 5),
            alpha=config.get('alpha', 0.3),
            reranker_model=config.get('reranker_model', 'BAAI/bge-reranker-v2-m3'),
            output_path=output_path
        )
        
        comparison_results.append({
            'config_name': f"config_{config_idx}",
            'top_k': config.get('top_k', 5),
            'alpha': config.get('alpha', 0.3),
            'reranker': config.get('reranker_model', 'BAAI/bge-reranker-v2-m3'),
            'hit@1': result['hit@1'],
            'hit@3': result['hit@3'],
            'hit@5': result['hit@5'],
            'mrr': result['mrr']
        })
    
    # 비교 결과 저장
    comparison_df = pd.DataFrame(comparison_results)
    comparison_path = os.path.join(output_dir, 'comparison_summary.xlsx')
    comparison_df.to_excel(comparison_path, index=False)
    
    print(f"\n{'='*80}")
    print("🏆 전체 비교 결과")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    print(f"\n💾 비교 결과 저장: {comparison_path}")
    
    return comparison_results


if __name__ == "__main__":
    # 단일 DB 평가
    DB_DIR = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB2/content_db"
    EVAL_DATA_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/evaluate_RAG/capstone_retrieval_eval_data.xlsx"
    OUTPUT_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/evaluate_RAG/capstone_retrieval_eval_data.xlsx"
    
    # 평가 실행
    result = evaluate_retrieval(
        db_dir=DB_DIR,
        eval_data_path=EVAL_DATA_PATH,
        top_k=5,
        alpha=0.3,
        reranker_model="BAAI/bge-reranker-v2-m3",
        output_path=OUTPUT_PATH
    )
    
    # 여러 설정 비교 (옵션)
    # configs = [
    #     {'top_k': 3, 'alpha': 0.3},
    #     {'top_k': 5, 'alpha': 0.3},
    #     {'top_k': 10, 'alpha': 0.3},
    #     {'top_k': 5, 'alpha': 0.5},
    #     {'top_k': 5, 'alpha': 0.7},
    # ]
    # compare_retrieval_configs(DB_DIR, EVAL_DATA_PATH, configs)