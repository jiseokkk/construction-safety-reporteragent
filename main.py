# main.py
import os
import pandas as pd

from core.agentstate import AgentState
from graph.workflow import graph_app
from core.query_builder import build_user_query_from_row, row_to_structured_fields


CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/train_preprocessing.csv"


def load_test_query_from_csv(row_index: int = 0) -> tuple[str, dict]:
    """train_preprocessing.csv의 한 row를 읽어 user_query와 구조화 필드를 반환"""
    df = pd.read_csv(CSV_PATH)
    row = df.iloc[row_index]

    user_query = build_user_query_from_row(row)
    fields = row_to_structured_fields(row)

    print("\n=== [CSV TEST INPUT] ===")
    print(f"선택된 ID: {row.get('ID', 'N/A')}")
    print(user_query)
    print("========================\n")

    return user_query, fields


def main():
    # 1) CSV에서 테스트용 user_query 생성 (0번째 row 사용)
    user_query, fields = load_test_query_from_csv(row_index=0)

    # 2) 초기 상태 구성
    state: AgentState = {
        "user_query": user_query,
        "raw_fields": fields,
    }

    print("\n" + "=" * 80)
    print("🚀 건설안전 Multi-Agent 보고서 생성 시스템 (CSV 테스트 모드)")
    print("=" * 80)

    # 3) LangGraph 워크플로우 실행 (Orchestrator + SubAgents)
    final_state = graph_app.invoke(state)

    print("\n" + "=" * 80)
    print("🎉 시스템 실행 완료!")
    print("=" * 80)

    # 4) 결과 요약 출력
    docs = final_state.get("retrieved_docs") or []
    report_text = final_state.get("report_text", "")
    docx_path = final_state.get("docx_path")

    print(f"\n📊 최종 결과 요약:")
    print(f"- 검색된 문서 수: {len(docs)}")
    print(f"- 보고서 텍스트 길이: {len(report_text)} 글자")
    print(f"- DOCX 파일: {docx_path}")

    if report_text:
        print("\n--- 보고서 내용 (처음 500자) ---")
        print(report_text[:500])
        print("\n------------------------------")


if __name__ == "__main__":
    main()
