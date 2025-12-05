"""
터미널 기반 건설안전 Intelligent Multi-Agent 시스템
Orchestrator + LangGraph 기반 완전 업데이트 버전
"""

import os
import pandas as pd

from core.agentstate import AgentState
from agents.orchestrator import OrchestratorAgent
from graph.workflow import create_graph


CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv"


# ==========================================
# CSV 로딩 함수
# ==========================================
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    df["발생일시_parsed"] = pd.to_datetime(
        df["발생일시"].str.split().str[0],
        format="%Y-%m-%d",
        errors="coerce"
    )

    print(f"✅ CSV 로드 완료: {len(df)}개 사고 기록")
    return df


# ==========================================
# 메인 인터랙션 루프
# ==========================================
def main():

    # 1) CSV 로드
    try:
        df = load_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    # 2) OrchestratorAgent 초기화
    print("🔧 Orchestrator 초기화 중...")
    orchestrator = OrchestratorAgent(df=df, csv_path=CSV_PATH)

    # 3) LangGraph 생성
    graph_app = create_graph()

    # 4) 사용자 안내
    print("\n" + "=" * 80)
    print("🏗️  건설안전 Intelligent Multi-Agent 시스템 (터미널 버전)")
    print("=" * 80)
    print("💬 자연어로 지시하세요.")
    print("예시:")
    print("  • '8월 8일 사고 정보 알려줘'")
    print("  • '2024-07-03 사고 관련 지침 검색해줘'")
    print("  • '2024-07-03 떨어짐 사고 보고서 작성해줘'")
    print("종료: exit / quit\n")

    # 5) 메인 루프
    while True:
        user_input = input("\n💬 입력: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\n👋 시스템 종료")
            break

        if not user_input:
            print("⚠️ 입력이 없습니다.")
            continue

        # 6) 초기 AgentState 구성
        state = AgentState()
        state["user_query"] = user_input  # 자연어 그대로 전달
        state["user_intent"] = None       # Intent는 Orchestrator가 결정

        print("\n🚀 시스템 실행 중...\n")

        # 7) LangGraph 실행
        final_state = graph_app.invoke(state)

        # 8) 출력 포맷
        print("\n" + "🎉" * 40)
        print("🎉 final_state:")
        print("🎉" * 40)

        intent = final_state.get("user_intent")
        print(f"🧭 수행 Intent: {intent}")

        if intent == "csv_info":
            info = final_state.get("meta", {}).get("csv_info")
            print("\n📄 CSV 정보:")
            print(info)

        if "retrieved_docs" in final_state:
            print(f"\n📚 검색된 문서 수: {len(final_state['retrieved_docs'])}")

        if "report_text" in final_state:
            print("\n📝 보고서 요약:")
            text = final_state["report_text"]
            print(text[:500] + ("..." if len(text) > 500 else ""))

        if final_state.get("docx_path"):
            print(f"\n📄 DOCX 생성됨: {final_state['docx_path']}")


if __name__ == "__main__":
    main()
