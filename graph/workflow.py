"""
LangGraph Workflow
단 하나의 노드(orchestrator_node)만 사용하여 전체 워크플로우 관리
"""
from langgraph.graph import StateGraph, END
from core.agentstate import AgentState

# 🚨 agents 폴더 내의 orchestrator.py에서 인스턴스를 가져옵니다.
from agents.orchestrator import orchestrator


# 🌟 orchestrator_node를 async 함수로 선언 (중요)
async def orchestrator_node(state: AgentState) -> AgentState:
    """
    LangGraph의 유일한 노드
    Orchestrator가 내부에서 SubAgent들을 조율
    """
    print(f"\n{'#'*80}")
    print("🎯 [ORCHESTRATOR NODE] 실행")
    print(f"{'#'*80}")
    
    # 🌟 Orchestrator의 run 메서드가 async이므로 await 필수
    updated_state = await orchestrator.run(state)
    
    return updated_state


def should_continue(state: AgentState) -> str:
    """
    다음 노드를 결정하는 조건부 엣지
    """
    # 1) STOP 상태면 그래프 반복만 멈춤 (사용자 입력 대기)
    #    Chainlit에서는 이 상태에서 사용자 입력을 기다리게 됩니다.
    if state.get("wait_for_user", False):
        print("⛔ STOP 상태: 다음 사용자 입력까지 대기합니다.")
        return "end"

    # 2) is_complete=True 면 진짜 끝 (워크플로우 종료)
    if state.get("is_complete", False):
        return "end"

    # 그 외에는 계속 루프 (Orchestrator가 다음 단계를 결정하도록 함)
    return "continue"


# ========================================
# LangGraph 정의
# ========================================
def create_graph():
    """
    워크플로우 그래프 생성
    """
    workflow = StateGraph(AgentState)
    
    # 단 하나의 노드만 추가
    workflow.add_node("orchestrator", orchestrator_node)
    
    # 시작점 설정
    workflow.set_entry_point("orchestrator")
    
    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "continue": "orchestrator", # 루프
            "end": END                  # 종료
        }
    )
    
    # 그래프 컴파일
    app = workflow.compile()
    
    print("✅ LangGraph 워크플로우 생성 완료")
    print("📊 구조: START → orchestrator_node ⟲ → END")
    
    return app


# 🚨 [매우 중요] 이 변수가 있어야 app_chainlit.py에서 import 할 수 있습니다.
graph_app = create_graph()