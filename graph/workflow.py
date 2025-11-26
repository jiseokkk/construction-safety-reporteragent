"""
LangGraph Workflow
단 하나의 노드(orchestrator_node)만 사용하여 전체 워크플로우 관리

✅ 수정사항: OrchestratorAgent.run()이 async 함수로 변경되었으므로,
           orchestrator_node를 async로 선언하고 await을 추가합니다.
"""
from langgraph.graph import StateGraph, END
from core.agentstate import AgentState
from agents.orchestrator import orchestrator


# 🌟 orchestrator_node를 async 함수로 선언
async def orchestrator_node(state: AgentState) -> AgentState: # 🌟 async 추가
    """
    LangGraph의 유일한 노드
    Orchestrator가 내부에서 SubAgent들을 조율
    """
    print(f"\n{'#'*80}")
    print("🎯 [ORCHESTRATOR NODE] 실행")
    print(f"{'#'*80}")
    
    # 🌟 Orchestrator 실행 시 await 추가
    updated_state = await orchestrator.run(state) # 🌟 await 추가
    
    return updated_state


def should_continue(state: AgentState) -> str:
    """
    다음 노드를 결정하는 조건부 엣지 (로직 유지)
    
    Returns:
        "continue": orchestrator_node를 다시 실행
        "end": 종료
    """
    # ✅ 1) STOP 상태면 그래프 반복만 멈춤 (사용자 입력 대기)
    if state.get("wait_for_user", False):
        print("⛔ STOP 상태: 다음 사용자 입력까지 대기합니다.")
        return "end"

    # ✅ 2) is_complete=True 면 진짜 끝
    if state.get("is_complete", False):
        return "end"

    # 그 외에는 계속
    return "continue"


# ========================================
# LangGraph 정의
# ========================================
def create_graph():
    """
    워크플로우 그래프 생성 (로직 유지)
    """
    workflow = StateGraph(AgentState)
    
    # 단 하나의 노드만 추가
    workflow.add_node("orchestrator", orchestrator_node)
    
    # 시작점
    workflow.set_entry_point("orchestrator")
    
    # 조건부 엣지: STOP 또는 완료되면 END, 아니면 다시 orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "continue": "orchestrator",
            "end": END
        }
    )
    
    # 그래프 컴파일
    app = workflow.compile()
    
    print("✅ LangGraph 워크플로우 생성 완료")
    print("📊 구조: START → orchestrator_node ⟲ → END")
    
    return app


# 전역 그래프 인스턴스
graph_app = create_graph()