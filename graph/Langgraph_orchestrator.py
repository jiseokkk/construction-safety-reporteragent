# langgraph_orchestrator.py
"""
LangGraph 기반 Multi-Agent HITL 시스템

핵심 구조:
1. 각 Agent는 독립적인 Node
2. LLM Router가 다음 행동 판단
3. interrupt_before를 통한 자동 HITL
4. 조건부 엣지로 동적 라우팅
"""

from typing import Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from core.agentstate import AgentState
from agents.intent_agent import IntentAgent
from agents.sql_agent import CSVSQLAgent
from agents.subagents import RAGAgent, WebSearchAgent, ReportWriterAgent
from core.llm_utils import call_llm

import pandas as pd
import json
import re


# =============================================================================
# Node Functions (각 Agent를 Node로 래핑)
# =============================================================================

def intent_node(state: AgentState) -> AgentState:
    """
    사용자 입력을 분석하여 Intent 판단
    """
    print("\n" + "="*80)
    print("? [INTENT NODE] 실행")
    print("="*80)
    
    agent = IntentAgent()
    user_input = state.get("user_query", "")
    
    # DataFrame은 외부에서 주입 (실제 구현에서는 state에서 가져오거나 전역 변수 사용)
    df = state.get("_df")  # 임시: 실제로는 다른 방법으로 주입
    
    result = agent.parse_and_decide(user_input, df)
    
    intent = result.get("intent")
    state["user_intent"] = intent
    state["meta"] = {"intent_result": result}
    
    # Intent별 다음 단계 설정
    if intent == "query_sql":
        state["next_node"] = "sql_query"
    elif intent == "csv_info":
        state["next_node"] = "show_accident"
        state["accident_row"] = result.get("accident_data").to_dict() if result.get("accident_data") is not None else None
    elif intent == "pure_guideline_search":
        state["next_node"] = "rag_search"
    elif intent in ("search_only", "generate_report"):
        state["next_node"] = "rag_search"
    elif intent == "ask_user_disambiguation":
        state["next_node"] = "disambiguation"
    elif intent == "select_accident":
        state["next_node"] = "accident_select"
    else:
        state["next_node"] = "end"
    
    print(f"? Intent 판단 완료: {intent}")
    print(f"   다음 Node: {state['next_node']}")
    
    return state


def disambiguation_node(state: AgentState) -> AgentState:
    """
    모호한 질문에 대한 사용자 재확인
    """
    print("\n" + "="*80)
    print("? [DISAMBIGUATION NODE] 실행")
    print("="*80)
    
    # 사용자에게 선택 요청 메시지 생성
    state["system_message"] = """
질문이 명확하지 않습니다. 다음 중 하나를 선택해주세요:

1. 사고 조회 (CSV 데이터베이스에서 사고 검색)
2. 지침 검색 (안전 규정 및 가이드라인 검색)

선택해주세요 (1 또는 2):
"""
    
    # 다음 노드는 사용자 입력 후 router에서 결정
    state["next_node"] = "router"
    state["wait_for_user"] = True
    
    return state


def sql_query_node(state: AgentState) -> AgentState:
    """
    SQL Agent를 통한 사고 데이터 조회
    """
    print("\n" + "="*80)
    print("?? [SQL QUERY NODE] 실행")
    print("="*80)
    
    # CSV 경로는 외부에서 주입
    csv_path = state.get("_csv_path")
    agent = CSVSQLAgent(csv_path)
    
    user_input = state.get("user_query", "")
    result = agent.query(user_input)
    
    state["sql_result"] = result
    
    if result.get("success"):
        rows = result.get("rows", [])
        print(f"? SQL 조회 완료: {len(rows)}건")
        
        if len(rows) > 1:
            state["next_node"] = "accident_select"
            state["system_message"] = f"{len(rows)}건의 사고가 검색되었습니다. 번호를 선택해주세요."
        elif len(rows) == 1:
            state["next_node"] = "show_accident"
            state["accident_row"] = rows[0]
        else:
            state["next_node"] = "end"
            state["system_message"] = "검색 결과가 없습니다."
    else:
        state["next_node"] = "end"
        state["system_message"] = f"SQL 조회 실패: {result.get('error')}"
    
    return state


def accident_select_node(state: AgentState) -> AgentState:
    """
    여러 사고 중 하나를 선택
    """
    print("\n" + "="*80)
    print("? [ACCIDENT SELECT NODE] 실행")
    print("="*80)
    
    # 사용자에게 사고 목록 표시
    sql_result = state.get("sql_result", {})
    rows = sql_result.get("rows", [])
    
    message_lines = ["검색된 사고 목록:"]
    for idx, row in enumerate(rows):
        date = row.get("발생일시", "")
        acc_type = row.get("인적사고", "")
        cause = row.get("사고원인", "")[:50] if row.get("사고원인") else ""
        message_lines.append(f"{idx}. {date} | {acc_type} | {cause}...")
    
    message_lines.append("\n번호를 선택해주세요:")
    
    state["system_message"] = "\n".join(message_lines)
    state["next_node"] = "router"
    state["wait_for_user"] = True
    
    return state


def show_accident_node(state: AgentState) -> AgentState:
    """
    선택된 사고의 상세 정보 표시
    """
    print("\n" + "="*80)
    print("? [SHOW ACCIDENT NODE] 실행")
    print("="*80)
    
    accident_row = state.get("accident_row", {})
    
    if not accident_row:
        state["system_message"] = "사고 정보를 찾을 수 없습니다."
        state["next_node"] = "end"
        return state
    
    # 사고 정보 포맷팅
    info_lines = ["=== 사고 상세 정보 ==="]
    for key, value in accident_row.items():
        info_lines.append(f"{key}: {value}")
    
    info_lines.append("\n다음 작업을 선택하세요:")
    info_lines.append("1. 지침 검색")
    info_lines.append("2. 보고서 생성")
    info_lines.append("3. 목록으로 돌아가기")
    info_lines.append("4. 종료")
    
    state["system_message"] = "\n".join(info_lines)
    state["next_node"] = "router"
    state["wait_for_user"] = True
    
    return state


def rag_node(state: AgentState) -> AgentState:
    """
    RAG Agent를 통한 문서 검색
    """
    print("\n" + "="*80)
    print("? [RAG NODE] 실행")
    print("="*80)
    
    agent = RAGAgent()
    state = agent.run(state)
    
    # RAG 완료 후 피드백 대기
    state["next_node"] = "rag_feedback"
    state["wait_for_user"] = True
    
    print("? RAG 검색 완료")
    
    return state


def rag_feedback_node(state: AgentState) -> AgentState:
    """
    RAG 결과에 대한 사용자 피드백 처리
    """
    print("\n" + "="*80)
    print("? [RAG FEEDBACK NODE] 실행")
    print("="*80)
    
    # 피드백 옵션 표시
    state["system_message"] = """
RAG 검색 결과를 확인하세요.

다음 작업을 선택할 수 있습니다:
1. 검색 재시도 (retry_rag)
2. 특정 문서 제외 (exclude_docs)
3. 웹 검색 추가 (add_web_search)
4. 보고서 생성 (generate_report)
5. 목록으로 돌아가기 (back_to_list)
6. 종료 (terminate)

선택해주세요:
"""
    
    state["next_node"] = "router"
    state["wait_for_user"] = True
    
    return state


def web_node(state: AgentState) -> AgentState:
    """
    웹 검색 Agent 실행
    """
    print("\n" + "="*80)
    print("? [WEB NODE] 실행")
    print("="*80)
    
    agent = WebSearchAgent()
    state = agent.run(state)
    
    # 웹 검색 후 다시 피드백으로
    state["next_node"] = "rag_feedback"
    
    print("? 웹 검색 완료")
    
    return state


def report_node(state: AgentState) -> AgentState:
    """
    보고서 생성 Agent 실행
    """
    print("\n" + "="*80)
    print("? [REPORT NODE] 실행")
    print("="*80)
    
    agent = ReportWriterAgent()
    state = agent.run(state)
    
    # 보고서 생성 후 DOCX 생성 옵션
    state["system_message"] = """
보고서 생성이 완료되었습니다.

다음 작업을 선택하세요:
1. DOCX 파일 생성
2. 종료

선택해주세요:
"""
    
    state["next_node"] = "router"
    state["wait_for_user"] = True
    
    print("? 보고서 생성 완료")
    
    return state


def docx_node(state: AgentState) -> AgentState:
    """
    DOCX 파일 생성
    """
    print("\n" + "="*80)
    print("? [DOCX NODE] 실행")
    print("="*80)
    
    agent = ReportWriterAgent()
    state = agent._create_docx_file(state)
    
    state["next_node"] = "end"
    state["is_complete"] = True
    
    print("? DOCX 파일 생성 완료")
    
    return state


# =============================================================================
# LLM Router Node (핵심: 다음 행동을 LLM이 판단)
# =============================================================================

def router_node(state: AgentState) -> AgentState:
    """
    LLM을 통해 다음 행동 결정
    
    이 노드는:
    1. 현재 상태와 사용자 입력을 분석
    2. LLM에게 다음 행동을 묻기
    3. 결정된 행동을 next_node에 설정
    """
    print("\n" + "="*80)
    print("? [ROUTER NODE] 실행 - LLM이 다음 행동 판단")
    print("="*80)
    
    # 현재 상태 요약
    user_query = state.get("user_query", "")
    user_intent = state.get("user_intent", "")
    phase = state.get("phase", "")
    has_accident = bool(state.get("accident_row"))
    has_rag_results = bool(state.get("retrieved_docs"))
    has_report = bool(state.get("report_text"))
    has_docx = bool(state.get("docx_path"))
    
    system_prompt = """
당신은 Multi-Agent 건설안전 시스템의 Router입니다.

현재 상태를 분석하여 다음에 실행할 Node를 결정하세요.

<가능한 Node 목록>
- intent_node: 사용자 입력 분석 (최초 진입점)
- disambiguation_node: 모호한 질문 명확화
- sql_query_node: SQL 사고 조회
- accident_select_node: 사고 선택
- show_accident_node: 사고 상세 정보
- rag_node: RAG 검색
- rag_feedback_node: RAG 피드백
- web_node: 웹 검색
- report_node: 보고서 생성
- docx_node: DOCX 파일 생성
- end: 종료

<판단 규칙>
1. 사용자가 "검색"이나 "지침"을 요청하면 → rag_node
2. 사용자가 사고 번호를 선택하면 → show_accident_node
3. 사용자가 "보고서"를 요청하면 → report_node
4. 사용자가 "웹 검색"을 요청하면 → web_node
5. 사용자가 "종료"를 요청하면 → end
6. 피드백 명령어(retry, exclude 등)는 해당 action 실행

<출력 형식>
<thinking>
현재 상태 분석...
사용자 의도 파악...
다음 행동 결정...
</thinking>
<output>
{
  "next_node": "node_name",
  "reason": "선택 이유"
}
</output>
"""
    
    user_message = f"""
<현재 상태>
- 사용자 입력: {user_query}
- Intent: {user_intent}
- Phase: {phase}
- 사고 정보: {"있음" if has_accident else "없음"}
- RAG 결과: {"있음" if has_rag_results else "없음"}
- 보고서: {"있음" if has_report else "없음"}
- DOCX: {"있음" if has_docx else "없음"}

다음 Node를 결정하세요.
"""
    
    try:
        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=500,
        )
        
        print(f"\n? LLM Router 응답:")
        print(response)
        
        # JSON 파싱
        m = re.search(r"<output>(.*?)</output>", response, re.S)
        if m:
            result = json.loads(m.group(1).strip())
            next_node = result.get("next_node", "end")
            reason = result.get("reason", "")
            
            print(f"\n? Router 판단:")
            print(f"   다음 Node: {next_node}")
            print(f"   이유: {reason}")
            
            state["next_node"] = next_node
            state["router_reason"] = reason
        else:
            print("?? Router 응답 파싱 실패 → end로 이동")
            state["next_node"] = "end"
    
    except Exception as e:
        print(f"? Router 오류: {e}")
        state["next_node"] = "end"
    
    return state


# =============================================================================
# Conditional Edge Functions (동적 라우팅)
# =============================================================================

def route_after_intent(state: AgentState) -> str:
    """Intent Node 이후 라우팅"""
    next_node = state.get("next_node", "end")
    print(f"? Intent 라우팅: {next_node}")
    return next_node


def route_after_sql(state: AgentState) -> str:
    """SQL Node 이후 라우팅"""
    next_node = state.get("next_node", "end")
    print(f"? SQL 라우팅: {next_node}")
    return next_node


def route_after_accident_select(state: AgentState) -> str:
    """Accident Select Node 이후 라우팅"""
    next_node = state.get("next_node", "end")
    print(f"? Accident Select 라우팅: {next_node}")
    return next_node


def route_after_show_accident(state: AgentState) -> str:
    """Show Accident Node 이후 라우팅"""
    next_node = state.get("next_node", "router")
    print(f"? Show Accident 라우팅: {next_node}")
    return next_node


def route_after_rag(state: AgentState) -> str:
    """RAG Node 이후 라우팅"""
    next_node = state.get("next_node", "rag_feedback")
    print(f"? RAG 라우팅: {next_node}")
    return next_node


def route_after_rag_feedback(state: AgentState) -> str:
    """RAG Feedback Node 이후 라우팅"""
    next_node = state.get("next_node", "router")
    print(f"? RAG Feedback 라우팅: {next_node}")
    return next_node


def route_after_router(state: AgentState) -> str:
    """Router Node 이후 라우팅"""
    next_node = state.get("next_node", "end")
    print(f"? Router 라우팅: {next_node}")
    return next_node


def route_after_web(state: AgentState) -> str:
    """Web Node 이후 라우팅"""
    next_node = state.get("next_node", "rag_feedback")
    print(f"? Web 라우팅: {next_node}")
    return next_node


def route_after_report(state: AgentState) -> str:
    """Report Node 이후 라우팅"""
    next_node = state.get("next_node", "router")
    print(f"? Report 라우팅: {next_node}")
    return next_node


def route_after_disambiguation(state: AgentState) -> str:
    """Disambiguation Node 이후 라우팅"""
    next_node = state.get("next_node", "router")
    print(f"? Disambiguation 라우팅: {next_node}")
    return next_node


# =============================================================================
# Graph Builder
# =============================================================================

def build_graph(csv_path: str, df: pd.DataFrame):
    """
    LangGraph 기반 Multi-Agent HITL 시스템 구축
    
    Args:
        csv_path: CSV 파일 경로
        df: pandas DataFrame
    
    Returns:
        compiled_graph: 실행 가능한 LangGraph
    """
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # =========================================================================
    # Nodes 추가
    # =========================================================================
    workflow.add_node("intent_node", intent_node)
    workflow.add_node("disambiguation_node", disambiguation_node)
    workflow.add_node("sql_query_node", sql_query_node)
    workflow.add_node("accident_select_node", accident_select_node)
    workflow.add_node("show_accident_node", show_accident_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("rag_feedback_node", rag_feedback_node)
    workflow.add_node("web_node", web_node)
    workflow.add_node("report_node", report_node)
    workflow.add_node("docx_node", docx_node)
    workflow.add_node("router_node", router_node)
    
    # =========================================================================
    # Edges 추가 (조건부 라우팅)
    # =========================================================================
    
    # 시작점 → Intent
    workflow.add_edge(START, "intent_node")
    
    # Intent → 조건부 라우팅
    workflow.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "sql_query": "sql_query_node",
            "show_accident": "show_accident_node",
            "rag_search": "rag_node",
            "disambiguation": "disambiguation_node",
            "accident_select": "accident_select_node",
            "router": "router_node",
            "end": END,
        }
    )
    
    # Disambiguation → Router
    workflow.add_conditional_edges(
        "disambiguation_node",
        route_after_disambiguation,
        {
            "router": "router_node",
            "end": END,
        }
    )
    
    # SQL → 조건부 라우팅
    workflow.add_conditional_edges(
        "sql_query_node",
        route_after_sql,
        {
            "accident_select": "accident_select_node",
            "show_accident": "show_accident_node",
            "end": END,
        }
    )
    
    # Accident Select → Router
    workflow.add_conditional_edges(
        "accident_select_node",
        route_after_accident_select,
        {
            "router": "router_node",
            "show_accident": "show_accident_node",
            "end": END,
        }
    )
    
    # Show Accident → Router
    workflow.add_conditional_edges(
        "show_accident_node",
        route_after_show_accident,
        {
            "router": "router_node",
            "end": END,
        }
    )
    
    # RAG → RAG Feedback
    workflow.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {
            "rag_feedback": "rag_feedback_node",
            "end": END,
        }
    )
    
    # RAG Feedback → Router
    workflow.add_conditional_edges(
        "rag_feedback_node",
        route_after_rag_feedback,
        {
            "router": "router_node",
            "end": END,
        }
    )
    
    # Web → RAG Feedback
    workflow.add_conditional_edges(
        "web_node",
        route_after_web,
        {
            "rag_feedback": "rag_feedback_node",
            "end": END,
        }
    )
    
    # Report → Router
    workflow.add_conditional_edges(
        "report_node",
        route_after_report,
        {
            "router": "router_node",
            "docx_node": "docx_node",
            "end": END,
        }
    )
    
    # DOCX → END
    workflow.add_edge("docx_node", END)
    
    # Router → 조건부 라우팅 (모든 노드로 갈 수 있음)
    workflow.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "intent_node": "intent_node",
            "disambiguation_node": "disambiguation_node",
            "sql_query_node": "sql_query_node",
            "accident_select_node": "accident_select_node",
            "show_accident_node": "show_accident_node",
            "rag_node": "rag_node",
            "rag_feedback_node": "rag_feedback_node",
            "web_node": "web_node",
            "report_node": "report_node",
            "docx_node": "docx_node",
            "router_node": "router_node",  # 재귀 가능
            "end": END,
        }
    )
    
    # =========================================================================
    # Checkpointer 설정 (HITL을 위해 필수)
    # =========================================================================
    memory = MemorySaver()
    
    # =========================================================================
    # 컴파일 및 반환
    # =========================================================================
    compiled_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=[
            "disambiguation_node",
            "accident_select_node", 
            "show_accident_node",
            "rag_feedback_node",
            "report_node",
        ]
    )
    
    print("\n" + "="*80)
    print("? LangGraph 컴파일 완료!")
    print("="*80)
    print(f"? 총 {len(workflow.nodes)} 개의 Node")
    print(f"? interrupt_before 설정: 5개 지점")
    print("="*80 + "\n")
    
    return compiled_graph


# =============================================================================
# 실행 함수
# =============================================================================

def run_hitl_system(csv_path: str, df: pd.DataFrame, initial_query: str):
    """
    HITL 시스템 실행
    
    Args:
        csv_path: CSV 파일 경로
        df: pandas DataFrame
        initial_query: 초기 사용자 질문
    """
    
    # 그래프 빌드
    graph = build_graph(csv_path, df)
    
    # 초기 상태
    initial_state = {
        "user_query": initial_query,
        "_csv_path": csv_path,
        "_df": df,
    }
    
    # Thread 설정 (HITL에 필수)
    config = {"configurable": {"thread_id": "user_session_1"}}
    
    print(f"\n? 초기 질문: {initial_query}")
    print("="*80 + "\n")
    
    # 실행
    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            print(f"\n[현재 상태]")
            print(f"  - next_node: {event.get('next_node')}")
            print(f"  - wait_for_user: {event.get('wait_for_user')}")
            
            # 시스템 메시지 출력
            if event.get("system_message"):
                print(f"\n? 시스템: {event.get('system_message')}")
            
            # HITL 대기 중이면 사용자 입력 받기
            if event.get("wait_for_user"):
                user_input = input("\n? 입력: ")
                
                # 상태 업데이트하여 재실행
                event["user_query"] = user_input
                event["wait_for_user"] = False
                
                graph.update_state(config, event)
                
                # 재실행
                for new_event in graph.stream(None, config, stream_mode="values"):
                    if new_event.get("is_complete"):
                        print("\n? 완료!")
                        return new_event
    
    except Exception as e:
        print(f"? 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n? 시스템 종료")


# =============================================================================
# 메인 (테스트용)
# =============================================================================

if __name__ == "__main__":
    # 테스트 데이터
    csv_path = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv"
    df = pd.read_csv(csv_path)
    
    # 실행
    run_hitl_system(
        csv_path=csv_path,
        df=df,
        initial_query="2024년 7월 사고 조회해줘"
    )