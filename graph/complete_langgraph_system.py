# complete_langgraph_system.py
"""
완전한 LangGraph 기반 Multi-Agent HITL 시스템 - 최종 완성본

특징:
1. 기존 Agent 클래스 재사용
2. LLM Router로 동적 라우팅
3. interrupt_before로 자동 HITL
4. 모든 피드백 루프 지원
5. Phase 이름 통일 (accident_select, show_accident, rag_feedback, report_approval)
"""

from typing import Literal, Dict, Any, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
import json
import re

from core.agentstate import AgentState
from agents.intent_agent import IntentAgent
from agents.sql_agent import CSVSQLAgent
from agents.subagents import RAGAgent, WebSearchAgent, ReportWriterAgent
from core.llm_utils import call_llm


# =============================================================================
# Node Functions (각 Agent를 독립 Node로 래핑)
# =============================================================================

class MultiAgentNodes:
    """모든 Agent Node를 관리하는 클래스"""
    
    def __init__(self, csv_path: str, df: pd.DataFrame):
        self.csv_path = csv_path
        self.df = df
        
        # Agent 인스턴스
        self.intent_agent = IntentAgent()
        self.sql_agent = CSVSQLAgent(csv_path)
        self.rag_agent = RAGAgent()
        self.web_agent = WebSearchAgent()
        self.report_agent = ReportWriterAgent()
    
    # -------------------------------------------------------------------------
    # Entry Node: Intent 분석
    # -------------------------------------------------------------------------
    
    def intent_node(self, state: AgentState) -> AgentState:
        """사용자 입력 분석"""
        print("\n" + "🎯"*40)
        print("🎯 [INTENT NODE] 실행")
        print("🎯"*40)
        
        user_input = state.get("user_query", "")
        result = self.intent_agent.parse_and_decide(user_input, self.df)
        
        intent = result.get("intent")
        state["user_intent"] = intent
        state["meta"] = {"intent_result": result}
        
        # 다음 노드 결정
        if intent == "query_sql":
            state["next_node"] = "sql_query"
        elif intent == "csv_info":
            accident_data = result.get("accident_data")
            if accident_data is not None:
                state["accident_row"] = accident_data.to_dict()
                state["next_node"] = "show_accident"
            else:
                state["next_node"] = "end"
        elif intent in ("pure_guideline_search", "search_only", "generate_report"):
            state["next_node"] = "rag"
        elif intent == "ask_user_disambiguation":  # ✅ 추가!
            state["next_node"] = "disambiguation"
        else:
            state["next_node"] = "end"
        
        print(f"\n✅ Intent: {intent}")
        print(f"➡️  다음: {state['next_node']}")
        
        return state
    
    # -------------------------------------------------------------------------
    # SQL Query Node
    # -------------------------------------------------------------------------
    
    def sql_query_node(self, state: AgentState) -> AgentState:
        """SQL 사고 조회"""
        print("\n" + "🗄️"*40)
        print("🗄️ [SQL QUERY NODE] 실행")
        print("🗄️"*40)
        
        user_input = state.get("user_query", "")
        result = self.sql_agent.query(user_input)
        
        state["sql_result"] = result
        
        if result.get("success"):
            rows = result.get("rows", [])
            print(f"\n✅ 조회 완료: {len(rows)}건")
            
            # SQL 결과 메시지 생성
            generated_sql = result.get("generated_sql", "")
            state["system_message"] = f"""## ✅ SQL 쿼리 결과

**📝 생성된 SQL:**
```sql
{generated_sql}
```

**📊 검색된 사고 수:** **{len(rows)}건**"""
            
            if len(rows) > 1:
                state["next_node"] = "accident_select"
                state["wait_for_user"] = True
                state["phase"] = "accident_select"  # ✅ 수정
            elif len(rows) == 1:
                state["accident_row"] = rows[0]
                state["next_node"] = "show_accident"
                state["wait_for_user"] = True
                state["phase"] = "show_accident"
            else:
                state["next_node"] = "end"
                state["system_message"] = "검색 결과가 없습니다."
        else:
            state["next_node"] = "end"
            state["system_message"] = f"SQL 오류: {result.get('error')}"
        
        print(f"➡️  다음: {state['next_node']}")
        
        return state
    
    # -------------------------------------------------------------------------
    # Accident Select Node
    # -------------------------------------------------------------------------
    
    def accident_select_node(self, state: AgentState) -> AgentState:
        """사고 선택 대기"""
        print("\n" + "📋"*40)
        print("📋 [ACCIDENT SELECT NODE] 실행")
        print("📋"*40)
        
        sql_result = state.get("sql_result", {})
        rows = sql_result.get("rows", [])
        
        if not rows:
            state["next_node"] = "end"
            return state
        
        # 사고 목록 포맷팅
        state["system_message"] = f"총 {len(rows)}건의 사고가 검색되었습니다."
        state["wait_for_user"] = True
        state["phase"] = "accident_select"  # ✅ 수정 (accident_selection → accident_select)
        state["next_node"] = "router"  # ✅ 추가!
        
        print("⏸️  사용자 입력 대기...")
        
        return state
    
    # -------------------------------------------------------------------------
    # Show Accident Node
    # -------------------------------------------------------------------------
    
    def show_accident_node(self, state: AgentState) -> AgentState:
        """사고 상세 정보 표시"""
        print("\n" + "📄"*40)
        print("📄 [SHOW ACCIDENT NODE] 실행")
        print("📄"*40)
        
        accident_row = state.get("accident_row", {})
        
        if not accident_row:
            state["next_node"] = "end"
            return state
        
        # 사고 정보 포맷팅
        lines = ["=== 사고 상세 정보 ==="]
        for key, value in accident_row.items():
            lines.append(f"{key}: {value}")
        
        lines.append("\n다음 작업을 선택하세요:")
        lines.append("1. 지침 검색")
        lines.append("2. 보고서 생성")
        lines.append("3. 종료")
        
        state["system_message"] = "\n".join(lines)
        state["wait_for_user"] = True
        state["phase"] = "show_accident"  # ✅ 확인
        state["next_node"] = "router"  # ✅ 추가!
        
        print("⏸️  사용자 입력 대기...")
        
        return state
    
    # -------------------------------------------------------------------------
    # Disambiguation Node
    # -------------------------------------------------------------------------
    
    def disambiguation_node(self, state: AgentState) -> AgentState:
        """모호한 질문 명확화"""
        print("\n" + "❓"*40)
        print("❓ [DISAMBIGUATION NODE] 실행")
        print("❓"*40)
        
        state["system_message"] = """
질문이 명확하지 않습니다. 다음 중 선택하세요:

1. 사고 조회 (CSV 데이터베이스)
2. 지침 검색 (안전 규정 문서)

선택 (1 또는 2):
"""
        state["wait_for_user"] = True
        state["phase"] = "disambiguation"  # ✅ 확인
        
        print("⏸️  사용자 입력 대기...")
        
        return state
    
    # -------------------------------------------------------------------------
    # RAG Node
    # -------------------------------------------------------------------------
    
    def rag_node(self, state: AgentState) -> AgentState:
        """RAG 검색"""
        print("\n" + "🔍"*40)
        print("🔍 [RAG NODE] 실행")
        print("🔍"*40)
        
        state = self.rag_agent.run(state)
        
        # 검색 결과 메시지 생성
        docs = state.get("retrieved_docs", [])
        if docs:
            state["system_message"] = f"✅ **{len(docs)}개의 관련 문서**를 찾았습니다."
        else:
            state["system_message"] = "⚠️ 관련 문서를 찾지 못했습니다."
        
        state["next_node"] = "rag_feedback"
        state["wait_for_user"] = True
        state["phase"] = "rag_feedback"  # ✅ 확인
        
        print("⏸️  피드백 대기...")
        
        return state
    
    # -------------------------------------------------------------------------
    # RAG Feedback Node
    # -------------------------------------------------------------------------
    
    def rag_feedback_node(self, state: AgentState) -> AgentState:
        """RAG 피드백 처리"""
        print("\n" + "💬"*40)
        print("💬 [RAG FEEDBACK NODE] 실행")
        print("💬"*40)
        
        state["system_message"] = """
RAG 검색 결과를 확인하세요.

다음 작업을 선택할 수 있습니다:
1. 검색 재시도 (retry)
2. 웹 검색 추가 (web)
3. 보고서 생성 (report)
4. 종료 (end)

선택:
"""
        state["wait_for_user"] = True
        state["phase"] = "rag_feedback"  # ✅ 확인
        
        print("⏸️  사용자 입력 대기...")
        
        return state
    
    # -------------------------------------------------------------------------
    # Web Search Node
    # -------------------------------------------------------------------------
    
    def web_node(self, state: AgentState) -> AgentState:
        """웹 검색"""
        print("\n" + "🌐"*40)
        print("🌐 [WEB SEARCH NODE] 실행")
        print("🌐"*40)
        
        state = self.web_agent.run(state)
        state["next_node"] = "rag_feedback"
        
        print("✅ 웹 검색 완료")
        
        return state
    
    # -------------------------------------------------------------------------
    # Report Writer Node
    # -------------------------------------------------------------------------
    
    def report_node(self, state: AgentState) -> AgentState:
        """보고서 생성"""
        print("\n" + "📝"*40)
        print("📝 [REPORT WRITER NODE] 실행")
        print("📝"*40)
        
        state = self.report_agent.run(state)
        
        if state.get("report_text"):
            state["next_node"] = "report_approval"
            state["wait_for_user"] = True
            state["phase"] = "report_approval"  # ✅ 확인
            print("⏸️  승인 대기...")
        else:
            state["next_node"] = "end"
            state["system_message"] = "보고서 생성 실패"
        
        return state
    
    # -------------------------------------------------------------------------
    # DOCX Node
    # -------------------------------------------------------------------------
    
    def docx_node(self, state: AgentState) -> AgentState:
        """DOCX 파일 생성"""
        print("\n" + "📄"*40)
        print("📄 [DOCX NODE] 실행")
        print("📄"*40)
        
        state = self.report_agent._create_docx_file(state)
        
        state["next_node"] = "end"
        state["is_complete"] = True
        
        print("✅ DOCX 생성 완료")
        
        return state
    
    # -------------------------------------------------------------------------
    # Router Node (LLM 기반)
    # -------------------------------------------------------------------------
    
    def router_node(self, state: AgentState) -> AgentState:
        """LLM 기반 라우터"""
        print("\n" + "🤖"*40)
        print("🤖 [ROUTER NODE] 실행")
        print("🤖"*40)
        
        user_query = state.get("user_query", "")
        user_intent = state.get("user_intent", "search_only")
        
        # 간단한 라우팅 로직
        if "rag" in user_query.lower() or user_intent == "search_only":
            state["next_node"] = "rag"
        elif "report" in user_query.lower() or user_intent == "generate_report":
            state["next_node"] = "report_writer"
        elif "web" in user_query.lower():
            state["next_node"] = "web"
        else:
            state["next_node"] = "rag"
        
        print(f"✅ Router 결정: {state['next_node']}")
        
        return state


# =============================================================================
# Router Functions
# =============================================================================

def route_from_intent(state: AgentState) -> str:
    """Intent에서 다음 노드 결정"""
    next_node = state.get("next_node", "end")
    return next_node


def route_from_sql(state: AgentState) -> str:
    """SQL 결과에서 다음 노드 결정"""
    return state.get("next_node", "end")


def route_after_accident_select(state: AgentState) -> str:
    """사고 선택 후 라우팅"""
    return state.get("next_node", "router")


def route_after_show_accident(state: AgentState) -> str:
    """사고 표시 후 라우팅"""
    return state.get("next_node", "end")


def route_after_disambiguation(state: AgentState) -> str:
    """명확화 후 라우팅"""
    return state.get("next_node", "end")


def route_after_rag_feedback(state: AgentState) -> str:
    """RAG 피드백 후 라우팅"""
    user_intent = state.get("user_intent", "search_only")
    
    if user_intent == "generate_report":
        return "report_writer"
    
    return state.get("next_node", "end")


def route_after_report(state: AgentState) -> str:
    """보고서 생성 후 라우팅"""
    return state.get("next_node", "end")


def route_from_router(state: AgentState) -> str:
    """Router에서 다음 노드 결정"""
    return state.get("next_node", "rag")


# =============================================================================
# Graph Builder
# =============================================================================

def build_complete_graph(csv_path: str, df: pd.DataFrame):
    """완전한 LangGraph 빌드"""
    
    nodes = MultiAgentNodes(csv_path, df)
    
    workflow = StateGraph(AgentState)
    
    # Nodes 추가
    workflow.add_node("intent", nodes.intent_node)
    workflow.add_node("sql_query", nodes.sql_query_node)
    workflow.add_node("accident_select", nodes.accident_select_node)
    workflow.add_node("show_accident", nodes.show_accident_node)
    workflow.add_node("disambiguation", nodes.disambiguation_node)
    workflow.add_node("rag", nodes.rag_node)
    workflow.add_node("rag_feedback", nodes.rag_feedback_node)
    workflow.add_node("web", nodes.web_node)
    workflow.add_node("report_writer", nodes.report_node)
    workflow.add_node("docx", nodes.docx_node)
    workflow.add_node("router", nodes.router_node)
    
    # Entry point
    workflow.add_edge(START, "intent")
    
    # Intent → 분기
    workflow.add_conditional_edges(
        "intent",
        route_from_intent,
        {
            "sql_query": "sql_query",
            "show_accident": "show_accident",
            "rag": "rag",
            "disambiguation": "disambiguation",  # ✅ 추가!
            "end": END,
        }
    )
    
    # SQL → 분기
    workflow.add_conditional_edges(
        "sql_query",
        route_from_sql,
        {
            "accident_select": "accident_select",
            "show_accident": "show_accident",
            "end": END,
        }
    )
    
    workflow.add_conditional_edges(
        "accident_select",
        route_after_accident_select,
        {
            "show_accident": "show_accident",
            "router": "router",
        }
    )
    
    workflow.add_conditional_edges(
        "show_accident",
        route_after_show_accident,
        {
            "rag": "rag",
            "router": "router",
            "end": END,
        }
    )
    
    workflow.add_conditional_edges(
        "disambiguation",
        route_after_disambiguation,
        {
            "sql_query": "sql_query",
            "rag": "rag",
            "show_accident": "show_accident",
            "router": "router",
            "end": END,
        }
    )
    
    workflow.add_edge("rag", "rag_feedback")
    
    workflow.add_conditional_edges(
        "rag_feedback",
        route_after_rag_feedback,
        {
            "rag": "rag",
            "web": "web",
            "report_writer": "report_writer",
            "end": END,
        }
    )
    
    workflow.add_edge("web", "rag_feedback")
    
    workflow.add_conditional_edges(
        "report_writer",
        route_after_report,
        {
            "report_approval": END,
            "docx": "docx",
            "end": END,
        }
    )
    
    workflow.add_edge("docx", END)
    
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "rag": "rag",
            "web": "web",
            "report_writer": "report_writer",
            "end": END,
        }
    )
    
    # Checkpointer
    memory = MemorySaver()
    
    # 컴파일 - interrupt_before에서 accident_select 제거
    # accident_select는 SQL Node에서 wait_for_user로 처리
    compiled = workflow.compile(
        checkpointer=memory,
        interrupt_before=[
            # "accident_select",  ← 제거! SQL Node가 이미 처리
            "show_accident",
            "rag_feedback",
            "report_writer",
            "disambiguation",
        ]
    )
    
    print("\n" + "="*80)
    print("✅ LangGraph 컴파일 완료")
    print("="*80)
    print(f"📊 총 {len(workflow.nodes)}개 Node")
    print(f"⏸️  Interrupt 지점: 4개")
    print("="*80 + "\n")
    
    return compiled