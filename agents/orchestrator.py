"""
🔥 FINAL OrchestratorAgent — LLM Driven Routing (Fully Orchestrated)
✅ 변경점: 
1. ChainlitContextException 해결을 위한 Lazy Loading 유지
2. IntentAgentWrapper 수정: 다중 결과('candidates') 반환 시 'sql_query_result'로 매핑하여 ASK_USER 트리거
"""

from typing import Optional, Literal, List, Dict, Any
from core.agentstate import AgentState
import json
import os
import chainlit as cl
import pandas as pd

# ✅ LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ✅ Factory Import
from core.llm_factory import get_llm

# ✅ 기존 Agent들 임포트
from agents.subagents import get_agent as get_subagent
from agents.intent_agent import IntentAgent
from agents.sql_agent import CSVSQLAgent

# ======================================================================
# 0. Wrappers (Lazy Loading 적용)
# ======================================================================

class IntentAgentWrapper:
    """IntentAgent를 Orchestrator가 쓸 수 있게 감싸는 래퍼"""
    def __init__(self):
        # IntentAgent는 세션 상태와 무관하므로 미리 생성해도 됨
        self.agent = IntentAgent()

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_query")
        # 실행 시점에 세션에서 df 가져오기 (Safe)
        df = cl.user_session.get("df")
        
        # IntentAgent 실행
        result = self.agent.parse_and_decide(user_input, df)
        
        # 결과 State에 반영
        state["user_intent"] = result.get("intent")
        state["accident_date"] = result.get("date")
        
        # [Case 1] 사고가 1개만 특정되어 바로 나온 경우
        if result.get("accident_data") is not None:
             acc_data = result["accident_data"]
             if isinstance(acc_data, pd.Series):
                 acc_data = acc_data.to_dict()
             state["selected_accident"] = acc_data
        
        # [Case 2] ✅ 다중 사고 후보(candidates)가 반환된 경우
        # 이를 SQL 결과인 것처럼 매핑하여 Orchestrator가 ASK_USER를 띄우도록 유도
        candidates = result.get("candidates")
        if candidates:
            state["sql_query_result"] = candidates
            state["sql_executed"] = True  # 실행된 것으로 간주

        return state

class CSVSQLAgentWrapper:
    """CSVSQLAgent를 Orchestrator가 쓸 수 있게 감싸는 래퍼"""
    def __init__(self):
        # ⚠️ 중요: 여기서 cl.user_session을 호출하면 안 됩니다!
        pass

    async def run(self, state: AgentState) -> AgentState:
        # ✅ 실행 시점(run)에는 세션이 존재하므로 여기서 가져옵니다.
        agent = cl.user_session.get("sql_agent")
        
        if not agent:
            # 혹시 세션에 없으면 Fallback (비상용)
            CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv"
            agent = CSVSQLAgent(CSV_PATH)

        user_query = state.get("user_query")
        
        # SQL 실행
        result = await cl.make_async(agent.query)(user_query)
        
        if result["success"]:
            rows = result.get("rows", [])
            state["sql_query_result"] = rows
            state["sql_executed"] = True
        else:
            state["sql_query_result"] = []
            state["sql_executed"] = True
            print(f"❌ SQL Error: {result.get('error')}")
            
        return state

# ======================================================================
# 1. Pydantic 모델 정의
# ======================================================================
class AgentDecision(BaseModel):
    """Orchestrator의 판단 결과"""
    
    next_agent: Literal[
        "IntentAgent", 
        "CSVSQLAgent", 
        "ASK_USER",        
        "RAGAgent", 
        "WebSearchAgent", 
        "ReportWriterAgent", 
        "FINISH"
    ] = Field(
        description="현재 상태를 기반으로 다음에 실행할 최적의 에이전트 또는 행동."
    )
    
    reason: str = Field(
        description="왜 이 에이전트를 선택했는지에 대한 논리적 근거 (Chain-of-Thought)."
    )

# ======================================================================
# 2. OrchestratorAgent 클래스
# ======================================================================
class OrchestratorAgent:
    def __init__(self):
        self.llm = get_llm(mode="smart") 
        self.parser = PydanticOutputParser(pydantic_object=AgentDecision)
        
        # Wrapper 인스턴스 생성
        self.agents = {
            "IntentAgent": IntentAgentWrapper(),
            "CSVSQLAgent": CSVSQLAgentWrapper(),
        }

    def _get_agent_instance(self, name: str):
        """이름으로 Agent 인스턴스 반환"""
        if name in self.agents:
            return self.agents[name]
        return get_subagent(name)

    def _summarize_state(self, state: AgentState) -> str:
        """LLM에게 보여줄 상태 요약"""
        sql_rows = state.get("sql_query_result")
        sql_count = len(sql_rows) if sql_rows is not None else None
        
        summary = {
            "user_query": state.get("user_query"),
            "current_intent": state.get("user_intent"),       
            "sql_executed": state.get("sql_executed", False), 
            "sql_result_count": sql_count,                    
            "selected_accident": bool(state.get("selected_accident")), 
            
            "retrieved_docs_count": len(state.get("retrieved_docs") or []), 
            "report_exist": bool(state.get("report_text")),   
            "docx_exist": bool(state.get("docx_path")),       
            
            "hitl_action": state.get("hitl_action"),          
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    async def decide_next_agent(self, state: AgentState) -> str:
        """오직 Prompt를 통해서만 다음 단계를 결정"""
        
        if state.get("wait_for_user", False):
            return "FINISH"

        summary_json = self._summarize_state(state)

        system_template = """
당신은 건설 안전 시스템의 지능형 Orchestrator입니다.
현재 상태(JSON)를 보고 다음에 실행할 **단 하나의 Agent**를 선택하세요.

[사용 가능한 에이전트 및 도구]
1. **IntentAgent**: 사용자의 첫 입력이 들어왔고, 아직 의도(current_intent)가 파악되지 않았을 때 실행.
2. **CSVSQLAgent**: 의도가 'query_sql' 이거나 날짜/통계 관련 질문인데, 아직 SQL을 실행하지 않았을 때(sql_executed=False) 실행.
3. **ASK_USER**: 
   - SQL 결과(sql_result_count)가 2건 이상이라서 사용자가 사고를 선택해야 할 때.
4. **RAGAgent**: 
   - 사고가 선택되었거나(selected_accident=True), 
   - 의도가 'search_only'이거나, 
   - SQL 결과가 0건이라서 지침 검색으로 넘어가야 할 때 (Fallback),
   - 사용자가 재검색(hitl_action='research_...')을 요청했을 때.
5. **ReportWriterAgent**: 문서 검색이 끝났고 보고서나 DOCX 파일을 생성해야 할 때.
6. **WebSearchAgent**: 내부 DB에 정보가 없고 웹 검색 요청이 있을 때.
7. **FINISH**: 모든 작업이 완료되었거나, 사용자 입력을 기다리는 중(ASK_USER 후)일 때.

[결정 논리 예시]
- Intent가 없으면? → IntentAgent
- Intent가 'query_sql'이고 SQL 실행 안 했으면? → CSVSQLAgent
- SQL 결과가 5개고 선택된 사고가 없으면? → ASK_USER (사용자 선택 필요)
- SQL 결과가 0개면? → RAGAgent (지침 검색으로 자동 전환)
- 문서 검색은 됐는데 보고서가 없으면? → ReportWriterAgent

반드시 아래 형식을 준수하여 JSON으로 응답하세요:
{format_instructions}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "현재 상태 JSON:\n{state_json}")
        ])

        chain = prompt | self.llm | self.parser

        try:
            decision: AgentDecision = await chain.ainvoke({
                "state_json": summary_json,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print(f"\n🧠 [Orchestrator] LLM 판단: {decision.next_agent}")
            print(f"   └─ 이유: {decision.reason}")
            
            return decision.next_agent

        except Exception as e:
            print(f"❌ 의사결정 실패: {e}")
            return "FINISH"

    async def run(self, state: AgentState) -> AgentState:
        
        next_agent_name = await self.decide_next_agent(state)
        state["next_agent"] = next_agent_name 
        
        if next_agent_name == "FINISH":
            state["is_complete"] = True
            return state
            
        if next_agent_name == "ASK_USER":
            print("🛑 Orchestrator: 사용자 입력 대기 (ASK_USER)")
            state["wait_for_user"] = True
            return state

        agent = self._get_agent_instance(next_agent_name)
        
        if agent:
            print(f"▶️ Agent 실행 시작: {next_agent_name}")
            returned_state = await agent.run(state)
            state.update(returned_state)
        else:
            print(f"⚠️ 알 수 없는 Agent 이름: {next_agent_name}")
            state["is_complete"] = True

        return state

# 싱글톤 인스턴스
orchestrator = OrchestratorAgent()