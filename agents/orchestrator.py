"""
🔥 FINAL OrchestratorAgent — LangChain LCEL & Pydantic V2 기반 (구조화된 출력 보장)
"""

from typing import Optional, Literal
from core.agentstate import AgentState
import json
import os
import chainlit as cl

# ✅ LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agents.subagents import get_agent 

# 🚨 [핵심 수정] Pydantic V1 호환 모듈 대신 표준 Pydantic(V2) 사용
# 이렇게 하면 'model_json_schema' 에러가 사라집니다.
from pydantic import BaseModel, Field


# ======================================================================
# 1. Pydantic 모델 정의 (LLM 출력 스키마 강제)
# ======================================================================
class AgentDecision(BaseModel):
    """Orchestrator가 다음 단계를 결정하기 위한 구조화된 출력 스키마"""
    
    next_agent: Literal["RAGAgent", "WebSearchAgent", "ReportWriterAgent", "FINISH"] = Field(
        description="다음에 실행할 Agent의 이름. 더 이상 수행할 작업이 없거나 완료되었으면 'FINISH'를 선택하세요."
    )
    reason: str = Field(
        description="현재 상태를 분석하여 왜 이 Agent(또는 FINISH)를 선택했는지에 대한 논리적인 이유(Chain-of-Thought)."
    )

class OrchestratorAgent:
    """Multi-Agent 시스템의 두뇌 (LCEL 기반)"""

    def __init__(self):
        # ✅ LangChain LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # ✅ Pydantic 파서 초기화
        self.parser = PydanticOutputParser(pydantic_object=AgentDecision)

    # ======================================================================
    # 상태 요약 (JSON)
    # ======================================================================
    def _create_state_summary(self, state: AgentState) -> str:
        summary = {
            "stop": state.get("wait_for_user", False),
            "user_intent": state.get("user_intent"),
            "hitl_action": state.get("hitl_action"),
            "retrieved_docs_count": len(state.get("retrieved_docs") or []),
            "report_created": bool(state.get("report_text")),
            "docx_created": bool(state.get("docx_path")),
            "web_search_requested": state.get("web_search_requested", False),
            "web_search_completed": state.get("web_search_completed", False),
        }
        return json.dumps(summary, ensure_ascii=False)

    # ======================================================================
    # 🔥 LCEL 기반 다음 Agent 결정 (가장 강력한 해결책)
    # ======================================================================
    async def decide_next_agent(self, state: AgentState) -> Optional[str]:

        if state.get("wait_for_user", False):
            return None

        summary_json = self._create_state_summary(state)

        # 1. 시스템 프롬프트 구성
        # {format_instructions} 부분에 LangChain이 자동으로 JSON 스키마를 삽입합니다.
        system_template = """
당신은 건설 안전 Multi-Agent 시스템의 Orchestrator입니다.
입력된 상태(JSON)를 분석하여 다음에 실행할 Agent를 결정하세요.

반드시 아래 형식을 준수하여 응답해야 합니다:
{format_instructions}

======================================================
📌 판단 규칙 (Priority Rules)
======================================================
1) HITL 액션 존재 (최우선)
    - research_keyword / research_db → RAGAgent
    - web_search → WebSearchAgent
    - accept_all / select_partial → ReportWriterAgent
    - exit → FINISH

2) user_intent == "search_only"
    - 문서 없음(0건) → RAGAgent
    - 문서 있음(>0건) → FINISH (STOP)

3) user_intent == "generate_report"
    - 문서 없음(0건) → RAGAgent
    - 보고서(report_text) 없음 → ReportWriterAgent
    - DOCX 파일(docx_path) 없음 → ReportWriterAgent
    - 보고서 + DOCX 모두 있음 → FINISH

4) 그 외 (초기 상태 등)
    - 문서 없음 → RAGAgent
    - 기타 불명확한 상태 → FINISH
"""

        # 2. 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "현재 상태 JSON:\n{state_json}")
        ])

        # 3. 🔥 LCEL 체인 조립: Prompt → LLM → Parser
        chain = prompt | self.llm | self.parser

        try:
            # 4. 체인 실행 (비동기)
            # format_instructions는 parser가 자동으로 제공
            decision: AgentDecision = await chain.ainvoke({
                "state_json": summary_json,
                "format_instructions": self.parser.get_format_instructions()
            })

            print(f"✅ Orchestrator Decision: {decision.next_agent}")
            print(f"🔍 Reason: {decision.reason}")

            # 5. 결과 반환 처리
            if decision.next_agent == "FINISH":
                return None
            
            return decision.next_agent

        except Exception as e:
            print(f"❌ LCEL Chain 결정 실패: {e}")
            # 파싱 실패 등 치명적 오류 시 Fallback 실행
            return self._fallback_decision(state)

    # ======================================================================
    # Fallback 로직 (비상용)
    # ======================================================================
    def _fallback_decision(self, state: AgentState) -> Optional[str]:
        print("\n🚨 FALLBACK 로직 실행 (LCEL 실패)")
        
        intent = state.get("user_intent")
        retrieved = state.get("retrieved_docs") or []

        if not retrieved:
            return "RAGAgent"
        if intent == "search_only":
            return None
        if not state.get("report_text"):
            return "ReportWriterAgent"
        if not state.get("docx_path"):
            return "ReportWriterAgent"
        return None
    
    # ======================================================================
    # Orchestrator 실행
    # ======================================================================
    async def run(self, state: AgentState) -> AgentState:
        
        intent = state.get("user_intent")
        hitl = state.get("hitl_action")

        # ---------------- HITL 처리 ----------------
        if hitl == "exit":
            state["is_complete"] = True
            return state

        if hitl in ["accept_all", "select_partial"] and intent == "search_only":
            state["user_intent"] = "generate_report"
            intent = "generate_report"

        # ---------------- search_only STOP ----------------
        if intent == "search_only" and state.get("retrieved_docs"):
            state["wait_for_user"] = True
            return state

        # ---------------- generate_report 완료 ----------------
        if (
            intent == "generate_report"
            and state.get("report_text")
            and state.get("docx_path")
        ):
            state["is_complete"] = True
            return state

        # ---------------- 다음 Agent 결정 (LCEL 호출) ----------------
        next_agent = await self.decide_next_agent(state)

        if next_agent is None:
            state["is_complete"] = True
            return state

        # HITL 초기화
        state["hitl_action"] = None
        state["hitl_payload"] = {}

        agent = get_agent(next_agent)
        if not agent:
            state["is_complete"] = True
            return state

        print(f"▶️ Agent 실행: {next_agent}")

        # Agent 실행
        returned_state = await agent.run(state)
        state.update(returned_state)

        return state

# 글로벌 인스턴스
orchestrator = OrchestratorAgent()