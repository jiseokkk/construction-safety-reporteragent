"""
🔥 FINAL OrchestratorAgent — 통합본 (한글)
✅ 수정 완료:
1. 문서 병합 중단 방지 (재검색 시 루프 유지)
2. [CRITICAL FIX] 에이전트 실행 전 HITL 정보 초기화 로직 제거 (RAGAgent에 정보 전달 보장)
✅ 기능: Pydantic V2 기반 구조화된 출력
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

# ✅ Pydantic V2 (표준)
from pydantic import BaseModel, Field


# ======================================================================
# 1. Pydantic 모델 정의 (구조화된 출력)
# ======================================================================
class AgentDecision(BaseModel):
    """Orchestrator가 다음 단계를 결정하기 위한 구조화된 스키마"""
    
    next_agent: Literal["RAGAgent", "WebSearchAgent", "ReportWriterAgent", "FINISH"] = Field(
        description="다음에 실행할 Agent의 이름. 더 이상 수행할 작업이 없거나 사용자 입력이 필요하면 'FINISH'를 선택하세요."
    )
    reason: str = Field(
        description="현재 상태를 분석하여 왜 이 Agent(또는 FINISH)를 선택했는지에 대한 논리적인 이유(Chain-of-Thought)."
    )

class OrchestratorAgent:
    """Multi-Agent 시스템의 두뇌 (LCEL 기반)"""

    def __init__(self):
        # ✅ LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # ✅ 파서 초기화
        self.parser = PydanticOutputParser(pydantic_object=AgentDecision)

    # ======================================================================
    # 상태 요약 헬퍼
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
    # 🔥 다음 Agent 결정 (LCEL)
    # ======================================================================
    async def decide_next_agent(self, state: AgentState) -> Optional[str]:

        if state.get("wait_for_user", False):
            return None

        summary_json = self._create_state_summary(state)

        # 1. 시스템 프롬프트 구성
        # HITL 재검색 요청 시 문서가 있어도 실행하도록 명시
        system_template = """
당신은 건설 안전 Multi-Agent 시스템의 Orchestrator입니다.
입력된 상태(JSON)를 분석하여 다음에 실행할 Agent를 결정하세요.

반드시 아래 형식을 준수하여 응답해야 합니다:
{format_instructions}

======================================================
📌 판단 규칙 (Priority Rules - 위에서부터 순서대로 적용)
======================================================
1. [최우선] HITL 재검색 요청이 있는 경우 (문서가 이미 있어도 무조건 실행)
    - hitl_action == "research_keyword" OR "research_db" → RAGAgent
    - hitl_action == "web_search" → WebSearchAgent

2. HITL 진행 요청
    - hitl_action == "accept_all" OR "select_partial" → ReportWriterAgent
    - hitl_action == "exit" → FINISH

3. user_intent == "search_only" (HITL 없음)
    - 문서 있음(retrieved_docs_count > 0) → FINISH
    - 문서 없음(retrieved_docs_count == 0) → RAGAgent

4. user_intent == "generate_report"
    - 문서 없음(0건) → RAGAgent
    - 보고서(report_text) 없음 → ReportWriterAgent
    - DOCX 파일(docx_path) 없음 → ReportWriterAgent
    - 보고서 + DOCX 모두 있음 → FINISH

5. 그 외 / Fallback
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
            # 4. 체인 실행
            decision: AgentDecision = await chain.ainvoke({
                "state_json": summary_json,
                "format_instructions": self.parser.get_format_instructions()
            })

            print(f"✅ Orchestrator Decision: {decision.next_agent}")
            print(f"🔍 Reason: {decision.reason}")

            if decision.next_agent == "FINISH":
                return None
            
            return decision.next_agent

        except Exception as e:
            print(f"❌ LCEL Chain 결정 실패: {e}")
            return self._fallback_decision(state)

    # ======================================================================
    # Fallback 로직 (비상용)
    # ======================================================================
    def _fallback_decision(self, state: AgentState) -> Optional[str]:
        print("\n🚨 FALLBACK 로직 실행 (LCEL 실패)")
        
        intent = state.get("user_intent")
        retrieved = state.get("retrieved_docs") or []
        hitl = state.get("hitl_action")

        # HITL 재검색인 경우 강제 RAGAgent
        if hitl in ["research_db", "research_keyword"]:
            return "RAGAgent"

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
    # Orchestrator 실행 (Main Run)
    # ======================================================================
    async def run(self, state: AgentState) -> AgentState:
        
        intent = state.get("user_intent")
        hitl = state.get("hitl_action")

        # ---------------- HITL: 종료 ----------------
        if hitl == "exit":
            state["is_complete"] = True
            return state

        # ---------------- HITL: 보고서 모드로 전환 ----------------
        if hitl in ["accept_all", "select_partial"] and intent == "search_only":
            state["user_intent"] = "generate_report"
            intent = "generate_report"

        # ---------------- [핵심 수정] search_only STOP 조건 완화 ----------------
        # 재검색 작업인지 확인
        is_researching = hitl in ["research_keyword", "research_db", "web_search"]
        
        # '재검색 중이 아닐 때'만 문서가 있으면 멈춤
        if intent == "search_only" and state.get("retrieved_docs") and not is_researching:
            state["wait_for_user"] = True
            return state

        # ---------------- generate_report 완료 조건 ----------------
        if (
            intent == "generate_report"
            and state.get("report_text")
            and state.get("docx_path")
        ):
            state["is_complete"] = True
            return state

        # ---------------- 다음 Agent 결정 ----------------
        next_agent = await self.decide_next_agent(state)

        if next_agent is None:
            state["is_complete"] = True
            return state

        # 🚨 [CRITICAL FIX] 에이전트 실행 전에 HITL 정보를 초기화하면 안 됨!
        # RAGAgent가 hitl_action과 hitl_payload를 읽어야 하므로,
        # 초기화 코드는 SubAgents 내부 또는 Orchestrator가 결과를 받은 후로 미뤄야 함.
        # 기존에 있던 아래 두 줄을 삭제함.
        # state["hitl_action"] = None
        # state["hitl_payload"] = {}

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