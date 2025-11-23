"""
Orchestrator Agent (STOP 지원 버전)
- user_intent 기반 판단 로직
- search_only: RAG 완료 후 STOP
- generate_report: RAG → (WebSearch) → ReportWriter → END
"""

from typing import Optional
from core.agentstate import AgentState
from core.llm_utils import call_llm_with_tools
from agents.subagents import get_agent
import json


class OrchestratorAgent:
    """
    전체 Multi-Agent 시스템의 두뇌
    """

    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "RAGAgent",
                    "description": "문서 검색을 수행하는 Agent입니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "이 Agent를 선택한 이유"}
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "WebSearchAgent",
                    "description": (
                        "Tavily API로 웹 검색을 수행하는 Agent입니다. "
                        "RAG 결과가 부족하거나 사용자 요청 시 호출됩니다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "이 Agent를 선택한 이유"}
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ReportWriterAgent",
                    "description": "보고서 생성 및 DOCX 생성을 담당합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "이 Agent를 선택한 이유"}
                        },
                        "required": ["reason"]
                    }
                }
            }
            # 🔴 END 툴은 굳이 필요 없어서 제거 (END는 우리가 직접 is_complete로 컨트롤)
        ]

    # ===========================
    #  상태 요약 (LLM 판단용)
    # ===========================
    def _create_state_summary(self, state: AgentState) -> str:
        retrieved = state.get("retrieved_docs")
        report_ready = state.get("report_text")
        docx_ready = state.get("docx_path")
        web_done = state.get("web_search_completed", False)
        web_req = state.get("web_search_requested", False)
        user_intent = state.get("user_intent", "generate_report")

        summary = f"""
[현재 상태]

STOP 상태: {state.get('wait_for_user', False)}
사용자 의도: {user_intent}

RAG 검색:
- 완료: {bool(retrieved)}
- 문서 수: {len(retrieved) if retrieved else 0}

웹 검색:
- 요청됨: {web_req}
- 완료됨: {web_done}

보고서:
- 생성됨: {bool(report_ready)}

DOCX:
- 생성됨: {bool(docx_ready)}
"""
        return summary

    # ===========================
    #  다음 Agent 결정 (LLM)
    # ===========================
    def decide_next_agent(self, state: AgentState) -> Optional[str]:
        # 이미 STOP 상태면 아무것도 하지 않음
        if state.get("wait_for_user", False):
            print("\n⏸ STOP 상태: 사용자 입력 대기 중...")
            return None

        state_summary = self._create_state_summary(state)

        system_message = {
            "role": "system",
            "content": """
당신은 Multi-Agent Orchestrator입니다.

search_only:
- RAGAgent로 검색만 수행
- 검색이 완료되면 보고서/웹검색/END를 호출하지 말고 멈춥니다.

generate_report:
- 기본 플로우: RAGAgent → (필요 시 WebSearchAgent) → ReportWriterAgent
- ReportWriterAgent는 보고서 생성 및 DOCX 생성을 담당합니다.

반드시 tool-calling 형식으로만 응답하세요.
            """
        }

        user_message = {"role": "user", "content": state_summary}

        try:
            response = call_llm_with_tools(
                messages=[system_message, user_message],
                tools=self.tools,
                temperature=0.0,
            )

            if response and response.tool_calls:
                tool_call = response.tool_calls[0]
                agent_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"✅ LLM 결정 Agent: {agent_name} / 이유: {args.get('reason','')}")
                return agent_name

            print("⚠️ LLM tool-call 없음 → fallback 사용")
            return self._fallback_decision(state)

        except Exception as e:
            print(f"❌ Orchestrator 오류: {e}")
            return self._fallback_decision(state)

    # ===========================
    #  Fallback 로직
    # ===========================
    def _fallback_decision(self, state: AgentState) -> Optional[str]:
        user_intent = state.get("user_intent", "generate_report")
        retrieved = state.get("retrieved_docs", [])
        web_req = state.get("web_search_requested", False)
        web_done = state.get("web_search_completed", False)

        # search_only 모드: RAG만 돌리고 STOP
        if user_intent == "search_only":
            if not retrieved:
                print("📌 [fallback] search_only: RAG 필요")
                return "RAGAgent"
            print("📌 [fallback] search_only: RAG 완료 → STOP")
            return None

        # generate_report 모드
        if not retrieved:
            print("📌 [fallback] generate_report: 우선 RAGAgent")
            return "RAGAgent"

        if len(retrieved) < 3 and not web_done:
            print("📌 [fallback] 문서 적음 → WebSearchAgent")
            return "WebSearchAgent"

        if web_req and not web_done:
            print("📌 [fallback] 사용자가 웹검색 요청 → WebSearchAgent")
            return "WebSearchAgent"

        if not state.get("report_text"):
            print("📌 [fallback] 보고서 없음 → ReportWriterAgent")
            return "ReportWriterAgent"

        if not state.get("docx_path"):
            print("📌 [fallback] DOCX 없음 → ReportWriterAgent")
            return "ReportWriterAgent"

        print("📌 [fallback] 모든 작업 완료 → 종료")
        return None

    # ===========================
    #  Orchestrator 실행
    # ===========================
    def run(self, state: AgentState) -> AgentState:
        user_intent = state.get("user_intent", "generate_report")

        # ✅ Rule 1: search_only + RAG 완료 → 여기서 강제 STOP
        if (
            user_intent == "search_only"
            and state.get("route") == "retrieve_complete"
            and state.get("retrieved_docs")
        ):
            print("\n################################################################################")
            print("📌 [Rule] search_only: RAG 완료 → STOP (사용자 입력 대기)")
            print("################################################################################")
            state["wait_for_user"] = True
            # is_complete 는 False → 나중에 보고서 생성/종료 선택 가능
            return state

        # ✅ Rule 2: generate_report 모드에서 report + docx 둘 다 있으면 종료
        if (
            user_intent == "generate_report"
            and state.get("report_text")
            and state.get("docx_path")
        ):
            print("\n🎉 모든 작업 완료! (보고서 + DOCX 생성 완료)")
            state["is_complete"] = True
            return state

        # 그 외에는 LLM/Rule 기반으로 다음 Agent 선택
        print("\n🧠 [Orchestrator] 다음 Agent 결정 중...")
        next_agent = self.decide_next_agent(state)

        # next_agent 가 None이면 → 더 할 일 없음 (완료로 처리)
        if next_agent is None:
            print("\nℹ️ 실행할 Agent가 없습니다. 워크플로우를 종료합니다.")
            state["is_complete"] = True
            return state

        agent = get_agent(next_agent)
        if not agent:
            print(f"❌ Agent '{next_agent}'를 찾을 수 없음 → 강제 종료")
            state["is_complete"] = True
            return state

        print(f"\n▶️ 다음 실행: {next_agent}")
        state["next_agent"] = next_agent

        return agent.run(state)


# 전역 인스턴스
orchestrator = OrchestratorAgent()