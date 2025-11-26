"""
Orchestrator Agent (LLM Prompt 기반 라우팅 강화)

✅ 최종 수정사항:
1. OrchestratorAgent.run()을 async로 선언했습니다. (Agent.run 호출을 위해 필수)
2. Agent.run(state) 호출 시 await을 추가했습니다. (TypeError 해결)
3. decide_next_agent 내의 LLM 호출을 cl.make_async로 감싸고 await을 추가했습니다.
4. decide_next_agent 자체를 async 함수로 변경했습니다.
5. HITL로 search_only → generate_report 전환 후 user_intent를 즉시 재평가하도록 수정했습니다.  🔥
"""

from typing import Optional, Any, Dict, List
from core.agentstate import AgentState
from core.llm_utils import call_llm_with_tools
from agents.subagents import get_agent
import json
import chainlit as cl  # cl.make_async를 사용하기 위해 추가 (Agent.run이 async이므로)


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
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 선택한 이유",
                            }
                        },
                        "required": ["reason"],
                    },
                },
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
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 선택한 이유",
                            }
                        },
                        "required": ["reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ReportWriterAgent",
                    "description": "보고서 생성 및 DOCX 생성을 담당합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 선택한 이유",
                            }
                        },
                        "required": ["reason"],
                    },
                },
            },
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
HITL 액션: {state.get('hitl_action', '없음')}
HITL 페이로드: {state.get('hitl_payload', {})}

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
    # 🌟 다음 Agent 결정 (LLM 프롬프트 기반) 🌟
    # ===========================
    async def decide_next_agent(self, state: AgentState) -> Optional[str]:
        """
        LLM + Tool Calling으로 다음 실행할 Agent를 결정 (async 버전)
        """
        if state.get("wait_for_user", False):
            print("\n⏸ STOP 상태: 사용자 입력 대기 중...")
            return None

        hitl_action = state.get("hitl_action")
        state_summary = self._create_state_summary(state)

        # 🌟 LLM 지침: HITL 피드백 & 플로우 규칙
        system_message = {
            "role": "system",
            "content": f"""
당신은 Multi-Agent Orchestrator입니다. 현재 상태 요약과 사용자 피드백(HITL 액션)을 기반으로 다음 실행 Agent를 결정하세요.

################################################################################
#  [최우선 START 규칙] 
################################################################################
1. HITL 액션이 '없음'이고 'RAG 검색 - 완료: False'이면, **RAGAgent**를 호출하여 최초 검색을 시작해야 합니다.
   이 규칙은 다른 HITL 처리 규칙이나 플로우 규칙보다 우선합니다. (검색 없이는 진행 불가)
2. HITL 액션이 'accept_all' 또는 'select_partial'이면 ReportWriterAgent를 반드시 호출해야 한다. (최우선 규칙)

################################################################################
# [HITL 피드백 처리 규칙 - 최우선]
################################################################################
2. HITL 액션이 존재한다면, 다음 Agent를 호출해야 합니다:
    - 'research_keyword' 또는 'research_db' → RAGAgent
    - 'web_search' → WebSearchAgent
    - 'accept_all' 또는 'select_partial' → ReportWriterAgent
    - 'exit' → None 반환 (종료)

################################################################################
# [기타 플로우 규칙]
################################################################################
3. 'user_intent'가 'generate_report'이고 '보고서: 생성됨: False'라면 ReportWriterAgent를 호출해야 합니다.
4. 'DOCX: 생성됨: False'라면 ReportWriterAgent를 호출해야 합니다.
5. 'search_only' 모드이고 RAG 완료(True) 상태라면, None을 반환하여 멈춰야 합니다.

반드시 tool-calling 형식으로만 응답하세요.
""",
        }

        user_message = {"role": "user", "content": state_summary}

        try:
            # ❗ call_llm_with_tools는 동기 함수 → cl.make_async로 감싸고 await
            response = await cl.make_async(call_llm_with_tools)(
                messages=[system_message, user_message],
                tools=self.tools,
                temperature=0.0,
            )

            if response and getattr(response, "tool_calls", None):
                tool_call = response.tool_calls[0]
                agent_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(
                    f"✅ LLM 결정 Agent: {agent_name} / 이유: {args.get('reason','')}"
                )
                return agent_name

            print("⚠️ LLM tool-call 없음 → fallback 사용")
            return self._fallback_decision(state)

        except Exception as e:
            print(f"❌ Orchestrator 오류: {e}")
            return self._fallback_decision(state)

    # ===========================
    #  Fallback 로직 (순수 Rule-based로 축소) 
    # ===========================
    def _fallback_decision(self, state: AgentState) -> Optional[str]:
        user_intent = state.get("user_intent", "generate_report")
        retrieved = state.get("retrieved_docs", [])
        web_req = state.get("web_search_requested", False)
        web_done = state.get("web_search_completed", False)

        # 📌 1. HITL 액션이 있다면, LLM이 판단하지 못했으므로 Fallback은 즉시 종료 (안전장치)
        if state.get("hitl_action"):
            print("⚠️ LLM이 HITL 액션을 처리하지 못했습니다. → 작업 종료 (재시도 방지)")
            state["hitl_action"] = None  # 상태 초기화
            return None  # run 함수에서 is_complete=True로 종료됨

        # 📌 2. 기존 플로우 유지 (RAG 호출을 Fallback에서 보장)

        # 문서가 없다면 RAGAgent 호출 (최초 search_only/generate_report 시)
        if not retrieved:
            print(f"📌 [fallback] {user_intent}: 문서 없음 → RAGAgent 호출")
            return "RAGAgent"

        # search_only 모드: RAG 완료 후 → NONE 반환 (STOP 상태 유지)
        if user_intent == "search_only":
            print("📌 [fallback] search_only: RAG 완료 상태. → NONE 반환")
            return None

        # generate_report 모드 (RAG 완료 후)
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
    #  Orchestrator 실행 (run)
    # ===========================
    async def run(self, state: AgentState) -> AgentState:  # 🌟 async 선언
        user_intent = state.get("user_intent", "generate_report")
        hitl_action = state.get("hitl_action")

        # 🌟🌟🌟 HITL 재개 시 모드 전환 및 최종 종료 🌟🌟🌟
        # HITL 액션이 'accept'류 이면, user_intent를 generate_report로 강제 변경 (LLM 판단 이전에 처리)
        if hitl_action in ["accept_all", "select_partial"] and user_intent == "search_only":
            print("\n🔄 HITL 확정: search_only → generate_report 모드로 전환.")
            state["user_intent"] = "generate_report"
            # 🔥 여기서 바로 user_intent를 다시 읽어서 아래 규칙들이 최신 의도 기준으로 동작하게 함
            user_intent = state.get("user_intent", "generate_report")

        # HITL 액션이 'exit'이면, 즉시 종료
        if hitl_action == "exit":
            print("\n👋 HITL 요청: 작업 종료. 워크플로우를 종료합니다.")
            state["is_complete"] = True
            state["hitl_action"] = None
            return state

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
        next_agent = await self.decide_next_agent(state)  # 🌟 async 호출이므로 await 필수

        # next_agent 가 None이면 → 더 할 일 없음 (완료로 처리)
        if next_agent is None:
            print("\nℹ️ 실행할 Agent가 없습니다. 워크플로우를 종료합니다.")
            state["is_complete"] = True
            return state

        # 🌟 Agent 실행 직전에 HITL 액션 소비 (재검색/웹검색/보고서 생성 플로우 시작)
        if state.get("hitl_action") not in [None, "exit"]:
            state["hitl_action"] = None
            state["hitl_payload"] = {}

        agent = get_agent(next_agent)
        if not agent:
            print(f"❌ Agent '{next_agent}'를 찾을 수 없음 → 강제 종료")
            state["is_complete"] = True
            return state

        print(f"\n▶️ 다음 실행: {next_agent}")
        state["next_agent"] = next_agent

        # 🎯🎯🎯 Agent의 run 메서드는 async이므로, 반드시 await 호출 🎯🎯🎯
        return await agent.run(state)


# 전역 인스턴스
orchestrator = OrchestratorAgent()
