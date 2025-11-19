"""
Orchestrator Agent (개선된 버전 v4)
- user_intent 기반 판단 로직
- WebSearchAgent 추가
- "search_only": RAG만 실행 후 종료
- "generate_report": RAG → (WebSearch) → ReportWriter → DOCX
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
                    "description": "문서 검색을 수행하는 Agent입니다. 검색이 필요하거나 불충분할 때 호출하세요.",
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
                        "RAG 검색 결과가 부족하거나(3개 미만), 최신 정보가 필요하거나, "
                        "사용자가 명시적으로 웹 검색을 요청한 경우에만 호출하세요."
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
                    "description": (
                        "보고서 생성, DOCX 생성을 담당하는 Agent입니다. "
                        "RAG 또는 웹 검색이 완료된 후 호출하세요."
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
                    "name": "END",
                    "description": "모든 작업이 완료되었을 때만 호출합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "종료하는 이유"}
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]


    def _create_state_summary(self, state: AgentState) -> str:
        """State를 LLM이 이해하기 쉬운 형식으로 요약"""
        
        retrieved = state.get("retrieved_docs")
        report_ready = state.get("report_text")
        docx_ready = state.get("docx_path")
        web_search_done = state.get("web_search_completed", False)
        web_search_requested = state.get("web_search_requested", False)  # ✅ 추가
        
        # ✅ 사용자 의도 확인
        user_intent = state.get("user_intent", "generate_report")

        summary = f"""
현재 시스템 상태:

[사용자 질의]
{state.get('user_query', 'N/A')}

[사용자 의도]
{user_intent}
- "search_only": 정보 검색만 원함 (RAG → END)
- "generate_report": 보고서 생성 원함 (RAG → (WebSearch) → ReportWriter → END)

[RAG 검색 상태]
- 문서 검색 완료: {'✅ 예' if retrieved else '❌ 아니오'}
- 검색된 문서 수: {len(retrieved) if retrieved else 0}

[웹 검색 상태]
- 웹 검색 완료: {'✅ 예' if web_search_done else '❌ 아니오'}
- 웹 검색 요청됨: {'✅ 예' if web_search_requested else '❌ 아니오'}

[보고서 상태]
- 보고서 생성 완료: {'✅ 예' if report_ready else '❌ 아니오'}

[DOCX 상태]
- DOCX 파일 생성 완료: {'✅ 예' if docx_ready else '❌ 아니오'}

[다음 Agent 선택 규칙]
**user_intent가 "search_only"인 경우:**
1. RAG 검색이 안 되었으면 → RAGAgent
2. 웹 검색이 요청되었고 완료 안 되었으면 → WebSearchAgent
3. 모두 완료되었으면 → END

**user_intent가 "generate_report"인 경우:**
1. RAG 검색이 안 되었으면 → RAGAgent
2. RAG 결과가 부족하고(<3개) 웹 검색 미완료면 → WebSearchAgent
3. 검색 완료되었지만 보고서 없으면 → ReportWriterAgent
4. 보고서 있지만 DOCX 없으면 → ReportWriterAgent
5. 모두 완료되었으면 → END

**중요: WebSearchAgent는 다음 경우에만 호출**
- RAG 검색 결과가 3개 미만
- 사용자가 명시적으로 웹 검색 요청 (web_search_requested=True)
- 최신 정보가 필요한 경우
"""
        return summary


    def decide_next_agent(self, state: AgentState) -> Optional[str]:
        """LLM을 사용하여 다음 Agent 결정 (user_intent 기반)"""
        
        state_summary = self._create_state_summary(state)

        system_message = {
            "role": "system",
            "content": """
당신은 Multi-Agent 시스템의 Orchestrator입니다.

선택 가능한 Agent:
- RAGAgent: 문서 검색
- ReportWriterAgent: 보고서 작성, 웹검색, DOCX 생성
- END: 모든 작업 완료

**중요: user_intent를 반드시 확인하세요!**

user_intent가 "search_only"이면:
- RAG 검색만 하고 바로 END

user_intent가 "generate_report"이면:
- 기존대로 RAG → ReportWriter → END

반드시 tool calling 형식으로 응답하세요.
"""
        }

        user_message = {"role": "user", "content": state_summary}

        try:
            print("\n🧠 [Orchestrator] 다음 Agent 결정 중...")
            print(state_summary)

            response = call_llm_with_tools(
                messages=[system_message, user_message],
                tools=self.tools,
                temperature=0.0,
            )

            if response and response.tool_calls:
                tool_call = response.tool_calls[0]
                agent_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                reason = args.get("reason", "")

                print(f"✅ 결정된 Agent: {agent_name}")
                print(f"💡 이유: {reason}")
                return agent_name
            else:
                print("⚠️ LLM tool 호출 실패 → fallback 사용")
                return self._fallback_decision(state)

        except Exception as e:
            print(f"❌ Orchestrator 오류: {e}")
            return self._fallback_decision(state)


    def _fallback_decision(self, state: AgentState) -> str:
        """
        Tool calling 실패 시 Rule-based fallback
        user_intent 및 웹 검색 요청 기반으로 판단
        """
        print("\n" + "⚠️ " * 40)
        print("⚠️  FALLBACK 모드 활성화 - LLM 판단 실패로 Rule-based 로직 사용")
        print("⚠️ " * 40)
        
        user_intent = state.get("user_intent", "generate_report")
        web_search_requested = state.get("web_search_requested", False)
        web_search_done = state.get("web_search_completed", False)
        retrieved_docs = state.get("retrieved_docs", [])
        
        # search_only 모드
        if user_intent == "search_only":
            if not retrieved_docs:
                print("📌 [Fallback Rule - search_only] RAG 검색 필요 → RAGAgent 선택")
                return "RAGAgent"
            elif web_search_requested and not web_search_done:
                print("📌 [Fallback Rule - search_only] 웹 검색 요청됨 → WebSearchAgent 선택")
                return "WebSearchAgent"
            else:
                print("📌 [Fallback Rule - search_only] 검색 완료 → END 선택")
                return "END"
        
        # generate_report 모드 (기존 로직 + 웹 검색)
        if not retrieved_docs:
            print("📌 [Fallback Rule 1] RAG 검색 필요 → RAGAgent 선택")
            return "RAGAgent"
        
        # RAG 결과가 부족하고 웹 검색이 안 되었으면 웹 검색
        if len(retrieved_docs) < 3 and not web_search_done:
            print("📌 [Fallback Rule 2] RAG 결과 부족(<3개) → WebSearchAgent 선택")
            return "WebSearchAgent"
        
        # 사용자가 웹 검색 요청했는데 안 되었으면
        if web_search_requested and not web_search_done:
            print("📌 [Fallback Rule 3] 웹 검색 요청됨 → WebSearchAgent 선택")
            return "WebSearchAgent"
        
        if not state.get("report_text"):
            print("📌 [Fallback Rule 4] 보고서 필요 → ReportWriterAgent 선택")
            return "ReportWriterAgent"
        
        if not state.get("docx_path"):
            print("📌 [Fallback Rule 5] DOCX 필요 → ReportWriterAgent 선택")
            return "ReportWriterAgent"
        
        print("📌 [Fallback Rule 6] 모든 작업 완료 → END 선택")
        return "END"


    def run(self, state: AgentState) -> AgentState:
        """Orchestrator 실행: 다음 Agent 결정 및 실행"""
        
        next_agent = self.decide_next_agent(state)

        if next_agent == "END":
            state["is_complete"] = True
            print("\n🎉 모든 작업 완료!")
            return state

        agent = get_agent(next_agent)
        if agent is None:
            print(f"❌ '{next_agent}' Agent를 찾을 수 없습니다.")
            state["is_complete"] = True
            return state

        # Agent 호출 전에 구분선 출력
        print(f"\n{'='*80}")
        print(f"▶️  다음 실행: {next_agent}")
        print(f"{'='*80}")
        
        state["next_agent"] = next_agent
        state = agent.run(state)

        return state


# 전역 인스턴스
orchestrator = OrchestratorAgent()