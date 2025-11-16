"""
Orchestrator Agent
LLM이 현재 state를 보고 다음에 실행할 SubAgent를 결정
"""
from typing import Optional
from core.agentstate import AgentState
from core.llm_utils import call_llm_with_tools
from agents.subagents import get_agent
import json


class OrchestratorAgent:
    """
    시스템의 두뇌 - 매 단계마다 현재 상태를 보고 다음 Agent 결정
    """
    
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "RAGAgent",
                    "description": "건설안전 DB에서 관련 문서를 검색합니다. 사용자 쿼리에 대한 정보가 필요할 때 사용하세요.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 호출하는 이유"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ReportWriterAgent",
                    "description": "검색된 문서를 바탕으로 건설 사고 재발 방지 보고서를 작성합니다. 문서 검색이 완료된 후 사용하세요.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 호출하는 이유"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "DocxWriterAgent",
                    "description": "보고서를 DOCX 파일로 생성합니다. 보고서 작성이 완료된 후 사용하세요.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "이 Agent를 호출하는 이유"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "END",
                    "description": "모든 작업이 완료되었을 때 호출합니다. 검색, 보고서 작성, DOCX 생성이 모두 끝났으면 이것을 호출하세요.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "작업을 종료하는 이유"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]
    
    def _create_state_summary(self, state: AgentState) -> str:
        """
        현재 state를 요약하여 LLM에게 전달
        """
        summary = f"""
현재 상태:
- 사용자 쿼리: {state.get('user_query', 'N/A')}
- 문서 검색 완료 여부: {'완료 ({} docs)'.format(len(state.get('retrieved_docs', []))) if state.get('retrieved_docs') else '미완료'}
- 보고서 작성 완료 여부: {'완료' if state.get('report_text') else '미완료'}
- DOCX 파일 생성 완료 여부: {'완료' if state.get('docx_path') else '미완료'}

작업 진행 단계:
"""
        if not state.get('retrieved_docs'):
            summary += "1. [대기중] 문서 검색 필요\n"
        else:
            summary += "1. [완료] 문서 검색\n"
        
        if not state.get('report_text'):
            summary += "2. [대기중] 보고서 작성 필요\n"
        else:
            summary += "2. [완료] 보고서 작성\n"
        
        if not state.get('docx_path'):
            summary += "3. [대기중] DOCX 생성 필요\n"
        else:
            summary += "3. [완료] DOCX 생성\n"
        
        return summary
    
    def decide_next_agent(self, state: AgentState) -> Optional[str]:
        """
        LLM을 사용하여 다음 Agent 결정
        
        Returns:
            다음 실행할 Agent 이름 또는 "END"
        """
        state_summary = self._create_state_summary(state)
        
        system_message = {
            "role": "system",
            "content": """
당신은 Multi-Agent 시스템의 Orchestrator입니다.
현재 상태를 보고 다음에 실행할 Agent를 결정하세요.

작업 순서:
1. RAGAgent: 문서 검색
2. ReportWriterAgent: 보고서 작성 (검색 완료 후)
3. DocxWriterAgent: DOCX 생성 (보고서 완료 후)
4. END: 모든 작업 완료

규칙:
- 순서대로 진행하세요
- 이미 완료된 작업은 다시 하지 마세요
- 모든 단계가 완료되면 END를 호출하세요
"""
        }
        
        user_message = {
            "role": "user",
            "content": state_summary
        }
        
        try:
            print(f"\n🧠 [Orchestrator] 다음 Agent 결정 중...")
            print(f"현재 상태:\n{state_summary}")
            
            response = call_llm_with_tools(
                messages=[system_message, user_message],
                tools=self.tools,
                temperature=0.0  # 결정론적 선택
            )
            
            if response and response.tool_calls:
                tool_call = response.tool_calls[0]
                agent_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                reason = arguments.get('reason', '')
                
                print(f"✅ 결정: {agent_name}")
                print(f"💡 이유: {reason}")
                
                return agent_name
            else:
                # Tool call이 없으면 기본 로직
                print("⚠️ LLM이 tool을 호출하지 않음. 기본 로직 사용")
                return self._fallback_decision(state)
        
        except Exception as e:
            print(f"❌ Orchestrator 오류: {e}")
            return self._fallback_decision(state)
    
    def _fallback_decision(self, state: AgentState) -> str:
        """
        LLM 실패 시 사용하는 폴백 로직
        """
        if not state.get('retrieved_docs'):
            return "RAGAgent"
        elif not state.get('report_text'):
            return "ReportWriterAgent"
        elif not state.get('docx_path'):
            return "DocxWriterAgent"
        else:
            return "END"
    
    def run(self, state: AgentState) -> AgentState:
        """
        Orchestrator 실행
        1. 다음 Agent 결정
        2. 해당 Agent 실행
        3. State 업데이트하여 반환
        """
        # 다음 Agent 결정
        next_agent_name = self.decide_next_agent(state)
        
        # END면 종료
        if next_agent_name == "END":
            state["is_complete"] = True
            state["next_agent"] = "END"
            print(f"\n{'='*80}")
            print("🎉 모든 작업 완료!")
            print(f"{'='*80}\n")
            return state
        
        # 해당 Agent 실행
        agent = get_agent(next_agent_name)
        if agent:
            state["next_agent"] = next_agent_name
            state = agent.run(state)
        else:
            print(f"❌ Agent '{next_agent_name}'을 찾을 수 없습니다.")
            state["is_complete"] = True
        
        return state


# Orchestrator 전역 인스턴스
orchestrator = OrchestratorAgent()
