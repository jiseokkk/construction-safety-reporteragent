"""
SubAgent 클래스들
각 Agent는 독립적으로 동작하며, state를 입력받아 작업 수행 후 state 반환
"""
from typing import Dict, Any
from core.agentstate import AgentState

# ✅ LLM 유틸 및 기존 모듈
from core.retriever import retriever_instance  # RerankRetriever 전역 인스턴스
from core.final_report import (
    summarize_accident_cause,
    generate_action_plan,
)
from core.docx_writer import create_accident_report_docx


# ========================================
# 1. RAGAgent - 문서 검색
# ========================================
class RAGAgent:
    """
    건설안전 DB에서 관련 문서를 검색하는 Agent
    기존 retriever.py의 RerankRetriever를 그대로 사용
    """
    name = "RAGAgent"
    
    def __init__(self):
        self.retriever = retriever_instance
    
    def run(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}")
        print(f"🔍 [{self.name}] 실행 중...")
        print(f"{'='*80}")
        
        query = state["user_query"]
        
        try:
            docs = self.retriever.retrieve(query)
            
            docs_text = "\n\n".join([
                f"[문서 {i+1}] ({doc.metadata.get('source', '?')} - {doc.metadata.get('section', '?')})\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            sources = [
                {
                    "idx": i + 1,
                    "filename": doc.metadata.get("source", ""),
                    "section": doc.metadata.get("section", "")
                }
                for i, doc in enumerate(docs)
            ]
            
            state["retrieved_docs"] = docs
            state["docs_text"] = docs_text
            state["sources"] = sources
            state["route"] = "retrieve_complete"
            
            print(f"✅ 검색 완료: {len(docs)}개 문서")
            
        except Exception as e:
            print(f"❌ RAGAgent 실행 실패: {e}")
            state["docs_text"] = ""
            state["sources"] = []
        
        return state


# ========================================
# 2. ReportWriterAgent - 요약 + 조치계획 생성
# ========================================
class ReportWriterAgent:
    """
    RAG 결과를 바탕으로
    - summary_cause (사고발생 경위)
    - summary_action_plan (조치사항 및 향후조치계획)
    을 생성하는 Agent
    """
    name = "ReportWriterAgent"
    
    def run(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}")
        print(f"📝 [{self.name}] 실행 중...")
        print(f"{'='*80}")
        
        rag_output = state.get("docs_text", "")
        user_query = state.get("user_query", "")
        
        if not rag_output:
            print("⚠️ 검색 결과가 없습니다. 보고서(요약/조치계획) 생성 불가")
            msg = "검색 결과가 없어 사고발생 경위 및 조치계획을 생성할 수 없습니다."
            state["summary_cause"] = msg
            state["summary_action_plan"] = msg
            state["report_text"] = msg
            return state
        
        try:
            # ① 사고발생 경위 요약
            summary_cause = summarize_accident_cause(rag_output, user_query)
            
            # ② 조치사항 및 향후조치계획 (길고 논리적인 보고서 스타일)
            action_plan = generate_action_plan(rag_output, user_query)
            
            # ③ 합쳐서 하나의 텍스트로도 보관 (호환용)
            combined = f"【사고발생 경위】\n{summary_cause}\n\n【조치사항 및 향후조치계획】\n{action_plan}"
            
            state["summary_cause"] = summary_cause
            state["summary_action_plan"] = action_plan
            state["report_text"] = combined
            state["report_summary"] = (combined[:200] + "...") if len(combined) > 200 else combined
            state["route"] = "report_complete"
            
            print("✅ 사고발생 경위 + 조치계획 생성 완료")
            
        except Exception as e:
            print(f"❌ ReportWriterAgent 실행 실패: {e}")
            msg = f"보고서(사고경위/조치계획) 생성 실패: {str(e)}"
            state["summary_cause"] = msg
            state["summary_action_plan"] = msg
            state["report_text"] = msg
        
        return state


# ========================================
# 3. DocxWriterAgent - DOCX 파일 생성
# ========================================
class DocxWriterAgent:
    """
    사고개요(user_query), summary_cause, summary_action_plan을 사용하여
    [별지 제2호 서식] 건설사고 발생현황 보고 양식을 DOCX로 생성.
    """
    name = "DocxWriterAgent"
    
    def run(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}")
        print(f"📄 [{self.name}] 실행 중...")
        print(f"{'='*80}")
        
        user_query = state.get("user_query", "")
        summary_cause = state.get("summary_cause", "")
        action_plan = state.get("summary_action_plan", "")
        
        if not user_query:
            print("⚠️ user_query가 없습니다. DOCX 생성 불가")
            return state
        
        try:
            # DOCX 파일 생성 (사고발생 경위 + 조치계획을 표에 그대로 채움)
            docx_path = create_accident_report_docx(
                user_query=user_query,
                cause_text=summary_cause,
                action_text=action_plan,
            )
            
            with open(docx_path, 'rb') as f:
                docx_bytes = f.read()
            
            state["docx_path"] = docx_path
            state["docx_bytes"] = docx_bytes
            state["route"] = "docx_complete"
            
            print(f"✅ DOCX 파일 생성 완료: {docx_path}")
            
        except Exception as e:
            print(f"❌ DocxWriterAgent 실행 실패: {e}")
            state["docx_path"] = None
        
        return state


# ========================================
# Agent Registry (Orchestrator가 사용)
# ========================================
AGENT_REGISTRY = {
    "RAGAgent": RAGAgent(),
    "ReportWriterAgent": ReportWriterAgent(),
    "DocxWriterAgent": DocxWriterAgent(),
}


def get_agent(agent_name: str):
    """Agent 이름으로 인스턴스 반환"""
    return AGENT_REGISTRY.get(agent_name)
