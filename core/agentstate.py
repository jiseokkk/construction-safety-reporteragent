# core/agentstate.py
from typing import TypedDict, NotRequired, Any
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Multi-Agent 기반 건설 사고 보고서 시스템의 공용 상태.
    Orchestrator ↔ SubAgents(RAG / Summary / DOCX)가 공유.
    """

    # 1) 사용자 입력
    user_query: str                      # Streamlit에서 to_human_query()로 만든 질의문

    # 2) RAGAgent 출력
    retrieved_docs: NotRequired[list[Document]]
    docs_text: NotRequired[str]          # 여러 문서를 합친 통합 텍스트
    sources: NotRequired[list[dict[str, Any]]]

    # (과거 버전 호환용 필드)
    rag_text: NotRequired[str]
    rag_sources: NotRequired[list[dict[str, Any]]]

    # 3) ReportWriterAgent 출력 (요약 + 조치 계획)
    summary_cause: NotRequired[str]              # 사고발생 경위(발생원인) 요약
    summary_action_plan: NotRequired[str]        # 조치사항 및 향후조치계획 (긴 보고서 스타일)
    report_text: NotRequired[str]                # summary_cause + summary_action_plan 합친 텍스트 (호환용)
    report: NotRequired[str]                     # 예전 필드명 호환
    report_summary: NotRequired[str]             # 앞 200자 요약

    # 4) DocxWriterAgent 출력
    docx_bytes: NotRequired[bytes]
    docx_path: NotRequired[str]

    # 5) 라우팅 / 메타
    next_agent: NotRequired[str]         # "RAGAgent" / "ReportWriterAgent" / "DocxWriterAgent" / "END"
    route: NotRequired[str]              # ex: "retrieve_complete", "report_complete", "docx_complete"
    meta: NotRequired[dict[str, Any]]

    # 6) 워크플로우 종료 여부
    is_complete: NotRequired[bool]
