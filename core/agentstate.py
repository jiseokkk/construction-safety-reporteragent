# core/agentstate.py (HITL 필드 추가 완료)
from typing import TypedDict, NotRequired, Any, Literal, Optional
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Multi-Agent 기반 건설 사고 보고서 시스템의 공용 상태.
    Orchestrator ↔ SubAgents(RAG / Summary / DOCX)가 공유.
    """

    # 1) 사용자 입력
    user_query: str                          # 사용자 질의문
    user_intent: NotRequired[Literal["search_only", "generate_report"]]  # ✅ 사용자 의도

    # 1-1) 사고 정보 (CSV에서 추출)
    accident_date: NotRequired[str]      # 사고 발생일시
    accident_type: NotRequired[str]      # 사고 유형 (끼임, 추락, 낙하 등)
    work_type: NotRequired[str]          # 공종
    work_process: NotRequired[str]       # 작업 프로세스
    accident_overview: NotRequired[str]  # 사고 개요

    # 2) RAGAgent 출력
    retrieved_docs: NotRequired[list[Document]]
    docs_text: NotRequired[str]          # 여러 문서를 합친 통합 텍스트
    sources: NotRequired[list[dict[str, Any]]]
    formatted_result: NotRequired[str]   # 가독성 좋게 포맷팅된 검색 결과
    source_references: NotRequired[list[dict[str, Any]]]  # 근거 자료 정보

    # (과거 버전 호환용 필드)
    rag_text: NotRequired[str]
    rag_sources: NotRequired[list[dict[str, Any]]]

    # 3) ReportWriterAgent 출력
    summary_cause: NotRequired[str]      # 사고발생 경위(발생원인) 요약
    summary_action_plan: NotRequired[str]  # 조치사항 및 향후조치계획
    report_text: NotRequired[str]        # 요약 합친 텍스트
    report: NotRequired[str]             # 예전 필드명 호환
    report_summary: NotRequired[str]     # 앞 200자 요약

    # 4) DOCX 출력
    docx_bytes: NotRequired[bytes]
    docx_path: NotRequired[str]

    # 5) Web 검색 관련
    web_docs: NotRequired[list[Document]]
    web_query: NotRequired[str]
    web_fallback: NotRequired[bool]
    web_error: NotRequired[str]
    web_search_count: NotRequired[int]
    web_search_completed: NotRequired[bool]  # 웹 검색 완료 플래그
    web_search_requested: NotRequired[bool]  # 웹 검색 요청 플래그

    # 6) 라우팅 / 메타
    next_agent: NotRequired[str]         # "RAGAgent" / "ReportWriterAgent" / "END"
    route: NotRequired[str]              # ex: "retrieve_complete", "report_complete", "docx_complete"
    meta: NotRequired[dict[str, Any]]

    # 7) 워크플로우 종료 여부
    is_complete: NotRequired[bool]

    # 8) STOP/WAIT 플래그
    wait_for_user: NotRequired[bool]

    # 9) 🌟 HITL 피드백 (Chainlit → Orchestrator) 🌟
    hitl_action: NotRequired[Optional[str]] # ✅ 추가: 사용자가 선택한 액션 (web_search, accept_all 등)
    hitl_payload: NotRequired[dict[str, Any]] # ✅ 추가: HITL에서 전달된 추가 데이터 (예: 재검색 키워드)