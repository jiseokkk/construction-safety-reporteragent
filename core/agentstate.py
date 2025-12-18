from typing import TypedDict, NotRequired, Any, Literal, Optional, List, Dict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Multi-Agent 기반 건설 사고 보고서 시스템의 공용 상태.
    Orchestrator ↔ SubAgents(Intent/SQL/RAG/Report)가 공유.
    """

    # =========================================================
    # 1. 사용자 입력 및 의도
    # =========================================================
    user_query: str                          # 사용자 질의문
    # ✅ Literal 확장: Orchestrator가 다루는 모든 의도 포함 (query_sql, csv_info 등)
    user_intent: NotRequired[Optional[str]]  

    # =========================================================
    # 2. 🔥 [NEW] SQL 검색 및 사고 선택 (Orchestrator 필수 필드)
    # =========================================================
    sql_executed: NotRequired[bool]          # SQL 에이전트 실행 여부 (True/False)
    sql_query_result: NotRequired[List[Dict[str, Any]]] # SQL 검색 결과 행(Row) 리스트
    selected_accident: NotRequired[Dict[str, Any]]      # 사용자가 선택한(또는 자동 선택된) 단일 사고 데이터
    needs_accident_selection: NotRequired[bool]         # UI에서 사고 선택이 필요한지 여부

    # =========================================================
    # 3. 사고 상세 정보 (CSV/선택된 사고에서 추출)
    # =========================================================
    accident_date: NotRequired[str]      # 사고 발생일시
    accident_type: NotRequired[str]      # 사고 유형 (끼임, 추락, 낙하 등)
    work_type: NotRequired[str]          # 공종
    work_process: NotRequired[str]       # 작업 프로세스
    accident_overview: NotRequired[str]  # 사고 개요
    
    # 메타 정보 (RAG 검색 필터링용)
    meta: NotRequired[Dict[str, Any]]

    # =========================================================
    # 4. RAGAgent 출력
    # =========================================================
    retrieved_docs: NotRequired[List[Document]]
    docs_text: NotRequired[str]          # 여러 문서를 합친 통합 텍스트
    sources: NotRequired[List[Dict[str, Any]]]
    formatted_result: NotRequired[str]   # 가독성 좋게 포맷팅된 검색 결과
    source_references: NotRequired[List[Dict[str, Any]]]  # 근거 자료 정보 (DocxWriter용)

    # (구버전 호환용)
    rag_text: NotRequired[str]
    rag_sources: NotRequired[List[Dict[str, Any]]]

    # =========================================================
    # 5. ReportWriterAgent / DOCX 출력
    # =========================================================
    summary_cause: NotRequired[str]      # 사고발생 경위(발생원인) 요약
    summary_action_plan: NotRequired[str]  # 조치사항 및 향후조치계획
    report_text: NotRequired[str]        # 요약 합친 텍스트
    report: NotRequired[str]             # (구버전 호환)
    report_summary: NotRequired[str]     # 앞 200자 요약

    docx_bytes: NotRequired[bytes]       # DOCX 바이너리 데이터
    docx_path: NotRequired[str]          # 생성된 DOCX 파일 경로

    # =========================================================
    # 6. Web 검색 관련
    # =========================================================
    web_docs: NotRequired[List[Document]]
    web_query: NotRequired[str]
    web_fallback: NotRequired[bool]
    web_error: NotRequired[str]
    web_search_count: NotRequired[int]
    web_search_completed: NotRequired[bool]  # 웹 검색 완료 플래그
    web_search_requested: NotRequired[bool]  # 웹 검색 요청 플래그

    # =========================================================
    # 7. 제어 흐름 (Flow Control)
    # =========================================================
    next_agent: NotRequired[str]         # 다음에 실행할 Agent 이름
    route: NotRequired[str]              # 현재 경로 상태 (로그용)
    is_complete: NotRequired[bool]       # 전체 워크플로우 종료 여부
    
    wait_for_user: NotRequired[bool]     # 🛑 실행을 멈추고 사용자 입력을 기다려야 함

    # =========================================================
    # 8. 🌟 HITL 피드백 (Chainlit ↔ Orchestrator)
    # =========================================================
    hitl_action: NotRequired[Optional[str]]    # 사용자가 UI에서 선택한 액션 (research_db, select_acc 등)
    hitl_payload: NotRequired[Dict[str, Any]]  # HITL에서 전달된 추가 데이터 (선택된 인덱스, 키워드 등)