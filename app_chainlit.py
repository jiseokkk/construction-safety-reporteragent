"""
Chainlit 기반 건설안전 Multi-Agent 시스템 - HITL 수정 버전

✅ 핵심 변경사항:
1. RAGAgent 검색 후 HITL을 app_chainlit.py에서 직접 처리
2. 비동기 함수 호출 문제 해결
3. 사용자에게 HITL UI가 제대로 표시됨
"""

import chainlit as cl
import pandas as pd
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
from langchain_core.documents import Document

from core.agentstate import AgentState
from graph.workflow import graph_app
from core.llm_utils import call_llm
from agents.intent_agent import IntentAgent 
from agents.sql_agent import CSVSQLAgent 
from agents.subagents import RAGAgent
from core.human_feedback_collector import HumanFeedbackCollector

# ========================================
# 전역 설정
# ========================================
CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv"


# ========================================
# 헬퍼 함수
# ========================================
def load_csv_data():
    """CSV 데이터 로드"""
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        # 발생일시 파싱
        df["발생일시_parsed"] = pd.to_datetime(
            df["발생일시"].str.split().str[0],
            format="%Y-%m-%d",
            errors="coerce",
        )

        return df
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return None


def row_to_user_query(row: pd.Series) -> str:
    """CSV row를 user_query로 변환"""
    query = "[사고 속성]\n"

    fields = {
        "발생일시": row.get("발생일시", "N/A"),
        "공종": row.get("공종(중분류)", "N/A"),
        "작업프로세스": row.get("작업프로세스", "N/A"),
        "사고 유형": row.get("인적사고", "N/A"),
        "사고 개요": row.get("사고원인", "N/A"),
        "사고객체(중분류)": row.get("사고객체(중분류)", "N/A"),
        "장소(중분류)": row.get("장소(중분류)", "N/A"),
    }

    for key, value in fields.items():
        if value and str(value) not in ["N/A", "nan"]:
            query += f"{key}: {value}\n"

    return query


def format_csv_details(row: pd.Series) -> str:
    """CSV 상세 정보 포맷 - 최종 수정 버전"""
    
    def safe_get(series, key, default='N/A'):
        try:
            value = series[key]  # ← 핵심!
            
            if pd.isna(value):
                return default
            
            if isinstance(value, str):
                value_stripped = value.strip()
                if value_stripped == '':
                    return default
                return value_stripped
            
            return str(value)
            
        except (KeyError, AttributeError, IndexError) as e:
            return default
    
    return f"""
## 📋 사고 상세 정보

### 🔍 기본 정보
- **ID**: {safe_get(row, 'ID')}
- **발생일시**: {safe_get(row, '발생일시')}
- **사고인지 시간**: {safe_get(row, '사고인지 시간')}

### 🌦️ 환경 정보
- **날씨**: {safe_get(row, '날씨')}
- **기온**: {safe_get(row, '기온')}
- **습도**: {safe_get(row, '습도')}

### 🏗️ 공사 정보
- **공사종류(대분류)**: {safe_get(row, '공사종류(대분류)')}
- **공사종류(중분류)**: {safe_get(row, '공사종류(중분류)')}
- **공종(대분류)**: {safe_get(row, '공종(대분류)')}
- **공종(중분류)**: {safe_get(row, '공종(중분류)')}
- **작업프로세스**: {safe_get(row, '작업프로세스')}

### ⚠️ 사고 정보
- **인적사고**: {safe_get(row, '인적사고')}
- **물적사고**: {safe_get(row, '물적사고')}
- **사고객체(대분류)**: {safe_get(row, '사고객체(대분류)')}
- **사고객체(중분류)**: {safe_get(row, '사고객체(중분류)')}
- **장소(대분류)**: {safe_get(row, '장소(대분류)')}
- **장소(중분류)**: {safe_get(row, '장소(중분류)')}

### 📝 사고 원인
{safe_get(row, '사고원인')}
"""


# ========================================
# 🔑 RAG 검색 + HITL 처리 (최종 수정 버전)
# ========================================
async def perform_rag_search_with_hitl(
    user_query: str,
    state: AgentState,
    rag_agent: RAGAgent,
    feedback_collector: HumanFeedbackCollector
) -> Dict[str, Any]:

    # 1) RAG 검색 (HITL 없이 1차 검색)
    await cl.Message(content="🔍 **관련 안전 지침 검색 중...**").send()

    try:
        docs = await cl.make_async(rag_agent.search_only)(user_query, state)

        if not docs:
            await cl.Message(content="⚠️ 관련 문서를 찾지 못했습니다.").send()
            return {
                "success": False,
                "docs": [],
                "feedback": {},
                "web_search_requested": False,
            }

        await cl.Message(
            content=f"✅ **{len(docs)}개의 관련 문서를 찾았습니다.**"
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ RAG 검색 오류: {e}").send()
        return {
            "success": False,
            "docs": [],
            "feedback": {},
            "web_search_requested": False,
        }

    # 2) HITL 루프 (최대 3번까지 재검색/수정 허용)
    max_feedback_loops = 3
    feedback_loop_count = 0
    last_feedback: Dict[str, Any] = {}

    while feedback_loop_count < max_feedback_loops:
        # 🔥 사용자 피드백 수집 (HumanFeedbackCollector)
        docs, feedback = await feedback_collector.process(
            docs=docs,
            query=user_query,
            available_dbs=rag_agent.available_dbs,
        )
        last_feedback = feedback or {}
        action = last_feedback.get("action")

        print(f"🔎 [HITL] action = {action}, feedback = {last_feedback}")

        # -------------------------------
        # 1) 웹 검색 요청 (웹 검색은 Orchestrator에서 WebSearchAgent로 처리)
        # -------------------------------
        if last_feedback.get("web_search_requested"):
            # 여기서는 플래그만 반환 → 나중에 state에 넣고 Orchestrator 호출
            return {
                "success": True,
                "docs": docs,
                "feedback": last_feedback,
                "web_search_requested": True,
            }

        # -------------------------------
        # 2) 키워드 기반 재검색 (research_keyword)
        # -------------------------------
        if action == "research_keyword":
            keywords = last_feedback.get("keywords", [])
            original_docs = last_feedback.get("original_docs", docs)

            if keywords:
                enhanced_query = user_query + "\n추가 키워드: " + ", ".join(keywords)
                await cl.Message(
                    content=f"🔁 추가 키워드로 재검색합니다: **{', '.join(keywords)}**"
                ).send()

                try:
                    new_docs = await cl.make_async(rag_agent.search_only)(
                        enhanced_query, state
                    )
                except Exception as e:
                    await cl.Message(content=f"❌ 키워드 재검색 오류: {e}").send()
                    # 실패 시 기존 문서만 사용하고 루프 종료
                    docs = original_docs
                    break

                # 기존 + 신규 문서 합치기 (너무 많아지지 않게 최대 15개)
                docs = (original_docs or []) + (new_docs or [])
                docs = docs[:15]

                feedback_loop_count += 1
                continue  # 다시 HITL로 돌아가서 새 문서 목록 보여주기

        # -------------------------------
        # 3) 다른 DB에서 재검색 (research_db)
        # -------------------------------
        if action == "research_db":
            selected_dbs = last_feedback.get("dbs", [])

            if selected_dbs:
                await cl.Message(
                    content=f"🗂️ 선택된 DB에서 재검색합니다: **{', '.join(selected_dbs)}**"
                ).send()

                try:
                    # RAGAgent의 내부 헬퍼를 그대로 활용
                    structured_query = rag_agent._build_structured_query(state)
                    new_docs = rag_agent._search_documents(
                        db_list=selected_dbs,
                        query=structured_query,
                        top_k=5,
                    )
                    docs = new_docs[:10]
                except Exception as e:
                    await cl.Message(content=f"❌ DB 재검색 오류: {e}").send()
                    # 실패 시 이전 문서 유지
                    pass

            feedback_loop_count += 1
            continue  # 다시 HITL로 돌아가서 새 문서 목록 보여주기

        # -------------------------------
        # 4) 문서 확정 (accept_all / select_partial)
        # -------------------------------
        if action in ("accept_all", "select_partial"):
            # HumanFeedbackCollector가 이미 docs를 확정된 목록으로 넘겨줌
            await cl.Message(
                content=f"✅ 선택된 문서 {len(docs)}개로 계속 진행합니다."
            ).send()
            break

        # -------------------------------
        # 5) 그 외 (no_docs, 오류 등) → 루프 종료
        # -------------------------------
        feedback_loop_count += 1
        if feedback_loop_count >= max_feedback_loops:
            await cl.Message(
                content="⚠️ 최대 HITL 반복 횟수에 도달하여 현재 문서로 진행합니다."
            ).send()
            break

    # 3) HITL 종료 → 상위 단계에서 후속 메뉴(보고서 생성/웹검색 등)로 이어짐
    return {
        "success": True,
        "docs": docs,
        "feedback": last_feedback,
        "web_search_requested": False,
    }


# ========================================
# Multi-Agent 실행 및 재개 함수
# ========================================
async def continue_to_report(state: AgentState) -> Dict[str, Any]:
    """HITL 완료 후 보고서 생성 계속"""
    
    state["user_intent"] = "generate_report"
    state["wait_for_user"] = False
    
    async with cl.Step(name="📝 보고서 생성 계속", type="run") as step:
        final_state = await cl.make_async(graph_app.invoke)(state)
        step.output = "보고서 생성 완료"
        return final_state


async def display_results(final_state: Dict[str, Any], intent: str):
    """결과 표시"""
    
    if intent == "search_only":
        docs = final_state.get("retrieved_docs") or []
        await cl.Message(
            content=f"📊 검색된 문서 수: **{len(docs)}개** (HITL 완료 후 종료)."
        ).send()
        
    else:  # generate_report
        docs = final_state.get("retrieved_docs") or []
        report_text = final_state.get("report_text", "")
        docx_path = final_state.get("docx_path")

        await cl.Message(
            content=f"""
## 📊 최종 결과

- **검색된 문서 수**: {len(docs)}개
- **보고서 텍스트 길이**: {len(report_text)} 글자
- **DOCX 파일**: {'✅ 생성됨' if docx_path else '❌ 생성 실패'}
"""
        ).send()

        if report_text:
            preview = report_text[:800] + ("..." if len(report_text) > 800 else "")
            await cl.Message(
                content=f"## 📄 보고서 미리보기\n\n```\n{preview}\n```"
            ).send()

        if docx_path and os.path.exists(docx_path):
            elements = [
                cl.File(
                    name=os.path.basename(docx_path),
                    path=docx_path,
                    display="inline",
                )
            ]
            await cl.Message(
                content="## 📥 DOCX 파일 다운로드", elements=elements
            ).send()


# ========================================
# 사고 선택 및 후속 작업 진행 함수
# ========================================
async def handle_accident_selection(
    df_result: pd.DataFrame, 
    accident_count: int, 
    current_intent: str = "list_view",
    original_intent: str = "query_sql"
):
    """사고 선택 및 후속 작업 처리"""
    
    rag_agent: RAGAgent = cl.user_session.get("rag_agent") 
    feedback_collector: HumanFeedbackCollector = cl.user_session.get("feedback_collector")
    
    # 3. 상세 정보 확인 후 후속 작업
    if current_intent == "show_detail":
        accident_data = cl.user_session.get("selected_accident_data")
        
        await cl.Message(content=format_csv_details(accident_data)).send()

        actions = [
            cl.Action(name="rag_search", value="search_only", label="🔍 관련 지침 검색", payload={"action": "search_only"}),
            cl.Action(name="gen_report", value="generate_report", label="📝 보고서 생성", payload={"action": "generate_report"}),
            cl.Action(name="back_to_list", value="back_to_list", label="⬅️ 목록으로 돌아가기", payload={"action": "back_to_list"}),
            cl.Action(name="exit", value="exit", label="❌ 종료", payload={"action": "exit"}),
        ]

        await cl.Message(
            content="**💬 추가 작업을 원하시나요?**", actions=actions
        ).send()

        res = await cl.AskActionMessage(
            content="", actions=actions, timeout=180
        ).send()

        if res:
            action_value = res.get("payload", {}).get("action") or res.get("value")
            
            if action_value == "back_to_list":
                await cl.Message(content="➡️ 사고 목록으로 돌아갑니다.").send()
                await handle_accident_selection(df_result, accident_count, current_intent="list_view", original_intent=original_intent)
                return
            
            elif action_value in ["search_only", "generate_report"]:
                user_query = row_to_user_query(accident_data)
                
                # 🔑 State 생성
                state: AgentState = {
                    "user_query": user_query,
                    "user_intent": action_value,
                    "accident_date": str(accident_data.get("발생일시", "N/A")),
                    "accident_type": str(accident_data.get("인적사고", "N/A")),
                    "work_type": str(accident_data.get("공종(중분류)", "N/A")),
                    "work_process": str(accident_data.get("작업프로세스", "N/A")),
                    "accident_overview": str(accident_data.get("사고원인", "N/A")[:200])
                }
                
                # 🔑 RAG 검색 + HITL (비동기로 직접 처리)
                rag_result = await perform_rag_search_with_hitl(
                    user_query=user_query,
                    state=state,
                    rag_agent=rag_agent,
                    feedback_collector=feedback_collector
                )
                                # ==========================================================
                # 🔥 HITL 종료 후: 무조건 후속 메뉴 제공
                # ==========================================================
                if rag_result["success"]:
                    docs = rag_result["docs"]
                    cl.user_session.set("rag_final_docs", docs)

                    actions = [
                        cl.Action(
                            name="full_report",
                            value="full_report",
                            label="📝 전체 문서로 보고서 생성",
                            payload={"action": "full_report"}
                        ),
                        cl.Action(
                            name="partial_report",
                            value="partial_report",
                            label="✂️ 일부 문서만 선택하여 보고서 생성",
                            payload={"action": "partial_report"}
                        ),
                        cl.Action(
                            name="db_research",
                            value="db_research",
                            label="🗂️ 다른 DB에서 재검색",
                            payload={"action": "db_research"}
                        ),
                        cl.Action(
                            name="web_search",
                            value="web_search",
                            label="🌐 웹 검색 추가",
                            payload={"action": "web_search"}
                        ),
                        cl.Action(
                            name="exit",
                            value="exit",
                            label="❌ 종료",
                            payload={"action": "exit"}
                        ),
                    ]

                    # -----------------------------------------------------
                    # ⚠️ [수정]: cl.Message(actions=...) 대신 AskActionMessage만 사용하여 중복 버튼 제거
                    # -----------------------------------------------------
                    res = await cl.AskActionMessage(
                        content="🔍 **HITL 완료! 다음 작업을 선택해주세요.**",
                        actions=actions,
                        timeout=180
                    ).send()
                    # -----------------------------------------------------

                    # AskActionMessage 결과에서 선택된 action value 추출 (value/payload/name 모두 처리)
                    if not res:
                        await cl.Message(content="⏹ 작업이 종료되었습니다.").send()
                        return

                    # 1) value 기반
                    choice = res.get("value")

                    # 2) payload 기반
                    if not choice:
                        choice = res.get("payload", {}).get("action")

                    # 3) name 기반 (Chainlit이 value/payload를 안 넣는 경우 대비)
                    if not choice:
                        name = res.get("name", "")
                        action_map = {
                            "full_report": "full_report",
                            "partial_report": "partial_report",
                            "db_research": "db_research",
                            "web_search": "web_search",
                            "exit": "exit",
                        }
                        if name in action_map:
                            choice = action_map[name]

                    if not choice:
                        await cl.Message(content="⏹ 선택이 취소되어 작업을 종료합니다.").send()
                        return


                    # === 선택 분기 ===
                    if choice == "full_report":
                        state["retrieved_docs"] = docs
                        await cl.Message(content="📝 전체 문서로 보고서를 생성합니다...").send()
                        # continue_to_report 호출 -> Orchestrator (ReportWriterAgent) 실행
                        final_state = await continue_to_report(state)
                        await display_results(final_state, "generate_report")
                        return

                    if choice == "partial_report":
                        await cl.Message(content="✂️ 일부 문서 선택 UI는 아직 준비중입니다.").send()
                        return

                    if choice == "db_research":
                        await cl.Message(content="🗂️ 다른 DB에서 재검색 기능은 추후 확장 예정입니다.").send()
                        return

                    if choice == "web_search":
                        await cl.Message(content="🌐 웹 검색을 진행합니다...").send()
                        state["web_search_requested"] = True
                        # continue_to_report 호출 -> Orchestrator (WebSearchAgent) 실행
                        final_state = await continue_to_report(state)
                        await display_results(final_state, "generate_report")
                        return

                    if choice == "exit":
                        await cl.Message(content="👋 작업을 종료합니다.").send()
                        return
                    
                    # ⚠️ 만약 위의 분기가 아닌 다른 선택(예: 시간 초과)이면 여기서 종료
                    await cl.Message(content="⏹ 작업이 종료되었습니다.").send()
                    return


                if not rag_result["success"]:
                    await cl.Message(content="❌ RAG 검색에 실패했습니다.").send()
                    return
                
                # State에 검색 결과 저장 (이 로직은 HITL 루프가 재검색 요청 없이 끝났을 때만 타야 함)
                state["retrieved_docs"] = rag_result["docs"]
                
                # search_only면 여기서 종료
                if action_value == "search_only":
                    await cl.Message(content="✅ 검색이 완료되었습니다.").send()
                    return
                
                # generate_report면 계속 진행 (이 로직은 위의 HITL 후속 메뉴가 없을 때의 fallback)
                if action_value == "generate_report":
                    # 웹 검색 요청 처리
                    if rag_result.get("web_search_requested"):
                        state["web_search_requested"] = True
                    
                    # 보고서 생성 확인
                    confirm_actions = [
                        cl.Action(name="confirm_yes", value="yes", label="✅ 예, 보고서 생성", payload={"action": "yes"}),
                        cl.Action(name="confirm_no", value="no", label="❌ 아니오, 취소", payload={"action": "no"}),
                    ]
                    
                    await cl.Message(
                        content="**📝 보고서 생성을 진행하시겠습니까?**",
                        actions=confirm_actions
                    ).send()
                    
                    confirm_res = await cl.AskActionMessage(
                        content="", actions=confirm_actions, timeout=60
                    ).send()
                    
                    if confirm_res and confirm_res.get("payload", {}).get("action") == "yes":
                        await cl.Message(content="📝 **보고서 생성을 시작합니다...**").send()
                        
                        # 보고서 생성 (LangGraph 재개)
                        final_state = await continue_to_report(state)
                        await display_results(final_state, "generate_report")
                    else:
                        await cl.Message(content="✅ 작업을 취소합니다.").send()
                
                return

            else:  # exit
                await cl.Message(content="✅ 작업을 종료합니다.").send()
                return
        
        else:
            await cl.Message(content="✅ 작업을 종료합니다.").send()
            return

    
    # 1. 목록 제시 및 선택
    elif current_intent == "list_view":
        display_columns = ['발생일시', '공종(중분류)', '작업프로세스', '인적사고', '사고원인']
        available_columns = [col for col in display_columns if col in df_result.columns]
        
        selected_df = df_result[available_columns].fillna('N/A').copy()
        
        # ✅ 번호를 1부터 시작하도록 수정
        selected_df.index = range(1, len(selected_df) + 1)
        selected_df.index.name = "번호"
        
        actions = []
        
        table_content = selected_df.to_markdown(index=True) 
        
        await cl.Message(
            content=f"### 📈 사고 기록 목록 (총 {accident_count}건)\n"
        ).send()

        await cl.Message(
            content=f"```markdown\n{table_content}\n```"
        ).send()

        for idx in range(accident_count):
            actions.append(
                cl.Action(
                    name=f"show_detail_{idx+1}",
                    value=str(idx),
                    label=f"[{idx+1}] 상세 확인",
                    payload={"index": idx, "action": "show_detail"}
                )
            )
        
        actions.append(cl.Action(name="exit_list", value="exit", label="❌ 목록 취소/종료", payload={"action": "exit"}))

        await cl.Message(
            content=f"**후속 작업을 위해 목록에서 사고 번호 (1~{accident_count})를 선택하거나 목록을 취소해주세요:**",
            actions=actions
        ).send()
        
        res = await cl.AskActionMessage(
            content="", actions=actions, timeout=300 
        ).send()

        if res:
            # ✅ 여러 방법으로 action 추출
            action_type = res.get("payload", {}).get("action")
            if not action_type:
                action_type = res.get("value")
            
            if action_type == "exit":
                await cl.Message(content="✅ 작업을 종료합니다.").send()
                return
                
            elif action_type == "show_detail":
                # ✅ payload의 index를 우선 사용
                selected_idx = res.get("payload", {}).get("index")
                if selected_idx is None:
                    selected_idx = int(res.get("value", 0))
                else:
                    selected_idx = int(selected_idx)
                
                # 🔥 FIXED: SQL 결과(df_result)가 아니라
                # 전체 CSV(df)에서 ID로 다시 조회해서 full row 사용
                df_full = cl.user_session.get("df")
                if df_full is not None and "ID" in df_full.columns and "ID" in df_result.columns:
                    selected_row = df_result.iloc[selected_idx]
                    accident_id = selected_row["ID"]
                    mask = df_full["ID"] == accident_id
                    if mask.any():
                        accident_data = df_full[mask].iloc[0]
                    else:
                        # 혹시 못 찾으면 fallback으로 df_result row 사용
                        accident_data = selected_row
                else:
                    # df_full이 없거나 ID 컬럼이 없으면 기존 방식 유지
                    accident_data = df_result.iloc[selected_idx]

                # 세션에 저장
                cl.user_session.set("selected_accident_data", accident_data)
                
                # 상세 정보 확인 단계로 이동 (재귀 호출)
                await cl.Message(content=f"🔍 **[{selected_idx + 1}]번 사고**의 상세 정보를 확인합니다.").send()
                await handle_accident_selection(df_result, accident_count, current_intent="show_detail", original_intent=original_intent)
                return
                
            else:
                await cl.Message(content="⚠️ 선택 시간이 초과되어 작업을 종료합니다.").send()
                return


@cl.on_chat_start
async def start():
    """채팅 시작 시 초기화"""

    df = load_csv_data()

    if df is None:
        await cl.Message(
            content="❌ 시스템 초기화 실패: CSV 파일을 로드할 수 없습니다."
        ).send()
        return

    cl.user_session.set("df", df)

    # CSVSQLAgent 및 IntentAgent 초기화
    try:
        sql_agent = CSVSQLAgent(CSV_PATH)
        cl.user_session.set("sql_agent", sql_agent)
    except Exception as e:
        await cl.Message(
            content=f"❌ SQL Agent 초기화 실패: {e}"
        ).send()
        return
        
    intent_agent = IntentAgent()
    cl.user_session.set("intent_agent", intent_agent)

    # RAGAgent 및 HumanFeedbackCollector 초기화
    try:
        rag_agent = RAGAgent() 
        feedback_collector = HumanFeedbackCollector()
        
        cl.user_session.set("rag_agent", rag_agent)
        cl.user_session.set("feedback_collector", feedback_collector)
        
    except Exception as e:
        await cl.Message(
            content=f"❌ RAG/Feedback 시스템 초기화 실패: {e}"
        ).send()
        return

    valid_dates = df["발생일시_parsed"].dropna()
    date_info = ""
    if len(valid_dates) > 0:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        date_info = f"\n📅 사고 기록 날짜 범위: {min_date} ~ {max_date}"

    await cl.Message(
        content=f"""
# 🏗️ 건설안전 Intelligent Multi-Agent 시스템

안녕하세요! 건설 사고 정보 조회 및 보고서 생성을 도와드립니다.

✅ 시스템 준비 완료
- 사고 기록: **{len(df)}건**{date_info}

## 💬 사용 방법

### 🔍 사고 기록 조회
- **"8월 8일 사고 정보 알려줘"**
- **"최근 3개월 낙상 사고 찾아줘"**
- **"2024년 철근콘크리트 사고는 몇 건이야?"**

### 📝 후속 작업
- 조회된 사고를 선택하여 관련 지침 검색 또는 보고서 생성을 할 수 있습니다.

자연어로 편하게 말씀해주세요! 🙂
"""
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """메시지 수신 시 처리"""

    user_input = message.content.strip()

    if not user_input:
        await cl.Message(content="⚠️ 메시지를 입력해주세요.").send()
        return

    df = cl.user_session.get("df")
    sql_agent: CSVSQLAgent = cl.user_session.get("sql_agent")
    intent_agent: IntentAgent = cl.user_session.get("intent_agent")

    if df is None or sql_agent is None or intent_agent is None:
        await cl.Message(content="❌ 시스템이 초기화되지 않았습니다.").send()
        return

    # 1단계: IntentAgent 처리
    intent_result = None
    
    async with cl.Step(name="🔍 의도 분석", type="tool") as step:
        step.input = user_input
        
        try:
            intent_result = await cl.make_async(intent_agent.parse_and_decide)(user_input, df)
            
            intent = intent_result.get("intent", "query_sql") 
            date_str = intent_result.get("date")
            
            step.output = f"의도: {intent}, 날짜: {date_str}"
            
        except Exception as e:
            step.output = f"파싱 오류: {e}"
            await cl.Message(content=f"❌ 의도 분석 중 오류 발생: {e}").send()
            return
            
        if not intent_result["success"] and intent != "query_sql":
             await cl.Message(content=f"❌ {intent_result.get('error')}").send()
             return

    # 2단계: SQL 쿼리 실행
    await cl.Message(content=f"**🎯 실행 모드**: **SQL 쿼리 조회**").send()
    async with cl.Step(name="📊 SQL 쿼리 실행", type="tool") as step:
        step.input = user_input
        
        sql_result = await cl.make_async(sql_agent.query)(user_input)
        
        if sql_result["success"]:
            df_result = pd.DataFrame(sql_result["rows"])
            accident_count = len(df_result)
            
            step.output = f"SQL 성공. {accident_count}건 검색됨."

            await cl.Message(
                content=f"## ✅ SQL 쿼리 결과\n\n**📝 생성된 SQL:**\n```sql\n{sql_result['generated_sql']}\n```\n\n**📊 검색된 사고 수:** **{accident_count}건**"
            ).send()
            
            if accident_count > 0:
                await handle_accident_selection(df_result, accident_count, current_intent="list_view", original_intent=intent)
                return 
            else:
                await cl.Message(content="✅ 검색 결과가 없습니다. 작업을 종료합니다.").send()
                return
        else:
            step.output = f"SQL 실패: {sql_result['error']}"
            await cl.Message(
                content=f"❌ SQL 쿼리 실행 실패: {sql_result['error']}\n\n**생성된 SQL:**\n```sql\n{sql_result.get('generated_sql', 'N/A')}\n```"
            ).send()
            return