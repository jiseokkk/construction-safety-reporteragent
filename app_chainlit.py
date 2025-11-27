"""
Chainlit 기반 건설안전 Multi-Agent 시스템 - LangGraph Orchestrator 중심 버전

✅ 최종 해결:
1. RAGAgent 직접 호출 및 HITL 루프 로직을 app_chainlit.py에서 제거.
2. LangGraph (workflow.py)를 호출하고, 'wait_for_user' 상태를 통해 HITL을 처리하는 루프 구현.
3. RAGAgent 인스턴스 대신 DB 목록만 저장하여 프론트엔드/백엔드 분리 강화.
4. [오류 해결] graph_app.invoke 대신 **await graph_app.ainvoke**를 사용하여 비동기 오류를 해결.
"""

import chainlit as cl
import pandas as pd 
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
from langchain_core.documents import Document

# 💡 core/agentstate는 그대로 사용
from core.agentstate import AgentState 
# 💡 graph/workflow에서 LangGraph 앱을 가져옴
from graph.workflow import graph_app 
from core.llm_utils import call_llm
from agents.intent_agent import IntentAgent
from agents.sql_agent import CSVSQLAgent
# RAGAgent는 이제 Orchestrator가 호출하지만, DB 목록 정보 추출을 위해 필요
from agents.subagents import RAGAgent 
from core.human_feedback_collector import HumanFeedbackCollector

# ========================================
# 전역 설정
# ========================================
# ⚠️ 주의: CSV_PATH는 시스템 환경에 맞게 수정해주세요
CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv" 


# ========================================
# 헬퍼 함수 (변경 없음)
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
    """CSV row를 user_query로 변환 (기존 로직 유지)"""
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
    """CSV 상세 정보 포맷 (기존 로직 유지)"""
    
    def safe_get(series, key, default="N/A"):
        try:
            value = series[key]

            if pd.isna(value):
                return default

            if isinstance(value, str):
                value_stripped = value.strip()
                if value_stripped == "":
                    return default
                return value_stripped

            return str(value)

        except (KeyError, AttributeError, IndexError):
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
# ❌ 제거된 함수: perform_rag_search_with_hitl
# ❌ 제거된 함수: continue_to_report
# ========================================

async def display_results(final_state: Dict[str, Any], intent: str):
    """결과 표시 (기존 로직 유지)"""

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
# 🔑 사고 선택 및 후속 작업 진행 함수 (핵심 수정)
# ========================================
async def handle_accident_selection(
    df_result: pd.DataFrame,
    accident_count: int,
    current_intent: str = "list_view",
    original_intent: str = "query_sql",
):
    """사고 선택 및 후속 작업 처리"""

    feedback_collector: HumanFeedbackCollector = cl.user_session.get("feedback_collector")
    # available_dbs는 HITL UI 구성에 필요하지만, process 호출 시 인수로 전달하지 않음 (self.available_dbs 사용)
    available_dbs: List[str] = cl.user_session.get("available_dbs") 

    # 3. 상세 정보 확인 후 후속 작업
    if current_intent == "show_detail":
        accident_data = cl.user_session.get("selected_accident_data")

        await cl.Message(content=format_csv_details(accident_data)).send()

        actions = [
            cl.Action(
                name="rag_search",
                value="search_only",
                label="🔍 관련 지침 검색",
                payload={"action": "search_only"},
            ),
            cl.Action(
                name="gen_report",
                value="generate_report",
                label="📝 보고서 생성",
                payload={"action": "generate_report"},
            ),
            cl.Action(
                name="back_to_list",
                value="back_to_list",
                label="⬅️ 목록으로 돌아가기",
                payload={"action": "back_to_list"},
            ),
            cl.Action(
                name="exit", value="exit", label="❌ 종료", payload={"action": "exit"}
            ),
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
                await handle_accident_selection(
                    df_result,
                    accident_count,
                    current_intent="list_view",
                    original_intent=original_intent,
                )
                return

            elif action_value in ["search_only", "generate_report"]:
                user_query = row_to_user_query(accident_data)

                # 🔑 State 생성 및 초기 설정
                state: AgentState = {
                    "user_query": user_query,
                    "user_intent": action_value,
                    "accident_date": str(accident_data.get("발생일시", "N/A")),
                    "accident_type": str(accident_data.get("인적사고", "N/A")),
                    "work_type": str(accident_data.get("공종(중분류)", "N/A")),
                    "work_process": str(accident_data.get("작업프로세스", "N/A")),
                    "accident_overview": str(
                        accident_data.get("사고원인", "N/A")[:200]
                    ),
                    "wait_for_user": False,
                    "is_complete": False,
                    "hitl_action": None, 
                    "hitl_payload": {},
                    "retrieved_docs": [],
                }

                # ==========================================================
                # 🔥 LangGraph Orchestrator 호출 및 HITL 루프 (핵심 로직)
                # ==========================================================
                max_loops = 10 
                loop_count = 0
                
                await cl.Message(content="🔄 **Orchestrator 워크플로우를 시작합니다...**").send()
                
                while not state.get("is_complete", False) and loop_count < max_loops:
                    loop_count += 1
                    
                    # 1. LangGraph 호출 (LangGraph 내부에서 RAG/WebSearch/Report 실행)
                    async with cl.Step(name=f"워크플로우 실행 {loop_count}", type="run") as step:
                        step.input = f"HITL 액션: {state.get('hitl_action') or '초기 실행'}"
                        
                        # 💡 오류 해결: graph_app.invoke 대신 graph_app.ainvoke 사용
                        final_state = await graph_app.ainvoke(state) 
                        
                        state = final_state # 상태 업데이트
                        step.output = f"상태: is_complete={state.get('is_complete')}, wait_for_user={state.get('wait_for_user')}"

                    # 2. ⛔ LangGraph가 HITL을 요청했을 때 (일시 중지)
                    if state.get("wait_for_user", False):
                        await cl.Message(content="---").send()
                        await cl.Message(content="🙋 **사용자 검토(HITL)가 필요합니다.** 관련 문서를 확인하고 피드백을 주세요.").send()
                        
                        docs_to_review = state.get("retrieved_docs", [])
                        
                        # HumanFeedbackCollector를 사용하여 UI 표시 및 피드백 수집
                        docs, feedback = await feedback_collector.process(
                            docs=docs_to_review,
                            query=state.get("user_query", ""),
                            # available_dbs=available_dbs, 👈 제거됨
                        )
                        
                        # 3. ➡️ 피드백을 상태에 반영하고 루프 재시작 (LangGraph 재개)
                        state["hitl_action"] = feedback.get("action", "accept_all")
                        state["hitl_payload"] = feedback
                        state["retrieved_docs"] = docs # 사용자가 선택한 최종 문서 목록
                        state["wait_for_user"] = False # 플래그 해제 -> LangGraph 재개
                        state["source_references"] = feedback.get("source_references", [])
                        
                        continue # while 루프 재시작 (LangGraph 재호출)

                    # 4. ✅ LangGraph가 최종 완료를 알렸을 때
                    elif state.get("is_complete", False):
                        break
                    
                    # 5. ⚠️ (선택 사항) 예상치 못한 종료 또는 루프 탈출 조건
                    elif not state.get("next_agent") and not state.get("is_complete") and loop_count > 1:
                        await cl.Message(content="⚠️ Orchestrator가 다음 단계를 결정하지 못하고 종료됩니다.").send()
                        state["is_complete"] = True # 강제 종료
                        break

                # ==========================================================
                # 🔥 HITL 루프 종료 후: 최종 결과 표시
                # ==========================================================
                if state.get("is_complete", False):
                    await display_results(state, state.get("user_intent"))
                elif loop_count >= max_loops:
                    await cl.Message(content="⚠️ 최대 워크플로우 실행 횟수에 도달하여 강제 종료됩니다.").send()
                else:
                    await cl.Message(content="⏹ 작업이 종료되었습니다.").send()
                
                return

            else:  # exit
                await cl.Message(content="✅ 작업을 종료합니다.").send()
                return

        else:
            await cl.Message(content="✅ 작업을 종료합니다.").send()
            return

    # 1. 목록 제시 및 선택 (기존 로직 유지)
    elif current_intent == "list_view":
        display_columns = ["발생일시", "공종(중분류)", "작업프로세스", "인적사고", "사고원인"]
        available_columns = [col for col in display_columns if col in df_result.columns]

        selected_df = df_result[available_columns].fillna("N/A").copy()

        selected_df.index = range(1, len(selected_df) + 1)
        selected_df.index.name = "번호"

        actions: List[cl.Action] = []

        table_content = selected_df.to_markdown(index=True)

        await cl.Message(
            content=f"### 📈 사고 기록 목록 (총 {accident_count}건)\n"
        ).send()

        await cl.Message(content=f"```markdown\n{table_content}\n```").send()

        for idx in range(accident_count):
            actions.append(
                cl.Action(
                    name=f"show_detail_{idx+1}",
                    value=str(idx),
                    label=f"[{idx+1}] 상세 확인",
                    payload={"index": idx, "action": "show_detail"},
                )
            )

        actions.append(
            cl.Action(
                name="exit_list",
                value="exit",
                label="❌ 목록 취소/종료",
                payload={"action": "exit"},
            )
        )

        await cl.Message(
            content=f"**후속 작업을 위해 목록에서 사고 번호 (1~{accident_count})를 선택하거나 목록을 취소해주세요:**",
            actions=actions,
        ).send()

        res = await cl.AskActionMessage(
            content="", actions=actions, timeout=300
        ).send()

        if res:
            action_type = res.get("payload", {}).get("action")
            if not action_type:
                action_type = res.get("value")

            if action_type == "exit":
                await cl.Message(content="✅ 작업을 종료합니다.").send()
                return
            
            elif action_type == "show_detail":
                selected_idx = res.get("payload", {}).get("index")
                if selected_idx is None:
                    selected_idx = int(res.get("value", 0))
                else:
                    selected_idx = int(selected_idx)

                df_full = cl.user_session.get("df")
                if (
                    df_full is not None
                    and "ID" in df_full.columns
                    and "ID" in df_result.columns
                ):
                    selected_row = df_result.iloc[selected_idx]
                    accident_id = selected_row["ID"]
                    mask = df_full["ID"] == accident_id
                    if mask.any():
                        accident_data = df_full[mask].iloc[0]
                    else:
                        accident_data = selected_row
                else:
                    accident_data = df_result.iloc[selected_idx]

                cl.user_session.set("selected_accident_data", accident_data)

                await cl.Message(
                    content=f"🔍 **[{selected_idx + 1}]번 사고**의 상세 정보를 확인합니다."
                ).send()
                await handle_accident_selection(
                    df_result,
                    accident_count,
                    current_intent="show_detail",
                    original_intent=original_intent,
                )
                return
            
            else:
                await cl.Message(
                    content="⚠️ 선택 시간이 초과되어 작업을 종료합니다."
                ).send()
                return


@cl.on_chat_start
async def start():
    """채팅 시작 시 초기화 (DB 목록 추출 로직으로 수정)"""

    df = load_csv_data()

    if df is None:
        await cl.Message(
            content="❌ 시스템 초기화 실패: CSV 파일을 로드할 수 없습니다."
        ).send()
        return

    cl.user_session.set("df", df)

    # 1. CSVSQLAgent 및 IntentAgent 초기화 (유지)
    try:
        sql_agent = CSVSQLAgent(CSV_PATH)
        cl.user_session.set("sql_agent", sql_agent)
    except Exception as e:
        await cl.Message(content=f"❌ SQL Agent 초기화 실패: {e}").send()
        return

    intent_agent = IntentAgent()
    cl.user_session.set("intent_agent", intent_agent)

    # 2. 🔑 RAG/Feedback 시스템 초기화 (RAG Agent 인스턴스 제거, DB 목록 추출)
    try:
        # DB 목록 정보 추출을 위한 임시 RAG Agent 인스턴스 생성
        rag_agent_for_info = RAGAgent()
        available_dbs = rag_agent_for_info.available_dbs 

        # HumanFeedbackCollector 초기화 시 DB 목록 직접 전달
        feedback_collector = HumanFeedbackCollector(available_dbs=available_dbs)

        cl.user_session.set("feedback_collector", feedback_collector)
        cl.user_session.set("available_dbs", available_dbs) # DB 목록만 세션에 저장

    except Exception as e:
        await cl.Message(
            content=f"❌ RAG/Feedback 시스템 초기화 실패: {e}"
        ).send()
        return

    # ... (기존 환영 메시지 로직 유지) ...
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
    """메시지 수신 시 처리 (기존 로직 유지)"""

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

    # 1단계: IntentAgent 처리 (유지)
    intent_result = None
    async with cl.Step(name="🔍 의도 분석", type="tool") as step:
        step.input = user_input

        try:
            intent_result = await cl.make_async(intent_agent.parse_and_decide)(
                user_input, df
            )

            intent = intent_result.get("intent", "query_sql")
            date_str = intent_result.get("date")

            step.output = f"의도: {intent}, 날짜: {date_str}"

        except Exception as e:
            step.output = f"파싱 오류: {e}"
            await cl.Message(
                content=f"❌ 의도 분석 중 오류 발생: {e}"
            ).send()
            return

        if not intent_result["success"] and intent != "query_sql":
            await cl.Message(content=f"❌ {intent_result.get('error')}").send()
            return

    # 2단계: SQL 쿼리 실행 (유지)
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
                # 3단계: 사고 선택 및 LangGraph 호출 루프 진입
                await handle_accident_selection(
                    df_result,
                    accident_count,
                    current_intent="list_view",
                    original_intent=intent,
                )
                return
            else:
                await cl.Message(
                    content="✅ 검색 결과가 없습니다. 작업을 종료합니다."
                ).send()
                return
        else:
            step.output = f"SQL 실패: {sql_result['error']}"
            await cl.Message(
                content=f"❌ SQL 쿼리 실행 실패: {sql_result['error']}\n\n**생성된 SQL:**\n```sql\n{sql_result.get('generated_sql', 'N/A')}\n```"
            ).send()
            return