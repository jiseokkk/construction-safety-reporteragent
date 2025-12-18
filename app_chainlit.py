"""
Chainlit 기반 건설안전 Multi-Agent 시스템 - Fully Orchestrated Version
✅ 통합 기능 명세:
1. 기존 기능 완벽 유지: 상세 보기(Detail View), 테이블 출력, 목록으로 돌아가기(Back)
2. 개선 기능 적용: SQL 결과 페이지네이션(10개씩), Payload 에러 수정
3. 로직 개선: HITL 문서 확정 시 보고서 모드 자동 전환, 무조건 결과 출력
4. 상세 보기 시 N/A 문제 해결: ID로 전체 데이터 조회
"""

import chainlit as cl
import pandas as pd 
from typing import Dict, Any, Optional, List
import os

# 💡 core/agentstate
from core.agentstate import AgentState 
# 💡 graph/workflow
from graph.workflow import graph_app 
from agents.intent_agent import IntentAgent
from agents.sql_agent import CSVSQLAgent
from agents.subagents import RAGAgent 
from core.human_feedback_collector import HumanFeedbackCollector

# ========================================
# 전역 설정
# ========================================
CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv" 

# ========================================
# 헬퍼 함수 (기존 로직 유지)
# ========================================
def load_csv_data():
    """CSV 데이터 로드"""
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df["발생일시_parsed"] = pd.to_datetime(
            df["발생일시"].str.split().str[0], format="%Y-%m-%d", errors="coerce"
        )
        return df
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return None

def row_to_user_query(row: dict) -> str:
    """선택된 사고 데이터를 자연어 쿼리 텍스트로 변환"""
    query = "[사고 속성]\n"
    fields = ["발생일시", "공종(중분류)", "작업프로세스", "인적사고", "사고원인", "사고객체(중분류)", "장소(중분류)"]
    for key in fields:
        val = row.get(key, "N/A")
        if val and str(val) not in ["N/A", "nan"]:
            query += f"{key}: {val}\n"
    return query
# ✅ [신규 추가] CSV 데이터를 보고서 양식(AgentState) 필드로 매핑하는 함수
def map_csv_to_state(row: dict) -> dict:
    """선택된 사고 데이터를 AgentState의 보고서 필드 포맷으로 변환"""
    
    def get_val(key, default="-"):
        val = row.get(key)
        if val is None or str(val).lower() in ['nan', 'n/a', 'null', '']:
            return default
        return str(val).strip()

    # 정보 조합
    weather_str = f"{get_val('날씨')}"
    if get_val('기온') != "-": weather_str += f", 기온: {get_val('기온')}"
    if get_val('습도') != "-": weather_str += f", 습도: {get_val('습도')}"

    loc_detail = get_val('장소(중분류)')
    if get_val('장소(대분류)') != "-":
        loc_detail = f"{get_val('장소(대분류)')} > {loc_detail}"

    return {
        # 보고서 필수 필드
        "accident_date": get_val('발생일시'),
        "weather_condition": weather_str,
        "project_name": f"{get_val('공사종류(중분류)')} 현장",
        "site_address": loc_detail,
        "accident_location_detail": get_val('작업프로세스'),
        "accident_type": get_val('인적사고'),
        "casualties": get_val('인적사고'),
        "equipment_damage": get_val('물적사고'),
        "structural_loss": get_val('물적사고'),
        "accident_overview": get_val('사고원인'),
        
        # 메타 정보
        "work_type": get_val('공종(중분류)'),
        "work_process": get_val('작업프로세스'),
        
        # 기본값 설정 (빈칸 방지)
        "damage_amount": "(조사 필요)",
        "construction_delay": "(조사 필요)",
        "safety_plan_status": "해당 (안전관리계획서 검토 필요)",
        "report_date": datetime.now().strftime("%Y년 %m월 %d일"),
        "reporter_name": "AI 안전 관리자"
    }

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

async def display_results(final_state: AgentState):
    """최종 결과 표시 (다운로드/미리보기 보장)"""
    docs = final_state.get("retrieved_docs") or []
    report_text = final_state.get("report_text", "")
    docx_path = final_state.get("docx_path")
    
    # 보고서가 없으면 검색 건수만 표시
    if not report_text and not docx_path:
        await cl.Message(content=f"📊 검색된 문서 수: **{len(docs)}개** (작업 완료).").send()
        return

    # 보고서 또는 파일이 있으면 결과 출력
    await cl.Message(
        content=f"""## 📊 최종 결과
- **검색된 문서 수**: {len(docs)}개
- **보고서 생성**: {'✅ 성공' if report_text else '❌ 없음'}
- **파일 생성**: {'✅ 성공' if docx_path else '❌ 없음'}"""
    ).send()

    if report_text:
        preview = report_text[:800] + ("..." if len(report_text) > 800 else "")
        await cl.Message(content=f"## 📄 보고서 미리보기\n\n```\n{preview}\n```").send()

    if docx_path and os.path.exists(docx_path):
        elements = [cl.File(name=os.path.basename(docx_path), path=docx_path, display="inline")]
        await cl.Message(content="## 📥 보고서 다운로드", elements=elements).send()


# ========================================
# 🔥 [핵심] 통합 워크플로우 루프 핸들러
# ========================================
async def run_orchestrator_loop(state: AgentState):
    feedback_collector: HumanFeedbackCollector = cl.user_session.get("feedback_collector")
    MAX_LOOPS = 15
    loop_count = 0
    
    await cl.Message(content="🔄 **AI 에이전트가 작업을 시작합니다...**").send()

    while loop_count < MAX_LOOPS:
        loop_count += 1
        
        # 1. Graph 실행 준비: 이전 단계에서 결정된 다음 에이전트 이름을 저장
        prev_agent_name = state.get('next_agent')
        
        # 2. Graph 실행
        async with cl.Step(name=f"Step {loop_count}", type="run") as step:
            step.input = f"Intent: {state.get('user_intent')}, Next: {state.get('next_agent')}"
            state = await graph_app.ainvoke(state)
            step.output = f"Wait: {state.get('wait_for_user')}, Complete: {state.get('is_complete')}"

        # 3. 🔥 [추가] WebSearchAgent 실행 후 요약 결과 출력
        # WebSearchAgent가 실행되었고 (prev_agent_name), 상태에 요약 결과가 남아있다면 출력
        if prev_agent_name == "WebSearchAgent" and state.get("web_search_summary"):
            summary = state.pop("web_search_summary") # 출력 후 state에서 제거
            
            # 사용자에게 웹 검색 결과 출력
            await cl.Message(
                content=f"""
## 🌐 웹 검색 결과 요약
{summary}

---
"""
            ).send()
        
        # 4. 종료 조건 확인
        if state.get("is_complete"):
            await display_results(state)
            break

        # 5. 🛑 사용자 입력 대기 (wait_for_user=True)
        if state.get("wait_for_user"):
            
            # ==================================================================
            # [CASE A] SQL 결과 목록 선택 (Pagination + 상세 보기)
            # ==================================================================
            if state.get("sql_query_result") and not state.get("selected_accident"):
                
                rows = state["sql_query_result"]
                total_count = len(rows)
                
                # A-1. 전체 목록 테이블 표시 (최초 1회)
                if loop_count == 1 or not state.get("table_shown"):
                    df_view = pd.DataFrame(rows)
                    cols = ["발생일시", "공종(중분류)", "인적사고", "사고원인"]
                    display_cols = [c for c in cols if c in df_view.columns]
                    display_df = df_view[display_cols].fillna("-")
                    display_df.index = range(1, total_count + 1)
                    
                    await cl.Message(content=f"### 📈 검색된 사고 목록 (총 {total_count}건)").send()
                    await cl.Message(content=f"```markdown\n{display_df.to_markdown()}\n```").send()
                    state["table_shown"] = True
                
                # A-2. 페이지네이션 루프 (목록 <-> 상세 보기 이동)
                page = 0
                ITEMS_PER_PAGE = 10 
                
                while True:
                    # --- 버튼 렌더링 ---
                    start_idx = page * ITEMS_PER_PAGE
                    end_idx = min((page + 1) * ITEMS_PER_PAGE, total_count)
                    current_batch = rows[start_idx:end_idx]
                    
                    msg_content = f"**분석할 사고를 선택해주세요 ({start_idx + 1}~{end_idx} / 총 {total_count}건):**"
                    actions = []
                    
                    for i, row in enumerate(current_batch):
                        real_idx = start_idx + i
                        actions.append(cl.Action(
                            name="select_acc", 
                            value=str(real_idx), 
                            label=f"[{real_idx + 1}]번 선택", 
                            payload={"value": str(real_idx)} # ✅ Payload 추가
                        ))
                    
                    if page > 0:
                        actions.append(cl.Action(name="prev_page", value="prev", label="⬅️ 이전", payload={"value": "prev"}))
                    if end_idx < total_count:
                        actions.append(cl.Action(name="next_page", value="next", label="➡️ 다음", payload={"value": "next"}))
                        
                    actions.append(cl.Action(name="cancel", value="cancel", label="❌ 취소", payload={"value": "cancel"}))

                    res = await cl.AskActionMessage(content=msg_content, actions=actions).send()
                    
                    # --- 값 추출 (Payload 우선) ---
                    if res:
                        val = res.get("payload", {}).get("value") or res.get("value")
                    else:
                        val = "cancel"

                    # --- 동작 처리 ---
                    if not res or val == "cancel":
                        await cl.Message(content="작업이 취소되었습니다.").send()
                        state["is_complete"] = True
                        return # 루프 및 함수 전체 종료

                    elif val == "next":
                        page += 1
                        continue # 다음 페이지
                    elif val == "prev":
                        page -= 1
                        continue # 이전 페이지
                    
                    else:
                        # --- [상세 보기 진입 - N/A 해결 로직] ---
                        sel_idx = int(val)
                        limited_row = rows[sel_idx] # SQL 결과 (일부 컬럼)
                        
                        # 🔥 전체 데이터(df)에서 ID로 다시 조회하여 완전한 정보 가져오기
                        full_df = cl.user_session.get("df")
                        target_id = limited_row.get("ID")
                        full_row_series = None # Series 객체 저장용
                        
                        if full_df is not None and target_id:
                            matched = full_df[full_df["ID"] == target_id]
                            if not matched.empty:
                                full_row_series = matched.iloc[0] # Series 객체 반환
                        
                        # 찾지 못했으면 SQL 결과라도 사용 (Series로 변환)
                        if full_row_series is None:
                            full_row_series = pd.Series(limited_row)
                        
                        # 1. 상세 정보 출력 (지정해주신 함수 사용)
                        await cl.Message(content=format_csv_details(full_row_series)).send()
                        
                        # 2. 후속 작업 질문 (기존 기능 복구)
                        detail_actions = [
                            cl.Action(name="rag", value="search_only", label="🔍 관련 지침 검색", payload={"value": "search_only"}),
                            cl.Action(name="report", value="generate_report", label="📝 보고서 생성", payload={"value": "generate_report"}),
                            cl.Action(name="back", value="back", label="⬅️ 목록으로 돌아가기", payload={"value": "back"})
                        ]
                        
                        sub_res = await cl.AskActionMessage(content="**💬 이 사고로 어떤 작업을 진행할까요?**", actions=detail_actions).send()
                        
                        sub_val = sub_res.get("payload", {}).get("value") if sub_res else "back"
                        
                        if sub_val == "back":
                            await cl.Message(content="목록으로 돌아갑니다.").send()
                            continue # 다시 목록 루프로 (while True 재시작)
                        
                        else:
                            # 3. 최종 확정 -> Graph 재개
                            state["selected_accident"] = full_row_series.to_dict() # dict로 저장
                            state["user_intent"] = sub_val 
                            state["user_query"] = row_to_user_query(full_row_series.to_dict())
                            state["wait_for_user"] = False

                            
                            
                            intent_label = "지침 검색" if sub_val == "search_only" else "보고서 생성"
                            await cl.Message(content=f"✅ **[{sel_idx+1}]번 사고**에 대해 **{intent_label}**을 시작합니다.").send()
                            break # 내부 while 종료 -> Main Loop 재개 (Graph 실행)

            # ==================================================================
            # [CASE B] RAG/Web 검색 결과 피드백 (HITL)
            # ==================================================================
            # WebSearchAgent는 검색 완료 후 wait_for_user=True를 설정하며 retrieved_docs가 존재함.
            elif state.get("retrieved_docs"):
                await cl.Message(content="🙋 **관련 문서를 확인해주세요.** (HITL)").send()
                
                # docs에는 '필터링된' 문서 리스트가 담겨옵니다 (select_partial 시)
                docs, feedback = await feedback_collector.process(
                    docs=state["retrieved_docs"],
                    query=state["user_query"]
                )
                
                # 🔥 [CRITICAL FIX] 필터링된 문서를 State에 반영!
                state["retrieved_docs"] = docs 
                
                # 선택된 근거자료 반영
                if feedback.get("source_references"):
                    state["source_references"] = feedback["source_references"]

                action = feedback.get("action", "accept_all")
                state["hitl_action"] = action
                state["hitl_payload"] = feedback
                state["wait_for_user"] = False 
                
                # ✅ [핵심 기능] 문서 확정 시 -> 보고서 모드로 자동 전환!
                if action in ["accept_all", "select_partial"]:
                    state["user_intent"] = "generate_report"
                    await cl.Message(content="✅ 문서가 확정되었습니다. 보고서 작성을 진행합니다.").send()

                # 메시지 표시
                action_map = {
                    "research_keyword": "🔄 키워드 추가 검색을 진행합니다.",
                    "research_db": "🔄 DB를 변경하여 검색합니다.",
                    "web_search": "🌐 웹 검색을 시도합니다.",
                    "accept_all": "📝 보고서 작성을 시작합니다.",
                    "select_partial": "📝 선택된 문서로 보고서 작성을 시작합니다.",
                    "exit": "종료합니다."
                }
                
                if action not in ["accept_all", "select_partial"]:
                    msg = action_map.get(action, "확인되었습니다.")
                    await cl.Message(content=msg).send()
            
            else:
                # 예외 상황 처리
                await cl.Message(content="⚠️ 입력이 필요하지만 처리할 수 없는 상태입니다. 종료합니다.").send()
                break

    if loop_count >= MAX_LOOPS:
        await cl.Message(content="⚠️ 최대 작업 횟수 초과로 종료됩니다.").send()


# ========================================
# Chainlit 이벤트 핸들러
# ========================================

@cl.on_chat_start
async def start():
    """초기화"""
    df = load_csv_data()
    if df is None:
        await cl.Message(content="❌ CSV 로드 실패: 경로를 확인해주세요.").send()
        return
    cl.user_session.set("df", df)

    try:
        sql_agent = CSVSQLAgent(CSV_PATH)
        cl.user_session.set("sql_agent", sql_agent)
        cl.user_session.set("intent_agent", IntentAgent())
        
        tmp_rag = RAGAgent()
        fb_collector = HumanFeedbackCollector(available_dbs=tmp_rag.available_dbs)
        cl.user_session.set("feedback_collector", fb_collector)
        cl.user_session.set("available_dbs", tmp_rag.available_dbs)
    except Exception as e:
        await cl.Message(content=f"❌ 초기화 실패: {e}").send()
        return

    valid_dates = df["발생일시_parsed"].dropna()
    date_info = f"\n📅 데이터 날짜: {valid_dates.min().date()} ~ {valid_dates.max().date()}" if len(valid_dates) > 0 else ""

    await cl.Message(content=f"""
# 🏗️ 건설안전 AI 에이전트 (Fully Orchestrated)

안녕하세요! **Orchestrator**가 탑재된 지능형 에이전트입니다.
제가 스스로 판단하여 SQL 검색, 지침 검색, 보고서 작성을 수행합니다.

✅ **준비 완료**: {len(df)}건의 사고 데이터 {date_info}

### 💡 이렇게 물어보세요
- "최근 떨어짐 사고 알려줘" 
- "2025년 5월 1일 사고 조회해줘"
""").send()

@cl.on_message
async def main(message: cl.Message):
    """메시지 핸들러"""
    user_input = message.content.strip()
    if not user_input: return

    # 초기 State 설정
    initial_state: AgentState = {
        "user_query": user_input,
        "user_intent": None,           # Orchestrator가 채움
        "sql_executed": False,
        "sql_query_result": [],
        "selected_accident": None,
        "retrieved_docs": [],
        "hitl_action": None,
        "wait_for_user": False,
        "is_complete": False,
        "report_text": "",
        "docx_path": None,
        "table_shown": False 
    }
    
    # 통합 루프 실행
    await run_orchestrator_loop(initial_state)