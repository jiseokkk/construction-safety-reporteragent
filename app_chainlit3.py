"""
Chainlit 프론트엔드 - 최종 완성 버전 v2

수정사항:
1. KeyError: 'rag' 해결
2. 사고 상세 정보에 모든 컬럼 표시
3. update_state as_node 사용
"""

import chainlit as cl
import pandas as pd
from typing import Dict, Any, Optional, List
import os

# graph 폴더에서 import
from graph.complete_langgraph_system import build_complete_graph


# ============================================================================
# 전역 설정
# ============================================================================

CSV_PATH = "data/test_preprocessing.csv"  # ← 실제 경로로 수정


# ============================================================================
# UI 헬퍼 함수들
# ============================================================================

def format_accident_table(df: pd.DataFrame) -> str:
    """사고 목록을 테이블로 포맷팅"""
    display_columns = ["발생일시", "공종(중분류)", "작업프로세스", "인적사고", "사고원인"]
    available_columns = [col for col in display_columns if col in df.columns]
    
    selected_df = df[available_columns].fillna("N/A").copy()
    selected_df.index = range(1, len(selected_df) + 1)
    selected_df.index.name = "번호"
    
    return selected_df.to_markdown(index=True)


def format_accident_details(row: Dict[str, Any]) -> str:
    """사고 상세 정보 포맷팅 - 모든 컬럼 포함"""
    def safe_get(key, default="N/A"):
        value = row.get(key, default)
        if pd.isna(value) or str(value).strip() == "":
            return default
        return str(value).strip()
    
    return f"""
## 📋 사고 상세 정보

### 🔍 기본 정보
- **ID**: {safe_get('ID')}
- **발생일시**: {safe_get('발생일시')}
- **사고인지 시간**: {safe_get('사고인지 시간')}

### 🌦️ 환경 정보
- **날씨**: {safe_get('날씨')}
- **기온**: {safe_get('기온')}
- **습도**: {safe_get('습도')}

### 🏗️ 공사 정보
- **공사종류(대분류)**: {safe_get('공사종류(대분류)')}
- **공사종류(중분류)**: {safe_get('공사종류(중분류)')}
- **공종(대분류)**: {safe_get('공종(대분류)')}
- **공종(중분류)**: {safe_get('공종(중분류)')}
- **작업프로세스**: {safe_get('작업프로세스')}

### ⚠️ 사고 정보
- **인적사고**: {safe_get('인적사고')}
- **물적사고**: {safe_get('물적사고')}
- **사고객체(대분류)**: {safe_get('사고객체(대분류)')}
- **사고객체(중분류)**: {safe_get('사고객체(중분류)')}
- **장소(대분류)**: {safe_get('장소(대분류)')}
- **장소(중분류)**: {safe_get('장소(중분류)')}

### 📝 사고 원인
{safe_get('사고원인')}
"""


def format_rag_results(docs: List) -> str:
    """RAG 검색 결과 포맷팅"""
    if not docs:
        return "검색 결과가 없습니다."
    
    result = f"## 📚 검색된 문서 ({len(docs)}개)\n\n"
    
    for idx, doc in enumerate(docs[:10], 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", f"문서 {idx}")
        content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        
        result += f"""
### [{idx}] {title}
- **출처**: {source}
- **내용**: {content_preview}

---
"""
    
    return result


# ============================================================================
# HITL 처리 함수
# ============================================================================

async def handle_accident_select(state: Dict[str, Any], graph, config) -> bool:
    """사고 선택 HITL"""
    
    sql_result = state.get("sql_result", {})
    rows = sql_result.get("rows", [])
    
    if not rows:
        await cl.Message(content="검색 결과가 없습니다.").send()
        return False
    
    # 테이블 표시
    df_result = pd.DataFrame(rows)
    table_content = format_accident_table(df_result)
    
    await cl.Message(content=f"### 📈 사고 기록 목록 (총 {len(rows)}건)\n").send()
    await cl.Message(content=f"```markdown\n{table_content}\n```").send()
    
    # Actions 생성
    actions = []
    for idx in range(len(rows)):
        actions.append(
            cl.Action(
                name=f"select_{idx}",
                value=str(idx),
                label=f"[{idx+1}] 상세 확인",
                payload={"index": idx}
            )
        )
    
    actions.append(
        cl.Action(
            name="cancel",
            value="cancel",
            label="❌ 취소",
            payload={"action": "cancel"}
        )
    )
    
    # 사용자 선택
    res = await cl.AskActionMessage(
        content="**사고를 선택하세요:**",
        actions=actions,
        timeout=300
    ).send()
    
    if res and res.get("value") != "cancel":
        selected_idx = int(res.get("value", 0))
        
        # ✅ 수정: as_node 제거, 그냥 상태만 업데이트
        new_state = {
            "accident_row": rows[selected_idx],
            "selected_accident_index": selected_idx,
            "phase": "show_accident",
            "wait_for_user": False,
        }
        # as_node 없이 업데이트
        graph.update_state(config, new_state)
        
        await cl.Message(
            content=f"✅ **[{selected_idx+1}]번 사고**를 선택했습니다."
        ).send()
        
        return True
    else:
        await cl.Message(content="작업을 취소했습니다.").send()
        return False


async def handle_show_accident(state: Dict[str, Any], graph, config) -> bool:
    """사고 상세 표시 및 다음 행동 선택 HITL"""
    
    accident_row = state.get("accident_row", {})
    
    if not accident_row:
        return False
    
    # 상세 정보 표시
    details = format_accident_details(accident_row)
    await cl.Message(content=details).send()
    
    # 다음 행동 선택
    actions = [
        cl.Action(
            name="search_guideline",
            value="rag",
            label="📚 안전 지침 검색",
            payload={"action": "rag"}
        ),
        cl.Action(
            name="generate_report",
            value="report",
            label="📝 보고서 생성",
            payload={"action": "report"}
        ),
        cl.Action(
            name="end",
            value="end",
            label="✅ 종료",
            payload={"action": "end"}
        )
    ]
    
    res = await cl.AskActionMessage(
        content="**다음 작업을 선택하세요:**",
        actions=actions,
        timeout=300
    ).send()
    
    if res:
        action = res.get("payload", {}).get("action", "end")
        
        if action == "rag":
            # ✅ 수정: as_node 제거
            new_state = {
                "user_intent": "search_only",
                "user_query": _accident_to_query(accident_row),
            }
            graph.update_state(config, new_state)
            
        elif action == "report":
            # ✅ 수정: as_node 제거
            new_state = {
                "user_intent": "generate_report",
                "user_query": _accident_to_query(accident_row),
            }
            graph.update_state(config, new_state)
        
        else:
            # 종료
            return False
        
        return action != "end"
    else:
        return False


async def handle_disambiguation(state: Dict[str, Any], graph, config) -> bool:
    """모호한 질문 명확화 HITL"""
    
    user_query = state.get("user_query", "")
    
    # 사용자에게 의도 확인
    actions = [
        cl.Action(
            name="sql",
            value="sql",
            label="🔍 사고 조회 (데이터베이스)",
            payload={"action": "sql"}
        ),
        cl.Action(
            name="guideline",
            value="guideline",
            label="📚 안전 지침 검색 (문서)",
            payload={"action": "guideline"}
        ),
        cl.Action(
            name="cancel",
            value="cancel",
            label="❌ 취소",
            payload={"action": "cancel"}
        )
    ]
    
    res = await cl.AskActionMessage(
        content=f"""**질문이 명확하지 않습니다:**
        
"{user_query}"

어떤 작업을 원하시나요?""",
        actions=actions,
        timeout=300
    ).send()
    
    if res:
        action = res.get("payload", {}).get("action", "cancel")
        
        if action == "sql":
            # 사고 조회
            new_state = {
                "user_intent": "query_sql",
            }
            graph.update_state(config, new_state)
            return True
            
        elif action == "guideline":
            # 지침 검색
            new_state = {
                "user_intent": "search_only",
            }
            graph.update_state(config, new_state)
            return True
        
        else:
            await cl.Message(content="작업을 취소했습니다.").send()
            return False
    
    return False


async def handle_rag_feedback(state: Dict[str, Any], graph, config) -> bool:
    """RAG 피드백 HITL"""
    
    docs = state.get("retrieved_docs", [])
    
    # 검색 결과 표시
    formatted = format_rag_results(docs)
    await cl.Message(content=formatted).send()
    
    # 피드백 옵션
    actions = [
        cl.Action(
            name="accept",
            value="accept",
            label="✅ 문서 확정",
            payload={"action": "accept"}
        ),
        cl.Action(
            name="retry",
            value="retry",
            label="🔁 키워드 추가 검색",
            payload={"action": "retry"}
        ),
        cl.Action(
            name="web",
            value="web",
            label="🌐 웹 검색 추가",
            payload={"action": "web"}
        ),
        cl.Action(
            name="report",
            value="report",
            label="📝 보고서 생성",
            payload={"action": "report"}
        )
    ]
    
    res = await cl.AskActionMessage(
        content="**피드백을 선택하세요:**",
        actions=actions,
        timeout=300
    ).send()
    
    if res:
        action = res.get("payload", {}).get("action", "accept")
        
        if action == "retry":
            # 키워드 추가 검색
            keyword_res = await cl.AskUserMessage(
                content="추가 검색 키워드를 입력하세요:",
                timeout=120
            ).send()
            
            if keyword_res:
                original_query = state.get("user_query", "")
                new_state = {
                    "user_query": original_query + f" {keyword_res['output']}",
                }
                graph.update_state(config, new_state)
            else:
                return False
        
        elif action == "web":
            # 웹 검색
            new_state = {
                "web_search_requested": True,
            }
            graph.update_state(config, new_state)
        
        elif action == "report":
            # 보고서 생성
            new_state = {
                "user_intent": "generate_report",
            }
            graph.update_state(config, new_state)
        
        else:  # accept
            user_intent = state.get("user_intent", "search_only")
            if user_intent == "generate_report":
                new_state = {}
                graph.update_state(config, new_state)
            else:
                return False
        
        return action != "accept" or state.get("user_intent") == "generate_report"
    
    return False


async def handle_report_approval(state: Dict[str, Any], graph, config) -> bool:
    """보고서 승인 HITL"""
    
    report_text = state.get("report_text", "")
    
    # 보고서 미리보기
    preview_length = 500
    preview = report_text[:preview_length] + "..." if len(report_text) > preview_length else report_text
    
    await cl.Message(
        content=f"## 📄 생성된 보고서\n\n{preview}"
    ).send()
    
    # DOCX 생성 여부
    actions = [
        cl.Action(
            name="create_docx",
            value="docx",
            label="📄 DOCX 파일 생성",
            payload={"action": "docx"}
        ),
        cl.Action(
            name="end",
            value="end",
            label="✅ 종료",
            payload={"action": "end"}
        )
    ]
    
    res = await cl.AskActionMessage(
        content="**DOCX 파일을 생성하시겠습니까?**",
        actions=actions,
        timeout=300
    ).send()
    
    if res and res.get("value") == "docx":
        new_state = {}
        graph.update_state(config, new_state)
        return True
    else:
        await cl.Message(content="✅ 작업을 완료했습니다.").send()
        return False


# ============================================================================
# 헬퍼 함수
# ============================================================================

def _accident_to_query(row: Dict[str, Any]) -> str:
    """사고 정보를 검색 쿼리로 변환"""
    
    query = "[사고 속성]\n"
    
    fields = {
        "발생일시": row.get("발생일시", ""),
        "공종": row.get("공종(중분류)", ""),
        "작업프로세스": row.get("작업프로세스", ""),
        "사고 유형": row.get("인적사고", ""),
        "사고객체": row.get("사고객체(중분류)", ""),
        "장소": row.get("장소(중분류)", "")
    }
    
    for key, value in fields.items():
        if value and str(value) not in ["N/A", "nan", ""]:
            query += f"{key}: {value}\n"
    
    return query


# ============================================================================
# 메인 HITL 라우터
# ============================================================================

async def route_hitl(state: Dict[str, Any], graph, config) -> bool:
    """HITL 이벤트를 적절한 핸들러로 라우팅"""
    
    phase = state.get("phase")
    
    print(f"🔀 route_hitl 호출: phase={phase}")
    
    if phase == "accident_select":
        return await handle_accident_select(state, graph, config)
    
    elif phase == "show_accident":
        return await handle_show_accident(state, graph, config)
    
    elif phase == "disambiguation":  # ✅ 추가!
        return await handle_disambiguation(state, graph, config)
    
    elif phase == "rag_feedback":
        return await handle_rag_feedback(state, graph, config)
    
    elif phase == "report_approval":
        return await handle_report_approval(state, graph, config)
    
    else:
        # 알 수 없는 phase
        print(f"⚠️  알 수 없는 phase: {phase}")
        return False


# ============================================================================
# Chainlit 이벤트 핸들러
# ============================================================================

@cl.on_chat_start
async def start():
    """채팅 시작 시 초기화"""
    
    session_id = cl.user_session.get("id")
    print(f"\n{'='*80}")
    print(f"🚀 [NEW SESSION] ID: {session_id}")
    print(f"{'='*80}\n")
    
    # CSV 로드
    try:
        if not os.path.exists(CSV_PATH):
            await cl.Message(
                content=f"❌ CSV 파일을 찾을 수 없습니다: {CSV_PATH}\n\n`app_chainlit.py`의 CSV_PATH를 수정하세요."
            ).send()
            return
        
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        
        # 날짜 파싱
        df["발생일시_parsed"] = pd.to_datetime(
            df["발생일시"].str.split().str[0],
            format="%Y-%m-%d",
            errors="coerce"
        )
        
    except Exception as e:
        await cl.Message(
            content=f"❌ CSV 로드 실패: {e}"
        ).send()
        return
    
    # 백엔드 그래프 빌드
    try:
        graph = build_complete_graph(CSV_PATH, df)
        
        # 세션에 저장
        cl.user_session.set("graph", graph)
        cl.user_session.set("df", df)
        
        # 날짜 범위
        valid_dates = df["발생일시_parsed"].dropna()
        date_info = ""
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            date_info = f"\n📅 사고 기록 날짜 범위: {min_date} ~ {max_date}"
        
        await cl.Message(
            content=f"""
# 🏗️ 건설안전 Multi-Agent HITL 시스템

안녕하세요! LangGraph 기반 시스템입니다.

✅ 시스템 준비 완료
- 사고 기록: **{len(df)}건**{date_info}
- 백엔드: `graph/complete_langgraph_system.py`
- 프론트엔드: Chainlit UI

## 💬 사용 방법

### 🔍 사고 조회
- **"8월 8일 사고 정보 알려줘"**
- **"최근 3개월 낙상 사고 찾아줘"**
- **"2025년 철근콘크리트 사고는 몇 건이야?"**

### 📝 후속 작업
1. 사고 선택 → 상세 확인
2. 안전 지침 검색 (RAG)
3. 보고서 생성 및 DOCX 다운로드

자연어로 편하게 말씀해주세요! 🙂
"""
        ).send()
    
    except Exception as e:
        await cl.Message(
            content=f"❌ 시스템 초기화 실패: {e}\n\n`graph/complete_langgraph_system.py`를 확인하세요."
        ).send()
        import traceback
        traceback.print_exc()


@cl.on_message
async def main(message: cl.Message):
    """메시지 처리 (백엔드와 통신)"""
    
    session_id = cl.user_session.get("id")
    print(f"\n{'='*80}")
    print(f"📨 [MESSAGE] Session: {session_id}")
    print(f"📨 Content: {message.content}")
    print(f"{'='*80}\n")
    
    user_input = message.content.strip()
    
    if not user_input:
        await cl.Message(content="⚠️ 메시지를 입력해주세요.").send()
        return
    
    # 백엔드 그래프 가져오기
    graph = cl.user_session.get("graph")
    
    if graph is None:
        await cl.Message(content="❌ 시스템이 초기화되지 않았습니다.").send()
        return
    
    # 실행 설정
    config = {"configurable": {"thread_id": session_id}}
    
    initial_state = {
        "user_query": user_input
    }
    
    try:
        # 백엔드 스트리밍 실행
        print(f"🔄 그래프 실행 시작...")
        
        event_count = 0
        last_event = None
        
        for event in graph.stream(initial_state, config, stream_mode="values"):
            event_count += 1
            last_event = event
            
            print(f"📦 Event #{event_count}: keys={list(event.keys())}")
            
            # 시스템 메시지 출력
            if event.get("system_message"):
                print(f"💬 System message found")
                await cl.Message(content=event["system_message"]).send()
            
            # HITL 이벤트 처리
            if event.get("wait_for_user"):
                phase = event.get("phase")
                print(f"⏸️  HITL 감지: phase={phase}")
                should_continue = await route_hitl(event, graph, config)
                
                if should_continue:
                    print(f"🔁 재실행 시작...")
                    # 백엔드 재실행
                    for new_event in graph.stream(None, config, stream_mode="values"):
                        
                        # 시스템 메시지
                        if new_event.get("system_message"):
                            await cl.Message(content=new_event["system_message"]).send()
                        
                        # 또 다른 HITL
                        if new_event.get("wait_for_user"):
                            should_continue_2 = await route_hitl(new_event, graph, config)
                            
                            if should_continue_2:
                                # 한 번 더 재실행 (최대 3단계)
                                for final_event in graph.stream(None, config, stream_mode="values"):
                                    if final_event.get("is_complete"):
                                        await cl.Message(content="✅ **작업이 완료되었습니다!**").send()
                                        break
                                    
                                    if final_event.get("wait_for_user"):
                                        await route_hitl(final_event, graph, config)
                            break
                        
                        # 완료
                        if new_event.get("is_complete"):
                            await cl.Message(content="✅ **작업이 완료되었습니다!**").send()
                            break
                
                break
            
            # 완료 확인
            if event.get("is_complete"):
                await cl.Message(content="✅ **작업이 완료되었습니다!**").send()
                break
            
            # DOCX 파일 생성 완료
            docx_path = event.get("docx_path")
            if docx_path and os.path.exists(docx_path):
                elements = [
                    cl.File(
                        name=os.path.basename(docx_path),
                        path=docx_path,
                        display="inline"
                    )
                ]
                
                await cl.Message(
                    content="✅ **DOCX 파일이 생성되었습니다!**",
                    elements=elements
                ).send()
        
        # 스트림 종료 후 마지막 이벤트 확인
        print(f"\n📊 스트림 종료: 총 {event_count}개 이벤트 처리")
        
        if last_event and last_event.get("wait_for_user"):
            print(f"⚠️  마지막 이벤트에 wait_for_user=True이지만 처리되지 않았습니다!")
            print(f"    phase={last_event.get('phase')}")
            print(f"    keys={list(last_event.keys())}")
            
            # 강제 HITL 처리
            await route_hitl(last_event, graph, config)
    
    except Exception as e:
        await cl.Message(content=f"❌ 실행 오류: {e}").send()
        import traceback
        traceback.print_exc()