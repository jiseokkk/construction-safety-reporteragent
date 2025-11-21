"""
Chainlit 기반 건설안전 Multi-Agent 시스템

✅ 핵심 기능
1. 날짜 + 사고 선택 + IntentAgent(csv_info / search_only / generate_report)
2. LangGraph 기반 Multi-Agent 실행
3. search_only → STOP → 사용자 "보고서 생성" 버튼 → generate_report 이어서 실행
"""

import chainlit as cl
import pandas as pd
from typing import Dict, Any
import os
from datetime import datetime

from core.agentstate import AgentState
from graph.workflow import graph_app
from core.llm_utils import call_llm

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


def format_accident_card(row: pd.Series, idx: int) -> str:
    """사고 정보를 카드 형식으로 포맷"""
    accident_cause = str(row.get("사고원인", "N/A"))
    if len(accident_cause) > 80:
        accident_cause = accident_cause[:80] + "..."

    return f"""
**[{idx}] 사고 정보**
- **ID**: {row.get('ID', 'N/A')}
- **발생일시**: {row.get('발생일시', 'N/A')}
- **공종**: {row.get('공종(중분류)', 'N/A')}
- **사고유형**: {row.get('인적사고', 'N/A')}
- **작업프로세스**: {row.get('작업프로세스', 'N/A')}
- **사고원인**: {accident_cause}
"""


def format_csv_details(row: pd.Series) -> str:
    """CSV 상세 정보 포맷"""
    return f"""
## 📋 사고 상세 정보

### 🔍 기본 정보
- **ID**: {row.get('ID', 'N/A')}
- **발생일시**: {row.get('발생일시', 'N/A')}
- **사고인지 시간**: {row.get('사고인지 시간', 'N/A')}

### 🌦️ 환경 정보
- **날씨**: {row.get('날씨', 'N/A')}
- **기온**: {row.get('기온', 'N/A')}
- **습도**: {row.get('습도', 'N/A')}

### 🏗️ 공사 정보
- **공사종류(대분류)**: {row.get('공사종류(대분류)', 'N/A')}
- **공사종류(중분류)**: {row.get('공사종류(중분류)', 'N/A')}
- **공종(대분류)**: {row.get('공종(대분류)', 'N/A')}
- **공종(중분류)**: {row.get('공종(중분류)', 'N/A')}
- **작업프로세스**: {row.get('작업프로세스', 'N/A')}

### ⚠️ 사고 정보
- **인적사고**: {row.get('인적사고', 'N/A')}
- **물적사고**: {row.get('물적사고', 'N/A')}
- **사고객체(대분류)**: {row.get('사고객체(대분류)', 'N/A')}
- **사고객체(중분류)**: {row.get('사고객체(중분류)', 'N/A')}
- **장소(대분류)**: {row.get('장소(대분류)', 'N/A')}
- **장소(중분류)**: {row.get('장소(중분류)', 'N/A')}

### 📝 사고 원인
{row.get('사고원인', 'N/A')}
"""


# ========================================
# Chainlit 이벤트 핸들러
# ========================================


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

### 📋 사고 정보 조회
- "8월 8일 사고 정보 알려줘"
- "2024-07-03 사고 어떤 거야?"

### 🔍 안전 지침 검색
- "8월 8일 사고 관련 지침 검색해줘"
- "관련 안전 규정 찾아줘"

### 📝 보고서 생성
- "8월 8일 사고 보고서 작성해줘"
- "DOCX 파일 만들어줘"

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

    if df is None:
        await cl.Message(content="❌ 시스템이 초기화되지 않았습니다.").send()
        return

    # ========================================
    # 1단계: IntentAgent 처리
    # ========================================
    async with cl.Step(name="🔍 의도 분석", type="tool") as step:
        step.input = user_input

        import json

        current_year = datetime.now().year

        system_prompt = f"""
당신은 건설안전 사고 관리 시스템의 IntentAgent입니다.

현재 연도: {current_year}

## 임무 1: 날짜 추출
사용자 입력에서 날짜를 추출하고 YYYY-MM-DD 형식으로 변환하세요.
연도가 없으면 {current_year}를 사용하세요.

예시:
- "7월 3일" → "{current_year}-07-03"
- "24년 8월 8일" → "2024-08-08"

## 임무 2: 사고 번호 추출 (선택사항)
사용자가 특정 번호를 언급하면 추출하세요.
예: "3번", "[3]", "세 번째" → 3

## 임무 3: 의도 파악 (매우 중요!)

### csv_info (CSV 정보만 조회)
키워드: "정보", "알려줘", "확인", "조회", "보여줘", "어떤", "뭐"
예: "8월 8일 사고 정보 알려줘", "어떤 사고야?"

### search_only (RAG 검색만)
키워드: "검색", "찾아줘", "지침", "규정"
예: "관련 지침 검색", "안전 규정 찾아줘"

### generate_report (보고서 생성)
키워드: "보고서", "작성", "문서", "DOCX", "만들어"
예: "보고서 작성", "DOCX 만들어줘"

## 판단 규칙:
1. "보고서/작성/문서/DOCX" → generate_report
2. "검색/찾아줘/지침" → search_only
3. "정보/알려줘/확인" → csv_info
4. 애매하면 → csv_info

## 출력 (JSON만):
{{
  "date": "2024-08-08",
  "accident_number": 3,
  "intent": "csv_info",
  "confidence": "high"
}}
"""

        try:
            response = await cl.make_async(call_llm)(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"사용자 입력: {user_input}"},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)

                date_str = parsed.get("date")
                accident_number = parsed.get("accident_number")
                intent = parsed.get("intent", "csv_info")
                confidence = parsed.get("confidence", "high")

                step.output = f"날짜: {date_str}, 사고번호: {accident_number}, 의도: {intent}"
            else:
                raise ValueError("JSON 파싱 실패")

        except Exception as e:
            step.output = f"파싱 실패: {e}"
            await cl.Message(content=f"❌ 입력을 이해할 수 없습니다: {e}").send()
            return

    # ========================================
    # 2단계: CSV에서 사고 검색
    # ========================================
    try:
        target_date = pd.to_datetime(date_str)
        filtered = df[df["발생일시_parsed"] == target_date]

        if filtered.empty:
            await cl.Message(
                content=f"❌ '{date_str}' 날짜에 사고 기록이 없습니다."
            ).send()
            return

    except Exception as e:
        await cl.Message(content=f"❌ 날짜 처리 오류: {e}").send()
        return

    # ========================================
    # 3단계: 사고 선택
    # ========================================
    if len(filtered) > 1:
        if accident_number is not None and 1 <= accident_number <= len(filtered):
            selected_idx = accident_number - 1
            accident_data = filtered.iloc[selected_idx]
            await cl.Message(
                content=f"✅ **[{accident_number}]번 사고**를 자동 선택했습니다."
            ).send()
        else:
            actions = []
            cards_text = f"✅ **{len(filtered)}건의 사고 기록:**\n\n"

            for idx, (_, row) in enumerate(filtered.iterrows(), 1):
                cards_text += format_accident_card(row, idx) + "\n"
                actions.append(
                    cl.Action(
                        name=f"select_{idx}",
                        value=str(idx - 1),
                        label=f"[{idx}] 선택",
                        payload={"index": idx - 1},
                    )
                )

            await cl.Message(
                content=cards_text + "\n**처리할 사고를 선택해주세요:**",
                actions=actions,
            ).send()

            res = await cl.AskActionMessage(
                content="", actions=actions, timeout=180
            ).send()

            if res:
                selected_idx = res.get("payload", {}).get("index")
                if selected_idx is None:
                    selected_idx = int(res.get("value", 0))

                accident_data = filtered.iloc[selected_idx]
                await cl.Message(
                    content=f"✅ **[{selected_idx + 1}]번 사고**를 선택하셨습니다."
                ).send()
            else:
                await cl.Message(content="⚠️ 선택이 취소되었습니다.").send()
                return
    else:
        accident_data = filtered.iloc[0]
        await cl.Message(content="✅ **1건의 사고**가 자동 선택되었습니다.").send()

    cl.user_session.set("accident_data", accident_data)

    # ========================================
    # 4단계: 의도별 처리
    # ========================================
    if intent == "csv_info":
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
                name="exit",
                value="exit",
                label="❌ 종료",
                payload={"action": "exit"},
            ),
        ]

        await cl.Message(
            content="**💬 추가 작업을 원하시나요?**", actions=actions
        ).send()

        action_response = await cl.AskActionMessage(
            content="", actions=actions, timeout=180
        ).send()

        if action_response:
            action_value = action_response.get("payload", {}).get("action") or action_response.get(
                "value"
            )
            if action_value and action_value != "exit":
                intent = action_value
            else:
                await cl.Message(content="✅ 작업을 종료합니다.").send()
                return
        else:
            await cl.Message(content="✅ 작업을 종료합니다.").send()
            return

    # ========================================
    # 5단계: Multi-Agent 실행
    # ========================================
    if intent in ["search_only", "generate_report"]:
        user_query = row_to_user_query(accident_data)

        await cl.Message(
            content=f"**📝 생성된 Query:**\n```\n{user_query}\n```\n\n**🎯 실행 모드**: {intent}"
        ).send()

        final_state = await execute_agents(user_query, intent, accident_data)
        await display_results(final_state, intent)

        # 🔁 search_only → STOP → "보고서 생성" 버튼 → generate_report
        if intent == "search_only":
            # 한 번 더 저장 (안전용)
            cl.user_session.set("last_state", final_state)

            actions = [
                cl.Action(
                    name="gen_report",
                    value="yes",
                    label="📝 보고서 생성",
                    payload={"action": "generate_report"},
                ),
                cl.Action(
                    name="exit",
                    value="no",
                    label="❌ 종료",
                    payload={"action": "exit"},
                ),
            ]

            await cl.Message(
                content="**💬 검색된 결과로 보고서를 생성하시겠습니까?**",
                actions=actions,
            ).send()

            action_response = await cl.AskActionMessage(
                content="", actions=actions, timeout=180
            ).send()

            if action_response:
                action_value = action_response.get("payload", {}).get("action") or action_response.get(
                    "value"
                )
                if action_value in ["generate_report", "yes"]:
                    await cl.Message(
                        content="📝 보고서 생성을 시작합니다..."
                    ).send()

                    # 🔑 STOP 당시 상태에서 이어서 실행
                    last_state = cl.user_session.get("last_state") or final_state
                    last_state["user_intent"] = "generate_report"
                    # ⭐ STOP 해제
                    last_state["wait_for_user"] = False

                    final_state = await continue_to_report(last_state)
                    await display_results(final_state, "generate_report")
                else:
                    await cl.Message(content="✅ 작업을 종료합니다.").send()
            else:
                await cl.Message(content="✅ 작업을 종료합니다.").send()


# ========================================
# Multi-Agent 실행 함수
# ========================================
async def execute_agents(
    user_query: str, intent: str, accident_data=None
) -> Dict[str, Any]:
    """Multi-Agent 시스템 실행"""

    mode_text = "정보 검색" if intent == "search_only" else "보고서 생성"

    async with cl.Step(name=f"🚀 {mode_text} 모드", type="run") as main_step:
        main_step.input = f"user_query: {user_query[:100]}..."

        state: AgentState = {
            "user_query": user_query,
            "user_intent": intent,
        }

        if accident_data is not None:
            state["accident_date"] = str(accident_data.get("발생일시", ""))
            state["accident_type"] = str(accident_data.get("인적사고", ""))
            state["work_type"] = str(accident_data.get("공종(중분류)", ""))
            state["work_process"] = str(accident_data.get("작업프로세스", ""))
            state["accident_overview"] = str(accident_data.get("사고원인", ""))

        final_state = await cl.make_async(graph_app.invoke)(state)

        # ⭐ STOP 상태면 세션에 저장 (나중에 이어서 사용)
        if final_state.get("wait_for_user", False):
            cl.user_session.set("last_state", final_state)
            main_step.output = "STOP 상태 → 사용자 입력 대기"
        else:
            main_step.output = "실행 완료"

        return final_state


async def continue_to_report(state: AgentState) -> Dict[str, Any]:
    """검색 후 보고서 생성 계속"""

    state["user_intent"] = "generate_report"
    # 🔑 매우 중요: STOP 상태 해제
    state["wait_for_user"] = False

    async with cl.Step(name="📝 보고서 생성 계속", type="run") as step:
        final_state = await cl.make_async(graph_app.invoke)(state)
        step.output = "보고서 생성 완료"
        return final_state


async def display_results(final_state: Dict[str, Any], intent: str):
    """결과 표시"""

    if intent == "search_only":
        formatted_result = final_state.get("formatted_result")

        if formatted_result:
            await cl.Message(
                content=f"## 🔍 검색 결과\n\n{formatted_result}"
            ).send()
        else:
            docs = final_state.get("retrieved_docs") or []
            await cl.Message(
                content=f"📊 검색된 문서 수: **{len(docs)}개**"
            ).send()
    else:
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


if __name__ == "__main__":
    pass
