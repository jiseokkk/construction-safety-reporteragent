# core/final_report.py (LLM Factory 적용)
from core.agentstate import AgentState
import traceback
import os

# ✅ Factory Import
from core.llm_factory import get_llm

# ✅ LangChain 관련 임포트
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === 1. 사고발생 경위 요약 생성 ===
def summarize_accident_cause(rag_output: str, user_query: str) -> str:
    """
    RAG 기반 사고 정보를 이용해 '사고발생 경위' 요약 (Qwen 사용)
    """
    
    # ✅ Qwen(Fast) 모델 사용 (보고서 초안 작성)
    llm = get_llm(mode="fast")
    
    system_template = """
당신은 건설 사고 조사 보고서를 작성하는 안전관리 담당자입니다.
아래 제공되는 RAG 문서와 사고 개요 정보를 바탕으로
'사고발생 경위(발생원인)'을 간결하게 작성하세요.

[작성 규칙]
- RAG 문서에 포함된 내용만 사용 (외부 지식/추측 금지)
- 원인과 상황이 드러나도록 4~6줄 정도로 작성
- 불필요한 수식어, 장황한 배경 설명은 줄이고 핵심만 기술
- 보고서 문체(존댓말 X, 서술형 문장)로 작성
"""
    
    user_template = """
[사고 개요]
{user_query}

[RAG 문서]
{rag_output}
"""

    try:
        print("🧠 [LLM 호출] 사고발생 경위 요약 생성 중...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # Qwen은 temperature 0 추천
        chain = prompt | llm.bind(temperature=0.0) | StrOutputParser()
        
        text = chain.invoke({
            "user_query": user_query, 
            "rag_output": rag_output
        })

        if not text:
            print("⚠️ 사고발생 경위 요약 생성 실패")
            return "RAG 문서를 바탕으로 사고발생 경위를 요약하는 데 실패했습니다."

        return text.strip()

    except Exception as e:
        print("❌ 사고발생 경위 요약 생성 중 예외 발생!")
        print(f"예외 메시지: {e}")
        return "사고발생 경위를 생성하는 과정에서 예외가 발생했습니다."


# === 2. 조치사항 및 향후조치계획 보고서 생성 ===
def generate_action_plan(rag_output: str, user_query: str, source_references: list = None) -> str:
    """
    '조치사항 및 향후조치계획' 생성 (Qwen 사용)
    """
    
    # ✅ Qwen(Fast) 모델 사용
    llm = get_llm(mode="smart")
    
    # 근거 자료 정보 구성 (기존 로직 유지)
    reference_info = ""
    if source_references and len(source_references) > 0:
        reference_info = "\n\n[참조 가능한 근거 문서 목록]\n"
        for ref in source_references:
            reference_info += f"- [문서 {ref['idx']}] {ref['filename']}"
            if ref.get('section'):
                reference_info += f" (섹션: {ref['section']})"
            reference_info += "\n"
            
            if ref.get('key_sentences'):
                reference_info += "  핵심 내용:\n"
                for sentence in ref['key_sentences'][:2]:  # 처음 2개만
                    reference_info += f"  • {sentence}\n"
    
    system_template = """
당신은 건설현장 안전관리 책임자로서,
상급자에게 제출할 '조치사항 및 향후조치계획' 보고서를 작성하는 역할입니다.

[요구사항]
- RAG 문서 내용을 근거로 사용할 것
- 문단 구조와 논리가 분명해야 함
- **각 조치사항마다 어떤 문서를 근거로 했는지 명시할 것** (예: "산업안전보건기준에 따라...")
- 한국어 보고서 문체(서술형)로 작성

[구성]
1. 즉시 조치
2. 원인 제거 조치
3. 재발 방지 대책
4. 관련 근거 요약

[분량]
- 최소 800자 이상 상세히 작성
"""

    user_template = """
아래는 사고 개요와 RAG 기반 근거 문서이다.
이를 바탕으로 '조치사항 및 향후조치계획'을 위 요구사항에 맞게 작성하라.

[사고 개요]
{user_query}

[근거가 되는 RAG 문서]
{rag_output}

{reference_info}
"""

    try:
        print("🧠 [LLM 호출] 조치사항 및 향후조치계획 생성 중...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # Qwen Context Length 고려 (필요시 max_tokens 조절)
        chain = prompt | llm.bind(temperature=0.1) | StrOutputParser()
        
        text = chain.invoke({
            "user_query": user_query,
            "rag_output": rag_output,
            "reference_info": reference_info
        })

        if not text:
            print("⚠️ 조치사항 생성 실패")
            return "조치사항을 생성하는 데 실패했습니다."

        return text.strip()

    except Exception as e:
        print("❌ 조치사항 생성 중 예외 발생!")
        print(f"예외 메시지: {e}")
        return "조치사항을 생성하는 과정에서 예외가 발생했습니다."


# === 3. (선택) LangGraph 용 Node (기존 유지) ===
def generate_accident_report_node(state: AgentState) -> AgentState:
    """
    LangGraph 호환용 노드 함수
    """
    rag_output = state.get("docs_text") or state.get("rag_text") or ""
    user_query = state.get("user_query", "")

    # ① 사고발생 경위 요약
    summary_cause = summarize_accident_cause(rag_output, user_query)

    # ② 조치사항 및 향후조치계획
    action_plan = generate_action_plan(
        rag_output,
        user_query,
        state.get("source_references", [])
    )

    # ③ 상태 업데이트
    combined = f"【사고발생 경위】\n{summary_cause}\n\n【조치사항 및 향후조치계획】\n{action_plan}"

    state["summary_cause"] = summary_cause
    state["summary_action_plan"] = action_plan
    state["report_text"] = combined
    state["report"] = combined
    state["report_summary"] = (combined[:200] + "...") if len(combined) > 200 else combined
    state["route"] = "grade_report_quality"

    print("🧾 [STATE UPDATE] 요약/조치계획 생성 완료")

    return state