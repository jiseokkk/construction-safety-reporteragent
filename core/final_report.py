# core/final_report.py (LangChain LCEL 적용 버전)
from core.agentstate import AgentState
import traceback
import os

# ✅ LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain LLM 초기화 (함수 밖에서 한 번만 초기화하여 재사용 권장)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3, # 기존 설정 유지
    api_key=os.getenv("OPENAI_API_KEY")
)

# === 1. 사고발생 경위 요약 생성 ===
def summarize_accident_cause(rag_output: str, user_query: str) -> str:
    """
    RAG 기반 사고 정보를 이용해 '사고발생 경위(발생원인)'을
    4~6줄 정도로 간단·명확하게 요약. (LangChain LCEL 적용)
    """
    
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
        
        # 🔥 LCEL Chain 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # temperature=0.2로 설정된 별도 체인 사용 (기존 로직 준수)
        chain = prompt | llm.bind(temperature=0.2, top_p=0.9, max_tokens=1600) | StrOutputParser()
        
        # 실행
        text = chain.invoke({
            "user_query": user_query, 
            "rag_output": rag_output
        })

        if not text or "⚠️" in text:
            print("⚠️ 사고발생 경위 요약 생성 실패:", text)
            return "RAG 문서를 바탕으로 사고발생 경위를 요약하는 데 실패했습니다."

        return text.strip()

    except Exception as e:
        print("❌ 사고발생 경위 요약 생성 중 예외 발생!")
        print(f"예외 타입: {type(e).__name__}")
        print(f"예외 메시지: {e}")
        print(traceback.format_exc())
        return "사고발생 경위를 생성하는 과정에서 예외가 발생했습니다."


# === 2. 조치사항 및 향후조치계획 보고서 생성 ===
def generate_action_plan(rag_output: str, user_query: str, source_references: list = None) -> str:
    """
    '조치사항 및 향후조치계획'을 상사 보고용 고품질 텍스트로 생성.
    (LangChain LCEL 적용)
    """
    
    # ✅ 근거 자료 정보 구성 (기존 로직 유지)
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
상급자(부서장 또는 발주처)에 제출할 '조치사항 및 향후조치계획' 보고서를 작성하는 역할입니다.

[전반적인 요구사항]
- 실제 보고서 문서에 그대로 삽입할 수 있는 수준의 완성도를 갖출 것
- RAG 문서에 포함된 내용만 근거로 사용 (외부 지식 추가 금지)
- 문단 구조와 논리가 분명해야 함 (단순 bullet 나열 금지)
- ✅ **각 조치사항마다 어떤 문서를 근거로 했는지 명시할 것**
  예: "산업안전보건기준에 관한 규칙에 따르면..."
  예: "교량공사 안전작업지침(문서 1)에 명시된 바와 같이..."
- 한국어 보고서 문체(서술형)로 작성할 것

[구성]
1. 즉시 조치 (Immediate Action)
   - 각 조치의 근거 문서 명시
   
2. 원인 제거 조치 (Corrective Action)
   - 각 조치의 근거 문서 명시
   
3. 재발 방지 대책 (Preventive Action)
   - 각 조치의 근거 문서 명시
   
4. 관련 근거 요약
   - 주요 참조 문서 및 조항 정리

[근거 명시 예시]
"교량공사(라멘교) 안전보건작업지침(문서 1)에 따르면, 높이 4미터를 초과하는 경우 수평연결재를 설치하도록 규정하고 있다. 따라서..."

[분량]
- 최소 800자 이상, 가능하면 1200~1800자 내외로 충분히 상세히 작성
- 각 항목은 하나 이상의 문단으로 구성
"""

    user_template = """
아래는 사고 개요와 RAG 기반 근거 문서이다.
이를 바탕으로 '조치사항 및 향후조치계획'을 위 요구사항에 맞게 작성하라.

**중요: 각 조치사항을 제시할 때, 반드시 어떤 문서를 근거로 했는지 명시하라.**

[사고 개요]
{user_query}

[근거가 되는 RAG 문서]
{rag_output}

{reference_info}
"""

    try:
        print("🧠 [LLM 호출] 조치사항 및 향후조치계획 생성 중 (근거 명시 포함)...")
        
        # 🔥 LCEL Chain 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # temperature=0.3 설정 (기존 로직 준수)
        chain = prompt | llm.bind(temperature=0.3, top_p=0.9, max_tokens=4000) | StrOutputParser()
        
        # 실행
        text = chain.invoke({
            "user_query": user_query,
            "rag_output": rag_output,
            "reference_info": reference_info
        })

        if not text or "⚠️" in text:
            print("⚠️ 조치사항 및 향후조치계획 생성 실패:", text)
            return "RAG 문서를 바탕으로 조치사항 및 향후조치계획을 생성하는 데 실패했습니다."

        return text.strip()

    except Exception as e:
        print("❌ 조치사항 및 향후조치계획 생성 중 예외 발생!")
        print(f"예외 타입: {type(e).__name__}")
        print(f"예외 메시지: {e}")
        print(traceback.format_exc())
        return "조치사항 및 향후조치계획을 생성하는 과정에서 예외가 발생했습니다."


# === 3. (선택) LangGraph 용 Node - 호환용 (기존 유지) ===
def generate_accident_report_node(state: AgentState) -> AgentState:
    """
    LangGraph에서 호출되는 보고서 생성 노드.
    - summary_cause
    - summary_action_plan
    을 생성하고, report_text에 합쳐둔다.
    """
    rag_output = state.get("docs_text") or state.get("rag_text") or ""
    user_query = state.get("user_query", "")

    # ① 사고발생 경위 요약
    summary_cause = summarize_accident_cause(rag_output, user_query)

    # ② 조치사항 및 향후조치계획
    action_plan = generate_action_plan(
        rag_output,
        user_query,
        state.get("source_references", [])     # ⭐ 근거자료 전달
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