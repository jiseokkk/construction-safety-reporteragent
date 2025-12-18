"""
IntentAgent (LangChain LCEL & Pydantic 적용 버전)
사용자의 자연어 입력을 분석하고 대화를 관리하는 Agent

역할:
1. 자연어에서 날짜 추출 (Pydantic 강제)
2. 사용자 의도 파악 (csv_info / search_only / generate_report / query_sql)
3. CSV 정보 직접 출력 (csv_info 모드)
4. 대화형 추가 작업 제안
"""

from typing import Dict, Optional, Literal, Any
import json
import os
from datetime import datetime
import pandas as pd

# ✅ LangChain & Pydantic 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from core.llm_factory import get_llm


# ======================================================================
# 1. Pydantic 모델 정의 (출력 스키마 강제)
# ======================================================================
class IntentAnalysis(BaseModel):
    """사용자 입력 분석 결과 스키마"""
    
    reasoning: str = Field(
        description="기간/특정일 여부와 조건(공종, 유형 등) 유무를 분석한 사고 과정."
    )
    date: Optional[str] = Field(
        description="추출된 날짜 정보 (YYYY-MM-DD 또는 YYYY-MM). 날짜가 없거나 불명확하면 null(None).",
        default=None
    )
    intent: Literal["csv_info", "search_only", "generate_report", "query_sql"] = Field(
        description="파악된 사용자 의도."
    )
    confidence: Literal["high", "low"] = Field(
        description="분석 결과의 확신도."
    )


# ======================================================================
# 2. IntentAgent 클래스
# ======================================================================
class IntentAgent:
    """자연어 입력을 처리하고 의도를 파악하는 Agent (LCEL 기반)"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.last_query = None
        
        # 🔄 [변경] Qwen(Fast) 모델 사용 (비용 절감 & 속도 향상)
        self.llm = get_llm(mode="smart") 
        self.parser = PydanticOutputParser(pydantic_object=IntentAnalysis)
    
    def parse_and_decide(self, user_input: str, df: pd.DataFrame) -> Dict:
        """
        사용자 입력을 LCEL로 파싱하고 의도에 따라 처리
        """
        
        # 1. 시스템 프롬프트 (조건 검색과 단순 조회를 구분하도록 강화)
        system_template = """
당신은 건설안전 사고 관리 시스템의 IntentAgent입니다.
현재 연도: {current_year}

## 임무 1: 날짜/기간 추출
- "11월 4일 사고" → "2024-11-04" (특정일)
- "11월 사고", "11월에 발생한" → "2024-11" (기간/월)
- "2023년 사고" → "2023" (기간/년)

## 임무 2: 의도 파악 (우선순위가 매우 중요함)

🔥 **[우선순위 1] SQL 검색 (query_sql)**
- **날짜 + 조건**이 결합된 경우 (예: "11월 철근콘크리트 사고", "작년 추락 사고")
- **특정 월(Month)이나 연도(Year)** 전체를 포괄적으로 물어보는 경우 (예: "11월 사고 보여줘")
- 통계나 집계를 물어보는 경우 (예: "가장 많이 발생한", "몇 건이야")

✅ **[우선순위 2] 상세 조회 (csv_info)**
- 오직 **특정 날짜(YYYY-MM-DD)** 하루의 사고만 물어볼 때 (예: "11월 4일 사고 알려줘")
- 다른 조건(공종, 사고유형 등) 없이 날짜만 명확할 때

🔍 **[우선순위 3] 지침 검색 (search_only)**
- "지침", "규정", "법규", "검색" 키워드 포함 (단, 사고 조회가 아닐 때)

📝 **[우선순위 4] 보고서 (generate_report)**
- "보고서", "작성", "문서", "DOCX"

반드시 아래 형식을 준수하여 JSON으로 응답해야 합니다:
{format_instructions}
"""
        
        # 2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{user_input}")
        ])
        
        # 3. LCEL 체인 생성
        chain = prompt | self.llm | self.parser
        
        try:
            # 4. 체인 실행 (동기 호출 invoke 사용)
            result: IntentAnalysis = chain.invoke({
                "current_year": self.current_year,
                "user_input": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            print("\n🧾 IntentAgent 분석 결과 (Pydantic):")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Date: {result.date}")
            print(f"   Intent: {result.intent}")
            
            # 5. 비즈니스 로직 처리 (기존 로직 유지)
            return self._process_intent(result, df)
            
        except Exception as e:
            print(f"❌ IntentAgent LCEL 오류: {e}")
            return self._default_result()
    
    def _process_intent(self, parsed: IntentAnalysis, df: pd.DataFrame) -> Dict:
        """의도에 따라 처리 (Pydantic 객체 사용)"""
        
        date_str = parsed.date
        intent = parsed.intent
        
        # 🔑 query_sql 의도는 사고 데이터 검색을 건너뛰고 바로 반환
        if intent == "query_sql":
            print(f"\n💡 의도: query_sql (기간/조건 검색). CSV 직접 검색 생략.")
            return {
                "success": True,
                "date": date_str, 
                "intent": intent,
                "confidence": parsed.confidence,
                "accident_data": None
            }

        # 단일 사고 처리가 필요한데 날짜가 없으면 실패 -> SQL로 유도
        if not date_str:
            return {
                "success": False,
                "error": "날짜를 추출할 수 없습니다.",
                "intent": "query_sql" # 날짜 없으면 SQL로 fallback
            }
        
        # CSV에서 날짜로 검색 (csv_info 로직)
        try:
            target_date = pd.to_datetime(date_str)
            filtered = df[df['발생일시_parsed'] == target_date]
            
            if filtered.empty:
                # 해당 날짜에 없으면 SQL로 넘겨서 비슷한 거라도 찾게 함
                return {
                    "success": True,
                    "date": date_str,
                    "intent": "query_sql",
                    "accident_data": None
                }
            
            # ✅ [수정됨] 다중 사고 발견 시 'candidates' 반환 (Orchestrator ASK_USER용)
            if len(filtered) > 1:
                print(f"⚠️ 다중 사고 발견: {len(filtered)}건 -> 목록 반환")
                return {
                    "success": True, 
                    "date": date_str,
                    "intent": intent,
                    "confidence": parsed.confidence,
                    "accident_data": None,
                    "candidates": filtered.to_dict(orient="records") # 후보 목록 반환
                }
            
            # 단일 사고 발견
            accident_data = self._select_accident(filtered)
            if accident_data is None: 
                # _select_accident 내부에서 다중 처리 시 None 반환할 수 있음
                return {
                    "success": True, 
                    "date": date_str,
                    "intent": intent,
                    "confidence": parsed.confidence,
                    "accident_data": None,
                    "candidates": filtered.to_dict(orient="records")
                }

            return {
                "success": True,
                "date": date_str,
                "intent": intent,
                "confidence": parsed.confidence,
                "accident_data": accident_data.to_dict() # Series -> Dict
            }
            
        except Exception as e:
            # 날짜 파싱 오류 등 발생 시 SQL로 안전하게 넘김
            return {
                "success": True,
                "date": date_str,
                "intent": "query_sql",
                "accident_data": None
            }
    
    def _select_accident(self, filtered: pd.DataFrame) -> Optional[pd.Series]:
        """여러 사고 중 선택 (콘솔 로깅용)"""
        print(f"\n✅ {len(filtered)}건의 사고 기록을 찾았습니다.")
        if len(filtered) > 1:
            print("⚠️ 다중 사고 발견. 목록을 반환합니다.")
            return None 
        else:
            print("✅ 1건의 사고가 자동 선택되었습니다.")
            return filtered.iloc[0]
    
    def _default_result(self) -> Dict:
        """파싱 실패 시 기본값"""
        return {
            "success": False,
            "error": "입력을 이해할 수 없습니다.",
            "intent": "query_sql" # 모르면 SQL로
        }