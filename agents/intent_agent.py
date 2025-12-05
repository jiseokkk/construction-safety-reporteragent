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


# ======================================================================
# 1. Pydantic 모델 정의 (출력 스키마 강제)
# ======================================================================
class IntentAnalysis(BaseModel):
    """사용자 입력 분석 결과 스키마"""
    
    reasoning: str = Field(
        description="날짜 추출 근거와 키워드 분석을 포함한 사고 과정(Chain-of-Thought)."
    )
    date: Optional[str] = Field(
        description="추출된 날짜 (YYYY-MM-DD 형식). 날짜가 없거나 불명확하면 null(None).",
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
        
        # ✅ LangChain 초기화 (temperature=0으로 일관성 확보)
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=IntentAnalysis)
    
    def parse_and_decide(self, user_input: str, df: pd.DataFrame) -> Dict:
        """
        사용자 입력을 LCEL로 파싱하고 의도에 따라 처리
        """
        
        # 1. 시스템 프롬프트 (기존 로직을 LangChain 템플릿으로 변환)
        system_template = """
당신은 건설안전 사고 관리 시스템의 IntentAgent입니다.
현재 연도: {current_year}

## 임무 1: 날짜 추출
사용자 입력에서 날짜를 추출하고 YYYY-MM-DD 형식으로 변환하세요.
- "7월 3일 사고" → "2024-07-03" (연도가 없으면 현재 연도 사용)
- "24년 8월 8일" → "2024-08-08"

## 임무 2: 의도 파악 (4가지 의도)
1. csv_info: "정보", "알려줘", "세부사항" + 명확한 날짜
2. search_only: "검색", "지침", "규정", "조회" (RAG 관련)
3. generate_report: "보고서", "작성", "문서", "DOCX"
4. query_sql: "최근", "통계", "몇 건", "가장 많은", "전체" (복합 쿼리)

## 우선순위 규칙
1. "보고서", "작성" → generate_report
2. "지침", "규정" → search_only
3. 복합 쿼리 키워드 발견 시 → query_sql
4. 날짜만 명확하고 다른 키워드가 없을 때 → csv_info
5. 날짜가 없거나 애매한 경우 → query_sql

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
            print(f"\n💡 의도: query_sql (복합 쿼리). CSV 검색 생략.")
            return {
                "success": True,
                "date": date_str, 
                "intent": intent,
                "confidence": parsed.confidence,
                "accident_data": None
            }

        # 단일 사고 처리가 필요한데 날짜가 없으면 실패
        if not date_str:
            return {
                "success": False,
                "error": "날짜를 추출할 수 없습니다.",
                "intent": intent
            }
        
        # CSV에서 날짜로 검색
        try:
            target_date = pd.to_datetime(date_str)
            filtered = df[df['발생일시_parsed'] == target_date]
            
            if filtered.empty:
                return {
                    "success": False,
                    "error": f"'{date_str}' 날짜에 사고 기록이 없습니다.",
                    "intent": intent
                }
            
            # 사고 선택
            accident_data = self._select_accident(filtered)
            
            if accident_data is None and len(filtered) > 1:
                # 다중 사고 발견 (Chainlit UI 처리용)
                return {
                    "success": True, 
                    "date": date_str,
                    "intent": intent,
                    "confidence": parsed.confidence,
                    "accident_data": None 
                }
            elif accident_data is None:
                # 선택 취소
                return {
                    "success": False,
                    "error": "사고 선택이 취소되었습니다.",
                    "intent": intent
                }
            
            return {
                "success": True,
                "date": date_str,
                "intent": intent,
                "confidence": parsed.confidence,
                "accident_data": accident_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"데이터 처리 오류: {e}",
                "intent": intent
            }
    
    def _select_accident(self, filtered: pd.DataFrame) -> Optional[pd.Series]:
        """여러 사고 중 선택 (콘솔 로깅용)"""
        print(f"\n✅ {len(filtered)}건의 사고 기록을 찾았습니다:")
        print("=" * 100)
        
        for idx, (_, row) in enumerate(filtered.iterrows(), 1):
            print(f"\n[{idx}] ID: {row.get('ID', 'N/A')}")
            print(f"    발생일시: {row.get('발생일시', 'N/A')}")
            print(f"    공종: {row.get('공종(중분류)', 'N/A')}")
            print(f"    사고유형: {row.get('인적사고', 'N/A')}")
            
            accident_cause = str(row.get('사고원인', 'N/A'))
            if len(accident_cause) > 50:
                accident_cause = accident_cause[:50] + "..."
            print(f"    사고원인: {accident_cause}")
        
        print("=" * 100)
        
        if len(filtered) > 1:
            print("\n⚠️ 다중 사고 발견. Chainlit 환경에서 선택합니다.")
            return None 
        else:
            print("\n✅ 1건의 사고가 자동 선택되었습니다.")
            return filtered.iloc[0]
    
    def _default_result(self) -> Dict:
        """파싱 실패 시 기본값"""
        return {
            "success": False,
            "error": "입력을 이해할 수 없습니다.",
            "intent": "csv_info"
        }