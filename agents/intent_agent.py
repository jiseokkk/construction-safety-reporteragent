
"""
IntentAgent
사용자의 자연어 입력을 분석하고 대화를 관리하는 Agent

역할:
1. 자연어에서 날짜 추출
2. 사용자 의도 파악 (csv_info / search_only / generate_report / query_sql)
3. CSV 정보 직접 출력 (csv_info 모드)
4. 대화형 추가 작업 제안
"""

from typing import Dict, Optional, Literal
from core.llm_utils import call_llm
import json
from datetime import datetime
import pandas as pd


class IntentAgent:
    """자연어 입력을 처리하고 의도를 파악하는 Agent"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.last_query = None  # 이전 쿼리 저장 (문맥 파악용)
    
    def parse_and_decide(self, user_input: str, df: pd.DataFrame) -> Dict:
        """
        사용자 입력을 파싱하고 의도 파악
        
        Returns:
            {
                "date": "2024-07-03",
                "intent": "csv_info" | "search_only" | "generate_report" | "query_sql",
                "confidence": "high" | "low",
                "accident_data": DataFrame row or None,
                "action": "csv_display" | "rag_search" | "full_report" | "sql_query"
            }
        """
        
        system_prompt = f"""
당신은 건설안전 사고 관리 시스템의 IntentAgent입니다.

현재 연도: {self.current_year}

## 임무 1: 날짜 추출
사용자 입력에서 날짜를 추출하고 YYYY-MM-DD 형식으로 변환하세요.

예시:
- "7월 3일 사고" → "2024-07-03"
- "24년 8월 8일" → "2024-08-08"
- "2024-06-03" → "2024-06-03"

연도가 없으면 {self.current_year}를 사용하세요.

## 임무 2: 의도 파악 (4가지 의도)

**1. csv_info (단일 사고의 CSV 정보 조회)**
- 키워드: "정보", "알려줘", "어떤 사고", "사고 내용", "세부사항"
- 날짜가 명확하게 추출되었고, 복합 쿼리 키워드가 없을 때
- 예: "8월 8일 사고 정보 알려줘"

**2. search_only (RAG 검색만)**
- 키워드: "검색", "찾아줘", "관련 지침", "안전 규정", "조회" (RAG 관련)
- 예: "관련 지침 검색해줘"

**3. generate_report (전체 보고서 생성)**
- 키워드: "보고서 작성", "문서 만들어", "리포트", "DOCX"
- 예: "보고서 작성해줘"

**4. query_sql (복합 쿼리 또는 통계)**
- 키워드: "최근", "가장 많은", "몇 건", "통계", "전체", "사고 찾아줘" (날짜 유무 관계 없음)
- 예: "최근 3개월 낙상 사고 찾아줘", "2024년 7월 가장 많은 사고 유형은?"

## 임무 3: 우선순위
1. "보고서", "작성" → "generate_report"
2. "지침", "규정" → "search_only"
3. 복합 쿼리 키워드 발견 시 → **"query_sql"**
4. 날짜만 명확하고 다른 키워드가 없을 때 → "csv_info"
5. 날짜가 없거나 애매한 모든 나머지 경우 → **"query_sql"**

## 출력 형식

<thinking>
1) 날짜 추출 과정
2) 키워드 분석
3) 의도 판단 근거
</thinking>

<output>
{{
  "date": "2024-07-03",
  "intent": "csv_info",
  "confidence": "high"
}}
</output>

규칙:
- intent는 반드시 "csv_info", "search_only", "generate_report", "query_sql" 중 하나
- confidence는 "high" 또는 "low"
"""
        
        user_message = f"""
사용자 입력: {user_input}

위 입력을 분석하여 날짜와 의도를 파악하세요.
"""
        
        try:
            response = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            print("\n🧾 IntentAgent LLM 응답:")
            print(response)
            
            # JSON 추출
            parsed = self._extract_json(response)
            
            if parsed:
                print(f"\n✅ 파싱 결과:")
                print(f"   날짜: {parsed.get('date')}")
                print(f"   의도: {parsed.get('intent')}")
                print(f"   확신도: {parsed.get('confidence')}")
                
                # CSV 검색 및 처리
                result = self._process_intent(parsed, df)
                return result
            else:
                print("⚠️ JSON 파싱 실패 - 기본값 사용")
                return self._default_result()
                
        except Exception as e:
            print(f"❌ IntentAgent 오류: {e}")
            return self._default_result()
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """LLM 응답에서 JSON 추출"""
        try:
            # <output> 태그 내부 추출
            if "<output>" in text and "</output>" in text:
                start = text.index("<output>") + len("<output>")
                end = text.index("</output>")
                json_str = text[start:end].strip()
                return json.loads(json_str)
        except:
            pass
        
        try:
            # <o> 태그 내부 추출
            if "<o>" in text and "</o>" in text:
                start = text.index("<o>") + len("<o>")
                end = text.index("</o>")
                json_str = text[start:end].strip()
                return json.loads(json_str)
        except:
            pass
        
        try:
            # 첫 { ~ 마지막 } 추출
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except:
            pass
        
        return None
    
    def _process_intent(self, parsed: Dict, df: pd.DataFrame) -> Dict:
        """의도에 따라 처리"""
        date_str = parsed.get("date")
        intent = parsed.get("intent", "csv_info")
        
        # 🔑 query_sql 의도는 사고 데이터 검색을 건너뛰고 바로 반환
        if intent == "query_sql":
            print(f"\n💡 의도: query_sql (복합 쿼리). CSV 검색 생략.")
            return {
                "success": True,
                "date": date_str, 
                "intent": intent,
                "confidence": parsed.get("confidence", "high"),
                "accident_data": None # SQL Agent가 처리하므로 None
            }

        # 단일 사고 처리가 필요한데 날짜가 없으면 실패
        if not date_str:
            return {
                "success": False,
                "error": "날짜를 추출할 수 없습니다.",
                "intent": intent
            }
        
        # CSV에서 날짜로 검색 (csv_info, search_only, generate_report만 이 로직을 탐)
        try:
            target_date = pd.to_datetime(date_str)
            filtered = df[df['발생일시_parsed'] == target_date]
            
            if filtered.empty:
                return {
                    "success": False,
                    "error": f"'{date_str}' 날짜에 사고 기록이 없습니다.",
                    "intent": intent
                }
            
            # 사고 선택 (Chainlit에서는 AskActionMessage로 처리되도록 None 반환 로직 추가)
            accident_data = self._select_accident(filtered)
            
            if accident_data is None and len(filtered) > 1:
                # 다중 사고가 발견되었으나 콘솔 input()을 피하기 위해 None 반환
                return {
                    "success": True, 
                    "date": date_str,
                    "intent": intent,
                    "confidence": parsed.get("confidence", "high"),
                    "accident_data": None # app_chainlit.py가 재선택하도록 유도
                }
            elif accident_data is None:
                # 사고 선택 취소 (콘솔 환경에서만 발생)
                return {
                    "success": False,
                    "error": "사고 선택이 취소되었습니다.",
                    "intent": intent
                }
            
            return {
                "success": True,
                "date": date_str,
                "intent": intent,
                "confidence": parsed.get("confidence", "high"),
                "accident_data": accident_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"처리 오류: {e}",
                "intent": intent
            }
    
    def _select_accident(self, filtered: pd.DataFrame) -> Optional[pd.Series]:
        """여러 사고 중 선택 (콘솔 테스트용)"""
        print(f"\n✅ {len(filtered)}건의 사고 기록을 찾았습니다:")
        print("=" * 100)
        
        for idx, (_, row) in enumerate(filtered.iterrows(), 1):
            print(f"\n[{idx}] ID: {row.get('ID', 'N/A')}")
            print(f"    발생일시: {row.get('발생일시', 'N/A')}")
            print(f"    공종: {row.get('공종(중분류)', 'N/A')}")
            print(f"    사고유형: {row.get('인적사고', 'N/A')}")
            print(f"    작업프로세스: {row.get('작업프로세스', 'N/A')}")
            
            accident_cause = str(row.get('사고원인', 'N/A'))
            if len(accident_cause) > 50:
                accident_cause = accident_cause[:50] + "..."
            print(f"    사고원인: {accident_cause}")
        
        print("=" * 100)
        
        # 여러 건인 경우 선택
        if len(filtered) > 1:
            print("\n⚠️ 다중 사고 발견. Chainlit 환경에서 선택합니다.")
            return None # Chainlit 환경으로 제어권 위임
        else:
            print("\n✅ 1건의 사고가 자동 선택되었습니다.")
            return filtered.iloc[0]
    
    def display_csv_info(self, row: pd.Series):
        """CSV 정보를 보기 좋게 출력"""
        # (기존 로직 유지 - 콘솔 출력용)
        # ... (생략)
        pass
    
    def ask_for_additional_action(self, current_intent: str) -> Optional[str]:
        """추가 작업 여부 물어보기"""
        # (기존 로직 유지 - 콘솔 출력용)
        # ... (생략)
        pass
    
    def _default_result(self) -> Dict:
        """파싱 실패 시 기본값"""
        return {
            "success": False,
            "error": "입력을 이해할 수 없습니다.",
            "intent": "csv_info"
        }