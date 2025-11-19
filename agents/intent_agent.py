"""
IntentAgent
사용자의 자연어 입력을 분석하고 대화를 관리하는 Agent

역할:
1. 자연어에서 날짜 추출
2. 사용자 의도 파악 (csv_info / search_only / generate_report)
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
                "intent": "csv_info" | "search_only" | "generate_report",
                "confidence": "high" | "low",
                "accident_data": DataFrame row or None,
                "action": "csv_display" | "rag_search" | "full_report"
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

## 임무 2: 의도 파악 (3가지 의도)

**1. csv_info (CSV 정보 조회만)**
- 키워드: "정보", "알려줘", "어떤 사고", "사고 내용", "세부사항"
- 사용자가 단순히 사고 정보를 알고 싶을 때
- 예: "8월 8일 사고 정보 알려줘", "어떤 사고야?"

**2. search_only (RAG 검색만)**
- 키워드: "검색", "찾아줘", "관련 지침", "안전 규정", "조회"
- 사고와 관련된 안전 지침/규정을 찾을 때
- 예: "관련 지침 검색해줘", "안전 규정 찾아줘"

**3. generate_report (전체 보고서 생성)**
- 키워드: "보고서 작성", "문서 만들어", "리포트", "DOCX"
- 공식 보고서가 필요할 때
- 예: "보고서 작성해줘", "DOCX 만들어줘"

## 임무 3: 우선순위
1. 명확한 키워드가 있으면 해당 의도 선택
2. 애매하면 "csv_info" (가장 안전)
3. "보고서", "작성", "문서"가 명확하면 "generate_report"

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
- date가 없으면 null
- intent는 반드시 "csv_info", "search_only", "generate_report" 중 하나
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
            
            if accident_data is None:
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
        """여러 사고 중 선택"""
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
            while True:
                choice = input(f"\n처리할 사고 번호를 선택하세요 (1-{len(filtered)}): ").strip()
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(filtered):
                        return filtered.iloc[choice_idx]
                    else:
                        print(f"⚠️ 1-{len(filtered)} 사이의 숫자를 입력하세요.")
                except ValueError:
                    print("⚠️ 숫자를 입력하세요.")
        else:
            print("\n✅ 1건의 사고가 자동 선택되었습니다.")
            return filtered.iloc[0]
    
    def display_csv_info(self, row: pd.Series):
        """CSV 정보를 보기 좋게 출력"""
        print("\n" + "📋" * 50)
        print("📋  사고 상세 정보 (CSV 데이터)")
        print("📋" * 50)
        print()
        
        # 주요 정보
        print("━" * 100)
        print("🔍 기본 정보")
        print("━" * 100)
        print(f"ID: {row.get('ID', 'N/A')}")
        print(f"발생일시: {row.get('발생일시', 'N/A')}")
        print(f"사고인지 시간: {row.get('사고인지 시간', 'N/A')}")
        
        print("\n" + "━" * 100)
        print("🌦️  환경 정보")
        print("━" * 100)
        print(f"날씨: {row.get('날씨', 'N/A')}")
        print(f"기온: {row.get('기온', 'N/A')}")
        print(f"습도: {row.get('습도', 'N/A')}")
        
        print("\n" + "━" * 100)
        print("🏗️  공사 정보")
        print("━" * 100)
        print(f"공사종류(대분류): {row.get('공사종류(대분류)', 'N/A')}")
        print(f"공사종류(중분류): {row.get('공사종류(중분류)', 'N/A')}")
        print(f"공종(대분류): {row.get('공종(대분류)', 'N/A')}")
        print(f"공종(중분류): {row.get('공종(중분류)', 'N/A')}")
        print(f"작업프로세스: {row.get('작업프로세스', 'N/A')}")
        
        print("\n" + "━" * 100)
        print("⚠️  사고 정보")
        print("━" * 100)
        print(f"인적사고: {row.get('인적사고', 'N/A')}")
        print(f"물적사고: {row.get('물적사고', 'N/A')}")
        print(f"사고객체(대분류): {row.get('사고객체(대분류)', 'N/A')}")
        print(f"사고객체(중분류): {row.get('사고객체(중분류)', 'N/A')}")
        print(f"장소(대분류): {row.get('장소(대분류)', 'N/A')}")
        print(f"장소(중분류): {row.get('장소(중분류)', 'N/A')}")
        
        print("\n" + "━" * 100)
        print("📝 사고 원인")
        print("━" * 100)
        print(row.get('사고원인', 'N/A'))
        
        print("\n" + "━" * 100)
    
    def ask_for_additional_action(self, current_intent: str) -> Optional[str]:
        """추가 작업 여부 물어보기"""
        print("\n" + "💬" * 50)
        
        if current_intent == "csv_info":
            print("💬 추가 작업을 원하시나요?")
            print("   1. RAG 검색 (관련 안전 지침 찾기)")
            print("   2. 보고서 생성 (전체 보고서 + DOCX)")
            print("   3. 종료")
            
            choice = input("\n선택 (1/2/3): ").strip()
            
            if choice == "1":
                return "search_only"
            elif choice == "2":
                return "generate_report"
            else:
                return None
        
        elif current_intent == "search_only":
            print("💬 검색 결과를 바탕으로 보고서를 생성하시겠습니까?")
            
            choice = input("   (y/n): ").strip().lower()
            
            if choice in ['y', 'yes', '예']:
                return "generate_report"
            else:
                return None
        
        return None
    
    def _default_result(self) -> Dict:
        """파싱 실패 시 기본값"""
        return {
            "success": False,
            "error": "입력을 이해할 수 없습니다.",
            "intent": "csv_info"
        }