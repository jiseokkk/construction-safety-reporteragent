"""
Query Parser
사용자의 자연어 입력을 분석하여:
1. 날짜 추출 (예: "7월 3일" → "2024-07-03")
2. 의도 파악 (예: "검색해줘" → "search_only", "보고서 작성" → "generate_report")
"""

from typing import Dict, Optional
from core.llm_utils import call_llm
import json
from datetime import datetime


class QueryParser:
    """자연어 쿼리를 파싱하여 날짜와 의도 추출"""
    
    def __init__(self):
        self.current_year = datetime.now().year
    
    def parse(self, user_input: str) -> Dict[str, Optional[str]]:
        """
        자연어 입력을 파싱
        
        Returns:
            {
                "date": "2024-07-03",
                "intent": "search_only" or "generate_report",
                "confidence": "high" or "low"
            }
        """
        
        system_prompt = f"""
당신은 사용자의 자연어 입력을 분석하는 Query Parser입니다.

현재 연도: {self.current_year}

## 작업 1: 날짜 추출
사용자 입력에서 날짜를 추출하고 YYYY-MM-DD 형식으로 변환하세요.

예시:
- "7월 3일 사고" → "2024-07-03"
- "6월 3일" → "2024-06-03"
- "2024년 8월 15일" → "2024-08-15"
- "24-07-03" → "2024-07-03"
- "240703" → "2024-07-03"

연도가 명시되지 않으면 현재 연도({self.current_year})를 사용하세요.

## 작업 2: 의도 파악
사용자가 원하는 것이 무엇인지 파악하세요.

**search_only** (정보 검색만):
- "검색해줘", "찾아줘", "알려줘", "조회해줘"
- "관련 지침", "안전 규정", "사고 정보"
- "어떤 사고야", "무슨 사고", "사고 내용"

**generate_report** (보고서 생성):
- "보고서 작성", "보고서 만들어", "리포트 작성"
- "문서 생성", "DOCX 만들어"
- "공문 작성", "양식 작성"

## 출력 형식
<thinking>
1) 날짜 추출 과정
2) 의도 파악 과정
</thinking>

<output>
{{
  "date": "2024-07-03",
  "intent": "search_only",
  "confidence": "high"
}}
</output>

규칙:
- date가 추출되지 않으면 null
- intent가 불명확하면 "generate_report" (기본값)
- confidence는 "high" 또는 "low"
"""
        
        user_message = f"사용자 입력: {user_input}"
        
        try:
            response = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            print("\n🧾 QueryParser LLM 응답:")
            print(response)
            
            # JSON 추출
            parsed = self._extract_json(response)
            
            if parsed:
                print(f"\n✅ 파싱 결과:")
                print(f"   날짜: {parsed.get('date')}")
                print(f"   의도: {parsed.get('intent')}")
                print(f"   확신도: {parsed.get('confidence')}")
                return parsed
            else:
                print("⚠️ JSON 파싱 실패 - 기본값 사용")
                return self._default_result()
                
        except Exception as e:
            print(f"❌ QueryParser 오류: {e}")
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
            # 첫 { ~ 마지막 } 추출
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except:
            pass
        
        return None
    
    def _default_result(self) -> Dict:
        """파싱 실패 시 기본값"""
        return {
            "date": None,
            "intent": "generate_report",  # 기본값: 보고서 생성
            "confidence": "low"
        }