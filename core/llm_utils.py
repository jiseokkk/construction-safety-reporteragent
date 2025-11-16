import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(
    messages: list,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 4000,
) -> str:
    """
    OpenAI LLM 호출 유틸리티 함수
    
    Args:
        messages: 대화 메시지 리스트 [{"role": "system/user/assistant", "content": "..."}]
        model: 사용할 모델
        temperature: 창의성 조절
        top_p: 샘플링 파라미터
        max_tokens: 최대 토큰 수
    
    Returns:
        LLM 응답 텍스트
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"❌ LLM 호출 실패: {e}")
        return f"⚠️ LLM 호출 중 오류 발생: {str(e)}"


def call_llm_with_tools(
    messages: list,
    tools: list,
    model: str = "gpt-4o",
    temperature: float = 0.7,
) -> dict:
    """
    Tool calling을 사용한 LLM 호출
    
    Args:
        messages: 대화 메시지 리스트
        tools: 사용 가능한 도구 정의 리스트
        model: 사용할 모델
        temperature: 창의성 조절
    
    Returns:
        전체 응답 객체 (tool_calls 포함)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        
        return response.choices[0].message
    
    except Exception as e:
        print(f"❌ LLM Tool Calling 실패: {e}")
        return None
