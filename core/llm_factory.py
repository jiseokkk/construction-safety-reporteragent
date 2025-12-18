# core/llm_factory.py

import os
from langchain_openai import ChatOpenAI

# ✅ 로컬 vLLM 서버 설정
LOCAL_LLM_URL = "http://localhost:8000/v1"
LOCAL_MODEL_NAME = "qwen-2.5-32b" 

def get_llm(mode: str = "fast"):
    """
    Args:
        mode (str): 
            - "fast": 일반 작업 (Qwen 32B Local) -> 라우팅, 초안 작성
            - "smart": 고지능 작업 (GPT-4o API) -> 평가(Self-Correction), SQL 생성
    """
    
    # 1. 고지능/평가 모델 (GPT-4o) - 'Editor' 역할
    if mode == "smart":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # 2. 로컬 오픈소스 모델 (Qwen) - 'Writer' 역할
    elif mode == "fast":
        return ChatOpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="EMPTY",
            model=LOCAL_MODEL_NAME,
            temperature=0,
            max_tokens=2048 
        )
    
    else:
        raise ValueError(f"Unknown LLM mode: {mode}")