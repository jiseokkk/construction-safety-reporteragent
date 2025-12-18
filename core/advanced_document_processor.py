"""
Phase 3: 고급 문서 처리 (LLM Factory 적용 + 비동기 ainvoke 사용)

✅ 수정사항:
1. LLM Factory (Qwen-Fast) 적용으로 비용 절감.
2. cl.make_async(call_llm) 대신 LangChain의 ainvoke() 사용으로 비동기 처리 최적화.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import chainlit as cl

# ✅ Factory Import
from core.llm_factory import get_llm

# ======================================================================
# 1. Pydantic 모델 정의 (출력 구조화)
# ======================================================================
class DuplicateCheck(BaseModel):
    is_duplicate: bool = Field(description="중복 여부 (true/false)")

class KeyInfoExtraction(BaseModel):
    relevance_summary: str = Field(description="문서와 사고의 관련성 요약 (한 문장)")
    key_sentences: List[str] = Field(description="핵심 문장 리스트 (최대 3개)")


# ======================================================================
# 2. AdvancedDocumentProcessor 클래스
# ======================================================================
class AdvancedDocumentProcessor:
    """고급 문서 처리: 중복 제거 & 핵심 추출"""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        # ✅ Qwen(Fast) 모델 사용 (문서 처리는 로컬로 충분)
        self.llm = get_llm(mode="fast")
    
    async def process_documents( 
        self, 
        docs: List[Document], 
        user_query: str,
        remove_duplicates: bool = True,
        extract_key_sentences: bool = True
    ) -> List[Dict[str, Any]]:
        """
        문서 고급 처리 (비동기)
        """
        if not docs:
            return []
        
        processed_docs = []
        print(f"\n🔍 고급 문서 처리 시작 ({len(docs)}개 문서)")
        
        # 1단계: 중복 제거
        if remove_duplicates:
            print("\n📊 1단계: 문서 간 중복 제거 중...")
            unique_docs = await self._remove_duplicates_llm(docs)
            print(f"   ✅ 중복 제거 완료: {len(docs)}개 → {len(unique_docs)}개")
        else:
            unique_docs = docs
        
        # 2단계: 각 문서 처리
        print("\n📝 2단계: 핵심 문장 추출 중...")
        for idx, doc in enumerate(unique_docs, 1):
            # print(f"   처리 중... [{idx}/{len(unique_docs)}]", end='\r')
            
            result = {
                "doc": doc,
                "is_duplicate": False,
                "key_sentences": [],
                "relevance_summary": ""
            }
            
            # 핵심 문장 추출
            if extract_key_sentences:
                key_info = await self._extract_key_info_llm(doc.page_content, user_query)
                result["key_sentences"] = key_info.get("key_sentences", [])
                result["relevance_summary"] = key_info.get("relevance_summary", "")
            
            processed_docs.append(result)
        
        print(f"\n   ✅ 핵심 추출 완료: {len(processed_docs)}개 문서")
        
        return processed_docs
    
    async def _remove_duplicates_llm(self, docs: List[Document]) -> List[Document]:
        """LLM 기반 중복 문서 제거 (비동기)"""
        if len(docs) <= 1:
            return docs
        
        unique_docs = [docs[0]]  # 첫 번째는 항상 포함
        
        for idx, new_doc in enumerate(docs[1:], 2):
            # 기존 문서들과 비교
            is_duplicate = await self._check_duplicate_with_llm(new_doc, unique_docs)
            
            if not is_duplicate:
                unique_docs.append(new_doc)
        
        return unique_docs
    
    async def _check_duplicate_with_llm(self, new_doc: Document, existing_docs: List[Document]) -> bool:
        """새 문서가 기존 문서들과 중복되는지 LLM으로 판단 (비동기)"""
        
        existing_summaries = []
        for doc in existing_docs[-3:]: # 최근 3개만 비교 (속도 최적화)
            metadata = doc.metadata
            summary = f"파일: {metadata.get('file', '?')}, 섹션: {metadata.get('section', '?')}"
            existing_summaries.append(summary)
        
        new_metadata = new_doc.metadata
        new_summary = f"파일: {new_metadata.get('file', '?')}, 섹션: {new_metadata.get('section', '?')}"
        
        system_template = """
기존 문서들 (최근 3개):
{existing_docs}

새 문서:
{new_doc}

판단: 새 문서가 기존 문서들과 **같은 내용**인가?

판단 기준:
- 같은 파일의 같은 섹션 → 중복 (true)
- 같은 파일의 다른 섹션 → 비중복 (false)
- 다른 파일 → 비중복 (false)

JSON 출력만:
{{ "is_duplicate": true/false }}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("user", system_template)
        ])
        
        # Pydantic Parser 사용 (구조화된 출력 보장)
        parser = JsonOutputParser(pydantic_object=DuplicateCheck)
        chain = prompt | self.llm | parser
        
        try:
            # ✅ 비동기 호출 (ainvoke)
            result = await chain.ainvoke({
                "existing_docs": "\n".join(f"- {s}" for s in existing_summaries),
                "new_doc": new_summary
            })
            return result.get("is_duplicate", False)
        
        except Exception as e:
            print(f"\n⚠️ 중복 판단 실패: {e} (비중복으로 간주)")
            return False
    
    async def _extract_key_info_llm(self, content: str, user_query: str) -> Dict[str, Any]:
        """LLM으로 핵심 정보 추출 (비동기)"""
        
        system_template = """
사용자가 다음 사고를 조사 중입니다:
{user_query}

문서 내용:
{content}

임무:
1. 이 문서가 사고와 어떻게 관련되는지 **한 문장**으로 요약
2. 사고 예방/대응에 도움되는 **핵심 문장 최대 3개** 추출 (원문 그대로)

JSON 출력 포맷을 엄수하세요:
{{
    "relevance_summary": "요약문",
    "key_sentences": ["문장1", "문장2", "문장3"]
}}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("user", system_template)
        ])
        
        parser = JsonOutputParser(pydantic_object=KeyInfoExtraction)
        chain = prompt | self.llm | parser
        
        try:
            # ✅ 비동기 호출 (ainvoke)
            result = await chain.ainvoke({
                "user_query": user_query,
                "content": content[:2000] # 토큰 절약 (앞부분만 사용)
            })
            return result
        
        except Exception as e:
            print(f"\n⚠️ 핵심 추출 실패: {e}")
            return {
                "relevance_summary": "정보 추출 실패",
                "key_sentences": []
            }