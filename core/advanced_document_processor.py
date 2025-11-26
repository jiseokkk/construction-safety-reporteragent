"""
Phase 3: 고급 문서 처리

✅ 수정사항: 모든 LLM 호출을 cl.make_async를 사용하여 비동기로 전환했습니다.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from core.llm_utils import call_llm
import json
import chainlit as cl # cl.make_async 사용을 위해 추가


class AdvancedDocumentProcessor:
    """고급 문서 처리: 중복 제거 & 핵심 추출"""
    
    def __init__(self):
        self.similarity_threshold = 0.85
    
    # 🌟 메서드 정의: async 추가 및 내부 await 처리
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
            unique_docs = await self._remove_duplicates_llm(docs) # 🌟 await 추가
            print(f"   ✅ 중복 제거 완료: {len(docs)}개 → {len(unique_docs)}개")
        else:
            unique_docs = docs
        
        # 2단계: 각 문서 처리
        print("\n📝 2단계: 핵심 문장 추출 중...")
        for idx, doc in enumerate(unique_docs, 1):
            print(f"   처리 중... [{idx}/{len(unique_docs)}]", end='\r')
            
            result = {
                "doc": doc,
                "is_duplicate": False,
                "key_sentences": [],
                "relevance_summary": ""
            }
            
            # 핵심 문장 추출
            if extract_key_sentences:
                key_info = await self._extract_key_info_llm(doc.page_content, user_query) # 🌟 await 추가
                result["key_sentences"] = key_info.get("key_sentences", [])
                result["relevance_summary"] = key_info.get("relevance_summary", "")
            
            processed_docs.append(result)
        
        print(f"\n   ✅ 핵심 추출 완료: {len(processed_docs)}개 문서")
        
        return processed_docs
    
    # 🌟 메서드 정의: async 추가 및 내부 await 처리
    async def _remove_duplicates_llm(self, docs: List[Document]) -> List[Document]:
        """LLM 기반 중복 문서 제거 (비동기)"""
        
        if len(docs) <= 1:
            return docs
        
        unique_docs = [docs[0]]  # 첫 번째는 항상 포함
        
        for idx, new_doc in enumerate(docs[1:], 2):
            # 기존 문서들과 비교
            is_duplicate = await self._check_duplicate_with_llm(new_doc, unique_docs) # 🌟 await 추가
            
            if not is_duplicate:
                unique_docs.append(new_doc)
        
        return unique_docs
    
    # 🌟 메서드 정의: async 추가 및 내부 await 처리
    async def _check_duplicate_with_llm(self, new_doc: Document, existing_docs: List[Document]) -> bool:
        """새 문서가 기존 문서들과 중복되는지 LLM으로 판단 (비동기)"""
        
        # ... (프롬프트 구성 로직 유지) ...
        existing_summaries = []
        for doc in existing_docs[-3:]:
            metadata = doc.metadata
            summary = f"파일: {metadata.get('file', '?')}, 섹션: {metadata.get('section', '?')}"
            existing_summaries.append(summary)
        
        new_metadata = new_doc.metadata
        new_summary = f"파일: {new_metadata.get('file', '?')}, 섹션: {new_metadata.get('section', '?')}"
        
        prompt = f"""
기존 문서들 (최근 3개):
{chr(10).join(f"- {s}" for s in existing_summaries)}

새 문서:
{new_summary}

판단: 새 문서가 기존 문서들과 **같은 내용**인가?

판단 기준:
- 같은 파일의 같은 섹션 → 중복
- 같은 파일의 다른 섹션 → 비중복
- 다른 파일 → 비중복

JSON 출력만:
{{"is_duplicate": true/false}}
"""
        
        try:
            # 🌟 LLM 호출을 비동기로 전환 (cl.make_async 사용)
            response = await cl.make_async(call_llm)(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            
            # JSON 파싱
            result = self._parse_json(response, {"is_duplicate": False})
            return result.get("is_duplicate", False)
        
        except Exception as e:
            print(f"\n⚠️ 중복 판단 실패: {e} (비중복으로 간주)")
            return False
    
    # 🌟 메서드 정의: async 추가 및 내부 await 처리
    async def _extract_key_info_llm(self, content: str, user_query: str) -> Dict[str, Any]:
        """LLM으로 핵심 정보 추출 (비동기)"""
        
        prompt = f"""
사용자가 다음 사고를 조사 중입니다:

{user_query}

문서 내용:
{content}

임무:
1. 이 문서가 사고와 어떻게 관련되는지 **한 문장**으로 요약
2. 사고 예방/대응에 도움되는 **핵심 문장 최대 3개** 추출 (원문 그대로)

JSON 출력:
{{
    "relevance_summary": "이 문서는 철근 작업 시 작업발판 설치 기준을 규정함",
    "key_sentences": [
        "작업발판은 견고한 구조로 설치되어야 한다.",
        "높이 2m 이상 작업 시 안전난간을 설치할 것.",
        "작업발판 폭은 최소 40cm 이상 확보할 것."
    ]
}}

규칙:
- relevance_summary: 반드시 한 문장
- key_sentences: 원문에서 정확히 추출, 최대 3개
- 관련 없으면 빈 배열
"""
        
        try:
            # 🌟 LLM 호출을 비동기로 전환 (cl.make_async 사용)
            response = await cl.make_async(call_llm)(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )
            
            result = self._parse_json(response, {
                "relevance_summary": "관련 정보 포함",
                "key_sentences": []
            })
            
            return result
        
        except Exception as e:
            print(f"\n⚠️ 핵심 추출 실패: {e}")
            return {
                "relevance_summary": "정보 추출 실패",
                "key_sentences": []
            }
    
    def _parse_json(self, text: str, default: dict) -> dict:
        """LLM 응답에서 JSON 추출 (로직 유지)"""
        
        if not text:
            return default
        
        # 1차: 전체 파싱
        try:
            return json.loads(text)
        except:
            pass
        
        # 2차: { } 추출
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except:
            pass
        
        return default