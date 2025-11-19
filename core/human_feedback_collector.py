"""
Human Feedback Collector (Chainlit 전용)
RAG 검색 결과에 대한 사용자 피드백을 Chainlit UI로 수집

✅ Chainlit 네이티브 방식
✅ wrapper 불필요
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from core.advanced_document_processor import AdvancedDocumentProcessor
import chainlit as cl


class HumanFeedbackCollector:
    """RAG 검색 결과에 대한 Human-in-the-Loop 피드백 수집 (Chainlit)"""
    
    def __init__(self, enable_advanced_processing: bool = True):
        self.feedback_history = []
        self.enable_advanced_processing = enable_advanced_processing
        self.processor = AdvancedDocumentProcessor() if enable_advanced_processing else None
    
    async def process(
        self, 
        docs: List[Document], 
        query: str,
        available_dbs: List[str]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        검색된 문서에 대한 사용자 피드백 수집 및 처리 (Chainlit UI)
        
        Args:
            docs: 검색된 Document 리스트
            query: 원본 쿼리
            available_dbs: 사용 가능한 DB 리스트
        
        Returns:
            (필터링된 문서 리스트, 피드백 정보)
        """
        
        if not docs:
            await cl.Message(content="⚠️ 검색된 문서가 없습니다.").send()
            return docs, {"action": "no_docs"}
        
        # ✅ Phase 3: 고급 처리
        processed_results = None
        if self.enable_advanced_processing and self.processor:
            processed_results = self.processor.process_documents(
                docs=docs,
                user_query=query,
                remove_duplicates=True,
                extract_key_sentences=True
            )
            
            # 중복 제거된 문서만 사용
            docs = [result["doc"] for result in processed_results]
        
        # 1) 문서 미리보기 (Chainlit UI)
        await self._preview_documents_chainlit(docs, processed_results)
        
        # 2) 사용자 선택 (Chainlit UI)
        action = await self._get_user_action_chainlit()
        
        if action == "1":  # 모두 사용
            await cl.Message(content="✅ 모든 문서를 사용하여 진행합니다.").send()
            return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
        
        elif action == "2":  # 일부 선택
            selected_docs = await self._select_documents_chainlit(docs)
            if selected_docs:
                await cl.Message(content=f"✅ {len(selected_docs)}개 문서를 선택했습니다.").send()
                return selected_docs, {"action": "select_partial", "count": len(selected_docs), "web_search_requested": False}
            else:
                await cl.Message(content="⚠️ 선택된 문서가 없습니다. 모든 문서를 사용합니다.").send()
                return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
        
        elif action == "3":  # 키워드 추가 재검색
            additional_keywords = await self._get_additional_keywords_chainlit()
            return docs, {
                "action": "research_keyword",
                "keywords": additional_keywords,
                "original_docs": docs,
                "web_search_requested": False
            }
        
        elif action == "4":  # DB 변경 재검색
            new_dbs = await self._select_databases_chainlit(available_dbs)
            return docs, {
                "action": "research_db",
                "dbs": new_dbs,
                "original_docs": docs,
                "web_search_requested": False
            }
        
        elif action == "5":  # 웹 검색
            await cl.Message(content="✅ 웹 검색을 요청하셨습니다.").send()
            return docs, {
                "action": "accept_all",
                "count": len(docs),
                "web_search_requested": True
            }
        
        else:
            await cl.Message(content="⚠️ 잘못된 선택입니다. 모든 문서를 사용합니다.").send()
            return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
    
    async def _preview_documents_chainlit(self, docs: List[Document], processed_results: List[Dict] = None):
        """검색된 문서 미리보기 (Chainlit UI)"""
        
        # 헤더
        header = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 **RAG 검색 결과 (Human-in-the-Loop + Phase 3 고급 처리)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

총 **{len(docs)}개의 문서**를 찾았습니다.
"""
        await cl.Message(content=header).send()
        
        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            # 메타데이터 추출
            file_name = metadata.get("file", "알 수 없음")
            section = metadata.get("section", "")
            hierarchy = metadata.get("hierarchy_str", "")
            db = metadata.get("db", "알 수 없음")
            
            # 관련도 계산
            score = metadata.get("score", 0)
            if score == 0:
                score = max(100 - (idx-1) * 5, 50)
            
            # 관련도 표시
            if score >= 80:
                relevance_icon = "✅ 높음"
            elif score >= 60:
                relevance_icon = "⚠️ 중간"
            else:
                relevance_icon = "❓ 낮음"
            
            # 기본 정보
            doc_info = f"""
**[{idx}] {relevance_icon}** (관련도: {score}%)

📄 **파일:** `{file_name}`
"""
            
            if hierarchy:
                doc_info += f"📍 **위치:** {hierarchy}\n"
            if section:
                section_clean = section.replace("##", "").replace("#", "").strip()
                doc_info += f"📌 **섹션:** {section_clean}\n"
            
            doc_info += f"🗂️  **DB:** {db}\n"
            
            # ✅ Phase 3: 고급 처리 결과
            if processed_results and idx <= len(processed_results):
                result = processed_results[idx - 1]
                
                relevance_summary = result.get("relevance_summary", "")
                if relevance_summary:
                    doc_info += f"\n💡 **관련성:** {relevance_summary}\n"
                
                key_sentences = result.get("key_sentences", [])
                if key_sentences:
                    doc_info += "\n🎯 **핵심 문장:**\n"
                    for i, sentence in enumerate(key_sentences, 1):
                        doc_info += f"   {i}) {sentence}\n"
            
            doc_info += "\n" + "─" * 80 + "\n"
            doc_info += f"\n📝 **원본 전체 내용:**\n```\n"
            
            # 내용 표시 (너무 길면 자르기)
            if len(content) > 800:
                doc_info += content[:800] + "...\n```\n"
                doc_info += f"\n*(전체 {len(content)}자 중 800자 표시)*"
            else:
                doc_info += content + "\n```"
            
            doc_info += "\n" + "─" * 80
            
            await cl.Message(content=doc_info).send()
        
        # 푸터
        await cl.Message(content="━" * 80).send()
    
    async def _get_user_action_chainlit(self) -> str:
        """사용자 행동 선택 (Chainlit UI - LLM 의도 파악)"""
        
        # 선택지 안내
        await cl.Message(content="""
**💬 다음 작업을 원하시나요?**

   [1] 모든 문서 사용하여 진행
   [2] 일부 문서만 선택
   [3] 키워드 추가하여 재검색
   [4] 다른 DB에서 재검색
   [5] 웹 검색 추가 (Tavily)

💡 **자유롭게 말씀해주세요!**
   예: "웹에서도 찾아봐", "이 문서들로 진행", "키워드 추가할게" 등
""").send()
        
        # 사용자 입력 받기
        res = await cl.AskUserMessage(
            content="**입력:**",
            timeout=180
        ).send()
        
        if res:
            user_input = res["output"].strip()
            
            # ✅ LLM으로 의도 파악
            choice = await self._parse_user_intent_with_llm(user_input)
            
            # 선택 확인 메시지
            choice_labels = {
                "1": "모든 문서 사용",
                "2": "일부 문서 선택",
                "3": "키워드 추가 재검색",
                "4": "DB 변경 재검색",
                "5": "웹 검색 추가"
            }
            
            if choice in choice_labels:
                await cl.Message(content=f"✅ **파악된 의도:** [{choice}] {choice_labels[choice]}").send()
            
            return choice
        
        return "1"  # 기본값
    
    async def _parse_user_intent_with_llm(self, user_input: str) -> str:
        """
        LLM을 사용하여 사용자 의도 파악
        
        Args:
            user_input: 사용자의 자연어 입력
        
        Returns:
            선택지 번호 ("1", "2", "3", "4", "5")
        """
        from core.llm_utils import call_llm
        import json
        
        system_prompt = """
당신은 사용자의 의도를 파악하는 AI입니다.

사용자가 RAG 검색 결과를 보고 다음 중 하나를 선택하려고 합니다:

1. 모든 문서 사용하여 진행
2. 일부 문서만 선택
3. 키워드 추가하여 재검색
4. 다른 DB에서 재검색
5. 웹 검색 추가 (Tavily)

사용자의 입력을 분석하여 어떤 선택지를 원하는지 파악하세요.

## 입력 예시와 결과:
- "1" → 1
- "웹에서도 찾아봐" → 5
- "이 문서들로 진행" → 1
- "몇 개만 골라서 쓸게" → 2
- "키워드 추가할게" → 3
- "다른 DB에서 검색" → 4
- "인터넷도 검색해줘" → 5
- "tavily 써봐" → 5
- "전부 사용" → 1
- "재검색" → 3

## 출력 형식 (JSON):
{{
  "choice": "1",
  "reason": "사용자가 모든 문서를 사용하겠다는 의도"
}}

숫자만 출력하지 말고 반드시 위 JSON 형식을 따르세요.
"""
        
        user_message = f"사용자 입력: {user_input}"
        
        try:
            # LLM 호출 (비동기)
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: call_llm(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
            )
            
            # JSON 파싱
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)
                
                choice = parsed.get("choice", "1")
                reason = parsed.get("reason", "")
                
                print(f"\n🤖 LLM 의도 파악: choice={choice}, reason={reason}")
                
                # 유효성 검사
                if choice in ["1", "2", "3", "4", "5"]:
                    return choice
            
        except Exception as e:
            print(f"⚠️ LLM 의도 파악 실패: {e}")
            # fallback: 키워드 기반 파싱
            return self._parse_user_choice_fallback(user_input)
        
        return "1"
    
    def _parse_user_choice_fallback(self, user_input: str) -> str:
        """
        LLM 실패 시 fallback: 키워드 기반 파싱
        """
        user_input = user_input.strip().lower()
        
        # 숫자 직접 입력
        if user_input in ["1", "2", "3", "4", "5"]:
            return user_input
        
        # 키워드 매칭
        if any(keyword in user_input for keyword in ["웹", "web", "인터넷", "tavily", "온라인"]):
            return "5"
        
        if any(keyword in user_input for keyword in ["모든", "전체", "모두", "all"]):
            return "1"
        
        if any(keyword in user_input for keyword in ["일부", "선택", "골라"]):
            return "2"
        
        if any(keyword in user_input for keyword in ["키워드", "재검색", "추가검색"]):
            return "3"
        
        if any(keyword in user_input for keyword in ["db", "데이터베이스", "디비"]):
            return "4"
        
        return "1"
    
    async def _select_documents_chainlit(self, docs: List[Document]) -> List[Document]:
        """사용자가 문서 선택 (Chainlit UI)"""
        
        selection_msg = await cl.AskUserMessage(
            content=f"""
📌 사용할 문서 번호를 입력하세요.

**예시:**
- `1,2,4,7` → 1, 2, 4, 7번 문서 선택
- `1-5,8,10` → 1~5번, 8번, 10번 문서 선택

**(총 {len(docs)}개 문서 중 선택)**
""",
            timeout=180
        ).send()
        
        if selection_msg:
            selection = selection_msg["output"].strip()
            
            try:
                indices = self._parse_selection(selection, len(docs))
                selected_docs = [docs[i-1] for i in indices if 1 <= i <= len(docs)]
                
                if selected_docs:
                    await cl.Message(content=f"✅ 선택된 문서: {', '.join(str(i) for i in indices)}").send()
                
                return selected_docs
            
            except Exception as e:
                await cl.Message(content=f"⚠️ 입력 오류: {e}").send()
                return []
        
        return []
    
    def _parse_selection(self, selection: str, max_num: int) -> List[int]:
        """문서 선택 입력 파싱"""
        indices = []
        
        for part in selection.split(","):
            part = part.strip()
            
            if "-" in part:
                start, end = part.split("-")
                start = int(start.strip())
                end = int(end.strip())
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        
        indices = sorted(set(indices))
        indices = [i for i in indices if 1 <= i <= max_num]
        
        return indices
    
    async def _get_additional_keywords_chainlit(self) -> List[str]:
        """추가 키워드 입력 (Chainlit UI)"""
        
        keyword_msg = await cl.AskUserMessage(
            content="""
🔍 추가로 검색할 키워드를 입력하세요.

**예시:** `안전대, 추락방지, 안전난간`

*(쉼표로 구분)*
""",
            timeout=180
        ).send()
        
        if keyword_msg:
            keywords_input = keyword_msg["output"].strip()
            
            if keywords_input:
                keywords = [k.strip() for k in keywords_input.split(",")]
                await cl.Message(content=f"✅ 추가 키워드: {', '.join(keywords)}").send()
                return keywords
        
        await cl.Message(content="⚠️ 키워드가 입력되지 않았습니다.").send()
        return []
    
    async def _select_databases_chainlit(self, available_dbs: List[str]) -> List[str]:
        """사용자가 DB 선택 (Chainlit UI)"""
        
        db_list_text = "\n".join([f"   [{i}] {db}" for i, db in enumerate(available_dbs, 1)])
        
        db_msg = await cl.AskUserMessage(
            content=f"""
🗂️  **사용 가능한 DB 목록:**

{db_list_text}

📌 검색할 DB 번호를 입력하세요.

**예시:** `1,3,5` 또는 `1-4`
""",
            timeout=180
        ).send()
        
        if db_msg:
            selection = db_msg["output"].strip()
            
            try:
                indices = self._parse_selection(selection, len(available_dbs))
                selected_dbs = [available_dbs[i-1] for i in indices if 1 <= i <= len(available_dbs)]
                
                if selected_dbs:
                    await cl.Message(content=f"✅ 선택된 DB: {', '.join(selected_dbs)}").send()
                
                return selected_dbs
            
            except Exception as e:
                await cl.Message(content=f"⚠️ 입력 오류: {e}").send()
                return []
        
        return []