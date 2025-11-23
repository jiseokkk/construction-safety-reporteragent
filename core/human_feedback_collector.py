from typing import List, Dict, Any, Tuple, Optional
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
        
        # 2) 사용자 선택 (Chainlit UI) - 🔑 버튼 기반 로직
        action = await self._get_user_action_chainlit_button()
        
        if action == "accept_all":  # 모두 사용
            await cl.Message(content="✅ 모든 문서를 사용하여 진행합니다.").send()
            return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
        
        elif action == "select_partial":  # 일부 선택
            selected_docs = await self._select_documents_chainlit(docs)
            if selected_docs:
                await cl.Message(content=f"✅ {len(selected_docs)}개 문서를 선택했습니다.").send()
                return selected_docs, {"action": "select_partial", "count": len(selected_docs), "web_search_requested": False}
            else:
                await cl.Message(content="⚠️ 선택된 문서가 없습니다. 모든 문서를 사용합니다.").send()
                return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
        
        elif action == "research_keyword":  # 키워드 추가 재검색
            additional_keywords = await self._get_additional_keywords_chainlit()
            return docs, {
                "action": "research_keyword",
                "keywords": additional_keywords,
                "original_docs": docs,
                "web_search_requested": False
            }
        
        elif action == "research_db":  # DB 변경 재검색
            new_dbs = await self._select_databases_chainlit(available_dbs)
            return docs, {
                "action": "research_db",
                "dbs": new_dbs,
                "original_docs": docs,
                "web_search_requested": False
            }
        
        elif action == "web_search":  # 웹 검색
            await cl.Message(content="✅ 웹 검색을 요청하셨습니다.").send()
            return docs, {
                "action": "accept_all",
                "count": len(docs),
                "web_search_requested": True
            }
        
        else:  # 취소/시간 초과 등
            await cl.Message(content="⚠️ 선택이 취소되었습니다. 모든 문서를 사용합니다.").send()
            return docs, {"action": "accept_all", "count": len(docs), "web_search_requested": False}
    
    # ------------------------------------------------------------
    # 🔑 수정된 버튼 기반 사용자 선택 메서드
    # ------------------------------------------------------------
    async def _get_user_action_chainlit_button(self) -> Optional[str]:
        """사용자 행동 선택 (Chainlit UI - 버튼 기반 AskActionMessage)"""
        
        # ✅ payload 필드 추가!
        actions = [
            cl.Action(
                name="action_1", 
                value="accept_all", 
                label="1️⃣ 모두 사용하여 진행", 
                description="검색된 문서를 모두 활용하여 다음 단계로 넘어갑니다.",
                payload={"action": "accept_all"}
            ),
            cl.Action(
                name="action_2", 
                value="select_partial", 
                label="2️⃣ 일부 문서만 선택", 
                description="문서 번호를 직접 지정하여 필터링합니다.",
                payload={"action": "select_partial"}
            ),
            cl.Action(
                name="action_3", 
                value="research_keyword", 
                label="3️⃣ 키워드 추가 재검색", 
                description="새 키워드를 추가하여 RAG 검색을 다시 수행합니다.",
                payload={"action": "research_keyword"}
            ),
            cl.Action(
                name="action_4", 
                value="research_db", 
                label="4️⃣ 다른 DB에서 재검색", 
                description="현재 DB가 아닌 다른 DB를 선택하여 다시 검색합니다.",
                payload={"action": "research_db"}
            ),
            cl.Action(
                name="action_5", 
                value="web_search", 
                label="5️⃣ 웹 검색 추가 (Tavily)", 
                description="내부 문서와 함께 웹 검색 결과를 추가로 요청합니다.",
                payload={"action": "web_search"}
            ),
        ]
        
        # 🔑 cl.AskActionMessage를 사용하여 사용자 응답 대기
        res = await cl.AskActionMessage(
            content="**💬 다음 작업을 선택해주세요.**", 
            actions=actions, 
            timeout=180  # 3분 대기
        ).send()
        
        if res:
            # 🔑 여러 방법으로 action 추출 시도
            print(f"DEBUG: res = {res}")
            print(f"DEBUG: res type = {type(res)}")
            
            # 방법 1: value에서 추출 (가장 확실)
            action = res.get("value")
            if action:
                print(f"DEBUG: Action from value = {action}")
                return action
            
            # 방법 2: payload에서 추출
            if isinstance(res, dict):
                action = res.get("payload", {}).get("action")
                if action:
                    print(f"DEBUG: Action from payload = {action}")
                    return action
                
                # 방법 3: name에서 추출
                name = res.get("name", "")
                if name.startswith("action_"):
                    action_map = {
                        "action_1": "accept_all",
                        "action_2": "select_partial",
                        "action_3": "research_keyword",
                        "action_4": "research_db",
                        "action_5": "web_search"
                    }
                    action = action_map.get(name)
                    if action:
                        print(f"DEBUG: Action from name = {action}")
                        return action
            
            # 방법 4: 문자열로 직접 반환된 경우
            elif isinstance(res, str):
                print(f"DEBUG: Action from string = {res}")
                return res
        
        print("DEBUG: No action found, returning None")
        return None  # 시간 초과 또는 취소
        
    # ------------------------------------------------------------
    # 나머지 헬퍼 메서드는 유지됩니다.
    # ------------------------------------------------------------
    
    async def _preview_documents_chainlit(self, docs: List[Document], processed_results: List[Dict] = None):
        """검색된 문서 미리보기 (Chainlit UI)"""
        
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