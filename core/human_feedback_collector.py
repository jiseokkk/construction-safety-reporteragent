from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from core.advanced_document_processor import AdvancedDocumentProcessor
import chainlit as cl


class HumanFeedbackCollector:
    """RAG 검색 결과에 대한 Human-in-the-Loop 피드백 수집 (Chainlit)"""

    def __init__(self, enable_advanced_processing: bool = True):
        self.feedback_history = []
        self.enable_advanced_processing = enable_advanced_processing
        self.processor = (
            AdvancedDocumentProcessor() if enable_advanced_processing else None
        )

    # =====================================================================================
    # ✅ DOCX용 근거 자료 생성 함수
    # =====================================================================================
    def _build_source_references(
        self,
        docs: List[Document],
        processed_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:

        refs = []
        iterable = processed_results or [{"doc": d} for d in docs]

        for idx, item in enumerate(iterable, 1):
            doc = item["doc"]
            md = getattr(doc, "metadata", {}) or {}

            refs.append(
                {
                    "idx": idx,
                    "filename": md.get("file")
                    or md.get("source")
                    or md.get("url")
                    or "알 수 없는 문서",
                    "hierarchy": md.get("hierarchy_str", ""),
                    "section": (md.get("section") or "").replace("#", "").strip(),
                    "relevance_summary": item.get("relevance_summary", ""),
                    "key_sentences": item.get("key_sentences", []),
                }
            )

        return refs

    # =====================================================================================
    async def process(
        self, docs: List[Document], query: str, available_dbs: List[str]
    ) -> Tuple[List[Document], Dict[str, Any]]:

        if not docs:
            await cl.Message(content="⚠️ 검색된 문서가 없습니다.").send()
            return docs, {"action": "no_docs"}

        # --------------------------------------
        # Phase 3 고급 처리
        # --------------------------------------
        processed_results = None
        if self.enable_advanced_processing and self.processor:
            processed_results = self.processor.process_documents(
                docs=docs,
                user_query=query,
                remove_duplicates=True,
                extract_key_sentences=True,
            )
            docs = [result["doc"] for result in processed_results]

        # --------------------------------------
        # 근거 목록 자동 생성
        # --------------------------------------
        source_references = self._build_source_references(docs, processed_results)

        # --------------------------------------
        # 문서 미리보기 UI
        # --------------------------------------
        await self._preview_documents_chainlit(docs, processed_results)

        # --------------------------------------
        # 사용자 행동 선택
        # --------------------------------------
        action = await self._get_user_action_chainlit_button()

        # =====================================================================================
        # 선택 분기 — 모든 return 값에 source_references 포함
        # =====================================================================================

        # 1) 전체 문서 사용
        if action == "accept_all":
            await cl.Message(content="✅ 모든 문서를 사용합니다.").send()
            return (
                docs,
                {
                    "action": "accept_all",
                    "count": len(docs),
                    "web_search_requested": False,
                    "source_references": source_references,
                },
            )

        # 2) 일부 문서만 선택
        elif action == "select_partial":
            selected_docs = await self._select_documents_chainlit(docs)
            if selected_docs:
                partial_refs = self._build_source_references(selected_docs)
                await cl.Message(
                    content=f"✂️ {len(selected_docs)}개 문서를 선택했습니다."
                ).send()

                return (
                    selected_docs,
                    {
                        "action": "select_partial",
                        "count": len(selected_docs),
                        "web_search_requested": False,
                        "source_references": partial_refs,
                    },
                )
            else:
                return (
                    docs,
                    {
                        "action": "accept_all",
                        "count": len(docs),
                        "web_search_requested": False,
                        "source_references": source_references,
                    },
                )

        # 3) 키워드 재검색
        elif action == "research_keyword":
            keywords = await self._get_additional_keywords_chainlit()
            return (
                docs,
                {
                    "action": "research_keyword",
                    "keywords": keywords,
                    "original_docs": docs,
                    "web_search_requested": False,
                    "source_references": source_references,
                },
            )

        # 4) 다른 DB에서 재검색
        elif action == "research_db":
            selected_dbs = await self._select_databases_chainlit(available_dbs)
            return (
                docs,
                {
                    "action": "research_db",
                    "dbs": selected_dbs,
                    "original_docs": docs,
                    "web_search_requested": False,
                    "source_references": source_references,
                },
            )

        # 5) 웹 검색 요청
        elif action == "web_search":
            await cl.Message(content="🌐 웹 검색 요청됨.").send()
            return (
                docs,
                {
                    "action": "accept_all",
                    "count": len(docs),
                    "web_search_requested": True,
                    "source_references": source_references,
                },
            )

        # 6) 취소 또는 timeout
        return (
            docs,
            {
                "action": "accept_all",
                "count": len(docs),
                "web_search_requested": False,
                "source_references": source_references,
            },
        )

    # =====================================================================================
    # 사용자 행동 선택 UI
    # =====================================================================================
    async def _get_user_action_chainlit_button(self) -> Optional[str]:

        actions = [
            cl.Action(
                name="action_1",
                value="accept_all",
                label="1️⃣ 모두 사용하여 진행",
                payload={"action": "accept_all"},
            ),
            cl.Action(
                name="action_2",
                value="select_partial",
                label="2️⃣ 일부 문서만 선택",
                payload={"action": "select_partial"},
            ),
            cl.Action(
                name="action_3",
                value="research_keyword",
                label="3️⃣ 키워드 추가 재검색",
                payload={"action": "research_keyword"},
            ),
            cl.Action(
                name="action_4",
                value="research_db",
                label="4️⃣ 다른 DB에서 재검색",
                payload={"action": "research_db"},
            ),
            cl.Action(
                name="action_5",
                value="web_search",
                label="5️⃣ 웹 검색 추가 (Tavily)",
                payload={"action": "web_search"},
            ),
        ]

        res = await cl.AskActionMessage(
            content="💬 다음 작업을 선택하세요.", actions=actions, timeout=180
        ).send()

        if not res:
            return None

        return (
            res.get("value")
            or res.get("payload", {}).get("action")
            or res.get("name")
        )

    # =====================================================================================
    # 문서 선택/미리보기 등 나머지 함수 — 기존 유지
    # =====================================================================================

    async def _preview_documents_chainlit(
        self, docs: List[Document], processed_results: List[Dict] = None
    ):
        header = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 **RAG 검색 결과 (HITL 고급 처리 적용)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총 **{len(docs)}개 문서**를 찾았습니다.
"""
        await cl.Message(content=header).send()

        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata
            content = doc.page_content

            file_name = metadata.get("file", "알 수 없음")
            section = metadata.get("section", "")
            hierarchy = metadata.get("hierarchy_str", "")
            db = metadata.get("db", "알 수 없음")

            score = metadata.get("score", 0)
            if score == 0:
                score = max(100 - (idx - 1) * 5, 50)

            if score >= 80:
                relevance_icon = "✅ 높음"
            elif score >= 60:
                relevance_icon = "⚠️ 중간"
            else:
                relevance_icon = "❓ 낮음"

            doc_info = f"""
**[{idx}] {relevance_icon}** (관련도: {score}%)
📄 파일: `{file_name}`
"""
            if hierarchy:
                doc_info += f"📍 위치: {hierarchy}\n"
            if section:
                doc_info += f"📌 섹션: {section}\n"

            doc_info += f"🗂️ DB: {db}\n"

            if processed_results and idx <= len(processed_results):
                r = processed_results[idx - 1]

                if r.get("relevance_summary"):
                    doc_info += f"\n💡 관련성: {r['relevance_summary']}\n"

                if r.get("key_sentences"):
                    doc_info += "\n🎯 핵심 문장:\n"
                    for i, s in enumerate(r["key_sentences"], 1):
                        doc_info += f"   {i}) {s}\n"

            content_preview = (
                content[:800] + "...\n(800자 표시)"
                if len(content) > 800
                else content
            )

            doc_info += f"""
────────────────────────────────────────

📝 원문:
{content_preview}

────────────────────────────────────────
"""

            await cl.Message(content=doc_info).send()

        await cl.Message(content="━" * 80).send()

    async def _select_documents_chainlit(
        self, docs: List[Document]
    ) -> List[Document]:

        msg = await cl.AskUserMessage(
            content=f"""
📌 사용할 문서 번호를 입력하세요.

예시:
- `1,2,4,7`
- `1-5,8`

총 {len(docs)}개 문서 중 선택
""",
            timeout=180,
        ).send()

        if not msg:
            return []

        selection = msg["output"].strip()

        try:
            indices = self._parse_selection(selection, len(docs))
            return [docs[i - 1] for i in indices if 1 <= i <= len(docs)]
        except:
            return []

    def _parse_selection(self, selection: str, max_num: int) -> List[int]:

        indices = []

        for part in selection.split(","):
            part = part.strip()

            if "-" in part:
                s, e = part.split("-")
                indices.extend(range(int(s), int(e) + 1))
            else:
                indices.append(int(part))

        indices = sorted(set(indices))
        return [i for i in indices if 1 <= i <= max_num]

    async def _get_additional_keywords_chainlit(self) -> List[str]:
        msg = await cl.AskUserMessage(
            content="🔍 추가 검색 키워드를 입력하세요 (쉼표로 구분)",
            timeout=180,
        ).send()

        if not msg:
            return []

        return [k.strip() for k in msg["output"].split(",") if k.strip()]

    async def _select_databases_chainlit(
        self, available_dbs: List[str]
    ) -> List[str]:

        db_list = "\n".join(
            [f"[{i}] {db}" for i, db in enumerate(available_dbs, 1)]
        )

        msg = await cl.AskUserMessage(
            content=f"""
🗂️ 사용 가능한 DB 목록:

{db_list}

📌 사용할 DB 번호를 입력하세요 예) 1,3 또는 2-5
""",
            timeout=180,
        ).send()

        if not msg:
            return []

        try:
            idxs = self._parse_selection(msg["output"], len(available_dbs))
            return [available_dbs[i - 1] for i in idxs]
        except:
            return []
