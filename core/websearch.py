from __future__ import annotations
import os
from typing import Any, List, Sequence
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import TavilySearchAPIRetriever

from core.agentstate import AgentState

load_dotenv()


class WebSearch:
    DEFAULT_TOP_K = 5
    DEFAULT_SEARCH_DEPTH = "advanced"
    DEFAULT_QUERY_SUFFIX = " 관련 법규 및 안전 기준"

    def __init__(self):
        pass

    # ------------------------------
    #  Private Utils
    # ------------------------------
    def _resolve_api_key(self, state: AgentState) -> str:
        """우선순위: state → 환경변수"""
        api_key = state.get("tavily_api_key") or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")
        return api_key

    def _extract_search_query(self, state: AgentState) -> str:
        """state에서 검색에 사용할 텍스트를 추출"""
        for key in ("query", "user_query"):
            value = state.get(key)
            if value:
                return str(value)

        messages = state.get("messages")
        if isinstance(messages, Sequence) and messages:
            last_message = messages[-1]

            if isinstance(last_message, dict):
                content = last_message.get("content") or last_message.get("text")
                if content:
                    return str(content)

            elif hasattr(last_message, "content"):
                return str(getattr(last_message, "content"))

        raise ValueError("검색 쿼리를 찾을 수 없습니다. state에 'user_query' 또는 'query'를 설정하세요.")

    def _append_system_message(self, state: AgentState, content: str) -> None:
        """state.messages가 dict 기반일 때만 로그 추가"""
        entry = {"role": "system", "content": content}
        messages = state.get("messages")

        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            messages.append(entry)
            return

        if not messages:
            state["messages"] = [entry]

    def _merge_documents(self, state: AgentState, new_docs: List[Document]) -> List[Document]:
        """
        기존 문서 + 신규 문서를 결합하여 state에 저장
        (과거 필드명과의 호환을 위해 retrieved/selected도 함께 갱신)
        """
        prev_docs: List[Document] = []

        if isinstance(state.get("retrieved_docs"), list):
            prev_docs = list(state.get("retrieved_docs") or [])
        elif isinstance(state.get("retrieved"), list):
            prev_docs = list(state.get("retrieved") or [])

        merged = prev_docs + new_docs

        if merged:
            state["retrieved_docs"] = merged
            state["retrieved"] = merged
            state["selected"] = merged

        return merged

    # -----------------------------------------
    #  Public Main Function
    # -----------------------------------------
    def run(self, state: AgentState) -> AgentState:
        """
        Tavily API를 이용해 웹 검색을 수행하고 검색된 문서를 state에 병합
        """
        api_key = self._resolve_api_key(state)
        query_text = self._extract_search_query(state)

        query_suffix = state.get("web_query_suffix", self.DEFAULT_QUERY_SUFFIX)
        expanded_query = state.get("web_query") or f"{query_text}{query_suffix}"

        # top_k 결정
        top_k = state.get("web_top_k")
        if top_k is None:
            env_top_k = os.getenv("TAVILY_TOP_K")
            top_k = int(env_top_k) if env_top_k else self.DEFAULT_TOP_K

        search_depth = (
            state.get("web_search_depth")
            or os.getenv("TAVILY_SEARCH_DEPTH")
            or self.DEFAULT_SEARCH_DEPTH
        )

        retriever = TavilySearchAPIRetriever(
            api_key=api_key,
            k=int(top_k),
            search_depth=search_depth,
        )

        # 실제 검색
        try:
            docs_web = retriever.get_relevant_documents(expanded_query)
        except Exception as exc:
            self._append_system_message(state, f"Tavily 검색 실패: {exc}")
            state["web_fallback"] = True
            state["web_error"] = str(exc)
            return state

        merged_docs = self._merge_documents(state, docs_web)

        # 상태 저장
        state["web_docs"] = docs_web
        state["web_query"] = expanded_query
        state["web_fallback"] = False

        # ✅ 웹 검색 횟수 카운트 증가
        prev_count = state.get("web_search_count", 0) or 0
        state["web_search_count"] = prev_count + 1

        self._append_system_message(state, f"Tavily 검색 결과 {len(docs_web)}건 추가됨.")

        return state
