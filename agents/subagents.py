"""
SubAgents - RAGAgent, WebSearchAgent, ReportWriterAgent 정의 (LangChain LCEL 적용 버전)

✅ 핵심 변경사항:
1. RAGAgent, ReportWriterAgent의 LLM 호출 및 파싱 로직을 LangChain LCEL + Pydantic으로 전면 교체.
2. 불안정한 `parse_json_with_recovery` 및 `call_llm` 의존성 제거.
3. RAGAgent.run()에 `source_references` 생성 로직 추가 (DOCX 13행 생성용).
4. 기존 로직(HITL 처리, 문서 검색 흐름 등)은 100% 유지.
"""

from typing import Any, Dict, List, Tuple, Literal, Optional
import torch
import gc
import json
import os
import chainlit as cl

# ✅ LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# 기존 모듈 임포트
from core.agentstate import AgentState
from core.docx_writer import create_accident_report_docx
from core.final_report import summarize_accident_cause, generate_action_plan
from core.websearch import WebSearch
from core.retriever import SingleDBHybridRetriever
from core.chunk_formatter import ChunkFormatter

DB_ROOT = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB"

# ======================================================================
# 1. Pydantic 모델 정의 (LLM 출력 스키마 강제)
# ======================================================================

class DBRoutingPlan(BaseModel):
    """RAGAgent의 DB 선택 계획"""
    db_list: List[str] = Field(description="검색할 데이터베이스 폴더 이름 목록 (예: ['01_bridge', '03_tunnel'])")
    fallback: bool = Field(description="검색 결과가 부족할 경우 Fallback DB를 사용할지 여부")
    fallback_db: str = Field(description="Fallback으로 사용할 DB 이름 (보통 '08_general')")
    reasoning: str = Field(description="이 DB들을 선택한 논리적 근거 (CoT)") 

class ReportAction(BaseModel):
    """ReportWriterAgent의 다음 행동 결정"""
    action: Literal["web_search", "final_report", "create_docx", "noop"] = Field(
        description="수행할 작업의 이름"
    )
    reason: str = Field(description="해당 작업을 선택한 이유")


# ========================================
# 헬퍼 함수
# ========================================
def load_db_descriptions():
    """DB 폴더의 description.json 로드"""
    db_info = {}
    if not os.path.exists(DB_ROOT):
        print(f"⚠️ 경고: DB 루트 경로를 찾을 수 없습니다: {DB_ROOT}")
        return {}
        
    for folder in os.listdir(DB_ROOT):
        desc_path = os.path.join(DB_ROOT, folder, "description.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                db_info[folder] = json.load(f)
    return db_info


# agents/subagents.py 내 RAGAgent 클래스

class RAGAgent:
    name = "RAGAgent"

    def __init__(self):
        self.db_info: Dict[str, Any] = load_db_descriptions() 
        self.available_dbs: List[str] = sorted(self.db_info.keys())
        self.formatter = ChunkFormatter()
        print(f"📚 사용 가능한 DB 목록: {self.available_dbs}")

        # ✅ LangChain 설정
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.parser = PydanticOutputParser(pydantic_object=DBRoutingPlan)

    def _build_structured_query(self, state: AgentState) -> str:
        """사용자 쿼리 + 구조화 정보 + HITL 키워드를 합친 텍스트"""
        user_query = state.get("user_query", "")
        
        gongsung = state.get("공종") or state.get("gongsung")
        process = state.get("작업프로세스") or state.get("process")
        acc_type = state.get("사고 유형") or state.get("accident_type")
        obj = state.get("사고객체(중분류)") or state.get("object")
        location = state.get("장소(중분류)") or state.get("location")

        extra_lines = []
        if gongsung: extra_lines.append(f"공종: {gongsung}")
        if process: extra_lines.append(f"작업프로세스: {process}")
        if acc_type: extra_lines.append(f"사고유형: {acc_type}")
        if obj: extra_lines.append(f"사고객체: {obj}")
        if location: extra_lines.append(f"장소: {location}")

        extra_block = "\n".join(extra_lines)
        
        # HITL 재검색 키워드 추가
        hitl_payload = state.get('hitl_payload', {})
        if hitl_payload.get('keywords'):
            extra_block += "\n[HITL 추가 키워드]\n" + ", ".join(hitl_payload['keywords'])
        
        structured_query = f"[User Query]\n{user_query}\n"
        if extra_block:
            structured_query += "\n[추가 구조화 정보]\n" + extra_block

        return structured_query

    async def _plan_db_selection(self, structured_query: str) -> Dict[str, Any]: 
        """LLM에게 DB 선택 계획 요청 (LangChain LCEL 적용)"""
        
        system_template = """
당신은 건설안전 RAG 시스템의 DB 라우팅을 담당하는 Agent입니다.

################################################################################
# 🔥임무: 사고 속성 기반으로 가장 적합한 DB를 1~3개 선택하고 Fallback 필요 여부 판단
################################################################################

사고 속성(객체, 공종, 프로세스 등)을 분석하여 아래 DB 목록 중 가장 적합한 것을 선택하세요.

[사용 가능한 DB 목록 및 설명]
{db_info}

반드시 아래 형식을 준수하여 JSON으로 응답해야 합니다:
{format_instructions}

판단 기준:
1. 사고객체/공종/작업프로세스와 DB 설명의 일치도
2. 관련성 높은 DB 1~3개 선택
3. 결과가 부족할 것 같으면 fallback=True, fallback_db="08_general" 설정
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "[사용자 사고 정보]\n{structured_query}")
        ])

        # 🔥 LCEL Chain: Prompt -> LLM -> Parser
        chain = prompt | self.llm | self.parser

        try:
            # Pydantic 객체 반환
            plan: DBRoutingPlan = await chain.ainvoke({
                "db_info": json.dumps(self.db_info, ensure_ascii=False, indent=2),
                "structured_query": structured_query,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Pydantic 객체를 dict로 변환
            return plan.dict()

        except Exception as e:
            print(f"⚠️ DB 선택 계획 수립 실패 (LCEL 오류): {e}")
            # Fallback Plan
            return {
                "db_list": ["08_general"] if "08_general" in self.available_dbs else (self.available_dbs[:1] or []),
                "fallback": False,
                "fallback_db": "08_general" if "08_general" in self.available_dbs else ""
            }

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """실제 존재하는 DB만 남기고 검증"""
        db_list = plan.get("db_list", []) or []
        fallback_flag = bool(plan.get("fallback", False))
        fallback_db_name = plan.get("fallback_db", "08_general")

        valid_db_list = [db for db in db_list if db in self.available_dbs]

        if not valid_db_list:
            if "08_general" in self.available_dbs:
                valid_db_list = ["08_general"]
            elif self.available_dbs:
                valid_db_list = [self.available_dbs[0]]
            else:
                valid_db_list = []

        if fallback_flag and fallback_db_name not in self.available_dbs:
            if "08_general" in self.available_dbs:
                fallback_db_name = "08_general"
            elif self.available_dbs:
                fallback_db_name = self.available_dbs[0]
            else:
                fallback_flag = False
                fallback_db_name = ""

        return {
            "db_list": valid_db_list,
            "fallback": fallback_flag,
            "fallback_db": fallback_db_name,
        }

    def _search_documents(self, db_list: List[str], query: str, top_k: int = 5) -> List[Document]:
        """여러 DB에서 문서 검색 (메모리 최적화 적용)"""
        all_docs = []
        
        # 🧹 시작 전 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for db_name in db_list:
            db_path = os.path.join(DB_ROOT, db_name)
            if not os.path.exists(os.path.join(db_path, "index.faiss")):
                continue
            
            print(f"📂 검색 대상 DB: {db_path}")
            try:
                # Retriever 생성 및 검색
                retriever = SingleDBHybridRetriever(db_dir=db_path, top_k=top_k, alpha=0.5)
                docs = retriever.retrieve(query) 
                
                # 메타데이터에 DB 출처 명시
                for d in docs: d.metadata['db'] = db_name
                all_docs.extend(docs)
                
                # 🧹 사용 완료한 Retriever 객체 삭제 및 메모리 정리 (OOM 방지 핵심)
                del retriever
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️ 검색 중 오류 발생 (DB: {db_name}): {e}")
                # 오류 발생 시에도 메모리 정리 시도
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        return all_docs

    async def search_only(self, user_query: str, state: AgentState) -> List[Document]:
        """HITL 없이 RAG 검색만 수행 (run()에서 호출됨)"""
        print("\n" + "="*80)
        print("📚 [RAGAgent] search_only - 검색 실행")
        print("="*80)

        structured_query = self._build_structured_query(state)

        hitl_payload = state.get('hitl_payload', {})
        hitl_action = state.get('hitl_action')
        
        # [Case A] DB 재검색: 사용자가 선택한 DB를 강제로 사용
        if hitl_action == 'research_db':
            selected_dbs = hitl_payload.get('dbs', [])
            print(f"🚨 [HITL Override] 사용자 요청으로 DB 강제 변경: {selected_dbs}")
            plan = {"db_list": selected_dbs, "fallback": False, "fallback_db": ""}
            
        # [Case B] 키워드 재검색: 쿼리가 변경되었으므로 LLM이 다시 DB를 계획
        elif hitl_action == 'research_keyword':
            print(f"🚨 [HITL Override] 키워드 추가됨 -> DB 재계획 수립")
            raw_plan = await self._plan_db_selection(structured_query)
            plan = self._sanitize_plan(raw_plan)
            
        # [Case C] 일반 검색 (초기 실행)
        else:
            raw_plan = await self._plan_db_selection(structured_query)
            plan = self._sanitize_plan(raw_plan)
        
        print(f"🧠 최종 사용 계획: {plan}")

        db_list = plan.get("db_list", []) or []
        fallback_flag = plan.get("fallback", False)
        fallback_db_name = plan.get("fallback_db", "08_general")

        # 3) 검색 (동기 함수를 cl.make_async로 감싸서 실행 권장)
        all_docs = await cl.make_async(self._search_documents)(db_list, structured_query, top_k=5)

        # 4) Fallback 검색
        if fallback_flag and len(all_docs) < 3 and fallback_db_name:
            fb_path = os.path.join(DB_ROOT, fallback_db_name)
            print(f"⚠️ Fallback DB 검색 실행 → {fb_path}")
            if os.path.exists(os.path.join(fb_path, "index.faiss")):
                # Fallback 검색도 비동기 처리
                fb_retriever = SingleDBHybridRetriever(db_dir=fb_path, top_k=5, alpha=0.5)
                fallback_docs = await cl.make_async(fb_retriever.retrieve)(structured_query)
                all_docs.extend(fallback_docs)

        final_docs = all_docs[:10]
        print(f"\n✅ RAG 검색 완료! (총 {len(final_docs)}개 문서)")
        return final_docs

    # ========================================
    # 🌟 run() 메서드 (Async)
    # ========================================
    async def run(self, state: AgentState) -> AgentState: 
        print("\n" + "="*80)
        print("📚 [RAGAgent] run - LangGraph 워크플로우 실행")
        print("="*80)

        user_query = state.get("user_query", "")
        
        # 1. 새로운 검색 실행 (새 DB 또는 키워드로 검색된 결과)
        new_docs = await self.search_only(user_query, state)
        
        # 2. 기존 문서 및 HITL 액션 확인
        existing_docs = state.get("retrieved_docs", []) or []
        hitl_action = state.get("hitl_action")
        
        # ---------------------------------------------------------
        # 🔥 [핵심] 문서 병합 로직 (DB 변경 OR 키워드 추가 시 병합)
        # ---------------------------------------------------------
        final_docs = []
        
        if hitl_action in ["research_db", "research_keyword"]:
            print(f"➕ [Merge] 기존 {len(existing_docs)}개 + 신규 {len(new_docs)}개 병합 시도")
            seen_content = set()
            
            # (A) 기존 문서 먼저 담기 (보존)
            for doc in existing_docs:
                # 중복 체크 키: 파일명 + 내용 앞부분 50자
                key = (doc.metadata.get("source", ""), doc.page_content[:50])
                seen_content.add(key)
                final_docs.append(doc)
            
            # (B) 새 문서 뒤에 붙이기 (중복 제외)
            duplicates = 0
            for doc in new_docs:
                key = (doc.metadata.get("source", ""), doc.page_content[:50])
                if key not in seen_content:
                    final_docs.append(doc)
                    seen_content.add(key)
                else:
                    duplicates += 1
            
            if duplicates > 0:
                print(f"   (중복된 문서 {duplicates}개는 제외되었습니다.)")
                
        else:
            # 그 외(초기 검색 등)는 결과 교체
            final_docs = new_docs

        # ---------------------------------------------------------

        # HITL 초기화
        state["hitl_action"] = None
        state["hitl_payload"] = {}
        
        # State 업데이트 (docs_text, sources)
        docs_text = "\n\n".join(
            f"[문서 {i+1}] ({doc.metadata.get('file', '?')}, {doc.metadata.get('section', '')})\n{doc.page_content}"
            for i, doc in enumerate(final_docs)
        )
        
        sources = [
            {"idx": i + 1, "filename": doc.metadata.get("file", ""), "section": doc.metadata.get("section", ""), "db": doc.metadata.get("db", "")}
            for i, doc in enumerate(final_docs)
        ]
        
        # DocxWriter용 상세 source_references 데이터 생성
        source_references = []
        for i, doc in enumerate(final_docs, 1):
            md = doc.metadata or {}
            
            ref_data = {
                "idx": i,
                "filename": md.get("file") or md.get("source", "알 수 없는 문서"),
                "hierarchy": md.get("hierarchy_str", ""),
                "section": md.get("section", ""),
                "db": md.get("db", ""),
                "relevance_summary": md.get("summary", ""), 
                "key_sentences": [] 
            }
            source_references.append(ref_data)

        # 상태 저장
        state["retrieved_docs"] = final_docs # 병합된 리스트 저장
        state["docs_text"] = docs_text
        state["sources"] = sources
        state["source_references"] = source_references
        state["route"] = "retrieve_complete"

        user_intent = state.get("user_intent", "generate_report")
        if user_intent == "search_only":
            state["wait_for_user"] = True
        
        return state

# ========================================
# ReportWriterAgent
# ========================================
class ReportWriterAgent:
    name = "ReportWriterAgent"

    def __init__(self):
        self.action_handlers = {
            "final_report": self._generate_final_report,
            "web_search": self._run_web_search,
            "create_docx": self._create_docx_file,
        }
        # ✅ LangChain 설정
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.parser = PydanticOutputParser(pydantic_object=ReportAction)

    def _summarize_state(self, state: AgentState) -> str:
        """State 요약"""
        doc_cnt = len(state.get("retrieved_docs") or [])
        docs_text_length = len(state.get("docs_text") or "")
        web_done = state.get("web_search_completed", False)
        web_count = state.get("web_search_count", 0)
        report_ready = bool(state.get("report_text"))
        docx_ready = bool(state.get("docx_path"))

        return f"""
현재 상태:
[사용자 질의] {state.get('user_query', 'N/A')}
[RAG 검색] 문서 수: {doc_cnt}, 텍스트 길이: {docs_text_length}
[웹 검색] 완료: {web_done}, 횟수: {web_count}
[보고서] 생성됨: {report_ready}
[DOCX] 생성됨: {docx_ready}
"""

    def _fallback_action(self, state: AgentState) -> Tuple[str, str]:
        """LLM 실패 시 Rule-based fallback"""
        print("\n⚠️ FALLBACK 모드 활성화 (ReportWriter)")
        if not state.get("report_text"): return "final_report", "[Fallback] 보고서 생성"
        if not state.get("docx_path"): return "create_docx", "[Fallback] DOCX 생성"
        return "noop", "[Fallback] 작업 완료"

    async def _decide_action(self, state: AgentState) -> Tuple[str, str]: 
        """LLM을 사용하여 다음 작업 결정 (LangChain LCEL 적용)"""
        
        system_template = """
당신은 ReportWriterAgent로서, 현재 상태를 분석하고 다음 작업을 결정합니다.

<available_actions>
1. web_search: RAG 결과 부족시 수행 (이미 완료된 경우 금지)
2. final_report: 보고서가 없을 때 수행
3. create_docx: 보고서가 있고 DOCX가 없을 때 수행
4. noop: 모든 작업 완료 시
</available_actions>

<decision_rules>
1. 보고서 없음 → final_report
2. 보고서 있음 + DOCX 없음 → create_docx
3. 보고서 있음 + DOCX 있음 → noop
4. web_search는 정보 부족 시에만
</decision_rules>

반드시 아래 형식을 준수하여 JSON으로 응답해야 합니다:
{format_instructions}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{state_summary}")
        ])

        # 🔥 LCEL Chain
        chain = prompt | self.llm | self.parser
        
        summary = self._summarize_state(state)

        try:
            # Pydantic 객체 반환
            decision: ReportAction = await chain.ainvoke({
                "state_summary": summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return decision.action, decision.reason
            
        except Exception as exc:
            print(f"⚠️ ReportWriter 의사결정 실패 (LCEL 오류): {exc}")
            return self._fallback_action(state)

    def _build_docs_text(self, docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """(로직 유지)"""
        if not docs: return "", []
        chunks = []
        sources = []
        for idx, doc in enumerate(docs, 1):
            metadata = getattr(doc, "metadata", {}) or {}
            chunks.append(f"[문서 {idx}] ({metadata.get('source','?')} - {metadata.get('section','')})\n{doc.page_content}")
            sources.append({"idx": idx, "filename": metadata.get('source'), "section": metadata.get('section')})
        return "\n\n".join(chunks), sources

    def _ensure_docs_text(self, state: AgentState) -> str:
        """(로직 유지)"""
        if state.get("docs_text"): return state.get("docs_text")
        docs_text, sources = self._build_docs_text(state.get("retrieved_docs") or [])
        state["docs_text"] = docs_text
        if sources: state["sources"] = sources
        return docs_text

    def _generate_final_report(self, state: AgentState) -> AgentState:
        """(로직 유지)"""
        rag_output = self._ensure_docs_text(state)
        user_query = state.get("user_query", "")
        # state에서 source_references를 가져옴 (RAGAgent가 생성한 것)
        source_references = state.get("source_references", [])

        if not rag_output:
            msg = "문서가 없어 보고서를 생성할 수 없습니다."
            state["summary_cause"] = msg; state["summary_action_plan"] = msg; state["report_text"] = msg
            return state

        try:
            # summarize_accident_cause, generate_action_plan은 내부적으로 ChatOpenAI를 쓰므로 동기 함수
            summary_cause = summarize_accident_cause(rag_output, user_query)
            action_plan = generate_action_plan(rag_output, user_query, source_references)
            combined = f"【사고발생 경위】\n{summary_cause}\n\n【조치사항 및 향후조치계획】\n{action_plan}"

            state["summary_cause"] = summary_cause
            state["summary_action_plan"] = action_plan
            state["report_text"] = combined
            state["report_summary"] = (combined[:200] + "...") if len(combined) > 200 else combined
            state["route"] = "report_complete"
        except Exception as exc:
            state["report_text"] = f"보고서 생성 실패: {exc}"
        return state

    def _run_web_search(self, state: AgentState) -> AgentState:
        return state 

    def _create_docx_file(self, state: AgentState) -> AgentState:
        """(로직 유지)"""
        user_query = state.get("user_query", "")
        summary_cause = state.get("summary_cause", "")
        action_plan = state.get("summary_action_plan", "")
        source_references = state.get("source_references", [])
        
        if not user_query or not summary_cause or not action_plan: return state

        try:
            docx_path = create_accident_report_docx(
                user_query=user_query,
                cause_text=summary_cause,
                action_text=action_plan,
                source_references=source_references, # ✅ 여기서 docx_writer에 전달됨
            )
            with open(docx_path, "rb") as f:
                docx_bytes = f.read()
            state["docx_path"] = docx_path
            state["docx_bytes"] = docx_bytes
            state["route"] = "docx_complete"
        except Exception as exc:
            print(f"❌ DOCX 생성 실패: {exc}")
            state["docx_path"] = None
        return state

    async def run(self, state: AgentState) -> AgentState: 
        print(f"\n{'='*80}\n📝 [{self.name}] 실행 중...\n{'='*80}")
        
        action, reason = await self._decide_action(state) 
        
        # HITL 초기화
        state["hitl_action"] = None
        state["hitl_payload"] = {}
        
        print(f"🤖 선택된 작업: {action} | 이유: {reason}")

        handler = self.action_handlers.get(action)
        if handler:
            # handler는 동기 함수이므로 그냥 호출 (필요시 cl.make_async 사용 가능)
            state = handler(state) 
        elif action == "noop":
            print("ℹ️ 수행할 작업이 없습니다.")
        else:
            print(f"⚠️ 알 수 없는 작업 '{action}'")

        return state

# ========================================
# WebSearchAgent (기존 유지)
# ========================================
class WebSearchAgent:
    def __init__(self):
        self.searcher = WebSearch()
    
    async def run(self, state: AgentState) -> AgentState: 
        print("\n" + "🌐"*50 + "\n🌐  WebSearchAgent 실행\n" + "🌐"*50)
        
        user_query = state.get("user_query", "")
        if not user_query:
            state["web_search_completed"] = False
            return state
        
        try:
            # WebSearch.run()이 동기 함수이므로 cl.make_async로 비동기 실행
            state = await cl.make_async(self.searcher.run)(state) 
            
            # HITL 초기화
            state["hitl_action"] = None
            state["hitl_payload"] = {}

            state["web_search_completed"] = True
            state["route"] = "web_search_complete"
            print("\n✅ WebSearchAgent 완료!")
            
        except Exception as e:
            print(f"❌ WebSearchAgent 오류: {e}")
            state["web_search_completed"] = False
            state["web_error"] = str(e)
            
        return state

# ========================================
# Agent Registry
# ========================================
AGENT_REGISTRY = {
    "RAGAgent": RAGAgent(),
    "WebSearchAgent": WebSearchAgent(),
    "ReportWriterAgent": ReportWriterAgent(),
}

def get_agent(agent_name: str):
    return AGENT_REGISTRY.get(agent_name)