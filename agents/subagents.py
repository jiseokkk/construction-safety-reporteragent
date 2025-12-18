"""
SubAgents - RAGAgent, WebSearchAgent, ReportWriterAgent (Self-Correction with GPT-4o)

✅ 수정 사항:
1. ReportEvaluation Pydantic 모델 추가 (Self-Correction용)
2. ReportSelfCorrector 클래스 추가 (GPT-4o 기반 평가/수정)
3. ReportWriterAgent: 초안 작성 -> 평가 -> 수정 루프 적용
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
from core.llm_factory import get_llm
from core.agentstate import AgentState
from core.docx_writer import create_accident_report_docx
from core.final_report import summarize_accident_cause, generate_action_plan
from core.websearch import WebSearch
from core.retriever import SingleDBHybridRetriever
from core.chunk_formatter import ChunkFormatter

DB_ROOT = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB"

# ======================================================================
# 1. Pydantic 모델 정의
# ======================================================================

class DBRoutingPlan(BaseModel):
    """RAGAgent의 DB 선택 계획"""
    db_list: List[str] = Field(description="검색할 데이터베이스 폴더 이름 목록")
    fallback: bool = Field(description="검색 결과가 부족할 경우 Fallback DB를 사용할지 여부")
    fallback_db: str = Field(description="Fallback으로 사용할 DB 이름")
    reasoning: str = Field(description="이 DB들을 선택한 논리적 근거") 

class ReportAction(BaseModel):
    """ReportWriterAgent의 다음 행동 결정"""
    action: Literal["web_search", "final_report", "create_docx", "noop"] = Field(...)
    reason: str = Field(...)

# 🔥 [NEW] 보고서 평가 모델 (먼저 정의되어야 함)
class ReportEvaluation(BaseModel):
    """보고서 품질 평가 결과"""
    faithfulness_score: int = Field(description="1~5점. 원문(Context)에 없는 내용을 지어내지 않았는지 평가.")
    clarity_score: int = Field(description="1~5점. 논리적 흐름과 문장이 명확한지 평가.")
    feedback: str = Field(description="점수가 낮다면 구체적으로 어떤 부분을 고쳐야 하는지 지적 (한글).")
    passed: bool = Field(description="두 점수 모두 4점 이상이면 True, 아니면 False")


# ========================================
# 헬퍼 함수
# ========================================
def load_db_descriptions():
    db_info = {}
    if not os.path.exists(DB_ROOT): return {}
    for folder in os.listdir(DB_ROOT):
        desc_path = os.path.join(DB_ROOT, folder, "description.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                db_info[folder] = json.load(f)
    return db_info


# ========================================
# RAGAgent (기존 유지)
# ========================================
class RAGAgent:
    name = "RAGAgent"

    def __init__(self):
        self.db_info: Dict[str, Any] = load_db_descriptions() 
        self.available_dbs: List[str] = sorted(self.db_info.keys())
        self.formatter = ChunkFormatter()
        # GPT-4o 사용
        self.llm = get_llm(mode="smart")
        
        self.parser = PydanticOutputParser(pydantic_object=DBRoutingPlan)
  
    def _build_structured_query(self, state: AgentState) -> str:
        user_query = state.get("user_query", "")
        extra_lines = []
        for k in ["공종", "작업프로세스", "사고 유형", "사고객체(중분류)", "장소(중분류)"]:
            val = state.get(k)
            if val: extra_lines.append(f"{k}: {val}")
        
        hitl_payload = state.get('hitl_payload', {})
        if hitl_payload.get('keywords'):
            extra_lines.append("\n[HITL 추가 키워드]\n" + ", ".join(hitl_payload['keywords']))
        
        return f"[User Query]\n{user_query}\n\n[구조화 정보]\n" + "\n".join(extra_lines)

    async def _plan_db_selection(self, structured_query: str) -> Dict[str, Any]: 
        system_template = """
당신은 건설안전 RAG 시스템의 DB 라우팅 Agent입니다.
사고 속성을 분석하여 가장 적합한 DB를 1~3개 선택하세요.

[DB 목록]
{db_info}

형식:
{format_instructions}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{structured_query}")
        ])
        chain = prompt | self.llm | self.parser

        try:
            plan = await chain.ainvoke({
                "db_info": json.dumps(self.db_info, ensure_ascii=False, indent=2),
                "structured_query": structured_query,
                "format_instructions": self.parser.get_format_instructions()
            })
            return plan.dict()
        except:
            return {"db_list": ["08_general"], "fallback": True, "fallback_db": "08_general"}

    def _sanitize_plan(self, plan: Dict) -> Dict:
        valid_list = [db for db in plan.get("db_list", []) if db in self.available_dbs]
        if not valid_list: valid_list = ["08_general"] if "08_general" in self.available_dbs else []
        return {"db_list": valid_list, "fallback": plan.get("fallback", False), "fallback_db": plan.get("fallback_db", "08_general")}

    def _search_documents(self, db_list: List[str], query: str, top_k: int = 5) -> List[Document]:
        all_docs = []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        for db_name in db_list:
            db_path = os.path.join(DB_ROOT, db_name)
            if not os.path.exists(os.path.join(db_path, "index.faiss")): continue
            try:
                retriever = SingleDBHybridRetriever(db_dir=db_path, top_k=top_k, alpha=0.5)
                docs = retriever.retrieve(query) 
                for d in docs: d.metadata['db'] = db_name
                all_docs.extend(docs)
                del retriever
                gc.collect()
            except: continue
        return all_docs

    async def search_only(self, user_query: str, state: AgentState) -> List[Document]:
        structured_query = self._build_structured_query(state)
        hitl_action = state.get('hitl_action')
        
        if hitl_action == 'research_db':
            plan = {"db_list": state.get('hitl_payload', {}).get('dbs', []), "fallback": False}
        else:
            raw_plan = await self._plan_db_selection(structured_query)
            plan = self._sanitize_plan(raw_plan)
        
        all_docs = await cl.make_async(self._search_documents)(plan['db_list'], structured_query)
        
        if plan.get('fallback') and len(all_docs) < 3:
            fb_path = os.path.join(DB_ROOT, plan['fallback_db'])
            if os.path.exists(os.path.join(fb_path, "index.faiss")):
                fb_retriever = SingleDBHybridRetriever(db_dir=fb_path, top_k=5)
                fb_docs = await cl.make_async(fb_retriever.retrieve)(structured_query)
                all_docs.extend(fb_docs)

        return all_docs[:10]

    async def run(self, state: AgentState) -> AgentState: 
        print(f"\n📚 [RAGAgent] 실행")
        new_docs = await self.search_only(state.get("user_query", ""), state)
        existing_docs = state.get("retrieved_docs", []) or []
        hitl_action = state.get("hitl_action")
        
        final_docs = []
        if hitl_action in ["research_db", "research_keyword"]:
            seen = set()
            for doc in existing_docs:
                key = (doc.metadata.get("source", ""), doc.page_content[:50])
                seen.add(key)
                final_docs.append(doc)
            for doc in new_docs:
                key = (doc.metadata.get("source", ""), doc.page_content[:50])
                if key not in seen:
                    final_docs.append(doc)
                    seen.add(key)
        else:
            final_docs = new_docs

        state["hitl_action"] = None
        state["hitl_payload"] = {}
        
        docs_text = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(final_docs)])
        
        source_references = []
        for i, doc in enumerate(final_docs, 1):
            md = doc.metadata
            source_references.append({
                "idx": i,
                "filename": md.get("file") or md.get("source", "Unknown"),
                "hierarchy": md.get("hierarchy_str", ""),
                "section": md.get("section", ""),
                "db": md.get("db", ""),
                "relevance_summary": md.get("summary", ""),
                "key_sentences": []
            })

        state["retrieved_docs"] = final_docs
        state["docs_text"] = docs_text
        state["source_references"] = source_references
        
        if state.get("user_intent") == "search_only":
            state["wait_for_user"] = True
        
        return state


# ========================================
# 🔥 [NEW] ReportSelfCorrector (GPT-4o 전용)
# ========================================
class ReportSelfCorrector:
    """보고서를 평가하고, 피드백을 반영해 수정하는 Helper Class"""
    
    def __init__(self):
        # ⚠️ GPT-4o 사용
        self.llm = get_llm(mode="smart")
        
        self.eval_parser = PydanticOutputParser(pydantic_object=ReportEvaluation)

    async def evaluate(self, report_text: str, context_text: str, user_query: str) -> ReportEvaluation:
        """보고서 평가 (Self-Correction)"""
        
        system_template = """
당신은 건설안전 보고서의 엄격한 편집장(Editor)입니다.
작성된 보고서가 제공된 "참고 문서(Context)"에 기반하여 사실에 입각해 작성되었는지 평가하세요.

[평가 기준]
1. Faithfulness (충실성): 보고서의 내용이 참고 문서에 근거하는가? (없는 말을 지어내면 감점)
2. Clarity (명확성): 문장이 명확하고 사고 원인과 대책이 논리적인가?

반드시 아래 JSON 형식으로 응답하세요:
{format_instructions}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "사용자 질문: {user_query}\n\n[참고 문서]\n{context}\n\n[작성된 보고서]\n{report}")
        ])
        
        chain = prompt | self.llm | self.eval_parser
        
        try:
            print("\n🧐 [Self-Correction] 보고서 품질 평가 중...")
            result = await chain.ainvoke({
                "user_query": user_query,
                "context": context_text[:15000], # 토큰 제한 고려
                "report": report_text,
                "format_instructions": self.eval_parser.get_format_instructions()
            })
            print(f"   📊 평가 점수: 충실성 {result.faithfulness_score}/5, 명확성 {result.clarity_score}/5")
            return result
        except Exception as e:
            print(f"❌ 평가 중 오류 발생 (통과 처리): {e}")
            return ReportEvaluation(faithfulness_score=5, clarity_score=5, feedback="", passed=True)

    async def refine(self, report_text: str, feedback: str, context_text: str) -> str:
        """피드백을 반영하여 보고서 수정 (Refinement)"""
        
        system_template = """
당신은 건설안전 보고서 수정 전문가입니다.
편집장의 피드백을 반영하여 보고서를 다시 작성하세요.

[지침]
1. **피드백 내용을 철저히 반영**하여 내용을 수정/보완할 것.
2. 참고 문서에 없는 내용(Hallucination)이 지적되었다면 반드시 삭제할 것.
3. 기존 보고서의 구조(사고발생 경위, 조치사항 등)는 유지할 것.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", """
[참고 문서]
{context}

[기존 보고서]
{report}

[편집장 피드백]
{feedback}

위 피드백을 반영하여 개선된 보고서를 작성해줘.
""")
        ])
        
        chain = prompt | self.llm
        
        print(f"🔧 [Self-Correction] 피드백 반영하여 보고서 수정 중...")
        response = await chain.ainvoke({
            "context": context_text[:15000],
            "report": report_text,
            "feedback": feedback
        })
        
        return response.content


# ========================================
# ReportWriterAgent (Self-Correction 루프 적용 - GPT-4o 전용)
# ========================================
class ReportWriterAgent:
    name = "ReportWriterAgent"

    def __init__(self):
        self.action_handlers = {
            "final_report": self._generate_final_report_with_correction, # ✅ 핸들러 이름 변경
            "web_search": self._run_web_search,
            "create_docx": self._create_docx_file,
        }
        # ✅ 실험용으로 GPT-4o 고정
        self.llm = get_llm(mode="smart")
        
        self.parser = PydanticOutputParser(pydantic_object=ReportAction)
        
        # ✅ Self-Correction 모듈 추가
        self.corrector = ReportSelfCorrector()

    def _summarize_state(self, state: AgentState) -> str:
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
        print("\n⚠️ FALLBACK 모드 활성화 (ReportWriter)")
        if not state.get("report_text"): return "final_report", "[Fallback] 보고서 생성"
        if not state.get("docx_path"): return "create_docx", "[Fallback] DOCX 생성"
        return "noop", "[Fallback] 작업 완료"

    async def _decide_action(self, state: AgentState) -> Tuple[str, str]: 
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

        chain = prompt | self.llm | self.parser
        summary = self._summarize_state(state)

        try:
            decision: ReportAction = await chain.ainvoke({
                "state_summary": summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            return decision.action, decision.reason
        except Exception as exc:
            print(f"⚠️ ReportWriter 의사결정 실패 (LCEL 오류): {exc}")
            return self._fallback_action(state)

    def _build_docs_text(self, docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        if not docs: return "", []
        chunks = []
        sources = []
        for idx, doc in enumerate(docs, 1):
            metadata = getattr(doc, "metadata", {}) or {}
            chunks.append(f"[문서 {idx}] ({metadata.get('source','?')} - {metadata.get('section','')})\n{doc.page_content}")
            sources.append({"idx": idx, "filename": metadata.get('source'), "section": metadata.get('section')})
        return "\n\n".join(chunks), sources

    def _ensure_docs_text(self, state: AgentState) -> str:
        if state.get("docs_text"): return state.get("docs_text")
        docs_text, sources = self._build_docs_text(state.get("retrieved_docs") or [])
        state["docs_text"] = docs_text
        if sources: state["sources"] = sources
        return docs_text

    # 🔥 [핵심 수정] 초안 생성 -> 평가 -> 수정 루프 구현
    async def _generate_final_report_with_correction(self, state: AgentState) -> AgentState:
        print("\n📝 [ReportWriter] 보고서 생성 프로세스 시작 (Self-Correction Enabled)")
        
        # 문서 텍스트 확보
        docs_text = self._ensure_docs_text(state)
        user_query = state.get("user_query", "")
        source_references = state.get("source_references", [])

        if not docs_text:
            state["report_text"] = "문서가 없어 보고서를 생성할 수 없습니다."
            return state

        # 1. 초안 생성 (Drafting)
        try:
            # summarize_accident_cause 등은 GPT-4o를 사용하는 외부 함수 (동기)
            summary_cause = summarize_accident_cause(docs_text, user_query)
            action_plan = generate_action_plan(docs_text, user_query, source_references)
            current_report = f"【사고발생 경위】\n{summary_cause}\n\n【조치사항 및 향후조치계획】\n{action_plan}"
        except Exception as e:
            print(f"❌ 초안 생성 실패: {e}")
            return state

        # 2. Self-Correction Loop (최대 2회 수정)
        MAX_RETRIES = 2
        
        for attempt in range(MAX_RETRIES):
            # (A) 평가 (Evaluate)
            evaluation = await self.corrector.evaluate(current_report, docs_text, user_query)
            
            if evaluation.passed:
                print(f"✅ 보고서 품질 통과 (시도 {attempt+1})")
                break
            
            # (B) 수정 (Refine) - 마지막 시도가 아닐 때만
            if attempt < MAX_RETRIES - 1:
                print(f"💡 피드백 반영: {evaluation.feedback}")
                current_report = await self.corrector.refine(current_report, evaluation.feedback, docs_text)
            else:
                print("⚠️ 최대 수정 횟수 도달. 현재 버전을 확정합니다.")

        # 3. 최종 결과 저장
        state["report_text"] = current_report
        # DOCX용 데이터는 구조 깨짐 방지를 위해 초안 데이터를 유지
        state["summary_cause"] = summary_cause 
        state["summary_action_plan"] = action_plan 
        
        state["route"] = "report_complete"
        return state

    def _run_web_search(self, state: AgentState) -> AgentState:
        return state 

    def _create_docx_file(self, state: AgentState) -> AgentState:
        user_query = state.get("user_query", "")
        summary_cause = state.get("summary_cause", "")
        action_plan = state.get("summary_action_plan", "")
        source_references = state.get("source_references", [])
        
        if not user_query: return state

        try:
            docx_path = create_accident_report_docx(
                user_query=user_query,
                cause_text=summary_cause,
                action_text=action_plan,
                source_references=source_references,
            )
            with open(docx_path, "rb") as f:
                state["docx_bytes"] = f.read()
            state["docx_path"] = docx_path
            state["route"] = "docx_complete"
        except Exception as exc:
            print(f"❌ DOCX 생성 실패: {exc}")
            state["docx_path"] = None
        return state

    async def run(self, state: AgentState) -> AgentState: 
        print(f"\n{'='*80}\n📝 [{self.name}] 실행 중...\n{'='*80}")
        
        action, reason = await self._decide_action(state) 
        state["hitl_action"] = None
        state["hitl_payload"] = {}
        
        print(f"🤖 선택된 작업: {action} | 이유: {reason}")

        if action == "final_report":
            state = await self._generate_final_report_with_correction(state)
        elif action == "create_docx":
            state = self._create_docx_file(state)
        elif action == "web_search":
            state = self._run_web_search(state)
        
        return state

# ========================================
# WebSearchAgent (최종 수정본: HITL 및 Source 통합)
# ========================================
class WebSearchAgent:
    def __init__(self):
        self.searcher = WebSearch()
        # 🔥 [수정] 요약(고지능 작업)은 get_llm("smart") (GPT-4o) 사용
        self.llm = get_llm("smart") 
    
    # 🔥 [추가] 웹 문서에서 Source Reference를 추출하는 헬퍼 함수
    def _extract_web_sources(self, docs_web: List[Document], existing_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tavily 검색 결과 Document를 source_references 형식에 맞게 변환하여 기존 리스트에 추가"""
        
        # 기존 source_references 리스트의 마지막 인덱스를 확인
        # RAG 문서와 웹 문서가 섞여서 들어가므로 인덱스를 이어서 부여합니다.
        start_idx = len(existing_sources) + 1
        
        new_sources = []
        for i, doc in enumerate(docs_web):
            metadata = doc.metadata
            
            # 웹 검색 결과는 'web'으로 명확히 구분
            source_entry = {
                "idx": start_idx + i,
                "filename": metadata.get("title", metadata.get("source", "웹 문서")), # 제목 또는 URL을 파일 이름으로 사용
                "hierarchy": "",
                "section": metadata.get("source", "N/A"), # URL을 섹션으로 사용
                "db": "web", # 웹 검색임을 명시
                "relevance_summary": doc.page_content[:150] + "...", # 내용의 일부를 요약으로 사용
                "key_sentences": []
            }
            new_sources.append(source_entry)
            
        return existing_sources + new_sources

    # 🔥 [추가] 웹 검색 결과를 요약하는 헬퍼 함수
    async def _summarize_web_docs(self, state: AgentState) -> str:
        web_docs: List[Document] = state.get("web_docs") or []
        if not web_docs:
            return "웹 검색 결과가 없습니다."
        
        doc_texts = "\n---\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in web_docs])
        query = state.get("web_query")
        
        system_template = "당신은 사용자 질문에 기반하여 웹 검색 결과를 간결하게 요약해주는 전문가입니다. 요약은 한국어로 작성하며, 검색 결과를 모두 포함하되 중복을 제거하고 핵심만 정리하세요."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "검색 질문: {query}\n\n[검색 결과]\n{doc_texts}\n\n이 검색 결과들을 바탕으로 질문에 답이 될 만한 내용을 2~3문장으로 요약하고, 관련 출처를 명시해줘.")
        ])
        
        chain = prompt | self.llm
        
        try:
            print("\n📰 [WebSearchAgent] 검색 결과 요약 중...")
            summary = await chain.ainvoke({"query": query, "doc_texts": doc_texts[:10000]})
            return summary.content
        except Exception as e:
            print(f"❌ 웹 검색 결과 요약 실패: {e}")
            return "웹 검색 결과가 있지만, 요약하는 데 실패했습니다. 원본 문서를 참조하세요."

    async def run(self, state: AgentState) -> AgentState: 
        print("\n" + "🌐"*50 + "\n🌐  WebSearchAgent 실행\n" + "🌐"*50)
        
        user_query = state.get("user_query", "")
        if not user_query:
            state["web_search_completed"] = False
            return state
        
        try:
            # 1. 웹 검색 실행 (state["web_docs"]와 state["retrieved_docs"]가 갱신됨)
            state = await cl.make_async(self.searcher.run)(state) 
            docs_web: List[Document] = state.get("web_docs") or [] # 검색된 웹 문서

            # 2. 🔥 [수정] 검색된 웹 문서를 source_references에 추가
            existing_sources = state.get("source_references", []) or []
            updated_sources = self._extract_web_sources(docs_web, existing_sources)
            state["source_references"] = updated_sources
            
            # 3. 검색 결과를 요약하여 상태에 저장
            summary_text = await self._summarize_web_docs(state)
            state["web_search_summary"] = summary_text # 사용자에게 보여줄 요약
            
            # --- 🔥 [핵심 수정] HITL 단계를 위해 사용자 대기 상태로 변경 ---
            state["hitl_action"] = None # 다음 루프에서 HITL이 실행되도록 초기화
            state["hitl_payload"] = {}
            
            state["web_search_completed"] = True
            state["wait_for_user"] = True # 사용자 피드백 대기
            state["route"] = "await_web_feedback"
            
            print("\n✅ WebSearchAgent 완료! (사용자 피드백 대기)")
            
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