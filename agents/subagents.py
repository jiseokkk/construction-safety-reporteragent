"""
SubAgents (개선된 버전 v5 - WebSearchAgent 독립)
- RAGAgent: 자동 판단 기반 Human-in-the-Loop
- WebSearchAgent: 웹 검색 전담 (신규)
- ReportWriterAgent: 보고서 생성
"""

from typing import Any, Dict, List, Tuple
import json
import os

from core.agentstate import AgentState
from core.llm_utils import call_llm
from core.docx_writer import create_accident_report_docx
from core.final_report import summarize_accident_cause, generate_action_plan
from core.websearch import WebSearch
from core.retriever import SingleDBHybridRetriever
from core.chunk_formatter import ChunkFormatter
from core.human_feedback_collector import HumanFeedbackCollector

DB_ROOT = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB"


def load_db_descriptions():
    """DB 폴더의 description.json 로드"""
    db_info = {}
    for folder in os.listdir(DB_ROOT):
        desc_path = os.path.join(DB_ROOT, folder, "description.json")
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                db_info[folder] = json.load(f)
    return db_info


def parse_json_with_recovery(raw: str, default: dict, context: str = "") -> dict:
    """LLM이 reasoning + JSON을 섞어서 내보내도 JSON 부분만 뽑아주는 복구용 파서"""
    if not raw or not isinstance(raw, str):
        print(f"⚠️ [{context}] 응답이 비어있음 → 기본값 사용")
        return default

    raw_stripped = raw.strip()

    # 1차: 전체를 그대로 파싱 시도
    try:
        return json.loads(raw_stripped)
    except Exception:
        pass

    # 2차: <o> 태그 내부 추출
    try:
        if "<o>" in raw_stripped and "</o>" in raw_stripped:
            start = raw_stripped.index("<o>") + len("<o>")
            end = raw_stripped.index("</o>")
            json_part = raw_stripped[start:end].strip()
            return json.loads(json_part)
    except Exception:
        pass

    # 3차: 첫 '{' ~ 마지막 '}' 사이만 추출
    try:
        start = raw_stripped.index("{")
        end = raw_stripped.rindex("}") + 1
        candidate = raw_stripped[start:end]
        return json.loads(candidate)
    except Exception as e:
        print(f"⚠️ [{context}] JSON 파싱 실패 → 기본값 사용: {e}")
        return default


def should_enable_feedback(state: AgentState, docs: List = None) -> bool:
    """
    Human-in-the-Loop 필요성 자동 판단
    
    우선순위:
    1. 사용자 명시 → 그대로 따름
    2. 보고서 생성 → 항상 확인
    3. 심각한 사고 → 항상 확인
    4. 검색 품질 낮음 → 확인
    5. 단순 검색 → 확인 안 함
    """
    
    # 1. 사용자가 명시적으로 설정했으면 우선
    if "enable_human_feedback" in state:
        return state["enable_human_feedback"]
    
    # 2. 의도별 기본값
    user_intent = state.get("user_intent", "generate_report")
    
    if user_intent == "csv_info":
        return False  # CSV 조회는 필요 없음
    
    elif user_intent == "search_only":
        # 검색 품질 확인
        if docs and len(docs) < 3:
            print("⚠️ 판단: 검색 결과 부족 → Human-in-the-Loop 활성화")
            return True
        return False  # 단순 검색은 기본적으로 자동
    
    elif user_intent == "generate_report":
        # 보고서는 사고 심각도 확인
        accident_type = str(state.get("사고 유형", ""))
        
        # 심각한 사고는 무조건 확인
        serious_keywords = ["사망", "중상", "끼임", "떨어짐", "맞음", "깔림"]
        if any(keyword in accident_type for keyword in serious_keywords):
            print(f"⚠️ 판단: 심각한 사고 ({accident_type}) → Human-in-the-Loop 활성화")
            return True
        
        # 보고서는 기본적으로 확인하는 게 안전
        print("ℹ️ 판단: 보고서 생성 → Human-in-the-Loop 활성화")
        return True
    
    # 3. 기본값: 안전하게 확인
    return True


# ========================================
# 1. RAGAgent - 문서 검색 (Human-in-the-Loop 통합)
# ========================================
class RAGAgent:
    name = "RAGAgent"

    def __init__(self):
        self.db_info: Dict[str, Any] = load_db_descriptions()
        self.available_dbs: List[str] = sorted(self.db_info.keys())
        self.formatter = ChunkFormatter()
        self.feedback_collector = HumanFeedbackCollector()  # ✅ 추가
        print(f"📚 사용 가능한 DB 목록: {self.available_dbs}")

    def _build_structured_query(self, state: AgentState) -> str:
        """사용자 쿼리 + 구조화 정보를 합친 텍스트"""
        user_query = state.get("user_query", "")
        
        gongsung = state.get("공종") or state.get("gongsung")
        process = state.get("작업프로세스") or state.get("process")
        acc_type = state.get("사고 유형") or state.get("accident_type")
        obj = state.get("사고객체(중분류)") or state.get("object")
        location = state.get("장소(중분류)") or state.get("location")

        extra_lines = []
        if gongsung:
            extra_lines.append(f"공종: {gongsung}")
        if process:
            extra_lines.append(f"작업프로세스: {process}")
        if acc_type:
            extra_lines.append(f"사고유형: {acc_type}")
        if obj:
            extra_lines.append(f"사고객체: {obj}")
        if location:
            extra_lines.append(f"장소: {location}")

        extra_block = "\n".join(extra_lines)
        structured_query = f"[User Query]\n{user_query}\n"
        if extra_block:
            structured_query += "\n[추가 구조화 정보]\n" + extra_block

        return structured_query

    def _plan_db_selection(self, structured_query: str) -> Dict[str, Any]:
        """LLM에게 DB 선택 계획 요청 (ReAct 스타일)"""
        
        system_prompt = """
당신은 건설안전 RAG 시스템의 DB 라우팅을 담당하는 Agent입니다.

먼저 <thinking> 블록에서 사고 분석 및 DB 선택 이유를 설명하세요:

<thinking>
1) 사용자 질의에서 핵심 키워드 추출
2) 각 DB의 특징과 사고 유형 매칭
3) 가장 관련 있는 DB 1~3개 선택
4) Fallback DB 필요 여부 판단
</thinking>

그 다음 <o> 블록에 JSON만 출력하세요:

<o>
{
  "db_list": ["01_bridge", "05_crane"],
  "fallback": true,
  "fallback_db": "08_general"
}
</o>

규칙:
- db_list: 검색할 DB 이름 리스트 (1~3개)
- fallback: 검색 결과가 부족할 때 추가 DB 사용 여부
- fallback_db: fallback 시 사용할 DB (보통 "08_general")
- <o> 블록 안에는 순수 JSON만 작성
"""

        user_prompt = f"""
[사용자 사고 정보]
{structured_query}

[사용 가능한 DB 목록 및 설명]
{json.dumps(self.db_info, ensure_ascii=False, indent=2)}
"""

        plan_raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )

        print("🧾 RAGAgent LLM 원시 응답:")
        print(plan_raw)

        default_plan = {
            "db_list": ["08_general"] if "08_general" in self.available_dbs else (self.available_dbs[:1] or []),
            "fallback": False,
            "fallback_db": "08_general" if "08_general" in self.available_dbs else (self.available_dbs[0] if self.available_dbs else "")
        }

        plan = parse_json_with_recovery(
            raw=plan_raw,
            default=default_plan,
            context="RAGAgent DB 선택"
        )
        return plan

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """실제 존재하는 DB만 남기고 검증"""
        
        db_list = plan.get("db_list", []) or []
        fallback_flag = bool(plan.get("fallback", False))
        fallback_db_name = plan.get("fallback_db", "08_general")

        # 실제 존재하는 DB만 필터링
        valid_db_list = [db for db in db_list if db in self.available_dbs]

        if not valid_db_list:
            print(f"⚠️ 선택된 DB가 존재하지 않음 → 기본값으로 보정")
            if "08_general" in self.available_dbs:
                valid_db_list = ["08_general"]
            elif self.available_dbs:
                valid_db_list = [self.available_dbs[0]]
            else:
                valid_db_list = []

        # Fallback DB 검증
        if fallback_flag and fallback_db_name not in self.available_dbs:
            print(f"⚠️ fallback_db '{fallback_db_name}' 존재하지 않음 → 보정")
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

    def _search_documents(self, db_list: List[str], query: str, top_k: int = 5) -> List[Any]:
        """여러 DB에서 문서 검색"""
        all_docs = []
        
        for db_name in db_list:
            db_path = os.path.join(DB_ROOT, db_name)

            if not os.path.exists(os.path.join(db_path, "index.faiss")):
                print(f"⚠️ {db_path}에 index.faiss 없음 → 스킵")
                continue

            print(f"📂 검색 대상 DB: {db_path}")
            retriever = SingleDBHybridRetriever(
                db_dir=db_path,
                top_k=top_k,
                alpha=0.5,
            )
            docs = retriever.retrieve(query)
            all_docs.extend(docs)
        
        return all_docs

    def run(self, state: AgentState) -> AgentState:
        print("\n" + "="*80)
        print("📚 [RAGAgent] 실행 - Self-Orchestrating DB Selection + Human-in-the-Loop")
        print("="*80)

        # 1) 구조화된 쿼리 생성
        structured_query = self._build_structured_query(state)
        user_query = state.get("user_query", "")

        # 2) LLM에게 DB 선택 계획 요청
        raw_plan = self._plan_db_selection(structured_query)
        plan = self._sanitize_plan(raw_plan)
        print(f"🧠 최종 사용 계획: {plan}")

        db_list = plan.get("db_list", []) or []
        fallback_flag = plan.get("fallback", False)
        fallback_db_name = plan.get("fallback_db", "08_general")

        # 3) 선택된 DB들에서 검색
        all_docs = self._search_documents(db_list, structured_query, top_k=5)

        # 4) Fallback 검색
        if fallback_flag and len(all_docs) < 3 and fallback_db_name:
            fb_path = os.path.join(DB_ROOT, fallback_db_name)
            print(f"⚠️ Fallback DB 검색 실행 → {fb_path}")

            if os.path.exists(os.path.join(fb_path, "index.faiss")):
                fb_retriever = SingleDBHybridRetriever(
                    db_dir=fb_path,
                    top_k=5,
                    alpha=0.5,
                )
                fallback_docs = fb_retriever.retrieve(structured_query)
                all_docs.extend(fallback_docs)
            else:
                print(f"⚠️ Fallback DB '{fallback_db_name}' index 없음")

        # 5) 최종 문서 정리
        final_docs = all_docs[:10]

        # ✅ 6) Human-in-the-Loop 필요성 자동 판단
        max_feedback_loops = 3  # 최대 3번까지 피드백 루프
        feedback_loop_count = 0
        processed_results = None  # ✅ Phase 3 결과 저장용
        
        while feedback_loop_count < max_feedback_loops:
            if should_enable_feedback(state, final_docs):
                print("\n" + "🔄" * 50)
                print(f"🔄  Human-in-the-Loop 모드 활성화 (루프 {feedback_loop_count + 1}/{max_feedback_loops})")
                print("🔄" * 50)
                
                # ✅ Phase 3 처리 결과 저장
                processed_results = None
                if self.feedback_collector.enable_advanced_processing:
                    processed_results = self.feedback_collector.processor.process_documents(
                        docs=final_docs,
                        user_query=user_query,
                        remove_duplicates=False,  # 이미 중복 제거됨
                        extract_key_sentences=True
                    )
                
                # ✅ Chainlit의 async 함수 호출
                import asyncio
                loop = asyncio.get_event_loop()
                
                final_docs, feedback = loop.run_until_complete(
                    self.feedback_collector.process(
                        final_docs,
                        user_query,
                        self.available_dbs
                    )
                )
                
                # ✅ 웹 검색 요청 처리 (신규)
                if feedback.get("web_search_requested", False):
                    state["web_search_requested"] = True
                    print("\n✅ 웹 검색이 요청되었습니다.")
                    print("   HITL을 종료하고 Orchestrator로 돌아갑니다...")
                    print("   Orchestrator가 WebSearchAgent를 호출합니다.")
                    break  # HITL 종료
                
                # 7) 피드백 처리
                needs_reloop = False
                
                if feedback["action"] == "research_keyword":
                    # 키워드 추가 재검색
                    additional_keywords = feedback.get("keywords", [])
                    if additional_keywords:
                        enhanced_query = structured_query + "\n키워드: " + ", ".join(additional_keywords)
                        print(f"\n🔍 키워드 추가 재검색: {', '.join(additional_keywords)}")
                        
                        research_docs = self._search_documents(db_list, enhanced_query, top_k=5)
                        
                        # 기존 문서와 합치기
                        final_docs = feedback["original_docs"] + research_docs
                        final_docs = final_docs[:15]  # 최대 15개
                        
                        print(f"✅ 재검색 완료: 총 {len(final_docs)}개 문서")
                        needs_reloop = True  # 재검색했으니 다시 피드백
                
                elif feedback["action"] == "research_db":
                    # DB 변경 재검색
                    new_dbs = feedback.get("dbs", [])
                    if new_dbs:
                        print(f"\n🔍 DB 변경 재검색: {', '.join(new_dbs)}")
                        
                        research_docs = self._search_documents(new_dbs, structured_query, top_k=5)
                        
                        # 기존 문서와 합치기
                        final_docs = feedback["original_docs"] + research_docs
                        final_docs = final_docs[:15]  # 최대 15개
                        
                        print(f"✅ 재검색 완료: 총 {len(final_docs)}개 문서")
                        needs_reloop = True  # 재검색했으니 다시 피드백
                
                elif feedback["action"] in ["accept_all", "select_partial"]:
                    # 사용자가 확정했으면 루프 종료
                    print(f"\n✅ 사용자 확정: 피드백 루프 종료")
                    break
                
                # 재검색했으면 루프 계속, 아니면 종료
                if needs_reloop:
                    feedback_loop_count += 1
                    print(f"\n🔄 재검색 결과를 다시 확인합니다...")
                    continue
                else:
                    break
            else:
                print("\n⚡ 자동 모드: 검색 결과 즉시 사용 (Human-in-the-Loop 생략)")
                break
        
        if feedback_loop_count >= max_feedback_loops:
            print(f"\n⚠️ 최대 피드백 루프 횟수({max_feedback_loops})에 도달했습니다. 현재 문서로 진행합니다.")

        # 8) 문서 텍스트 생성
        docs_text = "\n\n".join(
            f"[문서 {i+1}] ({doc.metadata.get('file', '?')}, {doc.metadata.get('section', '')})\n{doc.page_content}"
            for i, doc in enumerate(final_docs)
        )

        sources = [
            {
                "idx": i + 1,
                "filename": doc.metadata.get("file", ""),
                "section": doc.metadata.get("section", ""),
                "db": doc.metadata.get("db", "")
            }
            for i, doc in enumerate(final_docs)
        ]
        
        # ✅ 9) 근거 자료 정보 생성 (핵심 문장 포함)
        source_references = []
        if processed_results:
            for i, result in enumerate(processed_results, 1):
                doc = result["doc"]
                ref_info = {
                    "idx": i,
                    "filename": doc.metadata.get("file", "알 수 없음"),
                    "section": doc.metadata.get("section", ""),
                    "hierarchy": doc.metadata.get("hierarchy_str", ""),
                    "relevance_summary": result.get("relevance_summary", ""),
                    "key_sentences": result.get("key_sentences", [])
                }
                source_references.append(ref_info)
        else:
            # Phase 3 없을 때는 기본 정보만
            for i, doc in enumerate(final_docs, 1):
                ref_info = {
                    "idx": i,
                    "filename": doc.metadata.get("file", "알 수 없음"),
                    "section": doc.metadata.get("section", ""),
                    "hierarchy": doc.metadata.get("hierarchy_str", ""),
                    "relevance_summary": "",
                    "key_sentences": []
                }
                source_references.append(ref_info)

        # 10) 포맷팅 (search_only 모드용)
        user_intent = state.get("user_intent", "generate_report")
        if user_intent == "search_only" and final_docs:
            print("\n📝 검색 결과를 가독성 좋게 포맷팅 중...")
            formatted_result = self.formatter.format_chunks(final_docs, user_query)
            state["formatted_result"] = formatted_result

        # State 업데이트
        state["retrieved_docs"] = final_docs
        state["docs_text"] = docs_text
        state["sources"] = sources
        state["source_references"] = source_references  # ✅ 추가
        state["route"] = "retrieve_complete"

        print(f"\n✅ RAGAgent 검색 완료! (총 {len(final_docs)}개 문서)")
        return state


# ========================================
# 2. ReportWriterAgent - 보고서 작성
# ========================================
class ReportWriterAgent:
    """
    보고서 작성, 웹검색, DOCX 생성을 하나의 Agent에서 결정·수행
    """
    name = "ReportWriterAgent"

    def __init__(self):
        self.action_handlers = {
            "final_report": self._generate_final_report,
            "web_search": self._run_web_search,
            "create_docx": self._create_docx_file,
        }

    def _summarize_state(self, state: AgentState) -> str:
        """State를 LLM이 이해하기 쉽게 요약"""
        
        doc_cnt = len(state.get("retrieved_docs") or [])
        docs_text_length = len(state.get("docs_text") or "")
        web_done = state.get("web_search_completed", False)
        web_count = state.get("web_search_count", 0)
        report_ready = bool(state.get("report_text"))
        docx_ready = bool(state.get("docx_path"))

        summary = f"""
현재 상태:

[사용자 질의]
{state.get('user_query', 'N/A')}

[RAG 검색 결과]
- 문서 수: {doc_cnt}
- 문서 텍스트 길이: {docs_text_length} 글자

[웹 검색 상태]
- 웹 검색 완료: {'✅ 예' if web_done else '❌ 아니오'}
- 웹 검색 수행 횟수: {web_count}회

[보고서 상태]
- 보고서 생성: {'✅ 완료' if report_ready else '❌ 미완료'}

[DOCX 상태]
- DOCX 파일: {'✅ 완료' if docx_ready else '❌ 미완료'}

"""
        return summary

    def _fallback_action(self, state: AgentState) -> Tuple[str, str]:
        """LLM 실패 시 Rule-based fallback"""
        print("\n" + "⚠️ " * 40)
        print("⚠️  FALLBACK 모드 활성화 - LLM 작업 선택 실패로 Rule-based 로직 사용")
        print("⚠️ " * 40)
        
        if not state.get("report_text"):
            print("📌 [Fallback Rule 1] 보고서 필요 → final_report 선택")
            return "final_report", "[Fallback] 보고서가 없어 우선 생성합니다."
        
        if not state.get("docx_path"):
            print("📌 [Fallback Rule 2] DOCX 필요 → create_docx 선택")
            return "create_docx", "[Fallback] DOCX 파일이 없어 생성합니다."
        
        print("📌 [Fallback Rule 3] 모든 작업 완료 → noop 선택")
        return "noop", "[Fallback] 모든 작업이 완료되었습니다."

    def _decide_action(self, state: AgentState) -> Tuple[str, str]:
        """LLM을 사용하여 다음 작업 결정"""
        
        system_prompt = """
당신은 ReportWriterAgent로서, 현재 상태를 분석하고 다음 작업을 결정합니다.

<available_actions>
1. web_search
   - Tavily 웹 검색으로 부족한 정보 보완
   - RAG 결과가 부족하거나(3개 미만) 최신 정보가 필요할 때만 수행
   - ⚠️ 중요: 이미 웹 검색이 완료되었다면 절대 다시 수행하지 마세요

2. final_report
   - RAG/웹검색 결과로 사고 보고서 생성
   - 보고서가 없을 때 반드시 수행

3. create_docx
   - 보고서를 DOCX 파일로 변환
   - 보고서가 있지만 DOCX가 없을 때 수행

4. noop
   - 모든 작업 완료 시에만 선택
</available_actions>

<decision_rules>
1. 웹 검색 완료 여부를 반드시 확인하세요
2. 작업 순서: 보고서 없음 → final_report, DOCX 없음 → create_docx, 완료 → noop
3. 웹 검색은 매우 선택적으로만 수행
</decision_rules>

<output_format>
<thinking>
1) 웹 검색 완료 여부 확인
2) 보고서 존재 여부 확인
3) DOCX 존재 여부 확인
4) 다음 작업 결정
</thinking>

<o>
{
  "action": "final_report",
  "reason": "보고서가 없어 생성이 필요합니다."
}
</o>
</output_format>
"""
        
        summary = self._summarize_state(state)

        try:
            response_text = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": summary},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            
            print("🧾 ReportWriter LLM 응답 원문:")
            print(response_text)
            
            parsed = parse_json_with_recovery(
                raw=response_text,
                default={"action": "noop", "reason": "파싱 실패"},
                context="ReportWriterAgent 작업 선택"
            )
            
            action = parsed.get("action", "noop")
            reason = parsed.get("reason", "")
            
            return action, reason
            
        except Exception as exc:
            print(f"⚠️ LLM 의사결정 실패, fallback 사용: {exc}")
            return self._fallback_action(state)

    def _build_docs_text(self, docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """문서 리스트를 텍스트와 소스 정보로 변환"""
        if not docs:
            return "", []

        chunks = []
        sources = []
        for idx, doc in enumerate(docs, 1):
            metadata = getattr(doc, "metadata", {}) or {}
            page_content = getattr(doc, "page_content", "")
            filename = metadata.get("source") or metadata.get("url") or "?"
            section = metadata.get("section") or metadata.get("title") or ""

            chunks.append(f"[문서 {idx}] ({filename} - {section})\n{page_content}")
            sources.append({"idx": idx, "filename": filename, "section": section})

        return "\n\n".join(chunks), sources

    def _ensure_docs_text(self, state: AgentState) -> str:
        """docs_text가 없으면 retrieved_docs로부터 생성"""
        docs_text = state.get("docs_text")
        if docs_text:
            return docs_text

        docs = state.get("retrieved_docs") or []
        docs_text, sources = self._build_docs_text(docs)
        state["docs_text"] = docs_text
        if sources:
            state["sources"] = sources
        return docs_text

    def _generate_final_report(self, state: AgentState) -> AgentState:
        """사고 보고서 생성 (근거 자료 포함)"""
        rag_output = self._ensure_docs_text(state)
        user_query = state.get("user_query", "")
        source_references = state.get("source_references", [])  # ✅ 추가

        if not rag_output:
            print("⚠️ 문서가 없어 보고서를 생성할 수 없습니다.")
            msg = "문서가 없어 사고발생 경위 및 조치계획을 생성할 수 없습니다."
            state["summary_cause"] = msg
            state["summary_action_plan"] = msg
            state["report_text"] = msg
            return state

        try:
            summary_cause = summarize_accident_cause(rag_output, user_query)
            action_plan = generate_action_plan(rag_output, user_query, source_references)  # ✅ 추가
            combined = (
                f"【사고발생 경위】\n{summary_cause}\n\n"
                f"【조치사항 및 향후조치계획】\n{action_plan}"
            )

            state["summary_cause"] = summary_cause
            state["summary_action_plan"] = action_plan
            state["report_text"] = combined
            state["report_summary"] = (
                combined[:200] + "..." if len(combined) > 200 else combined
            )
            state["route"] = "report_complete"
            
            # 근거 자료 포함 여부 로그
            if source_references:
                print(f"✅ 사고발생 경위 + 조치계획 생성 완료 (근거 자료 {len(source_references)}개 참조)")
            else:
                print("✅ 사고발생 경위 + 조치계획 생성 완료")
                
        except Exception as exc:
            print(f"❌ 보고서 생성 실패: {exc}")
            msg = f"보고서 생성 실패: {exc}"
            state["summary_cause"] = msg
            state["summary_action_plan"] = msg
            state["report_text"] = msg

        return state

    def _run_web_search(self, state: AgentState) -> AgentState:
        """웹 검색 수행"""
        try:
            print("🌐 웹 검색 시작...")
            
            searcher = WebSearch()   
            updated_state = searcher.run(state)
            
            docs = updated_state.get("retrieved_docs") or []
            docs_text, sources = self._build_docs_text(docs)
            if docs_text:
                updated_state["docs_text"] = docs_text
                updated_state["sources"] = sources
            
            updated_state["web_search_completed"] = True
            updated_state["web_search_count"] = updated_state.get("web_search_count", 0) + 1
            updated_state["route"] = "websearch_complete"
            
            print("✅ 웹검색 완료 및 문서 갱신")
            return updated_state
            
        except Exception as exc:
            print(f"❌ 웹검색 실패: {exc}")
            state["web_error"] = str(exc)
            state["web_search_completed"] = True
            return state

    def _create_docx_file(self, state: AgentState) -> AgentState:
        """DOCX 파일 생성 (근거 자료 포함)"""
        user_query = state.get("user_query", "")
        summary_cause = state.get("summary_cause", "")
        action_plan = state.get("summary_action_plan", "")
        source_references = state.get("source_references", [])  # ✅ 추가

        if not user_query:
            print("⚠️ user_query가 없어 DOCX를 생성할 수 없습니다.")
            return state

        if not summary_cause or not action_plan:
            print("⚠️ 보고서 내용이 없어 DOCX를 생성할 수 없습니다.")
            return state

        try:
            docx_path = create_accident_report_docx(
                user_query=user_query,
                cause_text=summary_cause,
                action_text=action_plan,
                source_references=source_references,  # ✅ 추가
            )
            with open(docx_path, "rb") as f:
                docx_bytes = f.read()
            state["docx_path"] = docx_path
            state["docx_bytes"] = docx_bytes
            state["route"] = "docx_complete"
            
            # 근거 자료 포함 여부 로그
            if source_references:
                print(f"✅ DOCX 파일 생성 완료 (근거 자료 {len(source_references)}개 포함): {docx_path}")
            else:
                print(f"✅ DOCX 파일 생성 완료: {docx_path}")
                
        except Exception as exc:
            print(f"❌ DOCX 생성 실패: {exc}")
            state["docx_path"] = None
        return state

    def run(self, state: AgentState) -> AgentState:
        """ReportWriterAgent 실행"""
        print(f"\n{'='*80}")
        print(f"📝 [{self.name}] 실행 중...")
        print(f"{'='*80}")

        action, reason = self._decide_action(state)
        
        if reason.startswith("[Fallback]"):
            print(f"🤖 선택된 작업: {action} | 🔴 {reason}")
        else:
            print(f"🤖 선택된 작업: {action} | 이유: {reason}")

        handler = self.action_handlers.get(action)
        if handler:
            state = handler(state)
        elif action == "noop":
            print("ℹ️ 수행할 작업이 없습니다.")
        else:
            print(f"⚠️ 알 수 없는 작업 '{action}'")

        return state


# ========================================
# Agent Registry
# ========================================
AGENT_REGISTRY = {
    "RAGAgent": RAGAgent(),
    "ReportWriterAgent": ReportWriterAgent(),
}


def get_agent(agent_name: str):
    """Agent 이름으로 인스턴스 반환"""
    return AGENT_REGISTRY.get(agent_name)

# ========================================
# WebSearchAgent (신규)
# ========================================

class WebSearchAgent:
    """웹 검색 전담 Agent"""
    
    def __init__(self):
        self.searcher = WebSearch()
    
    def run(self, state: AgentState) -> AgentState:
        """
        웹 검색 실행
        
        Args:
            state: AgentState
                - user_query: 사용자 질의
                - retrieved_docs: 기존 RAG 문서 (선택)
        
        Returns:
            state: 업데이트된 AgentState
                - web_docs: 웹 검색 결과
                - retrieved_docs: RAG + Web 통합
                - web_search_completed: True
        """
        
        print("\n" + "🌐" * 50)
        print("🌐  WebSearchAgent 실행")
        print("🌐" * 50)
        
        user_query = state.get("user_query", "")
        
        if not user_query:
            print("⚠️ user_query가 없어 웹 검색을 수행할 수 없습니다.")
            state["web_search_completed"] = False
            return state
        
        try:
            # 검색 쿼리 생성 (사고 속성 기반)
            accident_date = state.get("accident_date", "")
            accident_type = state.get("accident_type", "")
            work_process = state.get("work_process", "")
            
            # 검색 쿼리 표시
            print(f"\n📋 검색 대상:")
            print(f"   - 사고 날짜: {accident_date}")
            print(f"   - 사고 유형: {accident_type}")
            print(f"   - 작업 프로세스: {work_process}")
            print(f"\n🔍 검색 쿼리:")
            print(f"   {user_query[:200]}...")
            
            # 사용자 확인
            print(f"\n💡 다음 키워드로 웹 검색을 수행합니다:")
            print(f"   '{accident_type}', '{work_process}', '안전 규정', '사고 예방'")
            
            # WebSearch.run() 호출 (state 전달)
            print(f"\n🌐 Tavily API 검색 중...")
            state = self.searcher.run(state)
            
            # 검색 결과 확인
            web_docs = state.get("web_docs", [])
            
            if not web_docs:
                print("⚠️ 웹 검색 결과가 없습니다.")
                state["web_search_completed"] = True
                return state
            
            print(f"\n✅ 웹 검색 완료: {len(web_docs)}개 결과")
            
            # 웹 검색 결과 미리보기
            print(f"\n" + "─" * 50)
            print("📰 웹 검색 결과 미리보기:")
            print("─" * 50)
            for idx, doc in enumerate(web_docs, 1):
                title = doc.metadata.get("title", "제목 없음")
                url = doc.metadata.get("url", "")
                content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                
                print(f"\n[{idx}] {title}")
                print(f"    🔗 {url}")
                print(f"    📝 {content_preview}")
            print("─" * 50)
            
            # 기존 RAG 문서 확인
            existing_docs = state.get("retrieved_docs", [])
            
            if existing_docs:
                print(f"\n📚 문서 통합:")
                print(f"  - 기존 RAG 문서: {len(existing_docs)}개")
                print(f"  - 웹 검색 결과: {len(web_docs)}개")
                print(f"  - 통합 결과: {len(state.get('retrieved_docs', []))}개")
            else:
                print(f"\n📚 웹 검색 결과만 사용: {len(web_docs)}개")
            
            # docs_text 업데이트
            all_docs = state.get("retrieved_docs", [])
            docs_text = "\n\n".join(
                f"[문서 {i+1}] ({doc.metadata.get('source', 'web')})\n{doc.page_content}"
                for i, doc in enumerate(all_docs)
            )
            
            state["docs_text"] = docs_text
            state["web_search_completed"] = True
            state["route"] = "web_search_complete"
            
            print("\n✅ WebSearchAgent 완료!")
            
        except Exception as e:
            print(f"❌ WebSearchAgent 오류: {e}")
            import traceback
            traceback.print_exc()
            
            state["web_search_completed"] = False
            state["web_error"] = str(e)
        
        return state


# ========================================
# Agent Registry
# ========================================

AGENT_REGISTRY = {
    "RAGAgent": RAGAgent(),
    "WebSearchAgent": WebSearchAgent(),  # ✅ 추가
    "ReportWriterAgent": ReportWriterAgent(),
}