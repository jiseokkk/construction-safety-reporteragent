"""
IntentAgent 기반 자연어 Multi-Agent 시스템
사용자 자연어 → IntentAgent → Orchestrator → SubAgents

기능:
1. CSV 정보 조회
2. RAG 검색
3. 보고서 생성
4. 대화형 추가 작업 제안
"""

import os
import pandas as pd
from typing import Dict, Any
from core.agentstate import AgentState
from agents.intent_agent import IntentAgent
from graph.workflow import graph_app


class IntelligentAgentSystem:
    """IntentAgent 기반 Multi-Agent 시스템"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.intent_agent = IntentAgent()
        self._load_data()
    
    def _load_data(self):
        """CSV 데이터 로드"""
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            self.df.columns = self.df.columns.str.strip()
            
            # 발생일시 파싱
            self.df['발생일시_parsed'] = pd.to_datetime(
                self.df['발생일시'].str.split().str[0],
                format='%Y-%m-%d',
                errors='coerce'
            )
            
            print(f"✅ CSV 데이터 로드 완료: {len(self.df)}개 사고 기록")
            
            # 날짜 범위 표시
            valid_dates = self.df['발생일시_parsed'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                print(f"📅 사고 기록 날짜 범위: {min_date.date()} ~ {max_date.date()}")
            
        except Exception as e:
            print(f"❌ CSV 로드 실패: {e}")
            self.df = None
    
    def row_to_user_query(self, row: pd.Series) -> str:
        """CSV row를 user_query로 변환"""
        query = "[사고 속성]\n"
        
        fields = {
            "발생일시": row.get("발생일시", "N/A"),
            "공종": row.get("공종(중분류)", "N/A"),
            "작업프로세스": row.get("작업프로세스", "N/A"),
            "사고 유형": row.get("인적사고", "N/A"),
            "사고 개요": row.get("사고원인", "N/A"),
            "사고객체(중분류)": row.get("사고객체(중분류)", "N/A"),
            "장소(중분류)": row.get("장소(중분류)", "N/A"),
        }
        
        for key, value in fields.items():
            if value and str(value) != "N/A" and str(value) != "nan":
                query += f"{key}: {value}\n"
        
        return query
    
    def process_user_input(self, user_input: str):
        """사용자 입력 처리"""
        
        print("\n" + "🔍" * 50)
        print("🔍  IntentAgent가 입력을 분석 중...")
        print("🔍" * 50)
        
        # 1) IntentAgent가 파싱 및 의도 파악
        result = self.intent_agent.parse_and_decide(user_input, self.df)
        
        if not result["success"]:
            print(f"\n❌ {result['error']}")
            return
        
        # 2) 선택된 사고 정보
        accident_data = result["accident_data"]
        intent = result["intent"]
        
        # 3) 의도에 따라 처리
        if intent == "csv_info":
            # CSV 정보만 출력
            self.intent_agent.display_csv_info(accident_data)
            
            # 추가 작업 제안
            additional_intent = self.intent_agent.ask_for_additional_action(intent)
            
            if additional_intent:
                print(f"\n🎯 추가 작업: {additional_intent}")
                intent = additional_intent  # 의도 변경
            else:
                print("\n✅ 작업을 종료합니다.")
                return
        
        # 4) RAG 검색 또는 보고서 생성
        if intent in ["search_only", "generate_report"]:
            user_query = self.row_to_user_query(accident_data)
            
            print(f"\n📝 생성된 Query:")
            print(user_query)
            print(f"\n🎯 실행 모드: {intent}")
            
            # Multi-Agent 실행
            final_state = self.execute_agents(user_query, intent)
            
            # 결과 출력
            self.display_results(final_state, intent)
            
            # RAG 검색 후 보고서 생성 제안
            if intent == "search_only":
                additional_intent = self.intent_agent.ask_for_additional_action(intent)
                
                if additional_intent == "generate_report":
                    print(f"\n🎯 추가 작업: 보고서 생성")
                    
                    # 동일한 user_query로 보고서 생성
                    final_state["user_intent"] = "generate_report"
                    
                    # ReportWriterAgent만 추가 실행
                    print("\n" + "🚀" * 50)
                    print("🚀  보고서 생성 모드로 전환")
                    print("🚀" * 50)
                    
                    final_state = self.continue_to_report(final_state)
                    self.display_results(final_state, "generate_report")
    
    def execute_agents(self, user_query: str, intent: str) -> Dict[str, Any]:
        """Multi-Agent 시스템 실행"""
        
        print("\n" + "🚀" * 50)
        if intent == "search_only":
            print("🚀  정보 검색 모드 - RAG 검색만 수행")
        else:
            print("🚀  보고서 생성 모드 - RAG + 보고서 + DOCX 생성")
        print("🚀" * 50)
        
        # AgentState 초기화
        state: AgentState = {
            "user_query": user_query,
            "user_intent": intent,
        }
        
        # LangGraph 워크플로우 실행
        print("\n▶️  Multi-Agent 시스템 실행 중...\n")
        final_state = graph_app.invoke(state)
        
        return final_state
    
    def continue_to_report(self, state: AgentState) -> Dict[str, Any]:
        """검색 후 보고서 생성 계속하기"""
        
        # user_intent를 generate_report로 변경
        state["user_intent"] = "generate_report"
        
        # 워크플로우 재실행 (이미 RAG는 완료되었으므로 ReportWriter만 실행됨)
        final_state = graph_app.invoke(state)
        
        return final_state
    
    def display_results(self, final_state: Dict[str, Any], intent: str):
        """결과 출력"""
        
        print("\n" + "🎉" * 50)
        print("🎉  작업 완료!")
        print("🎉" * 50)
        
        if intent == "search_only":
            # 정보 검색 모드: 포맷팅된 결과 출력
            formatted_result = final_state.get("formatted_result")
            if formatted_result:
                print("\n" + formatted_result)
            else:
                print("\n⚠️ 포맷팅된 결과가 없습니다.")
                docs = final_state.get("retrieved_docs") or []
                print(f"📊 검색된 문서 수: {len(docs)}")
        
        else:
            # 보고서 생성 모드: 보고서 및 DOCX 정보 출력
            docs = final_state.get("retrieved_docs") or []
            report_text = final_state.get("report_text", "")
            docx_path = final_state.get("docx_path")
            
            print(f"\n📊 최종 결과:")
            print(f"  - 검색된 문서 수: {len(docs)}")
            print(f"  - 보고서 텍스트 길이: {len(report_text)} 글자")
            print(f"  - DOCX 파일: {docx_path}")
            
            if report_text:
                print("\n--- 보고서 내용 (처음 500자) ---")
                print(report_text[:500])
                print("..." if len(report_text) > 500 else "")
                print("--------------------------------")


def main():
    """메인 실행 함수"""
    
    # CSV 경로
    CSV_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return
    
    # 시스템 초기화
    system = IntelligentAgentSystem(CSV_PATH)
    
    if system.df is None:
        print("❌ 시스템 초기화 실패")
        return
    
    print("\n" + "="*100)
    print("🏗️  건설안전 Intelligent Multi-Agent 시스템")
    print("="*100)
    print("\n💬 자연어로 입력하세요:")
    print("  📋 CSV 정보 조회: '8월 8일 사고 정보 알려줘'")
    print("  🔍 안전 지침 검색: '8월 8일 사고 관련 지침 검색해줘'")
    print("  📝 보고서 생성: '8월 8일 사고 보고서 작성해줘'")
    print("\n종료하려면 'exit' 또는 'quit'를 입력하세요.\n")
    
    while True:
        try:
            # 자연어 입력 받기
            user_input = input("\n💬 무엇을 도와드릴까요?: ").strip()
            
            # 종료 조건
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 시스템을 종료합니다.")
                break
            
            if not user_input:
                print("⚠️ 입력이 없습니다.")
                continue
            
            # IntentAgent로 처리
            system.process_user_input(user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print("계속 진행하려면 다시 입력하세요.")


if __name__ == "__main__":
    main()