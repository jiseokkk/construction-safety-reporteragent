# agents/sql_agent.py (LLM Factory 적용)

import os
import re
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

# ✅ Factory Import
from core.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

# 로깅
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CSVSQLAgent:
    """
    CSV 기반 건설사고 DB의 SQL Agent (Qwen 모델 사용)
    """

    def __init__(self, csv_path: str):
        print("\n" + "=" * 80)
        print("🔧 CSVSQLAgent 초기화 시작")
        print("=" * 80)

        if not os.path.exists(csv_path):
            cwd = os.getcwd()
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}\n  (cwd: {cwd})")

        self.csv_path = csv_path
        self.columns: List[str] = []

        # ✅ LLM 설정 (SQL 생성은 Qwen-32B가 잘함 -> 'fast' 모드)
        self.llm = get_llm(mode="smart")

        # ✅ 파일 DB 경로
        db_path = os.path.join(os.path.dirname(csv_path), "accidents_cache.sqlite")
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)

        # 테이블 준비
        self._ensure_table()

        print("✅ 초기화 완료!")
        print(f"   - DB 파일: {self.db_path}")
        print(f"   - 테이블: accidents")
        print(f"   - 컬럼 수: {len(self.columns)}")
        print("=" * 80 + "\n")

    # ---------------------------------------------------------------------
    # DB 준비 (기존 로직 유지)
    # ---------------------------------------------------------------------
    def _ensure_table(self):
        with self.engine.begin() as conn:
            exists = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='accidents'")
            ).fetchone()

            if not exists:
                self._load_csv_to_db(conn)
            else:
                cols = conn.execute(text("PRAGMA table_info('accidents')")).fetchall()
                self.columns = [c[1] for c in cols]

                if "발생일시_parsed" not in self.columns:
                    conn.execute(text("ALTER TABLE accidents ADD COLUMN 발생일시_parsed TEXT"))
                    conn.execute(
                        text("""
                        UPDATE accidents
                        SET 발생일시_parsed = substr(발생일시, 1, 10)
                        """)
                    )
                    self.columns.append("발생일시_parsed")

        with self.engine.connect() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM accidents")).scalar_one()
            logger.info(f"📦 accidents 테이블 준비 완료: {cnt} rows")

    def _load_csv_to_db(self, conn):
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        if "발생일시_parsed" not in df.columns and "발생일시" in df.columns:
            df["발생일시_parsed"] = pd.to_datetime(
                df["발생일시"].astype(str).str.split().str[0],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        self.columns = list(df.columns)
        df.to_sql("accidents", conn, if_exists="replace", index=False)

    # ---------------------------------------------------------------------
    # SQL 생성 (LLM Factory 적용)
    # ---------------------------------------------------------------------
    def _generate_sql(self, user_query: str) -> Optional[str]:
        """자연어를 SQL로 변환"""
        SELECT_COLUMNS = "ID, 발생일시, \"공종(중분류)\", 인적사고, 사고원인, \"사고객체(중분류)\", \"장소(중분류)\""

        system_prompt = f"""
당신은 건설사고 SQLite DB의 SQL 전문가입니다.

[테이블]
- accidents

[컬럼]
{', '.join(self.columns)}

[규칙]
1) SQLite 문법만 사용
2) 결과는 반드시 **SELECT {SELECT_COLUMNS} FROM accidents** 로 시작
3) 날짜 검색:
   - 특정 월 검색 (예: "11월 사고"): WHERE 발생일시_parsed LIKE 'YYYY-MM%'
   - 특정 연도 검색 (예: "2024년 사고"): WHERE 발생일시_parsed LIKE 'YYYY%'
   - 최근 기간 (예: "최근 3개월"): WHERE 발생일시_parsed >= date('now', '-3 months')
4) 조건 검색 (예: "철근콘크리트", "추락"):
   - 명확한 컬럼이 없으면 `공종(중분류)`, `사고원인`, `인적사고` 등에 LIKE 검색을 OR로 연결하세요.
   - 예: ( "공종(중분류)" LIKE '%철근%' OR 사고원인 LIKE '%철근%' )
5) 텍스트 검색은 LIKE '%키워드%' 사용
6) 여러 조건은 AND/OR로 결합

[출력]
- SQL만 출력 (설명, ```sql 등 마크다운 태그 포함 금지)
"""
        # LangChain Prompt Template 사용
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"사용자 질문: {user_query}\n\n위 질문에 대한 SQL을 생성하세요.")
        ])

        # Chain 실행
        chain = prompt | self.llm 

        try:
            response = chain.invoke({})
            sql = response.content.strip()

            # 마크다운 제거
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()

            if not sql.upper().startswith("SELECT"):
                logger.warning(f"유효하지 않은 SQL 생성: {sql}")
                return None
            
            # SELECT 필드 보정
            if SELECT_COLUMNS not in sql:
                 logger.warning(f"SELECT 필드가 지정되지 않아 {SELECT_COLUMNS}로 강제 대체합니다.")
                 sql = re.sub(r'SELECT\s+.*?\s+FROM', f'SELECT {SELECT_COLUMNS} FROM', sql, flags=re.IGNORECASE)

            return sql

        except Exception as e:
            logger.error(f"SQL 생성 오류: {e}")
            return None

    # ---------------------------------------------------------------------
    # 질의 실행 (유지)
    # ---------------------------------------------------------------------
    def query(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"🔍 사용자 입력: {user_query}")

        try:
            sql_query = self._generate_sql(user_query)
            if not sql_query:
                return {
                    "success": False,
                    "input": user_query,
                    "error": "SQL 쿼리를 생성할 수 없습니다."
                }

            logger.info(f"📝 생성된 SQL: {sql_query}")

            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql_query), conn)

            logger.info(f"✅ 검색 완료: {len(df)}건")

            return {
                "success": True,
                "input": user_query,
                "generated_sql": sql_query,
                "final_answer": f"{len(df)}건의 사고 기록을 찾았습니다.",
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
            }

        except Exception as e:
            logger.error(f"❌ SQL Agent 오류: {e}")
            return {
                "success": False,
                "input": user_query,
                "generated_sql": sql_query if 'sql_query' in locals() else None,
                "error": str(e),
            }

    def get_dataframe(self, user_query: str) -> Optional[pd.DataFrame]:
        res = self.query(user_query)
        if res["success"]:
            return pd.DataFrame(res["rows"])
        logger.error(f"DataFrame 생성 실패: {res.get('error')}")
        return None