# core/query_builder.py
"""
CSV(또는 프론트 입력)의 구조화된 필드를
시스템 공용 user_query 포맷으로 변환하는 유틸리티.
"""

from typing import Dict
import pandas as pd


def build_user_query_from_row(row: pd.Series) -> str:
    """
    train_preprocessing.csv 의 한 row를 받아서
    Orchestrator / RAG / DOCX 에서 공통으로 쓰는 user_query 문자열을 만든다.

    사용 컬럼:
        - 공사명: 공종(중분류)  (없으면 공사종류(중분류) 사용)
        - 작업프로세스: 작업프로세스
        - 사고 유형: 인적사고
        - 사고 개요: 사고원인
        - (추가 정보) 사고객체(중분류), 장소(중분류) 등은 참고용으로 아래에 붙인다.
    """
    def safe_get(col: str) -> str:
        return str(row.get(col, "")).strip() if col in row else ""

    work_type_mid = safe_get("공종(중분류)") or safe_get("공사종류(중분류)")
    process = safe_get("작업프로세스")
    accident_type = safe_get("인적사고")
    cause = safe_get("사고원인")
    object_mid = safe_get("사고객체(중분류)")
    location_mid = safe_get("장소(중분류)")

    # 👉 시스템에서 공통으로 쓰는 포맷 (이미 예시로 썼던 형태)
    # - RAG retriever: "공종:", "작업프로세스:" 줄을 사용
    # - DocxWriter.parse_user_query: "공종:", "작업프로세스:", "사고 유형:", "사고 개요:" 줄을 사용
    lines = [
        "[사고 속성]",
        f"공종: {work_type_mid}",
        f"작업프로세스: {process}",
        f"사고 유형: {accident_type}",
        f"사고 개요: {cause}",
    ]

    # 참고용 정보는 아래에 덧붙여 줌 (필수는 아님)
    if object_mid:
        lines.append(f"사고객체(중분류): {object_mid}")
    if location_mid:
        lines.append(f"장소(중분류): {location_mid}")

    return "\n".join(lines)


def row_to_structured_fields(row: pd.Series) -> Dict[str, str]:
    """
    나중에 필요하면 직접 state에 구조화된 필드로도 넣을 수 있도록
    딕셔너리 형태로 변환하는 헬퍼.
    (지금은 안 써도 되지만, 확장성 위해 같이 정의)
    """
    def safe_get(col: str) -> str:
        return str(row.get(col, "")).strip() if col in row else ""

    return {
        "공사명": safe_get("공종(중분류)") or safe_get("공사종류(중분류)"),
        "사고발생장소": safe_get("작업프로세스") or safe_get("장소(중분류)"),
        "사고종류": safe_get("인적사고"),
        "사고개요": safe_get("사고원인"),
        "작업프로세스": safe_get("작업프로세스"),
        "사고객체(중분류)": safe_get("사고객체(중분류)"),
        "장소(중분류)": safe_get("장소(중분류)"),
    }
