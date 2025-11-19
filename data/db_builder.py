import os
import json
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


# ----------------------------------------------------------
# 1) Qwen3 Embedding API 설정
# ----------------------------------------------------------
def get_qwen_api_embeddings():
    embedder_model_name = "Qwen/Qwen3-Embedding-4B"
    embedder_base_url = "http://211.47.56.71:15653/v1"
    embedder_api_key = "token-abc123"

    return OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )


# ----------------------------------------------------------
# 2) description.json 템플릿
# ----------------------------------------------------------
DESCRIPTION_MAP = {
    "01_bridge": {
        "name": "Bridge Construction Safety DB",
        "domain": "교량공사",
        "purpose": "교량 시공과 관련된 모든 공정에 대한 안전작업 지침과 사고 예방 기준 제공.",
        "covers": ["교량 상부공", "거더 설치", "슬래브 거푸집", "현수교 시공"],
        "common_accidents": ["거푸집 붕괴", "추락", "낙하물"],
        "best_for_queries": ["교량 거푸집 붕괴", "거더 인양 사고"]
    },
    "02_earth": {
        "name": "Earthwork & Excavation Safety DB",
        "domain": "토공사/굴착",
        "purpose": "굴착·지보공 관련 붕괴 및 매몰 예방 기준 제공.",
        "covers": ["터파기", "흙막이", "SCW", "CIP"],
        "common_accidents": ["붕괴", "매몰"],
        "best_for_queries": ["토사 붕괴", "흙막이 변형 사고"]
    },
    "03_tunnel": {
        "name": "Tunnel Construction Safety DB",
        "domain": "터널",
        "purpose": "NATM, TBM, 발파 등 터널 굴착 관련 안전지침 제공.",
        "covers": ["발파", "숏크리트", "록볼트", "지보공"],
        "common_accidents": ["낙석", "붕락", "가스 폭발"],
        "best_for_queries": ["터널 붕락 사고", "발파 작업 사고"]
    },
    "04_scaffold": {
        "name": "Scaffolding Safety DB",
        "domain": "비계/가설",
        "purpose": "비계, 달비계, 이동식 비계 등 고소작업 안전기준 제공.",
        "covers": ["비계 설치", "달비계", "이동식 비계"],
        "common_accidents": ["추락", "비계 붕괴"],
        "best_for_queries": ["비계 추락 사고"]
    },
    "05_crane": {
        "name": "Crane & Lifting Safety DB",
        "domain": "타워크레인/인양",
        "purpose": "타워크레인 및 중량물 인양 작업 안전지침 제공.",
        "covers": ["타워크레인", "이동식 크레인"],
        "common_accidents": ["전도", "낙하", "로프 파단"],
        "best_for_queries": ["크레인 전도", "인양물 낙하"]
    },
    "06_finishing": {
        "name": "Finishing Construction Safety DB",
        "domain": "마감",
        "purpose": "실내 마감공사 안전지침 제공.",
        "covers": ["석고보드", "창호", "내부 마감"],
        "common_accidents": ["사다리 전도", "낙하"],
        "best_for_queries": ["실내 사다리 사고"]
    },
    "07_concrete": {
        "name": "Concrete & Formwork Safety DB",
        "domain": "콘크리트",
        "purpose": "타설, 거푸집, 동바리 작업 안전기준 제공.",
        "covers": ["거푸집", "동바리", "타설"],
        "common_accidents": ["거푸집 붕괴", "동바리 좌굴"],
        "best_for_queries": ["동바리 붕괴", "타설 사고"]
    },
    "08_general": {
        "name": "General Construction Safety DB",
        "domain": "공통 안전",
        "purpose": "모든 공종에서 공통적으로 적용되는 안전 기준 제공.",
        "covers": ["PPE", "현장 안전 수칙"],
        "common_accidents": ["추락", "낙하"],
        "best_for_queries": ["현장 안전수칙"]
    }
}


# ----------------------------------------------------------
# 3) DB builder
# ----------------------------------------------------------
def build_faiss_db_for_folder(chunk_dir, output_dir):
    embeddings = get_qwen_api_embeddings()
    os.makedirs(output_dir, exist_ok=True)

    documents = []
    print(f"\n📌 Processing folder: {chunk_dir}")

    for file in os.listdir(chunk_dir):
        file_path = os.path.join(chunk_dir, file)

        # JSONL 파일 처리
        if file.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)

                    # 🧠 핵심: content → text → section fallback
                    text = (
                        item.get("content")
                        or item.get("text")
                        or item.get("section")
                        or ""
                    )

                    if not text.strip():  # 빈 문서 skip
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(chunk_dir),
                                "file": item.get("file"),
                                "section": item.get("section"),
                                "section_number": item.get("section_number"),
                                "hierarchy": item.get("hierarchy"),
                                "hierarchy_str": item.get("hierarchy_str"),
                            },
                        )
                    )

        # md 또는 txt 파일 처리
        elif file.endswith(".md") or file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(chunk_dir)},
                    )
                )

    print(f"→ Loaded {len(documents)} valid documents")

    if len(documents) == 0:
        print("⚠️ No valid documents found. Skipping...")
        return

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(output_dir)
    print(f"✅ Saved FAISS DB → {output_dir}")


# ----------------------------------------------------------
# 4) description.json 생성
# ----------------------------------------------------------
def create_description_file(folder_name, output_dir):
    desc = DESCRIPTION_MAP.get(folder_name)
    if desc:
        with open(
            os.path.join(output_dir, "description.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(desc, f, indent=4, ensure_ascii=False)
        print(f"📝 Created description.json at: {output_dir}")


# ----------------------------------------------------------
# 5) Main Builder Loop
# ----------------------------------------------------------
def build_all():
    BASE_CHUNK_DIR = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/chunks"
    BASE_DB_DIR = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB"

    os.makedirs(BASE_DB_DIR, exist_ok=True)

    for folder in sorted(os.listdir(BASE_CHUNK_DIR)):
        chunk_path = os.path.join(BASE_CHUNK_DIR, folder)
        if not os.path.isdir(chunk_path):
            continue

        output_path = os.path.join(BASE_DB_DIR, folder)
        build_faiss_db_for_folder(chunk_path, output_path)
        create_description_file(folder, output_path)

    print("\n🎉 All DBs built successfully!\n")


if __name__ == "__main__":
    build_all()
