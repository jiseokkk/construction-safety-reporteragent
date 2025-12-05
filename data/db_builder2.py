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
# 2) JSONL 파일로부터 DB 생성
# ----------------------------------------------------------
def build_faiss_db_from_jsonl(jsonl_path, output_dir):
    embeddings = get_qwen_api_embeddings()
    os.makedirs(output_dir, exist_ok=True)

    documents = []
    print(f"\n📌 Processing file: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        line_count = 0
        for line in f:
            line_count += 1
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
                        "source": item.get("source", "chunks2"),
                        "file": item.get("file"),
                        "section": item.get("section"),
                        "section_number": item.get("section_number"),
                        "hierarchy": item.get("hierarchy"),
                        "hierarchy_str": item.get("hierarchy_str"),
                    },
                )
            )

    print(f"→ Read {line_count} lines")
    print(f"→ Loaded {len(documents)} valid documents")

    if len(documents) == 0:
        print("⚠️ No valid documents found. Skipping...")
        return

    print("🔄 Creating FAISS index...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(output_dir)
    print(f"✅ Saved FAISS DB → {output_dir}")


# ----------------------------------------------------------
# 3) Main Builder
# ----------------------------------------------------------
def build_all():
    JSONL_PATH = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/chunks2/chunks.jsonl"
    OUTPUT_DIR = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB2"

    print(f"🔍 Processing JSONL file: {JSONL_PATH}")
    
    if not os.path.exists(JSONL_PATH):
        print(f"❌ File does not exist: {JSONL_PATH}")
        return

    build_faiss_db_from_jsonl(JSONL_PATH, OUTPUT_DIR)

    print("\n🎉 DB built successfully!\n")


if __name__ == "__main__":
    build_all()