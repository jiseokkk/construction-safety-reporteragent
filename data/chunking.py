import os
import re
import json
import pickle
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ===== 1️⃣ 경로 설정 =====
data_dir = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/data_md"
output_dir = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data"
os.makedirs(output_dir, exist_ok=True)

# ===== 2️⃣ Heading 기준 분리 함수 =====
def split_by_heading(text: str):
    """# 헤더(#)를 기준으로 문서를 섹션 단위로 나눕니다."""
    sections = re.split(r'(?=^# )', text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]

# ===== 3️⃣ 내부 청킹용 Splitter =====
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # 각 청크의 최대 길이
    chunk_overlap=100,  # 겹치는 부분 (문맥 유지)
    separators=["\n\n", "\n", ".", " "]  # 분리 기준
)

# ===== 4️⃣ 모든 Markdown 파일 불러오기 =====
md_files = [f for f in os.listdir(data_dir) if f.endswith(".md")]
print(f"✅ 총 Markdown 파일 수: {len(md_files)}")

chunks = []

for file_name in tqdm(md_files, desc="📄 Markdown 파일 처리 중"):
    file_path = os.path.join(data_dir, file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {file_name} ({e})")
        continue

    # 1️⃣ 헤더 기준 섹션 분리
    sections = split_by_heading(raw_text)

    # 2️⃣ 각 섹션 내부 청킹
    for sec in sections:
        heading_match = re.match(r"^#\s*(.*)", sec)
        heading = heading_match.group(1).strip() if heading_match else "본문"

        split_texts = splitter.split_text(sec)
        for chunk in split_texts:
            chunks.append({
                "file": file_name,
                "section": heading,
                "content": chunk
            })

print(f"✅ 총 생성된 청크 수: {len(chunks)}")

# ===== 5️⃣ 결과 저장 =====
json_path = os.path.join(output_dir, "chunks.json")
pkl_path = os.path.join(output_dir, "chunks.pkl")

# JSON 저장
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(chunks, jf, ensure_ascii=False, indent=2)

# Pickle 저장
with open(pkl_path, "wb") as pf:
    pickle.dump(chunks, pf)

print(f"\n✅ 청크 데이터 저장 완료!")
print(f"📄 JSON 파일: {json_path}")
print(f"📦 Pickle 파일: {pkl_path}")

# ===== 6️⃣ 예시 출력 =====
print("\n🧾 예시 2개:")
for c in chunks[:2]:
    print(f"[파일] {c['file']} | [섹션] {c['section']}")
    print(c['content'][:300], "...\n")
