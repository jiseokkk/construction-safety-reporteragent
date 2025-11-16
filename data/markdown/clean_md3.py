import os
import re

# === 1️⃣ 대상 폴더 경로 ===
target_dir = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/data_md"

# === 2️⃣ 모든 .md 파일 순회 ===
for filename in os.listdir(target_dir):
    if not filename.endswith(".md"):
        continue

    input_path = os.path.join(target_dir, filename)
    output_path = os.path.join(target_dir, filename)  # 원래 이름 그대로 덮어쓰기

    # === 3️⃣ 파일 읽기 ===
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # === 4️⃣ “목 차” 섹션 자동 제거 ===
    # - "목 차" 또는 "목차"로 시작해서
    # - 첫 번째 본문 시작(제1조 / 1. / # 제1장 / ## 등) 전까지 제거
    cleaned_text = re.sub(
        r"(?s)(#?\s*목\s*차.*?)(?=(제\s*1\s*조|#|##|\d+\.\s))",
        "",
        text
    )

    # === 5️⃣ 저장 (덮어쓰기) ===
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text.strip() + "\n")

    print(f"🧹 {filename} : 목차 제거 완료")

print("\n🎯 모든 .md 파일의 목차 섹션이 제거되었습니다!")
