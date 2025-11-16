import os
import re

# === 1️⃣ 대상 폴더 경로 ===
target_dir = "/home/user/Desktop/jiseok/capstone/RAG/data/modified_md"

# === 2️⃣ 모든 .md 파일 순회 ===
for filename in os.listdir(target_dir):
    if not filename.endswith(".md"):
        continue

    input_path = os.path.join(target_dir, filename)
    output_path = os.path.join(target_dir, filename.replace(".md", "_indented.md"))

    # === 3️⃣ 파일 읽기 ===
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # === 4️⃣ (가)(나)(다)… 앞에 들여쓰기 4칸 추가 ===
    # 라인 맨 앞의 공백을 모두 제거하고 정확히 4칸 들여쓰기 적용
    text = re.sub(
        r"(?m)^[ \t]*\((가|나|다|라|마|바|사|아|자|차|카|타|파|하)\)",
        r"    (\1)",
        text,
    )

    # === 5️⃣ 저장 ===
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ {filename} → {os.path.basename(output_path)} : (가) 계열 들여쓰기 적용 완료")

print("\n🎯 모든 .md 파일의 (가)(나)(다)… 들여쓰기 4칸으로 통일 완료!")
