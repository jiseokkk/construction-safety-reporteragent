import os
import re

# === 1️⃣ 입력 / 출력 경로 설정 ===
input_dir = "/home/user/Desktop/jiseok/capstone/RAG/data/data_md"
output_dir = "/home/user/Desktop/jiseok/capstone/RAG/data/modified_md"

# 출력 폴더가 없으면 자동 생성
os.makedirs(output_dir, exist_ok=True)

# === 2️⃣ 불필요 패턴 정의 ===
patterns = [
    r"_<그림.*?>.*?_",             
    r"_<사진.*?>.*?_",             
    r"!\[.*?\]\(.*?\)",            
    r"KOSHA GUIDE.*\n?",           
    r"^# KOSHA GUIDE.*$",          
    r"_{2,}.*?_{2,}",              
    r"안전보건기술지침의개요[\s\S]*?공표일자[:：]?\s*\d{4}년.*?(이사장)?"
]

# === 3️⃣ 모든 .md 파일 순회 ===
for filename in os.listdir(input_dir):
    if not filename.endswith(".md"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".md", "_modified.md"))

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # === 4️⃣ 불필요 요소 제거 ===
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)

    # === 5️⃣ (1)(2)..., (가)(나)... 줄바꿈 추가 ===
    text = re.sub(r"(?<!^)\s*\((\d{1,2})\)", r"\n\n(\1)", text)
    text = re.sub(r"(?<!^)\s*\((가|나|다|라|마|바|사|아|자|차|카|타|파|하)\)", r"\n\n(\1)", text)

    # === 6️⃣ 공백 정리 ===
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"[ \t]+", " ", text)

    # === 7️⃣ (1) 밑에 (가)(나)(다) 들여쓰기 적용 ===
    lines = text.split("\n")
    new_lines = []
    last_was_number = False  # (1)~(9) 직전 여부

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            new_lines.append("")  # 빈 줄 유지
            continue

        # (1)(2)(3)... → 상위 항목
        if re.match(r"^\(\d{1,2}\)", stripped):
            new_lines.append(stripped)
            last_was_number = True
            continue

        # (가)(나)(다)... → 하위 항목
        if re.match(r"^\((가|나|다|라|마|바|사|아|자|차|카|타|파|하)\)", stripped):
            # (1) 바로 다음이거나 이전에도 (가) 계열이면 들여쓰기 유지
            if last_was_number or (len(new_lines) > 0 and new_lines[-1].startswith("    (")):
                new_lines.append("    " + stripped)
            else:
                new_lines.append("    " + stripped)
            last_was_number = False
            continue

        # 일반 문장
        new_lines.append(stripped)
        last_was_number = False

    text = "\n".join(new_lines)

    # === 8️⃣ 중복 문단 제거 ===
    paragraphs = re.split(r"\n\s*\n", text.strip())
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        p_clean = re.sub(r"\s+", " ", p).strip()
        if p_clean not in seen:
            unique_paragraphs.append(p.strip())
            seen.add(p_clean)
    text = "\n\n".join(unique_paragraphs).strip()

    # === 9️⃣ 결과 저장 ===
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ {filename} → {os.path.basename(output_path)} 저장 완료 (개요 제거 + (1)/(가) 구조 정리 + 중복 제거)")

print("\n🎯 모든 .md 파일 정제 완료 → /home/user/Desktop/jiseok/capstone/RAG/data/modified_md 에 저장됨")
