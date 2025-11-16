import os
import re

# === 1️⃣ 폴더 경로 설정 ===
target_dir = "/home/user/Desktop/jiseok/capstone/RAG/data/modified_md"

# === 2️⃣ 파일 그룹화 (같은 원본 기준으로 비교) ===
files_by_base = {}

for filename in os.listdir(target_dir):
    if not filename.endswith(".md"):
        continue
    base = re.sub(r"(_modified.*)$", "", filename)  # 원본 이름 추출
    files_by_base.setdefault(base, []).append(filename)

# === 3️⃣ 각 그룹에서 가장 긴 modified_chain 파일만 남기고 나머지 삭제 ===
for base, files in files_by_base.items():
    # “modified” 개수 기준으로 내림차순 정렬 (긴 게 마지막 버전)
    files_sorted = sorted(files, key=lambda f: f.count("_modified"), reverse=True)

    # 가장 긴 modified_chain 중, indented가 붙은 파일만 남김
    keep_file = None
    for f in files_sorted:
        if f.endswith("_indented.md"):
            keep_file = f
            break

    # 삭제 로직
    for f in files:
        file_path = os.path.join(target_dir, f)
        if f != keep_file:
            try:
                os.remove(file_path)
                print(f"🗑️ 삭제 완료: {f}")
            except Exception as e:
                print(f"⚠️ 삭제 실패: {f} ({e})")
        else:
            # 마지막 버전 파일 이름을 깔끔하게 정리 (_modified_..._indented 제거)
            new_name = re.sub(r"(_modified)+(_indented)?", "", f).strip("_")
            new_path = os.path.join(target_dir, new_name)
            os.rename(file_path, new_path)
            print(f"✅ 유지 및 이름 변경: {f} → {os.path.basename(new_path)}")

print("\n🎯 정리 완료: 가장 긴 modified_chain + indented만 남기고 나머지 삭제됨!")
