import os

# === 1️⃣ 삭제 대상 폴더 경로 ===
target_dir = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/data_md"

# === 2️⃣ 폴더 내 모든 파일 순회 ===
deleted_count = 0
for filename in os.listdir(target_dir):
    if filename.endswith("_modified.md"):
        file_path = os.path.join(target_dir, filename)
        try:
            os.remove(file_path)
            print(f"🗑️ 삭제 완료: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ 삭제 실패: {filename} ({e})")

# === 3️⃣ 결과 출력 ===
if deleted_count == 0:
    print("❎ 삭제할 '_modified.md' 파일이 없습니다.")
else:
    print(f"\n✅ 총 {deleted_count}개의 '_modified.md' 파일을 삭제했습니다.")
