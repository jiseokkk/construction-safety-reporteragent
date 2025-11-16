import os
import json
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document


embedder_model_name = "Qwen/Qwen3-Embedding-4B"
embedder_base_url = "http://211.47.56.71:15653/v1"
embedder_api_key = "token-abc123"


def title_vector_store_save_from_folder(pdf_folder: str, output_folder: str = "title_vector_db"):
    """
    주어진 폴더 내의 모든 PDF 파일 이름을 기반으로 FAISS 벡터스토어 생성 및 저장
    
    Args:
        pdf_folder (str): PDF 파일들이 있는 폴더 경로
        output_folder (str): 저장할 FAISS DB 폴더 이름 (기본값: "title_vector_db")
    """
    # 1️⃣ 폴더 내 PDF 파일 수집
    file_list = [
        f for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]
    if not file_list:
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_folder}")
        return
    
    print(f"📂 총 {len(file_list)}개의 PDF 파일 감지됨")

    # 2️⃣ PDF 제목 리스트 생성
    pdf_titles = [f.replace(".pdf", "") for f in file_list]

    
    embedding_model = OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key
    )

    
    print("🚀 벡터스토어(DB) 생성 중...")
    db = FAISS.from_texts(
        texts=pdf_titles,
        embedding=embedding_model,
        metadatas=[{"path": os.path.join(pdf_folder, f)} for f in file_list]
    )
    print("DB 생성 완료")

    # DB 저장
    os.makedirs(output_folder, exist_ok=True)
    print("💾 DB 저장 중...")
    db.save_local(output_folder)
    print(f"🎉 DB 저장 완료: {output_folder}/")


if __name__ == "__main__":

    pdf_folder_path = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/건설안전지침"
    title_vector_store_save_from_folder(pdf_folder_path, output_folder="title_db")
