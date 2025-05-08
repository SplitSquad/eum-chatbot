import os
import pandas as pd
from loguru import logger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from tqdm import tqdm

def extract_text_from_pdf(pdf_path, use_ocr=False):
    """PDF에서 텍스트 추출"""
    try:
        if use_ocr:
            # OCR이 필요한 경우 이미지로 변환 후 텍스트 추출
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image, lang='kor+eng') + "\n"
        else:
            # 일반 PDF 텍스트 추출
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF 처리 중 오류 발생 ({pdf_path}): {str(e)}")
        return ""

def process_documents(domain_dir):
    """문서 처리 및 임베딩"""
    try:
        documents = []
        metadatas = []
        
        # PDF 파일 처리
        for pdf_file in os.listdir(domain_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(domain_dir, pdf_file)
                # 2024_서울생활안내_한국어.pdf는 OCR이 필요할 수 있음
                use_ocr = pdf_file == "2024_서울생활안내_한국어.pdf"
                text = extract_text_from_pdf(pdf_path, use_ocr)
                if text:
                    documents.append(text)
                    metadatas.append({
                        'source': pdf_file,
                        'type': 'pdf',
                        'ocr': use_ocr
                    })
        
        return documents, metadatas
    
    except Exception as e:
        logger.error(f"문서 처리 중 오류 발생: {str(e)}")
        return [], []

def create_vectorstore(domain_num, domain_name, documents, metadatas):
    """벡터스토어 생성"""
    try:
        # 벡터스토어 경로
        vectorstore_path = f"data/vectorstore{domain_num}"
        
        # 기존 벡터스토어 백업
        if os.path.exists(vectorstore_path):
            backup_path = f"{vectorstore_path}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(vectorstore_path, backup_path)
            logger.info(f"기존 벡터스토어 백업: {backup_path}")
        
        # 새 벡터스토어 생성
        client = chromadb.PersistentClient(
            path=vectorstore_path,
            settings=Settings(allow_reset=True)
        )
        
        # 임베딩 모델 초기화
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        # 문서 임베딩 생성
        embeddings = model.encode(documents, convert_to_numpy=True)
        
        # 컬렉션 생성 및 데이터 추가
        collection = client.create_collection(
            name=domain_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 데이터 추가
        collection.add(
            ids=[f"doc_{i}" for i in range(len(documents))],
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"벡터스토어 생성 완료: {len(documents)}개 문서")
        logger.info(f"임베딩 차원: {embeddings.shape[1]}")
        
    except Exception as e:
        logger.error(f"벡터스토어 생성 중 오류 발생: {str(e)}")
        raise

def main():
    """메인 함수"""
    # 도메인별 설정
    domains = {
        "1": ("1. 비자,법률", "visa_law"),
        "2": ("2. 사회보장제도", "social_security"),
        "3": ("3. 세금,금융", "tax_finance"),
        "4": ("4. 의료,건강", "medical_health"),
        "5": ("5. 취업관련", "employment"),
        "6": ("6. 일상생활", "daily_life")
    }
    
    for domain_num, (domain_dir, domain_name) in domains.items():
        logger.info(f"\n=== {domain_name} 처리 시작 ===")
        
        # 문서 처리
        documents, metadatas = process_documents(os.path.join("rawdata", domain_dir))
        
        if documents:
            # 벡터스토어 생성
            create_vectorstore(domain_num, domain_name, documents, metadatas)
        else:
            logger.warning(f"{domain_name}: 처리할 문서가 없습니다.")

if __name__ == "__main__":
    main() 