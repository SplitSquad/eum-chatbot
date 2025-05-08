import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
import os
import shutil
from datetime import datetime
import numpy as np

def backup_vectorstore(vectorstore_path):
    """벡터스토어 백업"""
    if os.path.exists(vectorstore_path):
        backup_path = f"{vectorstore_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(vectorstore_path, backup_path)
        logger.info(f"벡터스토어 백업 완료: {backup_path}")
        return backup_path
    return None

def migrate_embeddings(source_path, target_path, collection_name):
    """임베딩 차원 마이그레이션"""
    try:
        # 소스 클라이언트 초기화
        source_client = chromadb.PersistentClient(
            path=source_path,
            settings=Settings(allow_reset=True)
        )

        # 타겟 클라이언트 초기화 (새로운 벡터스토어)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        target_client = chromadb.PersistentClient(
            path=target_path,
            settings=Settings(allow_reset=True)
        )

        try:
            # 소스 컬렉션 가져오기
            source_collection = source_client.get_collection(collection_name)
            
            # 모든 데이터 가져오기
            result = source_collection.get()
            if not result['ids']:
                logger.warning(f"컬렉션 {collection_name}에 데이터가 없습니다.")
                return
        except Exception as e:
            logger.error(f"소스 컬렉션 접근 실패: {str(e)}")
            return

        # 새로운 임베딩 모델 초기화
        new_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        # 문서에서 새로운 임베딩 생성
        documents = result['documents']
        new_embeddings = new_model.encode(documents, convert_to_numpy=True)
        
        # 새로운 컬렉션 생성 및 데이터 추가
        target_collection = target_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 데이터 추가
        target_collection.add(
            ids=result['ids'],
            embeddings=new_embeddings.tolist(),
            documents=documents,
            metadatas=result['metadatas'] if 'metadatas' in result else None
        )

        logger.info(f"마이그레이션 완료: {len(documents)}개 문서")
        logger.info(f"임베딩 차원: {new_embeddings.shape[1]}")

    except Exception as e:
        logger.error(f"마이그레이션 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 모든 도메인 마이그레이션
    domains = {
        "data/vectorstore1": "visa_law",
        "data/vectorstore2": "social_security",
        "data/vectorstore3": "tax_finance",
        "data/vectorstore4": "medical_health",
        "data/vectorstore5": "employment",
        "data/vectorstore6": "daily_life"
    }
    
    for source_path, collection_name in domains.items():
        logger.info(f"\n=== {collection_name} 마이그레이션 시작 ===")
        
        # 기존 벡터스토어 백업
        backup_path = backup_vectorstore(source_path)
        if backup_path:
            logger.info(f"백업 생성됨: {backup_path}")
        
        # 마이그레이션 수행
        target_path = f"{source_path}_new"
        try:
            migrate_embeddings(source_path, target_path, collection_name)
            # 성공 시 새로운 벡터스토어로 교체
            shutil.rmtree(source_path)
            os.rename(target_path, source_path)
            logger.info(f"{collection_name} 마이그레이션 완료")
        except Exception as e:
            logger.error(f"{collection_name} 마이그레이션 실패: {str(e)}")
            continue 