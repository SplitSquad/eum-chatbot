import chromadb
from chromadb.config import Settings
from loguru import logger

def check_collection(vectorstore_path, collection_name):
    """벡터스토어 컬렉션 확인"""
    try:
        client = chromadb.PersistentClient(
            path=vectorstore_path,
            settings=Settings(allow_reset=True)
        )
        
        collection = client.get_collection(collection_name)
        result = collection.get()
        
        if result['ids']:
            sample_embedding = result['embeddings'][0]
            embedding_dim = len(sample_embedding)
            logger.info(f"{vectorstore_path} - {collection_name}:")
            logger.info(f"  문서 수: {len(result['ids'])}")
            logger.info(f"  임베딩 차원: {embedding_dim}")
        else:
            logger.warning(f"{vectorstore_path} - {collection_name}: 데이터 없음")
            
    except Exception as e:
        logger.error(f"{vectorstore_path} - {collection_name}: 오류 발생 - {str(e)}")

def main():
    """메인 함수"""
    domains = {
        "data/vectorstore1": "visa_law",
        "data/vectorstore2": "social_security",
        "data/vectorstore3": "tax_finance",
        "data/vectorstore4": "medical_health",
        "data/vectorstore5": "employment",
        "data/vectorstore6": "daily_life"
    }
    
    for path, collection in domains.items():
        check_collection(path, collection)

if __name__ == "__main__":
    main() 