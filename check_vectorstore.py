import chromadb
from chromadb.config import Settings
import json

def check_vectorstore():
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(
        path="data/vectorstore6",
        settings=Settings(allow_reset=True)
    )
    
    try:
        # 컬렉션 가져오기
        collection = client.get_collection("daily_life")
        
        # 컬렉션 정보 출력
        print("\nCollection Info:")
        print(f"Total documents: {collection.count()}")
        
        # 샘플 데이터 확인
        if collection.count() > 0:
            print("\nSample Documents:")
            results = collection.get()
            
            # 문서와 메타데이터 출력
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc[:200]}...")  # 처음 200자만 출력
                if metadata:
                    print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                print("-" * 80)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_vectorstore() 