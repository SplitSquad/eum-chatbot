import sqlite3
import chromadb
from chromadb.config import Settings
import json

def migrate_data():
    # 백업 데이터베이스에서 데이터 읽기
    backup_conn = sqlite3.connect('data/vectorstore6_backup/chroma.sqlite3')
    backup_cursor = backup_conn.cursor()
    
    # 임베딩과 문서 데이터 가져오기
    backup_cursor.execute("""
        SELECT 
            e.embedding_id,
            doc.string_value as document,
            tags.string_value as tags,
            source.string_value as source
        FROM embeddings e
        JOIN embedding_metadata doc ON e.id = doc.id AND doc.key = 'chroma:document'
        LEFT JOIN embedding_metadata tags ON e.id = tags.id AND tags.key = 'tags'
        LEFT JOIN embedding_metadata source ON e.id = source.id AND source.key = 'source'
    """)
    rows = backup_cursor.fetchall()
    
    if not rows:
        print("No data found in the backup database.")
        return
    
    # 새로운 ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(
        path="data/vectorstore6",
        settings=Settings(allow_reset=True)
    )
    
    # 새로운 컬렉션 생성
    collection = client.create_collection("daily_life")
    
    # 데이터 이전
    documents = []
    metadatas = []
    ids = []
    
    for row in rows:
        embedding_id = row[0]
        document = row[1]
        tags = row[2]
        source = row[3]
        
        metadata = {}
        if tags:
            metadata['tags'] = tags
        if source:
            metadata['source'] = source
        
        documents.append(document)
        metadatas.append(metadata)
        ids.append(str(embedding_id))
    
    if documents:
        # 데이터를 새 컬렉션에 추가
        # ChromaDB가 자동으로 새 임베딩을 생성할 것입니다
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    backup_conn.close()
    print(f"Successfully migrated {len(documents)} documents to the new database.")

if __name__ == "__main__":
    migrate_data() 