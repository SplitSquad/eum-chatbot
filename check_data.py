import sqlite3

def check_data():
    conn = sqlite3.connect('data/vectorstore6_backup/chroma.sqlite3')
    cursor = conn.cursor()
    
    # 임베딩 테이블 확인
    print("\nChecking embeddings table:")
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    count = cursor.fetchone()[0]
    print(f"Total embeddings: {count}")
    
    if count > 0:
        cursor.execute("SELECT * FROM embeddings LIMIT 1")
        columns = [description[0] for description in cursor.description]
        print("Columns:", columns)
        print("Sample row:", cursor.fetchone())
    
    # 메타데이터 테이블 확인
    print("\nChecking embedding_metadata table:")
    cursor.execute("SELECT COUNT(*) FROM embedding_metadata")
    count = cursor.fetchone()[0]
    print(f"Total metadata entries: {count}")
    
    if count > 0:
        cursor.execute("SELECT DISTINCT key FROM embedding_metadata")
        keys = cursor.fetchall()
        print("Available keys:", [k[0] for k in keys])
        
        cursor.execute("SELECT * FROM embedding_metadata LIMIT 1")
        columns = [description[0] for description in cursor.description]
        print("Columns:", columns)
        print("Sample row:", cursor.fetchone())
    
    conn.close()

if __name__ == "__main__":
    check_data() 