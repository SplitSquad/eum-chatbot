import sqlite3
from loguru import logger

def check_schema(db_path):
    """ChromaDB 스키마 확인"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 가져오기
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        logger.info("=== 테이블 목록 ===")
        for table in tables:
            table_name = table[0]
            logger.info(f"\n테이블: {table_name}")
            
            # 각 테이블의 스키마 확인
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                logger.info(f"  컬럼: {col[1]} ({col[2]})")
            
            # 샘플 데이터 확인 (첫 번째 행)
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
                row = cursor.fetchone()
                if row:
                    logger.info(f"  샘플 데이터: {row}")
            except Exception as e:
                logger.error(f"  샘플 데이터 조회 실패: {str(e)}")
                
    except Exception as e:
        logger.error(f"스키마 확인 중 오류 발생: {str(e)}")
    finally:
        if conn:
            conn.close()

def main():
    """메인 함수"""
    # 각 도메인별 데이터베이스 확인
    domains = {
        "visa_law": "data/vectorstore1/chroma.sqlite3",
        "social_security": "data/vectorstore2/chroma.sqlite3",
        "tax_finance": "data/vectorstore3/chroma.sqlite3",
        "medical_health": "data/vectorstore4/chroma.sqlite3",
        "employment": "data/vectorstore5/chroma.sqlite3",
        "daily_life": "data/vectorstore6/chroma.sqlite3"
    }
    
    for domain, path in domains.items():
        logger.info(f"\n도메인 스키마 확인: {domain}")
        check_schema(path)

if __name__ == "__main__":
    main() 