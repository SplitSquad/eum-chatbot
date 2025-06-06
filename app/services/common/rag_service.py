from typing import List, Optional, Dict, Union
from loguru import logger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config.rag_config import RAGConfig
from app.services.chatbot.chatbot_classifier import RAGType
import os

class RAGService:
    """RAG 서비스"""
    
    def __init__(self):
        self.config = RAGConfig()
        # 벡터 스토어 경로 검증
        self.config.validate_paths()
        
        self.embeddings = SentenceTransformer(self.config.EMBEDDING_MODEL)
        logger.info(f"[RAG] 임베딩 모델 사용: {self.config.EMBEDDING_MODEL}")
        
        # 도메인별 ChromaDB 클라이언트 초기화
        self.clients: Dict[RAGType, chromadb.PersistentClient] = {}
        self.collections: Dict[RAGType, chromadb.Collection] = {}
        
        # 벡터 스토어 검증 및 초기화
        self._validate_and_initialize_vectorstores()
    
    def _validate_and_initialize_vectorstores(self) -> None:
        """벡터 스토어를 검증하고 초기화합니다."""
        for rag_type, config in self.config.DOMAIN_CONFIGS.items():
            try:
                # 벡터 스토어 경로 검증
                vectorstore_path = config["vectorstore_path"]
                logger.info(f"[RAG] {rag_type.value} 도메인 벡터 스토어 경로: {vectorstore_path}")
                
                if not os.path.exists(vectorstore_path):
                    logger.warning(f"[RAG] {rag_type.value} 도메인의 벡터 스토어 경로가 존재하지 않습니다: {vectorstore_path}")
                    os.makedirs(vectorstore_path, exist_ok=True)
                    logger.info(f"[RAG] {rag_type.value} 도메인의 벡터 스토어 디렉토리를 생성했습니다: {vectorstore_path}")
                
                # ChromaDB 파일 확인
                chroma_db_file = os.path.join(vectorstore_path, "chroma.sqlite3")
                if os.path.exists(chroma_db_file):
                    logger.info(f"[RAG] {rag_type.value} 도메인의 ChromaDB 파일 크기: {os.path.getsize(chroma_db_file)} bytes")
                else:
                    logger.warning(f"[RAG] {rag_type.value} 도메인의 ChromaDB 파일이 존재하지 않습니다: {chroma_db_file}")
                
                # ChromaDB 클라이언트 생성
                client = chromadb.PersistentClient(
                    path=vectorstore_path,
                    settings=Settings(allow_reset=True)
                )
                self.clients[rag_type] = client
                
                # 컬렉션 초기화 및 검증
                try:
                    collection = client.get_collection(config["collection_name"])
                    count = collection.count()
                    
                    if count == 0:
                        logger.warning(f"[RAG] {rag_type.value} 도메인 컬렉션이 비어있습니다.")
                    else:
                        logger.info(f"[RAG] {rag_type.value} 도메인 컬렉션 로드 완료: {count}개의 문서")
                        # 컬렉션의 첫 번째 문서 샘플 확인
                        try:
                            sample = collection.peek(limit=1)
                            logger.info(f"[RAG] {rag_type.value} 도메인 컬렉션 샘플: {sample}")
                        except Exception as e:
                            logger.error(f"[RAG] {rag_type.value} 도메인 컬렉션 샘플 조회 실패: {str(e)}")
                    
                    # 컬렉션 메타데이터 검증
                    metadata = collection.metadata
                    if not metadata:
                        logger.warning(f"[RAG] {rag_type.value} 도메인 컬렉션의 메타데이터가 없습니다.")
                    else:
                        logger.info(f"[RAG] {rag_type.value} 도메인 컬렉션 메타데이터: {metadata}")
                    
                    self.collections[rag_type] = collection
                    
                except ValueError:
                    logger.warning(f"[RAG] {rag_type.value} 도메인 컬렉션이 존재하지 않습니다.")
                    # 임베딩 차원을 명시적으로 지정하여 컬렉션 생성
                    self.collections[rag_type] = client.create_collection(
                        name=config["collection_name"],
                        metadata={
                            "domain": rag_type.value,
                            "embedding_dimension": 768,  # sentence-transformers/paraphrase-multilingual-mpnet-base-v2 모델의 차원
                            "embedding_model": self.config.EMBEDDING_MODEL
                        }
                    )
                    logger.info(f"[RAG] {rag_type.value} 도메인 컬렉션을 생성했습니다.")
                
                logger.info(f"[RAG] {rag_type.value} 도메인 초기화 완료: {vectorstore_path}")
                
            except Exception as e:
                logger.error(f"[RAG] {rag_type.value} 도메인 초기화 중 오류 발생: {str(e)}")
                raise
    
    def _validate_collection(self, rag_type: RAGType) -> bool:
        """
        특정 도메인의 컬렉션이 유효한지 검증합니다.
        
        Args:
            rag_type: RAG 유형
            
        Returns:
            bool: 컬렉션이 유효하면 True, 아니면 False
        """
        try:
            if rag_type not in self.collections:
                logger.error(f"[RAG] {rag_type.value} 도메인 컬렉션이 존재하지 않습니다.")
                return False
            
            collection = self.collections[rag_type]
            count = collection.count()
            
            if count == 0:
                logger.warning(f"[RAG] {rag_type.value} 도메인 컬렉션이 비어있습니다.")
                return False
            
            # 컬렉션의 임베딩 차원 검증
            metadata = collection.metadata
            if not metadata or "embedding_dimension" not in metadata:
                logger.warning(f"[RAG] {rag_type.value} 도메인 컬렉션의 임베딩 차원 정보가 없습니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"[RAG] {rag_type.value} 도메인 컬렉션 검증 중 오류 발생: {str(e)}")
            return False
    
    def add_documents(self, rag_type: RAGType, documents: List[str]) -> None:
        """
        특정 도메인의 문서를 벡터 DB에 추가합니다.
        
        Args:
            rag_type: RAG 유형
            documents: 추가할 문서 리스트
        """
        try:
            logger.info(f"[RAG] {rag_type.value} 도메인에 문서 추가 시작: {len(documents)}개")
            
            # 문서 임베딩 생성
            embeddings = self.embeddings.encode(documents)
            
            # 문서 ID 생성
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 컬렉션에 문서 추가
            self.collections[rag_type].add(
                embeddings=embeddings.tolist(),
                documents=documents,
                ids=doc_ids
            )
            
            logger.info(f"[RAG] {rag_type.value} 도메인에 {len(documents)}개의 문서가 추가되었습니다.")
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            raise
    
    async def search(self, rag_type: RAGType, query: str, format_as_context: bool = False) -> Union[List[str], str]:
        """
        특정 도메인에서 질의와 관련된 문서를 검색합니다.
        
        Args:
            rag_type: RAG 유형
            query: 검색 질의
            format_as_context: True일 경우 검색 결과를 컨텍스트 문자열로 반환
            
        Returns:
            Union[List[str], str]: 검색된 문서 리스트 또는 컨텍스트 문자열
        """
        try:
            logger.info(f"[RAG] {rag_type.value} 도메인에서 문서 검색 시작: {query}")
            
            # 컬렉션 검증
            if not self._validate_collection(rag_type):
                logger.error(f"[RAG] {rag_type.value} 도메인 컬렉션이 유효하지 않습니다.")
                return [] if not format_as_context else ""
            
            # 질의 임베딩 생성
            query_embedding = self.embeddings.encode([query])[0]
            
            # 유사도 검색
            results = self.collections[rag_type].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.SEARCH_K,
                include=["documents", "distances", "metadatas"]
            )
            
            # 검색 결과 로깅
            logger.info(f"[RAG] {rag_type.value} 도메인 검색 결과:")
            for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
                logger.info(f"[RAG] 문서 {i+1} (유사도: {score:.4f}): {doc[:100]}...")
            
            # 임계값 이상의 문서만 반환 (임계값을 0.2로 낮춤)
            filtered_docs = []
            for doc, score in zip(results['documents'][0], results['distances'][0]):
                if score >= 0.2:  # 임계값을 0.2로 낮춤
                    filtered_docs.append(doc)
            
            if not filtered_docs:
                logger.warning(f"[RAG] {rag_type.value} 도메인에서 임계값({0.2}) 이상의 문서가 없습니다.")
                # 임계값을 만족하는 문서가 없으면 상위 2개 문서 반환
                filtered_docs = results['documents'][0][:2]
            
            logger.info(f"[RAG] {rag_type.value} 도메인에서 {len(filtered_docs)}개의 문서가 검색되었습니다.")
            
            # 컨텍스트 형식으로 반환
            if format_as_context:
                context = "\n\n".join(filtered_docs)
                logger.info(f"[RAG] 컨텍스트 생성 완료: {len(context)}자")
                return context
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return [] if not format_as_context else ""
    
    async def get_context(self, rag_type: RAGType, query: str) -> str:
        """
        특정 RAG 유형에서 질의에 대한 컨텍스트를 생성합니다.
        
        Args:
            rag_type: RAG 유형
            query: 질의
            
        Returns:
            str: 생성된 컨텍스트
        """
        return await self.search(rag_type, query, format_as_context=True) 