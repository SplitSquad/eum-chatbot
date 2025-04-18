from typing import List, Optional, Dict
from loguru import logger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config.rag_config import RAGConfig, RAGDomain
from app.services.chatbot.chatbot_classifier import RAGType

class ChatbotRAGService:
    """챗봇 RAG 서비스"""
    
    # RAGType과 RAGDomain 매핑
    RAG_TYPE_TO_DOMAIN: Dict[RAGType, RAGDomain] = {
        RAGType.VISA_LAW: RAGDomain.VISA_LAW,
        RAGType.SOCIAL_SECURITY: RAGDomain.SOCIAL_SECURITY,
        RAGType.TAX_FINANCE: RAGDomain.TAX_FINANCE,
        RAGType.MEDICAL_HEALTH: RAGDomain.MEDICAL_HEALTH,
        RAGType.EMPLOYMENT: RAGDomain.EMPLOYMENT,
        RAGType.DAILY_LIFE: RAGDomain.DAILY_LIFE
    }
    
    def __init__(self):
        self.config = RAGConfig()
        self.embeddings = SentenceTransformer(self.config.EMBEDDING_MODEL)
        logger.info(f"[RAG] 임베딩 모델 사용: {self.config.EMBEDDING_MODEL}")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )
        
        # 도메인별 컬렉션 초기화
        self.collections: Dict[RAGDomain, chromadb.Collection] = {}
        for domain, collection_name in self.config.COLLECTIONS.items():
            try:
                self.collections[domain] = self.client.get_collection(collection_name)
            except ValueError:
                self.collections[domain] = self.client.create_collection(collection_name)
            logger.info(f"[RAG] {domain.value} 도메인 컬렉션 초기화: {collection_name}")
        
        # 테스트 문서 추가
        test_documents = {
            RAGDomain.SOCIAL_SECURITY: [
                "사회 보장 급여 신청 절차는 다음과 같습니다: 1. 신청서 작성 2. 필요한 서류 준비 3. 주민센터 방문 4. 심사 대기 5. 결과 통보",
                "사회 보장 급여 신청 시 필요한 서류: 주민등록등본, 소득 증명서, 건강보험 자격 증명서, 은행 계좌 정보",
                "사회 보장 급여 수급 자격: 65세 이상, 소득 기준 미달, 기초생활 수급자 등이 해당됩니다."
            ]
        }
        
        for domain, docs in test_documents.items():
            try:
                self.add_documents(domain, docs)
            except Exception as e:
                logger.error(f"테스트 문서 추가 중 오류 발생: {str(e)}")
    
    async def add_documents(self, domain: RAGDomain, documents: List[str]) -> None:
        """
        특정 도메인의 문서를 벡터 DB에 추가합니다.
        
        Args:
            domain: 도메인
            documents: 추가할 문서 리스트
        """
        try:
            logger.info(f"[RAG] {domain.value} 도메인에 문서 추가 시작: {len(documents)}개")
            
            # 문서 임베딩 생성
            embeddings = self.embeddings.encode(documents)
            
            # 문서 ID 생성
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 컬렉션에 문서 추가
            self.collections[domain].add(
                embeddings=embeddings.tolist(),
                documents=documents,
                ids=doc_ids
            )
            
            logger.info(f"[RAG] {domain.value} 도메인에 {len(documents)}개의 문서가 추가되었습니다.")
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            raise
    
    async def search(self, domain: RAGDomain, query: str) -> List[str]:
        """
        특정 도메인에서 질의와 관련된 문서를 검색합니다.
        
        Args:
            domain: 도메인
            query: 검색 질의
            
        Returns:
            List[str]: 검색된 문서 리스트
        """
        try:
            logger.info(f"[RAG] {domain.value} 도메인에서 문서 검색 시작: {query}")
            
            # 질의 임베딩 생성
            query_embedding = self.embeddings.encode([query])[0]
            
            # 유사도 검색
            results = self.collections[domain].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.SEARCH_K
            )
            
            # 임계값 이상의 문서만 반환
            filtered_docs = []
            for doc, score in zip(results['documents'][0], results['distances'][0]):
                if score >= self.config.SEARCH_THRESHOLD:
                    filtered_docs.append(doc)
            
            logger.info(f"[RAG] {domain.value} 도메인에서 {len(filtered_docs)}개의 문서가 검색되었습니다.")
            for i, doc in enumerate(filtered_docs):
                logger.info(f"[RAG] 문서 {i+1}: {doc[:100]}...")
            
            return filtered_docs
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {str(e)}")
            return []
    
    async def get_context(self, rag_type: RAGType, query: str) -> str:
        """
        특정 RAG 유형에서 질의에 대한 컨텍스트를 생성합니다.
        
        Args:
            rag_type: RAG 유형
            query: 질의
            
        Returns:
            str: 생성된 컨텍스트
        """
        try:
            logger.info(f"[RAG] {rag_type.value} 유형에서 컨텍스트 생성 시작")
            
            # RAGType을 RAGDomain으로 변환
            if rag_type == RAGType.NONE:
                logger.info("[RAG] RAG 유형이 NONE이므로 컨텍스트를 생성하지 않습니다.")
                return ""
            
            domain = self.RAG_TYPE_TO_DOMAIN.get(rag_type)
            if not domain:
                logger.error(f"[RAG] {rag_type.value}에 해당하는 도메인이 없습니다.")
                return ""
            
            # 관련 문서 검색
            docs = await self.search(domain, query)
            
            if not docs:
                logger.info("[RAG] 관련 문서가 없습니다.")
                return ""
            
            # 컨텍스트 생성
            context = "\n\n".join(docs)
            logger.info(f"[RAG] 컨텍스트 생성 완료: {len(context)}자")
            
            return context
        except Exception as e:
            logger.error(f"컨텍스트 생성 중 오류 발생: {str(e)}")
            return "" 