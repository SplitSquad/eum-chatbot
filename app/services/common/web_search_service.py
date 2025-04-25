from typing import List, Dict, Any
import asyncio
from functools import partial
from loguru import logger
from app.config.app_config import get_env_var
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class WebSearchService:
    """웹 검색 서비스"""
    
    def __init__(self):
        self.search_provider = get_env_var("WEB_SEARCH_PROVIDER", "google")
        self.google_api_key = get_env_var("GOOGLE_API_KEY", "")
        self.google_cse_id = get_env_var("GOOGLE_CSE_ID", "")
        self.duckduckgo_api_key = get_env_var("DUCKDUCKGO_API_KEY", "")
        
        # Google Search API 초기화
        if self.search_provider == "google":
            if not self.google_api_key or not self.google_cse_id:
                logger.error("Google Search API를 사용하기 위해서는 GOOGLE_API_KEY와 GOOGLE_CSE_ID가 필요합니다.")
                raise ValueError("Google Search API credentials are required")
            try:
                self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
                logger.info("[웹 검색] Google Search API 초기화 완료")
                logger.debug(f"[웹 검색] Google CSE ID: {self.google_cse_id}")
            except Exception as e:
                logger.error(f"[웹 검색] Google Search API 초기화 실패: {str(e)}")
                raise
        else:
            logger.info("[웹 검색] DuckDuckGo API 초기화 완료")
    
    async def _google_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Google Custom Search API를 사용하여 검색을 수행합니다."""
        try:
            logger.info(f"[웹 검색] Google 검색 시작: {query}")
            logger.debug(f"[웹 검색] Google API 키: {self.google_api_key[:5]}...")
            logger.debug(f"[웹 검색] Google CSE ID: {self.google_cse_id}")
            
            # 동기 API 호출을 비동기로 실행
            loop = asyncio.get_event_loop()
            
            # API 요청 구성
            request = self.google_service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=max_results
            )
            
            # 요청 실행
            try:
                result = await loop.run_in_executor(None, request.execute)
                logger.debug(f"[웹 검색] Google 검색 응답: {result}")
            except Exception as e:
                logger.error(f"[웹 검색] Google API 호출 실패: {str(e)}")
                if hasattr(e, 'resp'):
                    logger.error(f"[웹 검색] 응답 상태: {e.resp.status}")
                    logger.error(f"[웹 검색] 응답 내용: {e.content}")
                return []
            
            # 결과 처리
            formatted_results = []
            if "items" in result:
                for item in result["items"]:
                    formatted_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                logger.info(f"[웹 검색] {len(formatted_results)}개의 결과 검색 완료")
            else:
                logger.warning("[웹 검색] 검색 결과가 없습니다.")
                if "error" in result:
                    logger.error(f"[웹 검색] API 오류: {result['error']}")
                elif "searchInformation" in result:
                    logger.info(f"[웹 검색] 검색 정보: {result['searchInformation']}")
            
            return formatted_results
            
        except HttpError as e:
            logger.error(f"[웹 검색] Google API 오류: {e.resp.status} - {e.content}")
            return []
        except Exception as e:
            logger.error(f"[웹 검색] Google 검색 중 오류 발생: {str(e)}")
            return []
    
    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        웹에서 질의에 대한 정보를 검색합니다.
        
        Args:
            query: 검색 질의
            max_results: 최대 검색 결과 수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
        """
        try:
            logger.info(f"[웹 검색] 검색 시작: {query}")
            logger.debug(f"[웹 검색] 검색 제공자: {self.search_provider}")
            
            if self.search_provider == "google":
                return await self._google_search(query, max_results)
            else:
                # DuckDuckGo API 사용 (기존 로직)
                import aiohttp
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "no_redirect": 1,
                    "skip_disambig": 1,
                    "api_key": self.duckduckgo_api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get("https://api.duckduckgo.com/", params=params) as response:
                        if response.status != 200:
                            logger.error(f"[웹 검색] API 요청 실패: {response.status}")
                            return []
                        
                        data = await response.json()
                        results = []
                        if "RelatedTopics" in data:
                            for topic in data["RelatedTopics"][:max_results]:
                                if "Text" in topic:
                                    results.append({
                                        "title": topic.get("Text", ""),
                                        "url": topic.get("FirstURL", ""),
                                        "snippet": topic.get("Text", "")
                                    })
                        logger.info(f"[웹 검색] {len(results)}개의 결과 검색 완료")
                        return results
                    
        except Exception as e:
            logger.error(f"[웹 검색] 검색 중 오류 발생: {str(e)}")
            return []
    
    async def get_context(self, query: str) -> str:
        """
        웹 검색 결과를 컨텍스트로 변환합니다.
        
        Args:
            query: 검색 질의
            
        Returns:
            str: 생성된 컨텍스트
        """
        try:
            logger.info(f"[웹 검색] 컨텍스트 생성 시작: {query}")
            
            # 웹 검색 실행
            results = await self.search_web(query)
            
            if not results:
                logger.info("[웹 검색] 검색 결과가 없습니다.")
                return ""
            
            # 컨텍스트 생성
            context = "\n\n".join([
                f"제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}"
                for result in results
            ])
            
            logger.info(f"[웹 검색] 컨텍스트 생성 완료: {len(context)}자")
            logger.debug(f"[웹 검색] 생성된 컨텍스트: {context[:200]}...")
            
            return context
        except Exception as e:
            logger.error(f"[웹 검색] 컨텍스트 생성 중 오류 발생: {str(e)}")
            return "" 