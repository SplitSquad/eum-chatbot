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
        # 환경 변수 로드
        self.search_provider = get_env_var("WEB_SEARCH_PROVIDER", "google").strip().lower()
        logger.info(f"[웹 검색] 검색 제공자: {self.search_provider}")
        
        if self.search_provider == "google":
            # Google Search API 초기화
            self.google_api_key = get_env_var("GOOGLE_API_KEY", "").strip()
            self.google_cse_id = get_env_var("GOOGLE_CSE_ID", "").strip()
            
            if not self.google_api_key or not self.google_cse_id:
                logger.error("[웹 검색] Google Search API를 사용하기 위해서는 GOOGLE_API_KEY와 GOOGLE_CSE_ID가 필요합니다.")
                raise ValueError("Google Search API credentials are required")
            
            try:
                self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
                logger.info("[웹 검색] Google Search API 초기화 완료")
                logger.debug(f"[웹 검색] Google CSE ID: {self.google_cse_id}")
                logger.debug(f"[웹 검색] Google API 키: {self.google_api_key[:5]}...")
            except Exception as e:
                logger.error(f"[웹 검색] Google Search API 초기화 실패: {str(e)}")
                raise
        elif self.search_provider == "duckduckgo":
            # DuckDuckGo API 초기화
            self.duckduckgo_api_key = get_env_var("DUCKDUCKGO_API_KEY", "").strip()
            if not self.duckduckgo_api_key:
                logger.error("[웹 검색] DuckDuckGo API를 사용하기 위해서는 DUCKDUCKGO_API_KEY가 필요합니다.")
                raise ValueError("DuckDuckGo API key is required")
            logger.info("[웹 검색] DuckDuckGo API 초기화 완료")
        else:
            logger.error(f"[웹 검색] 지원하지 않는 검색 제공자입니다: {self.search_provider}")
            raise ValueError(f"Unsupported search provider: {self.search_provider}")
    
    async def _google_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Google Custom Search API를 사용하여 검색을 수행합니다."""
        try:
            logger.info("[웹 검색] Google 검색 시작")
            logger.info(f"[웹 검색] 검색 질의: {query}")
            logger.debug(f"[웹 검색] 최대 결과 수: {max_results}")
            logger.debug(f"[웹 검색] Google CSE ID: {self.google_cse_id}")
            logger.debug(f"[웹 검색] Google API 키: {self.google_api_key[:5]}...")
            
            # 동기 API 호출을 비동기로 실행
            loop = asyncio.get_event_loop()
            
            # API 요청 구성
            try:
                request = self.google_service.cse().list(
                    q=query,
                    cx=self.google_cse_id,
                    num=max_results
                )
                logger.debug("[웹 검색] Google API 요청 구성 완료")
            except Exception as e:
                logger.error(f"[웹 검색] Google API 요청 구성 실패: {str(e)}")
                raise
            
            # 요청 실행
            try:
                logger.debug("[웹 검색] Google API 요청 실행 시작")
                result = await loop.run_in_executor(None, request.execute)
                logger.debug("[웹 검색] Google API 요청 실행 완료")
                logger.debug(f"[웹 검색] 응답: {result}")
            except HttpError as e:
                logger.error(f"[웹 검색] Google API HTTP 오류: {e.resp.status} - {e.content}")
                if e.resp.status == 403:
                    logger.error("[웹 검색] API 키 또는 권한 문제가 발생했습니다.")
                elif e.resp.status == 429:
                    logger.error("[웹 검색] API 할당량을 초과했습니다.")
                raise
            except Exception as e:
                logger.error(f"[웹 검색] Google API 요청 실행 실패: {str(e)}")
                raise
            
            # 결과 처리
            formatted_results = []
            if "items" in result:
                for item in result["items"]:
                    formatted_result = {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    }
                    formatted_results.append(formatted_result)
                    logger.info(f"[웹 검색] 검색 결과: {formatted_result['title']}")
                    logger.debug(f"[웹 검색] URL: {formatted_result['url']}")
                    logger.debug(f"[웹 검색] 요약: {formatted_result['snippet']}")
                logger.info(f"[웹 검색] {len(formatted_results)}개의 결과 검색 완료")
            else:
                logger.warning("[웹 검색] 검색 결과가 없습니다.")
                if "error" in result:
                    error_info = result["error"]
                    logger.error(f"[웹 검색] API 오류: {error_info.get('message', '')}")
                    logger.error(f"[웹 검색] 오류 코드: {error_info.get('code', '')}")
                    logger.error(f"[웹 검색] 오류 상태: {error_info.get('status', '')}")
                elif "searchInformation" in result:
                    logger.info(f"[웹 검색] 검색 정보: {result['searchInformation']}")
            
            return formatted_results
            
        except HttpError as e:
            logger.error(f"[웹 검색] Google API 오류: {e.resp.status} - {e.content}")
            return []
        except Exception as e:
            logger.error(f"[웹 검색] Google 검색 중 오류 발생: {str(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"[웹 검색] 오류 상세 정보: {e.__dict__}")
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
            logger.info("[웹 검색] 검색 시작")
            logger.info(f"[웹 검색] 검색 질의: {query}")
            logger.debug(f"[웹 검색] 검색 제공자: {self.search_provider}")
            logger.debug(f"[웹 검색] 최대 결과 수: {max_results}")
            
            if self.search_provider == "google":
                return await self._google_search(query, max_results)
            elif self.search_provider == "duckduckgo":
                return await self._duckduckgo_search(query, max_results)
            else:
                logger.error(f"[웹 검색] 지원하지 않는 검색 제공자입니다: {self.search_provider}")
                return []
                    
        except Exception as e:
            logger.error(f"[웹 검색] 검색 중 오류 발생: {str(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"[웹 검색] 오류 상세 정보: {e.__dict__}")
            return []
    
    async def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """DuckDuckGo API를 사용하여 검색을 수행합니다."""
        try:
            import aiohttp
            
            logger.info("[웹 검색] DuckDuckGo 검색 시작")
            logger.info(f"[웹 검색] 검색 질의: {query}")
            logger.debug(f"[웹 검색] 최대 결과 수: {max_results}")
            
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
                        logger.error(f"[웹 검색] DuckDuckGo API 요청 실패: {response.status}")
                        return []
                    
                    try:
                        data = await response.json()
                    except Exception as e:
                        logger.error(f"[웹 검색] DuckDuckGo API 응답 파싱 실패: {str(e)}")
                        return []
                    
                    results = []
                    if "RelatedTopics" in data:
                        for topic in data["RelatedTopics"][:max_results]:
                            if "Text" in topic:
                                result = {
                                    "title": topic.get("Text", ""),
                                    "url": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", "")
                                }
                                results.append(result)
                                logger.info(f"[웹 검색] 검색 결과: {result['title']}")
                                logger.debug(f"[웹 검색] URL: {result['url']}")
                                logger.debug(f"[웹 검색] 요약: {result['snippet']}")
                        logger.info(f"[웹 검색] {len(results)}개의 결과 검색 완료")
                        return results
                    else:
                        logger.warning("[웹 검색] 검색 결과가 없습니다.")
                        return []
                    
        except Exception as e:
            logger.error(f"[웹 검색] DuckDuckGo 검색 중 오류 발생: {str(e)}")
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
            logger.info("[웹 검색] 컨텍스트 생성 시작")
            logger.info(f"[웹 검색] 검색 질의: {query}")
            
            # 웹 검색 실행
            results = await self.search_web(query)
            
            if not results:
                logger.info("[웹 검색] 검색 결과가 없습니다.")
                return ""
            
            # 검색 결과 분석 및 요약
            context_parts = []
            price_info = []
            date_info = []
            airline_info = []
            booking_info = []
            
            for i, result in enumerate(results, 1):
                # 가격 정보 추출
                if "가격" in result['title'] or "가격" in result['snippet'] or "요금" in result['title'] or "요금" in result['snippet']:
                    price_info.append(f"검색 결과 {i}:\n제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}")
                
                # 날짜 정보 추출
                if "날짜" in result['title'] or "날짜" in result['snippet'] or "기간" in result['title'] or "기간" in result['snippet']:
                    date_info.append(f"검색 결과 {i}:\n제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}")
                
                # 항공사 정보 추출
                if "항공사" in result['title'] or "항공사" in result['snippet'] or "항공" in result['title'] or "항공" in result['snippet']:
                    airline_info.append(f"검색 결과 {i}:\n제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}")
                
                # 예약 정보 추출
                if "예약" in result['title'] or "예약" in result['snippet'] or "예매" in result['title'] or "예매" in result['snippet']:
                    booking_info.append(f"검색 결과 {i}:\n제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}")
                
                # 기타 정보
                context_parts.append(f"검색 결과 {i}:\n제목: {result['title']}\n요약: {result['snippet']}\nURL: {result['url']}")
            
            # 컨텍스트 구성
            context = []
            
            if price_info:
                context.append("=== 가격 정보 ===\n" + "\n\n".join(price_info))
            
            if date_info:
                context.append("=== 날짜/기간 정보 ===\n" + "\n\n".join(date_info))
            
            if airline_info:
                context.append("=== 항공사 정보 ===\n" + "\n\n".join(airline_info))
            
            if booking_info:
                context.append("=== 예약/예매 정보 ===\n" + "\n\n".join(booking_info))
            
            if not context:
                context.append("=== 검색 결과 ===\n" + "\n\n".join(context_parts))
            
            final_context = "\n\n".join(context)
            
            logger.info(f"[웹 검색] 컨텍스트 생성 완료: {len(final_context)}자")
            logger.debug(f"[웹 검색] 생성된 컨텍스트: {final_context[:200]}...")
            
            return final_context
        except Exception as e:
            logger.error(f"[웹 검색] 컨텍스트 생성 중 오류 발생: {str(e)}")
            return "" 