# app/services/common/preprocessor.py

from typing import Dict
import json
import time
from loguru import logger
from app.core.llm_client import get_llm_client

PROMPT_TEMPLATE = """
Detect the language of the following query, then translate it to English.

Return the result ONLY in this JSON format:

```json
{{
  "translated_query": "...",
  "lang_code": "..."
}}
Query: "{query}" """

async def translate_query(query: str) -> Dict[str, str]:
    """
    Translates the given query to English and returns language code.

    Args:
        query (str): The query text to translate

    Returns:
        Dict[str, str]: Dictionary containing 'translated_query' and 'lang_code'
                        - 'translated_query': Query translated to English
                        - 'lang_code': IETF language code of original language (e.g., 'ko' for Korean)

    Raises:
        ValueError: When translation response cannot be parsed
    """
    start_time = time.time()
    result = None
    
    try:
        # Log original query
        logger.info(f"[PREPROCESS] Original query: {query}")
        
        # Initialize lightweight model client
        init_start = time.time()
        llm_client = get_llm_client(is_lightweight=True)
        init_time = time.time() - init_start
        logger.info(f"[Preprocess] Client initialization time: {init_time:.2f} seconds")
        logger.info(f"[Preprocess] Model used: {llm_client.model}")
        
        # Check server connection
        conn_start = time.time()
        if not await llm_client.check_connection():
            raise ConnectionError("Failed to connect to LLM server")
        conn_time = time.time() - conn_start
        logger.info(f"[Preprocess] Server connection check time: {conn_time:.2f} seconds")
        
        # Translation request
        gen_start = time.time()
        result = await llm_client.generate(
            prompt=PROMPT_TEMPLATE.format(query=query)
        )
        gen_time = time.time() - gen_start
        logger.info(f"[Preprocess] LLM generation time: {gen_time:.2f} seconds")
        
        # Extract and process JSON
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Preprocess] Translation response: {json_str}")
        
        data = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Preprocess] JSON parsing time: {parse_time:.2f} seconds")
        
        # Log the preprocessed result
        logger.info(f"[PREPROCESS] Translated query: {data['translated_query']}")
        logger.info(f"[PREPROCESS] Detected language: {data['lang_code']}")
        
        total_time = time.time() - start_time
        logger.info(f"[Preprocess] Total execution time: {total_time:.2f} seconds")
        
        return {
            "translated_query": data["translated_query"],
            "lang_code": data["lang_code"]
        }
    except json.JSONDecodeError as e:
        logger.error(f"[Preprocess] Failed to parse translation response JSON: {str(e)}")
        logger.error(f"[Preprocess] Original response: {result}")
        raise ValueError(f"Translation response parsing error: {e}\nOriginal response: {result}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Preprocess] Error during translation processing: {str(e)} (Time elapsed: {elapsed:.2f} seconds)")
        raise ValueError(f"Error during translation processing: {str(e)}") 