#!/usr/bin/env python3
"""
Updated scraper to work with the new HTTP-based LightRAG server.
"""

import asyncio
import json
import os
import re
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import aiohttp  # Используем aiohttp и для запросов к сайту, и к нашему серверу
import logging

# Настройки
BASE_URL = "http://filonov.net"
LIGHTRAG_HTTP_ENDPOINT = os.getenv("LIGHTRAG_HTTP_URL", "http://localhost:8000/api/insert")
LIGHTRAG_BATCH_ENDPOINT = os.getenv("LIGHTRAG_HTTP_BATCH_URL", "http://localhost:8000/api/insert/batch")

# ⏱️ Настройки устойчивости к разрывам сети (для обоих сайтов и нашего сервера)
TIMEOUT_CONNECT = 10      # секунд на установку соединения
TIMEOUT_READ = 60         # секунд на получение ответа от сайта
TIMEOUT_LIGHTRAG_READ = 120 # секунд на обработку вставки в LightRAG (может быть долго)
MAX_RETRIES = 3           # попыток при сбое
BACKOFF_BASE = 2          # база для экспоненциальной задержки

CATEGORIES = [
    f"{BASE_URL}/statji/",
    # f"{BASE_URL}/statji/vidyi-golodaniya/",
    # f"{BASE_URL}/statji/podgotovka-k-golodu/",
    # f"{BASE_URL}/statji/osnonovi/",
    # f"{BASE_URL}/statji/suh-golod-effektivno/",
    # f"{BASE_URL}/statji/protivoparazitarnoe-lechenie/",
    # f"{BASE_URL}/statji/primeryi-iz-praktiki/",
    # f"{BASE_URL}/statji/dietologicheskie-mifi-o-vrede-golodaniya/",
    # f"{BASE_URL}/statji/interesnie-svedeniya-o-suhom-golodanii/",
    # f"{BASE_URL}/statji/poleznie-svedeniya-o-golodanii/",
    # f"{BASE_URL}/statji/primenenie-lechebnogo-golodaniya-u-detey/",
    # f"{BASE_URL}/statji/drugie-metodi-lecheniya/",
]

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_article_url(url: str) -> bool:
    if "filonov.net" not in url:
        return False
    if any(x in url for x in ['/page/', '/category/', '/tag/', '/author/', '/feed/', '/wp-', '/?']):
        return False
    if url.rstrip('/') in ["http://filonov.net", "http://filonov.net/", "http://filonov.net/statji"]:
        return False
    # Предположим, статьи — это URL с 4-9 сегментами пути (http://.../seg1/seg2/../segN/)
    return 4 <= url.count('/') <= 9


async def fetch_with_retry(session: aiohttp.ClientSession, url: str, target='site') -> str | None:
    """Загрузка страницы с таймаутами и повторами"""
    timeout_val = TIMEOUT_READ if target == 'site' else TIMEOUT_LIGHTRAG_READ
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_val, connect=TIMEOUT_CONNECT)) as resp:
                resp.raise_for_status()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"❌ Failed {url} after {MAX_RETRIES} retries: {type(e).__name__}")
                return None
            delay = BACKOFF_BASE ** attempt
            logger.warning(f"⚠️ Retry {attempt+1}/{MAX_RETRIES} for {url} in {delay}s... ({type(e).__name__})")
            await asyncio.sleep(delay)


async def get_all_article_links() -> list[str]:
    article_links = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    async with aiohttp.ClientSession(headers=headers) as session:
        for cat_url in CATEGORIES:
            logger.info(f"🔍 Scanning: {cat_url}")
            html = await fetch_with_retry(session, cat_url, target='site')
            if not html:
                continue
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                full_url = urljoin("http://filonov.net", link['href']).rstrip('/')
                if is_article_url(full_url):
                    article_links.add(full_url)
    return list(article_links)


async def extract_text_from_html(html: str) -> str:
    """Извлекает читаемый текст из HTML, убирая мусор"""
    soup = BeautifulSoup(html, 'html.parser')
    # Удаляем скрипты, стили, навигацию, футеры
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()
    # Берём основной контент или article/main
    content = soup.find('article') or soup.find('main') or soup.find('div', class_='post-content') or soup.body
    return content.get_text(separator='\n', strip=True) if content else ""


# --- НОВАЯ ФУНКЦИЯ: отправка в LightRAG через HTTP ---
async def send_to_lightrag_via_http(text: str, session: aiohttp.ClientSession) -> dict:
    """
    Отправляет один чанк текста в LightRAG через HTTP API.
    """
    payload = {
        "text": text
    }
    try:
        async with session.post(LIGHTRAG_HTTP_ENDPOINT, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result
    except aiohttp.ClientResponseError as e:
        error_detail = await e.response.text()
        logger.error(f"❌ HTTP {e.status} error from LightRAG: {error_detail}")
        return {"error": f"HTTP {e.status}: {error_detail}"}
    except Exception as e:
        logger.error(f"❌ Unexpected error sending to LightRAG: {e}")
        return {"error": str(e)}

async def send_batch_to_lightrag_via_http(texts: list[str], session: aiohttp.ClientSession) -> dict:
    """
    Отправляет список чанков в LightRAG через HTTP API (batch endpoint).
    """
    payload = {
        "texts": texts
    }
    try:
        async with session.post(LIGHTRAG_BATCH_ENDPOINT, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result
    except aiohttp.ClientResponseError as e:
        error_detail = await e.response.text()
        logger.error(f"❌ Batch HTTP {e.status} error from LightRAG: {error_detail}")
        return {"error": f"Batch HTTP {e.status}: {error_detail}"}
    except Exception as e:
        logger.error(f"❌ Unexpected error sending batch to LightRAG: {e}")
        return {"error": str(e)}


# --- ОСНОВНОЙ ЦИКЛ СКРАПИНГА ---
async def load_and_index_filonov_articles():
    # 1. Сбор ссылок
    logger.info("🔍 Collecting article links...")
    article_urls = await get_all_article_links()
    logger.info(f"✅ Found {len(article_urls)} unique article URLs")

    if not article_urls:
        logger.error("❌ No articles found. Check site structure or categories.")
        return

    # 2. Асинхронная загрузка и очистка текста
    logger.info("📥 Loading & cleaning articles...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    texts = []
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch_with_retry(session, url, target='site') for url in article_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, html in zip(article_urls, results):
            if isinstance(html, Exception) or not html:
                logger.warning(f"  ⚠️ Skipped {url}")
                continue
            text = await extract_text_from_html(html)
            if len(text) > 100:  # пропускаем пустые/битые страницы
                texts.append({"url": url, "text": text})
            else:
                logger.warning(f"  ⚠️ Empty content: {url}")

    logger.info(f"✅ Loaded {len(texts)} valid articles")

    # 3. Сплиттинг
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = []
    for item in texts:
        docs = splitter.create_documents([item["text"]], [{"source": item["url"]}])
        chunks.extend([d.page_content for d in docs if len(d.page_content.strip()) > 50])
    logger.info(f"✂️ Generated {len(chunks)} chunks")

    # 4. Отправка чанков в LightRAG через HTTP
    logger.info(f"💾 Sending chunks to LightRAG server at {LIGHTRAG_HTTP_ENDPOINT}...")

    # Создаём отдельную сессию для запросов к нашему HTTP-серверу
    # Увеличиваем timeout для сервера LightRAG
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_LIGHTRAG_READ, connect=TIMEOUT_CONNECT)
    async with aiohttp.ClientSession(timeout=timeout) as rag_session:
        failed = 0
        successful = 0

        # Вариант 1: Отправка по одному (медленнее, но точнее обработка ошибок)
        for i, chunk in enumerate(chunks, 1):
            res = await send_to_lightrag_via_http(chunk, rag_session)
            if "error" in res:
                failed += 1
                logger.error(f"  [{i}/{len(chunks)}] ⚠️ Error: {res['error'][:60]}...")
            else:
                successful += 1
                logger.info(f"  [{i}/{len(chunks)}] ✅ OK")
            # Щадящий режим для Ollama и сервера
            await asyncio.sleep(0.1)

        # Вариант 2: Отправка батчами (эффективнее для большого количества чанков)
        # BATCH_SIZE = 10 # Отправлять по 10 чанков за раз
        # total_chunks = len(chunks)
        # for i in range(0, total_chunks, BATCH_SIZE):
        #     batch = chunks[i:i+BATCH_SIZE]
        #     res = await send_batch_to_lightrag_via_http(batch, rag_session)
        #     if "error" in res:
        #         failed += len(batch)
        #         logger.error(f"  Batch [{i//BATCH_SIZE + 1}] ⚠️ Errors: {res['error'][:100]}...")
        #     else:
        #         successful += len(batch)
        #         logger.info(f"  Batch [{i//BATCH_SIZE + 1}] ✅ Inserted {len(batch)} chunks")
        #     # Щадящий режим
        #     await asyncio.sleep(0.5)

    logger.info("\n--- Summary ---")
    logger.info(f"Total Chunks: {len(chunks)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    if failed == 0:
        logger.info("🎉 All chunks processed successfully!")
    else:
        logger.warning(f"⚠️ {failed} chunks failed. Check your LightRAG server logs.")


async def main():
    logger.info("🚀 Starting robust scraper for http://filonov.net with HTTP API...")
    logger.info("💡 Tip: Ensure your LightRAG HTTP server (mcp_lightrag_server.py) is running.")
    logger.info("💡 Tip: Run with OLLAMA_KEEP_ALIVE=-1 to prevent model unloading. можно 60m")
    await load_and_index_filonov_articles()


if __name__ == "__main__":
    asyncio.run(main())
