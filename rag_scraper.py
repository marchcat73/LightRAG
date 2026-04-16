#!/usr/bin/env python3
import asyncio
import json
import os
import re
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import aiohttp

MCP_SERVER_PATH = "/home/marchcat/ollama/LightRAG/mcp_lightrag_server.py"
MCP_SERVER_CMD = ["python", MCP_SERVER_PATH]
BASE_URL = "http://filonov.net"

# ⏱️ Настройки устойчивости к разрывам сети
TIMEOUT_CONNECT = 10      # секунд на установку соединения
TIMEOUT_READ = 60         # секунд на получение ответа
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

def is_article_url(url: str) -> bool:
    if BASE_URL not in url:
        return False
    if any(x in url for x in ['/page/', '/category/', '/tag/', '/author/', '/feed/', '/wp-', '/?']):
        return False
    if url.rstrip('/') in [BASE_URL, f"{BASE_URL}/", f"{BASE_URL}/statji"]:
        return False
    return 4 <= url.count('/') <= 9

async def fetch_with_retry(session: aiohttp.ClientSession, url: str) -> str | None:
    """Загрузка страницы с таймаутами и повторами"""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_READ, connect=TIMEOUT_CONNECT)) as resp:
                resp.raise_for_status()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if attempt == MAX_RETRIES - 1:
                print(f"❌ Failed {url} after {MAX_RETRIES} retries: {type(e).__name__}")
                return None
            delay = BACKOFF_BASE ** attempt
            print(f"⚠️ Retry {attempt+1}/{MAX_RETRIES} for {url} in {delay}s... ({type(e).__name__})")
            await asyncio.sleep(delay)

async def get_all_article_links() -> list[str]:
    article_links = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    async with aiohttp.ClientSession(headers=headers) as session:
        for cat_url in CATEGORIES:
            print(f"🔍 Scanning: {cat_url}")
            html = await fetch_with_retry(session, cat_url)
            if not html:
                continue

            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                full_url = urljoin(BASE_URL, link['href']).rstrip('/')
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

async def send_to_lightrag_via_mcp(text: str, process: asyncio.subprocess.Process) -> dict:
    message = json.dumps({"method": "insert", "params": {"text": text}})
    process.stdin.write((message + "\n").encode())
    await process.stdin.drain()

    response_line = await process.stdout.readline()
    try:
        return json.loads(response_line)
    except Exception:
        return {"error": response_line.decode().strip() if response_line else "Empty response"}

async def load_and_index_filonov_articles():
    # 1. Сбор ссылок
    print("🔍 Collecting article links...")
    article_urls = await get_all_article_links()
    print(f"✅ Found {len(article_urls)} unique article URLs")
    if not article_urls:
        print("❌ No articles found. Check site structure or categories.")
        return

    # 2. Асинхронная загрузка и очистка текста
    print("📥 Loading & cleaning articles...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    texts = []

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [fetch_with_retry(session, url) for url in article_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, html in zip(article_urls, results):
            if isinstance(html, Exception) or not html:
                print(f"  ⚠️ Skipped {url}")
                continue
            text = await extract_text_from_html(html)
            if len(text) > 100:  # пропускаем пустые/битые страницы
                texts.append({"url": url, "text": text})
            else:
                print(f"  ⚠️ Empty content: {url}")

    print(f"✅ Loaded {len(texts)} valid articles")

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
    print(f"✂️ Generated {len(chunks)} chunks")

    # 4. Запуск MCP-сервера и отправка
    print("💾 Starting LightRAG MCP server & sending chunks...")
    process = await asyncio.create_subprocess_exec(
        *MCP_SERVER_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL  # логи сервера идут в терминал 1, чтобы не мешать
    )

    failed = 0
    try:
        for i, chunk in enumerate(chunks, 1):
            res = await send_to_lightrag_via_mcp(chunk, process)
            if "error" in res:
                failed += 1
                print(f"  [{i}/{len(chunks)}] ⚠️ Error: {res['error'][:60]}")
            else:
                print(f"  [{i}/{len(chunks)}] ✅ OK")
            await asyncio.sleep(0.3)  # щадящий режим для Ollama
    finally:
        print("\n⏳ Отправка завершена. Закрываю stdin для корректного завершения LightRAG...")
        process.stdin.close()  # сигналим серверу, что входных данных больше нет
        print("⏳ Ожидаю завершения фоновых задач (запись в Neo4j/Faiss)...")
        await process.wait()   # ждём естественного выхода процесса
        print("✅ Индексация полностью завершена.")

    if failed:
        print(f"⚠️ {failed} chunks failed. Check Ollama logs for NaN/timeout errors.")
    else:
        print("🎉 All chunks processed successfully!")

async def main():
    print(f"🚀 Starting robust scraper for {BASE_URL}...")
    print("💡 Tip: Run with OLLAMA_KEEP_ALIVE=-1 to prevent model unloading.")
    await load_and_index_filonov_articles()

if __name__ == "__main__":
    asyncio.run(main())
