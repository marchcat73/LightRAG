#!/usr/bin/env python3
import asyncio
import codecs
import io
import json
import os
import sys
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import setup_logger, EmbeddingFunc
import numpy as np
import logging
logger = logging.getLogger(__name__)

setup_logger("lightrag", level="INFO")

class LightRAGMCPServer:
    def __init__(self):
        self.rag = None
        self.initialized = False

    async def initialize(self):
        working_dir = os.getenv("DATA_DIR", "./rag_storage")
        os.makedirs(working_dir, exist_ok=True)

        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        timeout = int(os.getenv("LLM_TIMEOUT", "600"))
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        llm_model = os.getenv("LLM_MODEL", "qwen3-coder:latest")
        reranker_model = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-4B:Q5_K_M")

        print(f"Initializing LightRAG: ctx={num_ctx}, timeout={timeout}s, embed_dim={embedding_dim}",
              file=sys.stderr, flush=True)

        # Ваша кастомная функция (без изменений)
        async def _embedding_impl(texts: list[str]) -> np.ndarray:
            """Асинхронная обертка для синхронного Ollama embeddings"""
            import ollama
            import asyncio
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor

            def sync_embed():
                embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
                embeddings = []

                for text in texts:
                    cleaned = text.strip()
                    if not cleaned:
                        embeddings.append([0.0] * embedding_dim)
                        continue
                    try:
                        response = ollama.embeddings(
                            model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                            prompt=cleaned
                        )
                        emb = response['embedding']
                        if any(np.isnan(emb)) or any(np.isinf(emb)):
                            embeddings.append([0.0] * embedding_dim)
                        else:
                            embeddings.append(emb)
                    except Exception as e:
                        if "nan" in str(e).lower() or "500" in str(e):
                            embeddings.append([0.0] * embedding_dim)
                        else:
                            raise
                return np.array(embeddings, dtype=np.float32)

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=4) as executor:
                return await loop.run_in_executor(executor, sync_embed)


        embedding_func = EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=_embedding_impl
        )


        async def rerank_model_func(query: str, documents: list, top_n: int = None) -> list:
            """
            Rerank documents and return reranked document objects (not strings!)

            LightRAG передает список объектов Document/Chunk, а не строки.
            Нужно вернуть ТЕ ЖЕ САМЫЕ объекты в новом порядке или отфильтрованные.
            """
            import ollama
            import re

            if not documents:
                return documents

            ollama_host = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            client = ollama.Client(host=ollama_host)
            reranker_model = os.getenv("RERANKER_MODEL", "dengcao/Qwen3-Reranker-4B:Q5_K_M")

            # Извлекаем текст из документов для ранжирования
            doc_texts = []
            for doc in documents:
                # Пытаемся получить текст из документа
                if hasattr(doc, 'content'):
                    text = doc.content
                elif hasattr(doc, 'text'):
                    text = doc.text
                elif isinstance(doc, str):
                    text = doc
                else:
                    text = str(doc)
                doc_texts.append(text[:500])  # Ограничиваем длину

            # Вычисляем scores
            scores = []
            for text in doc_texts:
                prompt = f"Query: {query}\nDocument: {text}\nRelevance score (0-1):"
                try:
                    response = client.generate(
                        model=reranker_model,
                        prompt=prompt,
                        options={"temperature": 0, "num_predict": 10}
                    )
                    score_text = response['response'].strip()
                    match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    score = float(match.group(1)) if match else 0.5
                    score = max(0.0, min(1.0, score))
                except Exception as e:
                    print(f"⚠️ Reranker error: {e}", file=sys.stderr)
                    score = 0.5
                scores.append(score)

            # Сортируем исходные документы по score
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

            if top_n:
                # Возвращаем топ-N документов (сохраняя их исходный тип!)
                return [doc for doc, _ in scored_docs[:top_n]]
            else:
                # Возвращаем все документы в отсортированном порядке
                return [doc for doc, _ in scored_docs]

        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            llm_model_kwargs={
                "host": ollama_host,
                "system_prompt": (
                    "Ты — эксперт по извлечению сущностей из текстов на русском языке.\n"
                    "КРИТИЧЕСКИ ВАЖНО: Сохраняй пробелы между словами в названиях сущностей."
                    "Пиши 'сухое голодание', а не 'сухоеголодание'. "
                    "Пиши 'время года', а не 'времягода'. "
                    "Не используй слитное написание, camelCase, snake_case или нижние подчёркивания. "
                    "Соблюдай естественное русское написание с пробелами. "
                    "ПРАВИЛА:\n"
                    "1. Верни ТОЛЬКО валидный JSON. Никакого markdown, никаких пояснений, никаких тегов <think>.\n"
                    "2. Не используй блоки кода (```json). Начинай ответ сразу с { и заканчивай }.\n"
                    "3. Сущности: сохраняй оригинальное написание (имена, названия), но приводи к начальной форме (именительный падеж), если это возможно.\n"
                    "4. Отношения: описывай глаголами в настоящем времени (например, 'влияет на', 'является частью').\n"
                    "5. Если сущностей нет — верни {\"entities\": [], \"relationships\": []}.\n\n"
                    "Формат ответа:\n"
                    "{\"entities\": [{\"name\": \"Имя\", \"type\": \"Тип\"}], \"relationships\": [{\"source\": \"Имя1\", \"target\": \"Имя2\", \"relation\": \"связь\"}]}"
                ),
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 20,
                    "num_predict": 1024,
                    "num_think": 0,
                    # ✅ Только маркеры конца генерации (EOS). Без \n и без ```!
                    "stop": ["</answer>", "<|endoftext|>", "<|end|>", "<|im_end|>"]
                },
                "timeout": timeout
            },
            llm_model_max_async=1,
            embedding_func=embedding_func,
            embedding_func_max_async=2,
            chunk_token_size=1000,
            chunk_overlap_token_size=80,
            entity_extract_max_gleaning=1,
            rerank_model_func=None,
            graph_storage=os.getenv("GRAPH_STORE", "Neo4JStorage"),
            vector_storage=os.getenv("VECTOR_STORE", "FaissVectorDBStorage"),
        )

        await self.rag.initialize_storages()
        self.initialized = True
        print("LightRAG MCP Server initialized successfully", file=sys.stderr)

    async def process_message(self, message: dict) -> dict:
        logger.info(f"Processing {message.get('method')} request")
        if not self.initialized:
            await self.initialize()

        method = message.get("method")
        params = message.get("params", {})

        try:
            if method == "query":
                result = await self.rag.aquery(
                    params.get("query", ""),
                    param=QueryParam(
                        mode=params.get("mode", "hybrid"),
                        stream=False,
                        only_need_context=params.get("only_context", False),
                        enable_rerank=True if self.rag.rerank_model_func else False
                    )
                )
                if isinstance(result, str):
                    result = codecs.decode(result, 'unicode_escape')
                return {"result": result}

            elif method == "query_stream":
                query = params.get("query", "")
                async for chunk in self.rag.aquery_stream(
                    query,
                    param=QueryParam(mode=params.get("mode", "hybrid"), stream=True)
                ):
                    print(json.dumps({"chunk": chunk}), flush=True)
                return {"result": "Stream completed"}

            elif method == "insert":
                text = params.get("text", "")
                if not text:
                    return {"error": "No text provided"}
                await self.rag.ainsert(text)
                return {"result": "Document inserted successfully"}

            elif method == "insert_batch":
                texts = params.get("texts", [])
                results = await asyncio.gather(*[self.rag.ainsert(t) for t in texts])
                return {"result": f"Inserted {len(results)} documents"}

            elif method == "health":
                return {
                    "status": "ok",
                    "initialized": self.initialized,
                    "stats": {
                        "chunks": len(self.rag.chunk_db.get_all()),
                        "entities": len(self.rag.entity_db.get_all())
                    } if self.initialized else None
                }

            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            return {"error": str(e)}


def setup_utf8_output():
    """Настраивает stdout для корректного вывода UTF-8"""
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)


async def main():
    server = LightRAGMCPServer()

    # ⚠️ ВАЖНО: Не меняем sys.stdout, а работаем через buffer напрямую
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue

            message = json.loads(line)
            response = await server.process_message(message)

            # ✅ Правильный способ вывода UTF-8 в JSON
            json_str = json.dumps(response, ensure_ascii=False, separators=(',', ':'))

            # ✅ Пишем байты напрямую
            sys.stdout.buffer.write(json_str.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()

        except json.JSONDecodeError as e:
            err = json.dumps({"error": f"Invalid JSON: {e}"}, ensure_ascii=False)
            sys.stdout.buffer.write(err.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()
        except Exception as e:
            err = json.dumps({"error": f"Unexpected error: {e}"}, ensure_ascii=False)
            sys.stdout.buffer.write(err.encode('utf-8'))
            sys.stdout.buffer.write(b'\n')
            sys.stdout.buffer.flush()

if __name__ == "__main__":
    asyncio.run(main())
