#!/usr/bin/env python3
"""
HTTP Server for LightRAG with SSE support.
Replaces MCP protocol with REST API.
"""

import asyncio
import json
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import setup_logger, EmbeddingFunc

# HTTP server imports
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse, PlainTextResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn


logger = logging.getLogger(__name__)
setup_logger("lightrag", level="INFO")


class LightRAGHTTPServer:
    def __init__(self):
        self.rag: Optional[LightRAG] = None
        self.initialized = False
        self.ollama_client = None  # Клиент для rerank/embeddings

    async def initialize(self):
        """Initialize LightRAG with all components"""
        if self.initialized:
            return

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

        # Создаём Ollama клиент один раз
        import ollama
        self.ollama_client = ollama.AsyncClient(host=ollama_host)

        # Эмбеддинг функция
        async def _embedding_impl(texts: List[str]) -> np.ndarray:
            embeddings = []
            for text in texts:
                cleaned = text.strip()
                if not cleaned:
                    embeddings.append([0.0] * embedding_dim)
                    continue
                try:
                    response = await self.ollama_client.embeddings(
                        model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                        prompt=cleaned
                    )
                    emb = response['embedding']
                    if any(np.isnan(emb)) or any(np.isinf(emb)):
                        embeddings.append([0.0] * embedding_dim)
                    else:
                        embeddings.append(emb)
                except Exception as e:
                    print(f"⚠️ Embedding error: {e}", file=sys.stderr)
                    embeddings.append([0.0] * embedding_dim)
            return np.array(embeddings, dtype=np.float32)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=_embedding_impl
        )

        # Reranker функция (опционально)
        rerank_func = None
        if reranker_model:
            import ollama
            import re

            async def rerank_model_func(query: str, documents: list, **kwargs) -> list:
                if not documents:
                    return documents
                # Извлекаем текст из документов для ранжирования
                doc_texts = []
                for doc in documents:
                    if hasattr(doc, 'content'):
                        text = doc.content
                    elif hasattr(doc, 'text'):
                        text = doc.text
                    elif isinstance(doc, str):
                        text = doc
                    else:
                        text = str(doc)
                    doc_texts.append(text[:500])  # Ограничиваем длину

                scores = []
                for text in doc_texts:
                    prompt = f"Query: {query}\nDocument: {text}\nRelevance score (0-1):"
                    try:
                        response = await self.ollama_client.generate(
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

                scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

                # Возвращаем отсортированные документы (или только scores, в зависимости от вызова)
                top_n = kwargs.get('top_n')
                if top_n is not None:
                    return [doc for doc, _ in scored_docs[:top_n]]
                else:
                    # Возвращаем все документы в отсортированном порядке
                    return [doc for doc, _ in scored_docs]

            rerank_func = rerank_model_func
        else:
            rerank_func = None

        # === Системный промпт для экстракции сущностей (JSON) ===
        EXTRACTION_PROMPT = (
            "Ты — экстрактор сущностей для графа знаний. Твоя задача — извлекать данные строго в формате JSON.\n"
            "ПРАВИЛА:\n"
            "1. Верни ТОЛЬКО валидный JSON. Никакого markdown, никаких пояснений, никаких тегов <think>.\n"
            "2. Не используй блоки кода (```json). Начинай ответ сразу с { и заканчивай }.\n"
            "3. Сущности: сохраняй оригинальное написание (имена, названия), но приводи к начальной форме (именительный падеж), если это возможно.\n"
            "4. Отношения: описывай глаголами в настоящем времени (например, 'влияет на', 'является частью').\n"
            "5. Если сущностей нет — верни {\"entities\": [], \"relationships\": []}.\n\n"
            "Формат ответа:\n"
            "{\"entities\": [{\"name\": \"Имя\", \"type\": \"Тип\"}], \"relationships\": [{\"source\": \"Имя1\", \"target\": \"Имя2\", \"relation\": \"связь\"}]}"
        )

        # === Системный промпт для генерации ответа ===
        GENERATION_PROMPT = (
            "Ты — опытный научный журналист и редактор медицинской тематики. "
            "Твоя задача — создать краткую, информативную и легко читаемую статью "
            "на русском языке, опираясь на предоставленный контекст. "
            "Тема: Лечебное голодание, его история, методики, польза, риски и современные исследования. "
            "Структура статьи (если применимо): Заголовок, Вступление, Основная часть (1-3 абзаца), Заключение. "
            "Стиль: Нейтральный, научно-популярный, доступный широкому кругу читателей. "
            "Если контекст не содержит информации по теме, честно скажи: "
            "'Информация по этому вопросу в базе данных отсутствует.' "
            "Не выдумывай факты. Избегай повторений. "
            "Если в контексте есть английские термины — используй их русские эквиваленты или поясняй в скобках. "
            "Не упоминай, что ты опираешься на базу знаний — пиши, как будто это твой собственный анализ."
            "\nОтвечай ТОЛЬКО на русском языке."
        )

        # LightRAG инициализация
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            llm_model_kwargs={
                "host": ollama_host,
                "system_prompt": GENERATION_PROMPT,  # Для экстракции
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": 0.6,  # Для стабильности JSON
                    "top_p": 0.95,
                    "top_k": 20,
                    "num_predict": 1024,  # Может быть нужно увеличить, если статья должна быть длиннее
                    #"num_think": 0,
                    "stop": ["</answer>", "<|endoftext|>", "<|end|>", "<|im_end|>"]
                },
                "timeout": timeout
            },
            llm_model_max_async=2,
            embedding_func=embedding_func,
            embedding_func_max_async=4,
            chunk_token_size=1000,
            chunk_overlap_token_size=80,
            entity_extract_max_gleaning=1,
            rerank_model_func=None,
            graph_storage=os.getenv("GRAPH_STORE", "Neo4JStorage"),
            vector_storage=os.getenv("VECTOR_STORE", "FaissVectorDBStorage"),
        )

        await self.rag.initialize_storages()
        self.initialized = True
        print("LightRAG HTTP Server initialized successfully", file=sys.stderr)

    async def query(self, query_text: str, mode: str = "hybrid", only_context: bool = False) -> str:
        """Execute a query"""
        if not self.initialized:
            await self.initialize()

        result = await self.rag.aquery(
            query_text,
            param=QueryParam(
                mode=mode,
                stream=False,
                only_need_context=only_context,
                # Включаем rerank, если функция определена
                enable_rerank=self.rag.rerank_model_func is not None
            )
        )
        return result

    # ... внутри класса LightRAGHTTPServer ...

    async def query_stream(self, query_text: str, mode: str = "hybrid"):
        """Имитируем потоковую передачу через обычный aquery."""
        if not self.initialized:
            await self.initialize()

        # Выполняем обычный запрос
        try:
            result = await self.rag.aquery(
                query_text,
                param=QueryParam(
                    mode=mode,
                    stream=False,  # ❗ Обычный запрос
                    only_need_context=False,
                    enable_rerank=self.rag.rerank_model_func is not None
                )
            )
        except Exception as e:
            logger.error(f"Error in query_stream (sync fallback): {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            return

        # Имитируем поток, отправляя результат целиком как один "чанк"
        yield f"data: {json.dumps({'chunk': result}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def insert(self, text: str) -> str:
        """Insert a document"""
        if not self.initialized:
            await self.initialize()
        await self.rag.ainsert(text)
        return "Document inserted successfully"

    async def insert_batch(self, texts: List[str]) -> str:
        """Insert multiple documents"""
        if not self.initialized:
            await self.initialize()
        await asyncio.gather(*[self.rag.ainsert(t) for t in texts])
        return f"Inserted {len(texts)} documents"

    async def health(self) -> Dict[str, Any]:
        """Health check"""
        stats = {}
        if self.initialized and self.rag:
            try:
                # Пример получения статистики (может отличаться в вашей версии LightRAG)
                # stats = {
                #     "chunks": len(self.rag.chunk_db.get_all()),
                #     "entities": len(self.rag.entity_db.get_all())
                # }
                pass  # Пока без статистики
            except:
                pass

        return {
            "status": "ok",
            "initialized": self.initialized,
            "version": "2.0.0-http",
            "stats": stats if stats else None
        }


# ==================== HTTP Handlers ====================

server = LightRAGHTTPServer()


async def health_handler(request):
    """Health check endpoint"""
    return JSONResponse(await server.health())


async def query_handler(request):
    """Handle query requests"""
    try:
        data = await request.json()
        query_text = data.get("query", "")
        mode = data.get("mode", "hybrid")
        only_context = data.get("only_context", False)

        if not query_text:
            return JSONResponse({"error": "Query text is required"}, status_code=400)

        result = await server.query(query_text, mode, only_context)
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"Error in query_handler: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def query_stream_handler(request):
    """Handle streaming query requests (SSE)"""
    try:
        data = await request.json()
        query_text = data.get("query", "")
        mode = data.get("mode", "hybrid")

        if not query_text:
            return JSONResponse({"error": "Query text is required"}, status_code=400)

        async def event_generator():
            async for chunk in server.query_stream(query_text, mode):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Отключить буферизацию для nginx
            }
        )
    except Exception as e:
        logger.error(f"Error in query_stream_handler: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def insert_handler(request):
    """Handle insert requests"""
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            return JSONResponse({"error": "Text is required"}, status_code=400)

        result = await server.insert(text)
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"Error in insert_handler: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def insert_batch_handler(request):
    """Handle batch insert requests"""
    try:
        data = await request.json()
        texts = data.get("texts", [])

        if not texts or not isinstance(texts, list):
            return JSONResponse({"error": "List of texts is required"}, status_code=400)

        result = await server.insert_batch(texts)
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"Error in insert_batch_handler: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def root_handler(request):
    """Root endpoint with API info"""
    return JSONResponse({
        "name": "LightRAG HTTP Server",
        "version": "2.0.0",
        "description": "RAG server with Neo4j graph storage",
        "endpoints": {
            "health": "/health (GET)",
            "query": "/api/query (POST)",
            "query_stream": "/api/query/stream (POST)",
            "insert": "/api/insert (POST)",
            "insert_batch": "/api/insert/batch (POST)"
        }
    })


# ==================== Application Setup ====================

def create_app():
    # CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Измените на конкретные домены в продакшене
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/", root_handler, methods=["GET"]),
        Route("/health", health_handler, methods=["GET"]),
        Route("/api/query", query_handler, methods=["POST"]),
        Route("/api/query/stream", query_stream_handler, methods=["POST"]),
        Route("/api/insert", insert_handler, methods=["POST"]),
        Route("/api/insert/batch", insert_batch_handler, methods=["POST"]),
        # Mount("/static", StaticFiles(directory="static"), name="static"), # если нужно
    ]

    app = Starlette(
        debug=False,
        middleware=middleware,
        routes=routes,
    )
    return app


app = create_app()


# ==================== Main Entry Point ====================

def main():
    """Start HTTP server"""
    host = os.getenv("HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("HTTP_PORT", "8000"))

    print(f"Starting LightRAG HTTP Server on http://{host}:{port}", file=sys.stderr)
    print(f"API endpoints available at http://{host}:{port}/api/", file=sys.stderr)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
