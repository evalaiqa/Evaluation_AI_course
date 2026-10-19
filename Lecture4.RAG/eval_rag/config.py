"""
Централизованная конфигурация для оценки RAG.

Все значения читаются из файла .env, который лежит рядом с этим файлом
(см. .env.example — скопируйте его в .env и подставьте свои значения).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # .env лежит в папке eval_rag, рядом с этим файлом
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    # python-dotenv не установлен — читаем переменные окружения как есть
    pass


# --- RAG-система, которую оцениваем ---
# endpoint чат-API и X-API-Key для доступа к нему
endpoint_url = os.getenv("RAG_ENDPOINT_URL", "http://5.11.83.110:8002/api/v1/chat/")
api_key = os.getenv("RAG_API_KEY", "rag-api-key")

# --- Прокси для LLM-судьи и эмбеддингов (OpenAI-совместимый) ---
# Через этот прокси метрики (DeepEval, RAGAS, Opik, promptfoo) обращаются к LLM
proxy_base_url = os.getenv("PROXY_BASE_URL", "http://5.11.83.110:8000")
proxy_api_key = os.getenv("PROXY_API_KEY", "sk-proxy-your-key")

# --- Модель-судья, через которую считаются метрики ---
model = os.getenv("MODEL", "gpt-4o-mini")
