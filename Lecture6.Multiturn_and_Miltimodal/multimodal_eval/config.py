"""
Централизованная конфигурация для мультимодальной оценки.

Все значения читаются из файла .env, который лежит рядом с этим файлом
(см. .env.example — скопируйте его в .env и подставьте свои значения).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # .env лежит в папке multimodal_eval, рядом с этим файлом
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    # python-dotenv не установлен — читаем переменные окружения как есть
    pass


# --- Прокси для LLM-судьи (OpenAI-совместимый) ---
# Через этот прокси мультимодальные метрики DeepEval обращаются к MLLM-судье
proxy_base_url = os.getenv("PROXY_BASE_URL", "http://5.11.83.110:8000")
proxy_api_key = os.getenv("PROXY_API_KEY", "sk-proxy-your-key")

# --- Модель-судья, через которую считаются метрики (должна поддерживать vision) ---
model = os.getenv("MODEL", "gpt-4o-mini")
