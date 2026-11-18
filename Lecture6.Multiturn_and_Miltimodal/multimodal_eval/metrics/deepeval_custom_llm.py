"""
Прокси-модель DeepEval для мультимодальных метрик (DeepEval 4.x).

В 4.x отдельный DeepEvalBaseMLLM удалён, а нативный OpenAIModel уже умеет
мультимодальность и принимает base_url / api_key. Поэтому прокси-доступ к
OpenAI (например, для российской аудитории) делается просто: возвращаем
OpenAIModel, настроенный на прокси-сервер. Кастомный ProxyMLLM больше не нужен.

Настройки прокси (PROXY_BASE_URL / PROXY_API_KEY / MODEL) берутся из .env
через config.py (см. multimodal_eval/.env.example).
"""
import sys
from pathlib import Path

# Корень multimodal_eval на sys.path, чтобы импортировать config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional  # noqa: E402
from deepeval.models import OpenAIModel  # noqa: E402
from config import proxy_base_url, proxy_api_key, model as default_model  # noqa: E402


def create_proxy_model(
    model: str = default_model,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
) -> OpenAIModel:
    """
    Создаёт нативный мультимодальный OpenAIModel, настроенный на прокси.

    Args:
        model:       Название модели (должна поддерживать vision, напр. gpt-4o-mini)
        api_key:     Ключ прокси (по умолчанию из .env через config.proxy_api_key)
        base_url:    Адрес прокси (по умолчанию из .env через config.proxy_base_url)
        temperature: Температура генерации судьи

    Returns:
        OpenAIModel, готовый к передаче в мультимодальные метрики DeepEval.
    """
    return OpenAIModel(
        model=model,
        api_key=api_key or proxy_api_key,
        base_url=base_url or proxy_base_url,
        temperature=temperature,
    )


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("🔄 Тестирование прокси-модели (OpenAIModel через прокси)...")

    proxy_model = create_proxy_model()

    print(f"🤖 Модель:  {proxy_model.get_model_name()}")
    print(f"🌐 Прокси:  {proxy_model.base_url}")
    print(f"🖼  Vision:  {proxy_model.supports_multimodal()}")

    try:
        response = proxy_model.generate("Explain what is RAG in one sentence.")
        # generate возвращает кортеж (text, cost) для нативных моделей
        text = response[0] if isinstance(response, tuple) else response
        print(f"\n✅ Ответ: {text}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
