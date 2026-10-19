"""
RAGAS с поддержкой прокси через LangChain.

Настройки прокси (PROXY_BASE_URL / PROXY_API_KEY / MODEL) берутся из .env
через config.py (см. eval_rag/.env.example).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from config import proxy_base_url, proxy_api_key, model as default_model  # noqa: E402


def create_proxy_llm(
    model: str = default_model,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.0
):
    """
    Создать LLM для RAGAS через прокси

    Args:
        model: Название модели
        api_key: API ключ прокси (по умолчанию из .env)
        base_url: URL прокси сервера (по умолчанию из .env)
        temperature: Температура

    Returns:
        LangchainLLMWrapper для RAGAS
    """
    # LangChain OpenAI с кастомным base_url
    llm = ChatOpenAI(
        model=model,
        api_key=api_key or proxy_api_key,
        base_url=base_url or proxy_base_url,  # ← Ваш прокси!
        temperature=temperature
    )

    # Оборачиваем для RAGAS
    return LangchainLLMWrapper(llm)


def create_proxy_embeddings(
    api_key: str = None,
    base_url: str = None
):
    """
    Создать Embeddings для RAGAS через прокси

    Args:
        api_key: API ключ прокси (по умолчанию из .env)
        base_url: URL прокси сервера (по умолчанию из .env)

    Returns:
        LangchainEmbeddingsWrapper для RAGAS
    """
    embeddings = OpenAIEmbeddings(
        api_key=api_key or proxy_api_key,
        base_url=base_url or proxy_base_url  # ← Ваш прокси!
    )

    return LangchainEmbeddingsWrapper(embeddings)


# # ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

# if __name__ == "__main__":
#     from ragas.dataset_schema import SingleTurnSample
#     from ragas.metrics import Faithfulness
#     import asyncio

#     print("🔄 Тестирование RAGAS с прокси...")

#     # Создаем прокси LLM (настройки из .env)
#     proxy_llm = create_proxy_llm()

#     print(f"✅ Прокси LLM создан")

#     # Создаем метрику с прокси
#     metric = Faithfulness(llm=proxy_llm)

#     # Тестовые данные
#     sample = SingleTurnSample(
#         user_input="Какая столица Франции?",
#         response="Париж — столица Франции.",
#         retrieved_contexts=["Париж — столица и крупнейший город Франции."]
#     )

#     # Оцениваем
#     async def test():
#         score = await metric.single_turn_ascore(sample)
#         print(f"\n✅ Faithfulness score: {score:.3f}")

#     asyncio.run(test())
