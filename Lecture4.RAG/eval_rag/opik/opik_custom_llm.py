"""
Кастомная модель Opik для работы через прокси сервер.
Аналог DeepEval ProxyLLM, но для Opik.

Настройки прокси (PROXY_BASE_URL / PROXY_API_KEY / MODEL) берутся из .env
через config.py (см. eval_rag/.env.example).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, List, Dict, Any, Type  # noqa: E402
import pydantic  # noqa: E402
from openai import OpenAI  # noqa: E402
from opik.evaluation.models.base_model import OpikBaseModel  # noqa: E402
from config import proxy_base_url, proxy_api_key, model as default_model  # noqa: E402


class OpikProxyLLM(OpikBaseModel):
    """
    Кастомная модель для Opik, которая работает через прокси сервер.
    Наследует OpikBaseModel — совместима с метриками Opik.
    """

    def __init__(
        self,
        model: str = default_model,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Инициализация прокси модели

        Args:
            model: Название модели (gpt-4o-mini, gpt-3.5-turbo и т.д.)
            api_key: API ключ прокси (по умолчанию из .env)
            base_url: URL прокси сервера (по умолчанию из .env)
            temperature: Температура генерации
        """
        super().__init__(model_name=model)
        self.temperature = temperature

        # Берём настройки из аргументов или из .env (через config.py)
        self.api_key = api_key or proxy_api_key
        self.base_url = base_url or proxy_base_url

        # Создаем клиент OpenAI с прокси настройками
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_string(
        self,
        input: str,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Генерация строки через прокси (основной метод для метрик Opik)
        """
        messages = [{"role": "user", "content": input}]
        return self._chat(messages, response_format=response_format, **kwargs)

    def generate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Генерация ответа ассистента из списка сообщений с ролями.
        Этот метод вызывают метрики answer_relevance / hallucination и др.
        Возвращает dict в формате {"role": "assistant", "content": <str>}.
        """
        content = self._chat(messages, response_format=response_format, **kwargs)
        return {"role": "assistant", "content": content}

    async def agenerate_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Клиент OpenAI синхронный — для простоты переиспользуем sync-версию.
        return self.generate_chat_completion(
            messages, response_format=response_format, **kwargs
        )

    async def agenerate_string(
        self,
        input: str,
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        return self.generate_string(input, response_format=response_format, **kwargs)

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[pydantic.BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Общий вызов chat completions. Если задан response_format (pydantic-схема) —
        используем structured output, чтобы получить валидный JSON (его потом
        парсят метрики Opik). Иначе — обычная генерация.
        """
        try:
            if response_format is not None:
                try:
                    # Нативный structured output OpenAI
                    response = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        response_format=response_format,
                    )
                    return response.choices[0].message.content
                except Exception:
                    # Фолбэк для прокси без parse: JSON-mode
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Ошибка при генерации через прокси: {e}")
            raise

    def generate_provider_response(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        """
        Генерация ответа через прокси в формате OpenAI response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return response

        except Exception as e:
            print(f"❌ Ошибка при генерации через прокси: {e}")
            raise

    def __call__(self, prompt: str) -> str:
        """
        Прямой вызов модели (для простоты использования)
        """
        return self.generate_string(prompt)


def create_opik_proxy_model(
    model: str = default_model,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> OpikProxyLLM:
    """
    Helper функция для создания прокси модели для Opik

    Args:
        model: Название модели
        api_key: API ключ прокси (ваш Bearer токен)
        base_url: URL прокси сервера
        temperature: Температура генерации

    Returns:
        Экземпляр OpikProxyLLM
    """
    return OpikProxyLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )


# Тестирование модели при запуске файла напрямую
if __name__ == "__main__":
    print("🔄 Тестирование OpikProxyLLM...")

    # Создаем модель (настройки берутся из .env)
    proxy_model = create_opik_proxy_model()

    # Тестовый промпт
    test_prompt = "Объясни одним предложением, что такое RAG."

    print(f"\n📝 Промпт: {test_prompt}")
    print(f"🤖 Модель: {proxy_model.model_name}")
    print(f"🌐 Прокси: {proxy_model.base_url}")
    print(f"🔑 API Key: {proxy_model.api_key[:20]}...")

    # Генерируем ответ
    try:
        response = proxy_model(test_prompt)
        print(f"\n✅ Ответ: {response}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
