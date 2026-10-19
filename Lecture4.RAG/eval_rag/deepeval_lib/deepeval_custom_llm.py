"""
Кастомная модель DeepEval для работы через прокси сервер.
Следует официальной документации DeepEval.

Настройки прокси (PROXY_BASE_URL / PROXY_API_KEY / MODEL) берутся из .env
через config.py (см. eval_rag/.env.example).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional  # noqa: E402
from openai import OpenAI  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from config import proxy_base_url, proxy_api_key, model as default_model  # noqa: E402


class ProxyLLM(DeepEvalBaseLLM):
    """
    Кастомная модель для DeepEval, которая работает через прокси сервер
    """

    def __init__(
        self,
        model: str = default_model,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        # Сохраняем название модели
        self._model_name = model
        self.temperature = temperature

        # Берём настройки из аргументов или из .env (через config.py)
        self.api_key = api_key or proxy_api_key
        self.base_url = base_url or proxy_base_url

        # Создаем клиент OpenAI с прокси настройками
        # ВАЖНО: прокси использует Bearer токен в Authorization
        self.client = OpenAI(
            # Это попадет в Authorization: Bearer {api_key}
            api_key=self.api_key,
            base_url=self.base_url
        )

        # НЕ вызываем super().__init__() - вместо этого устанавливаем self.model вручную
        self.model = self.client

    def load_model(self):
        """
        Загрузка модели (требуется DeepEval)
        Возвращает клиент OpenAI
        """
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Ошибка при генерации через прокси: {e}")
            raise

    async def a_generate(self, prompt: str) -> str:
        # DeepEval может использовать async для параллельной оценки
        # Для простоты используем синхронную версию
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """
        Получить название модели (требуется DeepEval)
        """
        return self._model_name


def create_proxy_model(
    model: str = default_model,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> ProxyLLM:

    return ProxyLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
