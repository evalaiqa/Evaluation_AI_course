"""
RAGAS Evaluator с поддержкой прокси (упрощенная версия).

По умолчанию работает через прокси; настройки берутся из .env
через config.py (см. eval_rag/.env.example).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional  # noqa: E402
import pandas as pd  # noqa: E402
import asyncio  # noqa: E402

from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.metrics import (  # noqa: E402
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    Faithfulness
)
from ragas_proxy import create_proxy_llm, create_proxy_embeddings  # noqa: E402
from config import model as default_model  # noqa: E402


class RagasEvaluator:
    """RAGAS Evaluator с поддержкой прокси"""

    def __init__(
        self,
        model: str = default_model,
        temperature: float = 0.0,
        use_proxy: bool = True,
        proxy_api_key: Optional[str] = None,
        proxy_base_url: Optional[str] = None
    ):
        """
        Инициализация

        Args:
            model: Название модели
            temperature: Температура
            use_proxy: Использовать прокси (по умолчанию True)
            proxy_api_key: API ключ прокси (по умолчанию из .env)
            proxy_base_url: URL прокси (по умолчанию из .env)
        """
        self.model = model
        self.temperature = temperature
        self.metric_configs = {}

        # Создаем LLM и embeddings
        if use_proxy:
            # Настройки прокси берутся из .env (через config.py), если не переданы явно
            self.evaluator_llm = create_proxy_llm(
                model=model,
                api_key=proxy_api_key,
                base_url=proxy_base_url,
                temperature=temperature
            )
            self.evaluator_embeddings = create_proxy_embeddings(
                api_key=proxy_api_key,
                base_url=proxy_base_url
            )
            print(f"🌐 Используется прокси для RAGAS")
        else:
            # Обычный OpenAI
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = ChatOpenAI(model=model, temperature=temperature)
            self.evaluator_llm = LangchainLLMWrapper(llm)

            embeddings = OpenAIEmbeddings()
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

        print(f"✅ RAGAS настроен: {model}")

    def configure_metrics(self, metrics_config: Dict[str, Dict[str, Any]]):
        """Настройка метрик"""
        self.metric_configs = metrics_config

    def _create_metric(self, metric_name: str):
        """Создать метрику"""
        if metric_name == 'context_precision':
            return LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        elif metric_name == 'context_recall':
            return LLMContextRecall(llm=self.evaluator_llm)
        elif metric_name == 'response_relevancy':
            return ResponseRelevancy(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
        elif metric_name == 'faithfulness':
            return Faithfulness(llm=self.evaluator_llm)

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Оценить один случай

        Args:
            question: Вопрос пользователя
            answer: Ответ RAG системы
            contexts: Список извлеченных контекстов
            ground_truth: Правильный ответ

        Returns:
            Словарь с оценками метрик
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
            retrieved_contexts=contexts
        )

        results = {}

        for metric_name, config in self.metric_configs.items():
            if not config.get('enabled', True):
                continue

            print(f"   Оценка: {metric_name}")
            metric = self._create_metric(metric_name)
            score = await metric.single_turn_ascore(sample)
            results[metric_name] = float(score)

        return results

    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Пакетная оценка

        Args:
            test_cases: Список словарей с ключами:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: str

        Returns:
            DataFrame с результатами
        """
        async def evaluate_all():
            all_results = []
            for i, case in enumerate(test_cases, 1):
                print(f"\n📝 {i}/{len(test_cases)}: {case['question'][:50]}...")

                scores = await self.evaluate_single(
                    question=case['question'],
                    answer=case['answer'],
                    contexts=case['contexts'],
                    ground_truth=case['ground_truth']
                )

                scores['question'] = case['question']
                all_results.append(scores)

            return all_results

        results = asyncio.run(evaluate_all())
        return pd.DataFrame(results)


# # ==================== ПРИМЕР ====================

# if __name__ == "__main__":

#     # Просто список словарей!
#     test_cases = [
#         {
#             'question': "Какая столица Франции?",
#             'answer': "Париж — столица Франции.",
#             'contexts': ["Париж — столица и крупнейший город Франции."],
#             'ground_truth': "Париж"
#         },
#         {
#             'question': "Кто изобрёл телефон?",
#             'answer': "Александр Белл изобрёл телефон в 1876 году.",
#             'contexts': [
#                 "Александр Грэм Белл — изобретатель, которому приписывают изобретение телефона.",
#                 "Первый успешный телефонный звонок состоялся 10 марта 1876 года."
#             ],
#             'ground_truth': "Александр Грэм Белл"
#         }
#     ]

#     # Пример: через прокси (по умолчанию, настройки из .env)
#     print("="*60)
#     print("📊 Пример: RAGAS через прокси")

#     evaluator = RagasEvaluator()

#     evaluator.configure_metrics({
#         'faithfulness': {'enabled': True, 'threshold': 0.7},
#         'response_relevancy': {'enabled': True, 'threshold': 0.7}
#     })

#     df = evaluator.evaluate_batch(test_cases)
#     print("\n✅ Результаты:")
#     print(df)
