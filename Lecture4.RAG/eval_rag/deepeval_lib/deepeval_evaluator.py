import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Union  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval import evaluate  # noqa: E402
from deepeval_custom_llm import create_proxy_model, ProxyLLM  # noqa: E402


def create_metrics(
    model: Union[str, ProxyLLM],
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    if metrics_list is None:
        metrics_list = ['answer_relevancy', 'faithfulness']

    METRICS = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_relevancy': ContextualRelevancyMetric,
        'contextual_recall': ContextualRecallMetric,
        'contextual_precision': ContextualPrecisionMetric
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            metric = METRICS[metric_name](
                threshold=threshold,
                model=model,
                include_reason=True,
                async_mode=False
            )
            metrics.append(metric)

    return metrics


if __name__ == "__main__":

    # ============= ПРИМЕР 1: Через кастомную прокси модель =============
    print("📊 Пример 1: Метрики через прокси модель")

    # Прокси-модель как судья (настройки берутся из .env)
    proxy_model = create_proxy_model()

    # Создаем метрики с прокси моделью
    metrics = create_metrics(
        model=proxy_model,
        metrics_list=['answer_relevancy', 'faithfulness'],
        threshold=0.7
    )

    # Создаем тест кейс
    test_case = LLMTestCase(
        input="Что делать, если эти туфли мне не подойдут?",
        actual_output="Мы предлагаем полный возврат средств в течение 30 дней без дополнительной платы.",
        expected_output="Вы можете вернуть товар в течение 30 дней с полным возвратом средств без дополнительной платы.",
        retrieval_context=[
            "Все клиенты могут вернуть товар в течение 30 дней с полным возвратом средств без дополнительной платы."]
    )

    # Оцениваем (DeepEval сам выведет результаты)
    evaluate(test_cases=[test_case], metrics=metrics)

    # ============= ПРИМЕР 2: Без wrapper функции =============
    print("\n" + "="*60)
    print("📊 Пример 2: Прямое использование DeepEval (без обертки)")

    # Прямо создаем метрику с прокси моделью
    metric = AnswerRelevancyMetric(
        model=proxy_model,
        threshold=0.7,
        include_reason=True,
        async_mode=False
    )

    # Измеряем
    metric.measure(test_case)

    # Смотрим результаты
    print(f"\nScore: {metric.score:.3f}")
    print(f"Reason: {metric.reason}")
