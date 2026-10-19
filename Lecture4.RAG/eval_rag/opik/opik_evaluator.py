# opik_evaluator.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Union  # noqa: E402
from opik.evaluation.metrics import (  # noqa: E402
    AnswerRelevance,
    Hallucination,
    ContextPrecision,
    ContextRecall
)
from opik_custom_llm import OpikProxyLLM  # noqa: E402


def create_metrics(
    model: Union[str, OpikProxyLLM],
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    """
    Создать метрики Opik

    Args:
        model: Название модели ИЛИ экземпляр OpikProxyLLM (прокси)
        metrics_list: ['answer_relevance', 'hallucination', 'context_precision', 'context_recall']
        threshold: Порог для метрик

    Returns:
        Список метрик Opik
    """
    if metrics_list is None:
        metrics_list = ['answer_relevance', 'hallucination']

    METRICS = {
        'answer_relevance': AnswerRelevance,
        'hallucination': Hallucination,
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            kwargs = {"name": metric_name}
            if model is not None:
                kwargs["model"] = model
            metric = METRICS[metric_name](**kwargs)
            metrics.append(metric)

    return metrics


def simple_rag_task(item: Dict) -> Dict:
    """
    Простая task функция для эксперимента.
    В реальности здесь был бы вызов вашей RAG-системы.
    """
    return {
        "input": item["input"],
        "output": item["output"],
        "context": item.get("context", []),
        "expected_output": item.get("expected_output", "")
    }
