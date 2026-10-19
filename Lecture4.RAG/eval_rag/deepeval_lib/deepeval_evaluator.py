import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Union  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
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
        'contextual_relevancy': ContextualRelevancyMetric
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
