"""
Оценка RAG системы с использованием Opik метрик.
Использует готовый датасет из Opik UI и запускает оценку.

Судья (LLM для метрик) работает через прокси; адрес RAG и прокси берутся
из .env через config.py (см. eval_rag/.env.example).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time  # noqa: E402
from typing import List, Optional, Union  # noqa: E402
import opik  # noqa: E402
from opik.evaluation import evaluate  # noqa: E402
from opik import Opik  # noqa: E402
from opik_evaluator import create_metrics, simple_rag_task  # noqa: E402
from opik_custom_llm import create_opik_proxy_model, OpikProxyLLM  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from config import endpoint_url, api_key  # noqa: E402


def evaluate_rag_with_opik(
    dataset_name: str,
    rag_connector: RAGConnector,
    metrics_list: List[str],
    experiment_name: str = "v_2_RAG_Evaluation",
    project_name: str = "Test evaluation",
    model: Optional[Union[str, OpikProxyLLM]] = None,
    sleep_time: float = 0.1
):
    # По умолчанию судья — прокси модель (настройки из .env)
    if model is None:
        model = create_opik_proxy_model()

    opik.configure(use_local=False)
    client = Opik()

    try:
        source_dataset = client.get_dataset(name=dataset_name)
        source_items = list(source_dataset.get_items())
    except Exception as e:
        print(f"❌ Ошибка загрузки датасета: {e}")
        return

    rag_results = []

    for i, item in enumerate(source_items, 1):
        question = item.get("input") or item.get("question", "")
        expected_output = item.get("expected_output", "")

        print(f"  [{i}/{len(source_items)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            continue

        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        rag_results.append({
            "input": question,
            "output": actual_output,
            "context": retrieval_context,
            "expected_output": expected_output
        })

        time.sleep(sleep_time)

    new_dataset_name = f"{dataset_name}_results_{int(time.time())}"

    try:
        new_dataset = client.create_dataset(
            name=new_dataset_name,
            description=f"Результаты оценки RAG для датасета '{dataset_name}'"
        )
        new_dataset.insert(rag_results)
    except Exception as e:
        print(f"❌ Ошибка создания датасета: {e}")
        return

    metrics = create_metrics(
        model=model,
        metrics_list=metrics_list
    )

    try:
        result = evaluate(
            dataset=new_dataset,
            task=simple_rag_task,
            scoring_metrics=metrics,
            experiment_name=experiment_name,
            project_name=project_name
        )

    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")


if __name__ == "__main__":

    # Инициализируем RAG коннектор (адрес и ключ из .env)
    rag_connector = RAGConnector(
        endpoint_url=endpoint_url,
        api_key=api_key,
        timeout=30
    )

    dataset_name = "evaluation_dataset_happy_path_2026-08-21"
    experiment_name = "RAG_Evaluation_From_Dataset_v8"
    project_name = "Test evaluation"

    metrics_to_use = [
        'answer_relevance',
        'hallucination'
    ]

    # Судья — прокси модель (настройки из .env)
    proxy_model = create_opik_proxy_model()

    evaluate_rag_with_opik(
        dataset_name=dataset_name,
        rag_connector=rag_connector,
        metrics_list=metrics_to_use,
        experiment_name=experiment_name,
        project_name=project_name,
        model=proxy_model
    )
