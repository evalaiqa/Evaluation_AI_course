"""
Оценка RAG системы с использованием RAGAS метрик.
Загружает датасет из Excel, получает ответы от RAG и запускает оценку.

Судья (LLM/эмбеддинги для метрик) работает через прокси; адрес RAG и прокси
берутся из .env через config.py (см. eval_rag/.env.example).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time  # noqa: E402
from typing import Dict, Any  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from ragas_evaluator import RagasEvaluator  # noqa: E402
from config import endpoint_url, api_key, model as default_model  # noqa: E402


def evaluate_rag_with_ragas(
    excel_path: str,
    rag_connector: RAGConnector,
    metrics_config: Dict[str, Dict[str, Any]],
    model: str = default_model,
    use_proxy: bool = True,
    proxy_api_key: str = None,
    proxy_base_url: str = None,
    sleep_time: float = 0.1
):

    # ============= ШАГ 1: Парсим Excel =============
    print(f"\n Шаг 1: Парсинг Excel файла...")

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Ошибка загрузки датасета")
        return

    # Превью
    parser.preview_dataset(df, n=2)

    # ============= ШАГ 2: Получаем вопросы и expected ответы =============
    print(f"\n Шаг 2: Извлечение вопросов из датасета...")

    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"Извлечено {len(questions)} вопросов")

    # ============= ШАГ 3: Задаем вопросы в RAG =============
    print(f"\n Шаг 3: Получение ответов от RAG системы...")

    test_cases = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        # Запрос к RAG
        rag_response = rag_connector.query(question)

        # Проверяем на ошибки
        if 'error' in rag_response:
            print(f"      ⚠️  Ошибка RAG: {rag_response['error']}")
            continue

        # Извлекаем данные
        answer = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        contexts = [s.get('content', '') for s in sources if s.get('content')]

        print(f"      ✅ Ответ: {answer[:60]}...")
        print(f"      📚 Контекстов: {len(contexts)}")

        # Получаем ground_truth
        ground_truth = expected_responses[i-1] if i - \
            1 < len(expected_responses) else ""

        # Создаем тест кейс для RAGAS
        test_case = {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth
        }

        test_cases.append(test_case)

        # Пауза между запросами
        time.sleep(sleep_time)

    print(f"\n✅ Создано {len(test_cases)} тест кейсов")

    # ============= ШАГ 4: Инициализируем RAGAS Evaluator =============
    print(f"\n📊 Шаг 4: Инициализация RAGAS Evaluator...")

    evaluator = RagasEvaluator(
        model=model,
        use_proxy=use_proxy,
        proxy_api_key=proxy_api_key,
        proxy_base_url=proxy_base_url
    )

    # Настраиваем метрики
    evaluator.configure_metrics(metrics_config)

    print(f"✅ Настроены метрики:")
    for metric_name, config in metrics_config.items():
        if config.get('enabled', True):
            print(f"   - {metric_name}")

    # ============= ШАГ 5: Запускаем оценку =============
    print(f"\n Шаг 5: Запуск оценки RAGAS...")

    try:
        # Запускаем пакетную оценку
        results_df = evaluator.evaluate_batch(test_cases)

        print(f"\n✅ Оценка завершена!")

        # ============= ШАГ 6: Показываем результаты =============

        # Показываем статистику по метрикам
        print("\n Средние значения метрик:")
        for metric_name in metrics_config.keys():
            if metric_name in results_df.columns:
                avg_score = results_df[metric_name].mean()
                print(f"   {metric_name}: {avg_score:.3f}")

        # Показываем детальные результаты
        print("\n Детальные результаты:")
        print(results_df.to_string(index=False))

        # Сохраняем результаты в файл
        output_path = "ragas_evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Результаты сохранены в: {output_path}")

        return results_df

    except Exception as e:
        print(f"❌ Ошибка при оценке: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    # ============= НАСТРОЙКА =============

    # 1. Инициализируем RAG коннектор (адрес и ключ из .env)
    rag_connector = RAGConnector(
        endpoint_url=endpoint_url,
        api_key=api_key,
        timeout=30
    )

    # 2. Указываем путь к Excel файлу
    excel_path = "data/evaluation_dataset.xlsx"

    # 3. Конфигурация метрик RAGAS
    metrics_config = {
        'faithfulness': {
            'enabled': True,
            'threshold': 0.7
        },
        'response_relevancy': {
            'enabled': True,
            'threshold': 0.7
        },
        'context_precision': {
            'enabled': True,
            'threshold': 0.7
        }
    }

    # 4. Прокси включён по умолчанию; настройки берутся из .env
    use_proxy = True  # Измените на False для обычной OpenAI

    # ============= ЗАПУСК ОЦЕНКИ =============

    results = evaluate_rag_with_ragas(
        excel_path=excel_path,
        rag_connector=rag_connector,
        metrics_config=metrics_config,
        use_proxy=use_proxy,
        sleep_time=0.1
    )

    if results is not None:
        print("\n🎉 Оценка успешно завершена!")
        print(f"📊 Обработано {len(results)} вопросов")
