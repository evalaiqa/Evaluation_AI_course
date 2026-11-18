"""
DeepEval мультимодальные метрики для оценки изображений (DeepEval 4.x API).

Метрики на основе MLLM-судьи (по умолчанию GPT-4o / gpt-4o-mini):
- ImageCoherence:   насколько изображения согласуются с окружающим текстом
- ImageHelpfulness: насколько изображения помогают понять текст
- ImageReference:   корректно ли текст ссылается на изображения
- TextToImage:      качество изображения, сгенерированного по prompt

Все метрики возвращают score в диапазоне 0-1 (чем выше, тем лучше).

Все метрики ходят к MLLM-судье через прокси (create_proxy_model):
в 4.x нативный OpenAIModel умеет vision и принимает base_url / api_key,
поэтому кастомный ProxyMLLM больше не нужен. Настройки прокси
(PROXY_BASE_URL / PROXY_API_KEY / MODEL) берутся из .env через config.py
(см. multimodal_eval/.env.example).
"""

from pathlib import Path
from typing import List, Union

# DeepEval imports (4.x)
from deepeval.test_case import LLMTestCase, MLLMImage
from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
    TextToImageMetric,
)

# Прокси-модель (поддержка запуска и как модуля, и напрямую)
try:
    from .deepeval_custom_llm import create_proxy_model
except ImportError:
    from deepeval_custom_llm import create_proxy_model


def _is_image_source(item) -> bool:
    """True, если `item` указывает на изображение (локальный файл или http(s)-URL)."""
    if isinstance(item, MLLMImage):
        return True
    source = str(item)
    if source.startswith(("http://", "https://")):
        return True
    try:
        return Path(source).is_file()
    except OSError:
        return False


def _to_mllm_image(source: Union[str, Path, MLLMImage]) -> MLLMImage:
    """
    Создаёт MLLMImage из пути / URL / MLLMImage.
    """
    if isinstance(source, MLLMImage):
        return source

    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        return MLLMImage(url=source_str, local=False)

    abs_path = str(Path(source_str).resolve())
    return MLLMImage(url=abs_path, local=True)


def _build_multimodal_output(items: List) -> str:
    """
    Чередует текст и изображения в едином `actual_output` (строке).

    Принимает список, смешивающий обычный текст (str) и изображения
    (пути к файлам, URL или MLLMImage). Каждое изображение вставляется как
    его DeepEval-плейсхолдер (через `str(MLLMImage(...))`), чтобы судья
    видел текст до/после изображения.

    Пример:
        ["Вот рабочее место:", "img.png", "Такая настройка."]
        -> "Вот рабочее место:\\n[DEEPEVAL:IMAGE:...]\\nТакая настройка."
    """
    parts: List[str] = []
    for item in items:
        if _is_image_source(item):
            parts.append(str(_to_mllm_image(item)))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _metric_result(metric) -> dict:
    """Извлекает чистый сериализуемый результат из измеренной метрики."""
    return {
        "metric": metric.__name__,
        "score": metric.score,
        "reason": metric.reason,
        "threshold": metric.threshold,
        "success": metric.is_successful(),
    }


def _measure_single_turn(metric, input_prompt: str, actual_output: List) -> dict:
    """Общий поток для метрик «изображение в контексте» (coherence/helpfulness/reference)."""
    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=_build_multimodal_output(actual_output),
    )
    metric.measure(test_case)
    return _metric_result(metric)


def evaluate_image_coherence(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Оценивает, насколько согласованно каждое изображение вписано в текст."""
    metric = ImageCoherenceMetric(
        model=create_proxy_model(model=model), threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_image_helpfulness(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Оценивает, насколько каждое изображение помогает понять текст."""
    metric = ImageHelpfulnessMetric(
        model=create_proxy_model(model=model), threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_image_reference(
    input_prompt: str,
    actual_output: List,
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """Оценивает, корректно ли текст ссылается на сопровождающие изображения."""
    metric = ImageReferenceMetric(
        model=create_proxy_model(model=model), threshold=threshold)
    return _measure_single_turn(metric, input_prompt, actual_output)


def evaluate_text_to_image(
    prompt: str,
    generated_image_path: Union[str, Path, MLLMImage],
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Оценивает сгенерированное изображение относительно его prompt.

    TextToImageMetric требует ровно 0 изображений в `input` (чистый текст prompt)
    и ровно 1 изображение в `actual_output`.
    """
    image = _to_mllm_image(generated_image_path)
    test_case = LLMTestCase(input=prompt, actual_output=str(image))

    metric = TextToImageMetric(
        model=create_proxy_model(model=model), threshold=threshold)
    metric.measure(test_case)
    return _metric_result(metric)


def _print_result(result: dict) -> None:
    """Красиво печатает результат метрики."""
    status = "✅ PASS" if result["success"] else "❌ FAIL"
    print(f"   {result['metric']}: {result['score']:.2f} "
          f"(threshold {result['threshold']}) {status}")
    if result.get("reason"):
        print(f"   Reason: {result['reason']}")


# Тестирование при запуске напрямую
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование DeepEval мультимодальных метрик (DeepEval 4.x)")
    print("=" * 50)

    # Используем готовые sample_images
    sample_dir = Path(__file__).parent.parent / "sample_images"

    original = sample_dir / "original.png"
    edited = sample_dir / "edited.png"
    different = sample_dir / "different.png"

    if not original.exists():
        print(f"\n⚠️  Тестовые изображения не найдены в {sample_dir}")
        print("   Добавьте изображения в папку sample_images/")
        exit(1)

    print(f"\n📁 Используем изображения из {sample_dir}")

    # ---------------------------------------------------
    # 1. TextToImage
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("1. TextToImageMetric")
    print("-" * 40)

    prompt = "a man with glasses working on a laptop with code on monitors in an office"
    result = evaluate_text_to_image(
        prompt=prompt, generated_image_path=original)
    _print_result(result)

    # ---------------------------------------------------
    # 2. ImageCoherence
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("2. ImageCoherenceMetric")
    print("-" * 40)

    result = evaluate_image_coherence(
        input_prompt="Show me a software developer at work",
        actual_output=[
            "Here is a software developer at work:",
            str(original),
        ],
    )
    _print_result(result)

    # ---------------------------------------------------
    # 3. ImageHelpfulness
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("3. ImageHelpfulnessMetric")
    print("-" * 40)

    result = evaluate_image_helpfulness(
        input_prompt="Show me what a programmer's workspace looks like",
        actual_output=[
            "A typical programmer's workspace looks like this:",
            str(edited),
        ],
    )
    _print_result(result)

    # ---------------------------------------------------
    # 4. ImageReference
    # ---------------------------------------------------
    print("\n" + "-" * 40)
    print("4. ImageReferenceMetric")
    print("-" * 40)

    result = evaluate_image_reference(
        input_prompt="What does a software developer's workspace look like?",
        actual_output=[
            "As shown in the image below, a developer's workspace has multiple monitors:",
            str(original),
            "The edited version below shows a cleaner, more modern setup:",
            str(edited),
        ],
    )
    _print_result(result)

    print("\n✅ Тестирование завершено!")
