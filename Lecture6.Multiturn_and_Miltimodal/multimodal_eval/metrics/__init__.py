"""
Метрики для мультимодальной оценки изображений
"""

from .clip_score import compute_clip_score_text, compute_clip_score_images
from .lpips_metric import compute_lpips
from .deepeval_custom_llm import create_proxy_model
from .deepeval_multimodal import (
    evaluate_image_coherence,
    evaluate_image_helpfulness,
    evaluate_image_reference,
    evaluate_text_to_image,
)

__all__ = [
    # CLIP Score
    "compute_clip_score_text",
    "compute_clip_score_images",
    # LPIPS
    "compute_lpips",
    # DeepEval прокси-модель
    "create_proxy_model",
    # DeepEval метрики
    "evaluate_image_coherence",
    "evaluate_image_helpfulness",
    "evaluate_image_reference",
    "evaluate_text_to_image",
]
