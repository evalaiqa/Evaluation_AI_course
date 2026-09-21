import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

from classification import (
    device,
    load_cifar10_data,
    SimpleCNN_CIFAR,
    train_model,
    evaluate_model_with_metrics,
    calculate_confidence_scores,
    calculate_top_k_accuracy,
    calculate_entropy,
    calculate_calibration_error,
    plot_training_history,
    plot_confusion_matrix,
)


def run_model():
    """
    Главная функция, демонстрирующая полный конвейер обучения и оценки
    """
    print("=" * 80)
    print("PRACTICE: DEEP LEARNING CLASSIFICATION AND METRICS")
    print("=" * 80)

    # 1. Загрузка данных
    print("\n1. Loading the CIFAR-10 data...")
    train_loader, val_loader, test_loader, classes = load_cifar10_data(
        batch_size=128, val_split=0.1)
    print(f"Classes: {classes}")

    # 2. Построение модели
    print("\n2. Building the model...")
    model = SimpleCNN_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Считаем число параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # 3. Обучение модели
    print("\n3. Training the model with validation...")
    print("=" * 60)
    training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10
    )

    print(f"\n🎯 TRAINING SUMMARY:")
    print(
        f"Best validation accuracy: {training_history['best_val_accuracy']:.2f}%")

    # 4. Финальная оценка на тестовой выборке
    print("\n4. Final evaluation on the test set...")
    print("⚠️  IMPORTANT: The test set is used ONLY for the final evaluation!")
    print("🔒 This set was NOT used during training or validation")
    predictions, true_labels, softmax_probs, logits = evaluate_model_with_metrics(
        model, test_loader, device, classes
    )

    # Базовая точность на тестовой выборке
    test_accuracy = accuracy_score(true_labels, predictions) * 100
    val_accuracy = training_history['best_val_accuracy']

    print(f"\n📊 VALIDATION VS TEST COMPARISON:")
    print(f"Best validation accuracy: {val_accuracy:.2f}%")
    print(f"Final test accuracy:      {test_accuracy:.2f}%")

    accuracy_diff = abs(val_accuracy - test_accuracy)
    if accuracy_diff < 2:
        print(f"✅ Excellent agreement (difference: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 5:
        print(f"✅ Good agreement (difference: {accuracy_diff:.2f}%)")
    elif accuracy_diff < 10:
        print(f"⚠️  Moderate discrepancy (difference: {accuracy_diff:.2f}%)")
    else:
        print(f"❌ Significant discrepancy (difference: {accuracy_diff:.2f}%)")
        print(
            "   Possible causes: overfitting on the validation set, data distribution shift")

    # ========================================================================
    # 5. ВЫЧИСЛЕНИЕ И АНАЛИЗ МЕТРИК
    # ========================================================================

    print("\n" + "=" * 50)
    print("FINAL METRICS ON THE TEST SET")
    print("=" * 50)
    print("🔬 All metrics below are computed on the independent test set")

    # 5.1 Базовая точность (Top-1 Accuracy)
    basic_accuracy = accuracy_score(true_labels, predictions) * 100
    print(f"\n📊 BASIC METRICS:")
    print(f"Top-1 Accuracy: {basic_accuracy:.2f}%")

    # 5.2 Оценки уверенности (Confidence Scores)
    confidence_scores = calculate_confidence_scores(softmax_probs)
    avg_confidence = np.mean(confidence_scores)
    print(f"Average Confidence Score: {avg_confidence:.3f}")
    print(f"Min. Confidence Score: {np.min(confidence_scores):.3f}")
    print(f"Max. Confidence Score: {np.max(confidence_scores):.3f}")

    # 5.3 Top-K Accuracy
    print(f"\n🎯 TOP-K ACCURACY:")
    for k in [1, 3, 5]:
        top_k_acc = calculate_top_k_accuracy(softmax_probs, true_labels, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.2f}%")

    # 5.4 Энтропия
    entropy_values = calculate_entropy(softmax_probs)
    avg_entropy = np.mean(entropy_values)
    print(f"\n🔀 ENTROPY (uncertainty):")
    print(f"Average entropy: {avg_entropy:.3f}")
    print(f"Min. entropy: {np.min(entropy_values):.3f}")
    print(f"Max. entropy: {np.max(entropy_values):.3f}")
    print(f"Standard deviation: {np.std(entropy_values):.3f}")

    # 5.5 Калибровка
    ece, bin_data = calculate_calibration_error(
        softmax_probs, predictions, true_labels)
    print(f"\n⚖️ CALIBRATION:")
    print(
        f"Expected Calibration Error (ECE), the difference between stated confidence and actual accuracy: {ece:.3f}")
    print(f"How to interpret: if the model says 'I'm 80% sure', it is right in about 76-80% of those cases (almost perfect).")
    print("ECE interpretation:")
    print("  - 0.0-0.05: Excellently calibrated")
    print("  - 0.05-0.1: Well calibrated")
    print("  - 0.1-0.2: Moderately calibrated")
    print("  - >0.2: Poorly calibrated")

    # 5.6 Анализ по классам
    print(f"\n📋 DETAILED PER-CLASS REPORT:")
    class_report = classification_report(true_labels, predictions,
                                         target_names=classes, digits=3)
    print(class_report)

    # ========================================================================
    # 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    # ========================================================================

    print("\n" + "=" * 50)
    print("VISUALIZATION OF RESULTS")
    print("=" * 50)

    # График обучения с валидацией
    overfitting_gap = plot_training_history(training_history)

    # Матрица ошибок
    plot_confusion_matrix(true_labels, predictions, classes)


if __name__ == "__main__":
    run_model()
