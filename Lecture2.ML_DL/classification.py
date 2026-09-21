import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, top_k_accuracy_score
from scipy.stats import entropy as scipy_entropy
from torchmetrics.classification import MulticlassCalibrationError, MulticlassAccuracy
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Выбираем устройство для вычислений (GPU, если доступен, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (CIFAR-10)
# ============================================================================


def load_cifar10_data(batch_size=128, val_split=0.1):
    """
    Загружает и готовит датасет CIFAR-10 с разбиением на train/validation/test.

    CIFAR-10 содержит 60 000 цветных изображений 32x32 в 10 классах:
    - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    Args:
        batch_size: размер батча
        val_split: доля обучающих данных для валидации (0.1 = 10%)
    """

    # Определяем преобразования для обучающей выборки (с аугментацией)
    transform_train = transforms.Compose([
        # RandomHorizontalFlip: случайно отражает изображение по горизонтали (например, самолёт влево/вправо).
        transforms.RandomHorizontalFlip(p=0.5),
        # RandomCrop: случайно обрезает часть изображения и добавляет отступы (чтобы имитировать разные ракурсы).
        transforms.RandomCrop(32, padding=4),
        # ToTensor: переводит изображение из PIL/NumPy в тензор PyTorch.
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))  # Нормализация
    ])

    # Преобразования для валидационной и тестовой выборок (без аугментации)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Загружаем исходные данные
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val_test
    )

    # Разбиваем обучающую выборку на train и validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Создаём генератор для воспроизводимого разбиения
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Собираем отдельную валидационную выборку без аугментации
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_val_test
    )

    # Получаем индексы для валидационной выборки
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Создаём загрузчики данных
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Названия классов CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Dataset sizes:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, classes


def load_mnist_data(batch_size=128, val_split=0.1):
    """
    Загружает и готовит датасет MNIST с разбиением на train/validation/test.

    MNIST содержит 70 000 чёрно-белых изображений 28x28 рукописных цифр:
    - цифры от 0 до 9

    Args:
        batch_size: размер батча
        val_split: доля обучающих данных для валидации (0.1 = 10%)
    """

    # Преобразования обучающей выборки (с аугментацией)
    transform_train = transforms.Compose([
        # Случайное горизонтальное отражение (для цифр менее уместно)
        transforms.RandomHorizontalFlip(p=0.1),  # Пониженная вероятность
        # Случайная обрезка с отступами
        transforms.RandomCrop(28, padding=2),     # Меньший отступ для 28x28
        # Случайный поворот (полезно для цифр)
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),                    # Перевод в тензор
        # нормализация для MNIST
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Преобразования валидационной/тестовой выборок (без аугментации)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        # нормализация для MNIST
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Загружаем MNIST вместо CIFAR-10
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_val_test
    )

    # Разбиваем обучающую выборку на train и validation (БЕЗ ИЗМЕНЕНИЙ)
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Генератор для воспроизводимого разбиения
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Отдельная валидационная выборка без аугментации
    val_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transform_val_test
    )

    # Получаем индексы для валидационной выборки
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Создаём загрузчики данных (БЕЗ ИЗМЕНЕНИЙ)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Названия классов MNIST
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    print(f"Dataset sizes:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, classes

# ============================================================================
# 2. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ
# ============================================================================


class SimpleCNN_CIFAR(nn.Module):
    """
    Простая свёрточная нейронная сеть для классификации CIFAR
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN_CIFAR, self).__init__()

        # Первый слой принимает 3 канала вместо 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 32x32x32 -> 32x32x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 16x16x64 -> 16x16x128

        # Пулинг и активация
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Полносвязные слои, подогнанные под форму 8x8x128
        # 8x8x128 -> 256 (меньше нейронов)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)           # 256 -> 64
        self.fc3 = nn.Linear(64, num_classes)   # 64 -> 10 (число классов)

    def forward(self, x):
        # Свёрточные слои с активацией и пулингом
        x = self.pool(self.relu(self.conv1(x)))  # 28x28x32 -> 14x14x32
        x = self.pool(self.relu(self.conv2(x)))  # 14x14x64 -> 7x7x64
        x = self.relu(self.conv3(x))             # 7x7x128 (БЕЗ пулинга)

        # Разворачиваем в одномерный вектор
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (7*7*128)

        # Полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Выходной слой (логиты)

        return x


# class VerySimpleCNN(nn.Module):
#     """
#     Очень простая модель для сравнения
#     """

#     def __init__(self, num_classes=10):
#         super(VerySimpleCNN, self).__init__()

#         self.conv1 = nn.Conv2d(1, 16, 5)  # 28x28 -> 24x24
#         self.pool = nn.MaxPool2d(2, 2)    # 24x24 -> 12x12
#         self.conv2 = nn.Conv2d(16, 32, 5)  # 12x12 -> 8x8
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 4 * 4)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class SimpleCNN(nn.Module):
    """
    Простая свёрточная нейронная сеть для классификации изображений CIFAR-10
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 1) число входных каналов = 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 7x7   -> 7x7
        # 3x3   -> 3x3 (после 3 пулингов)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # 2) размер после трёх пулингов для 28x28 равен 3x3
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Свёрточные слои с активацией и пулингом
        x = self.pool(self.relu(self.conv1(x)))  # 32x32x32 -> 16x16x32
        x = self.pool(self.relu(self.conv2(x)))  # 16x16x64 -> 8x8x64
        x = self.pool(self.relu(self.conv3(x)))  # 8x8x128 -> 4x4x128
        x = self.relu(self.conv4(x))             # 4x4x256

        # Разворачиваем в одномерный вектор
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (4*4*256)

        # Полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Выходной слой (логиты)

        return x

# ============================================================================
# 3. ФУНКЦИИ ОБУЧЕНИЯ И ОЦЕНКИ
# ============================================================================


def validate_model(model, val_loader, criterion, device):
    """
    Валидация модели (оценка на валидационной выборке)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return avg_val_loss, val_accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Обучение модели с валидацией и отслеживанием метрик
    """
    # Списки для хранения метрик
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Для ранней остановки (early stopping)
    best_val_accuracy = 0.0
    patience = 3  # Число эпох без улучшения
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ====================================================================
        # ОБУЧЕНИЕ
        # ====================================================================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Переносим данные на устройство
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход (forward pass)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и шаг оптимизации
            loss.backward()
            optimizer.step()

            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Метрики обучения за эту эпоху
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ====================================================================
        # ВАЛИДАЦИЯ
        # ====================================================================
        val_loss, val_accuracy = validate_model(
            model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Выводим метрики эпохи
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(
            f'  Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(
            f'  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

        # Проверяем наличие улучшения (для ранней остановки)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            # Здесь можно было бы сохранять лучшую модель
            print(
                f'  ✅ New best validation accuracy: {best_val_accuracy:.2f}%')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f'  ⏹️ Early stopping: {patience} epochs without improvement')
                break

        print()

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }


def evaluate_model_with_metrics(model, test_loader, device, classes):
    """
    Комплексная оценка модели с различными метриками
    """
    model.eval()

    # Списки для хранения результатов
    all_predictions = []
    all_labels = []
    all_softmax_probs = []
    all_logits = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Получаем логиты (сырые выходы)
            logits = model(inputs)

            # Применяем softmax, чтобы получить вероятности
            softmax_probs = torch.softmax(logits, dim=1)

            # Получаем предсказания
            _, predicted = torch.max(logits, 1)

            # Сохраняем результаты
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_softmax_probs.extend(softmax_probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # Переводим в numpy-массивы для удобства
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_softmax_probs = np.array(all_softmax_probs)
    all_logits = np.array(all_logits)

    return all_predictions, all_labels, all_softmax_probs, all_logits

# ============================================================================
# 4. ФУНКЦИИ ВЫЧИСЛЕНИЯ МЕТРИК
# ============================================================================


def calculate_confidence_scores(softmax_probs):
    """
    Вычисляет оценки уверенности как максимальную вероятность
    """
    confidence_scores = np.max(softmax_probs, axis=1)
    return confidence_scores


def calculate_top_k_accuracy(softmax_probs, true_labels, k=5):
    """
    Вычисляет Top-K accuracy
    """
    acc = top_k_accuracy_score(true_labels, softmax_probs, k=k)
    return acc * 100.0


def calculate_entropy(softmax_probs):
    """
    Вычисляет энтропию как меру неопределённости.
    H = -sum(p_i * log(p_i))
    """
    # Добавляем небольшое значение, чтобы избежать log(0)
    ent = scipy_entropy(softmax_probs, axis=1)
    return ent


def calculate_calibration_error(softmax_probs, predictions, true_labels, n_bins=10):
    """
    Вычисляет ожидаемую ошибку калибровки (Expected Calibration Error, ECE).
    Показывает, насколько уверенность модели соответствует её реальной точности.
    """
    probs_t = torch.from_numpy(softmax_probs).float()      # форма: [N, C]
    targets_t = torch.from_numpy(true_labels).long()       # форма: [N]

    # ECE с нормой L1 (|conf - acc|), как и раньше
    ece_metric = MulticlassCalibrationError(
        num_classes=softmax_probs.shape[1],
        n_bins=n_bins,
        norm='l1'
    )
    ece = float(ece_metric(probs_t, targets_t).item())

    # bin_data не нужен — возвращаем пустой список, чтобы сохранить сигнатуру
    return ece, []

# ============================================================================
# 5. ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================================


def plot_training_history(training_history):
    """
    Строит кривые обучения и валидации (с полосами разброса)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- единый стиль
    FIGSIZE = (12, 5)
    GRID_ALPHA = 0.3
    BAND_ALPHA = 0.15
    LINEWIDTH = 2

    train_accuracies = np.asarray(
        training_history['train_accuracies'], dtype=float)
    val_accuracies = np.asarray(
        training_history['val_accuracies'],   dtype=float)
    train_losses = np.asarray(
        training_history['train_losses'],     dtype=float)
    val_losses = np.asarray(training_history['val_losses'],       dtype=float)
    epochs = np.arange(1, len(train_accuracies) + 1)

    def _smooth(x, k=3):
        if len(x) < k:
            return x
        w = np.ones(k)/k
        y = np.convolve(x, w, mode='valid')
        pad = (len(x) - len(y)) // 2
        return np.pad(y, (pad, len(x)-len(y)-pad), mode='edge')

    def _roll_minmax(x, k=5):
        if k < 2 or len(x) < k:
            return x, x
        from collections import deque
        dmin, dmax, qmin, qmax, xs = [], [], deque(), deque(), x.tolist()
        for i, v in enumerate(xs):
            while qmin and xs[qmin[-1]] >= v:
                qmin.pop()
            qmin.append(i)
            while qmax and xs[qmax[-1]] <= v:
                qmax.pop()
            qmax.append(i)
            if qmin[0] <= i-k:
                qmin.popleft()
            if qmax[0] <= i-k:
                qmax.popleft()
            dmin.append(xs[qmin[0]])
            dmax.append(xs[qmax[0]])
        return np.array(dmin), np.array(dmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # --- Accuracy (точность)
    tr_s = _smooth(train_accuracies, k=3)
    va_s = _smooth(val_accuracies,   k=3)
    ax1.plot(epochs, tr_s, linewidth=LINEWIDTH, label='Training')
    ax1.plot(epochs, va_s, linewidth=LINEWIDTH, label='Validation')
    tr_lo, tr_hi = _roll_minmax(train_accuracies, k=5)
    va_lo, va_hi = _roll_minmax(val_accuracies,   k=5)
    ax1.fill_between(epochs, tr_lo, tr_hi, alpha=BAND_ALPHA)
    ax1.fill_between(epochs, va_lo, va_hi, alpha=BAND_ALPHA)
    ax1.set_title('Accuracy per epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc='lower right')

    # --- Loss (функция потерь)
    tl_s = _smooth(train_losses, k=3)
    vl_s = _smooth(val_losses,   k=3)
    ax2.plot(epochs, tl_s, linewidth=LINEWIDTH, label='Training')
    ax2.plot(epochs, vl_s, linewidth=LINEWIDTH, label='Validation')
    tl_lo, tl_hi = _roll_minmax(train_losses, k=5)
    vl_lo, vl_hi = _roll_minmax(val_losses,   k=5)
    ax2.fill_between(epochs, tl_lo, tl_hi, alpha=BAND_ALPHA)
    ax2.fill_between(epochs, vl_lo, vl_hi, alpha=BAND_ALPHA)
    ax2.set_title('Loss per epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # --- Анализ переобучения (без изменений)
    print("\n🔍 OVERFITTING ANALYSIS:")
    final_train_acc = float(train_accuracies[-1])
    final_val_acc = float(val_accuracies[-1])
    gap = final_train_acc - final_val_acc
    print(f"Final training accuracy: {final_train_acc:.2f}%")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Gap (Train - Val): {gap:.2f}%")
    if gap < 3:
        print("✅ Virtually no overfitting")
    elif gap < 8:
        print("⚠️ Mild overfitting")
    elif gap < 15:
        print("🔶 Moderate overfitting")
    else:
        print("❌ Strong overfitting")
    return gap


def plot_confusion_matrix(true_labels, predictions, classes):
    """
    Строит матрицу ошибок (проценты + абсолютные значения, нормировка по истинному классу)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    FIGSIZE = (8, 6)
    GRID_ALPHA = 0.3

    cm_abs = confusion_matrix(true_labels, predictions)
    with np.errstate(invalid='ignore'):
        cm = cm_abs / cm_abs.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm)

    plt.figure(figsize=FIGSIZE)
    ax = sns.heatmap(cm, annot=False, cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     vmin=0.0, vmax=1.0, cbar_kws={"label": "Share per true class"})
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]*100:.1f}%\n({cm_abs[i, j]})"
            ax.text(j+0.5, i+0.5, txt, ha='center',
                    va='center', fontsize=9, color='black')

    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.grid(False)  # у heatmap своя сетка
    plt.tight_layout()
    plt.show()
