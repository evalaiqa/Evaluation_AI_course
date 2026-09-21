# regression.py

# =============================================================================
# MSE ДЛЯ ГЛУБОКОГО ОБУЧЕНИЯ — РЕАЛЬНЫЙ ДАТАСЕТ О ЖИЛЬЕ (только стандартная модель)
# =============================================================================
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time


class TimeCallback(keras.callbacks.Callback):
    """Колбэк для отслеживания времени обучения"""

    def __init__(self, print_every=5):
        super().__init__()
        self.print_every = print_every

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every == 0:
            elapsed_time = time.time() - self.start_time
            print(f"⏱️ Epoch {epoch + 1}: {elapsed_time:.1f}s total time")


class DataLoader:
    """Класс для загрузки и предобработки данных"""

    def __init__(self):
        self.data = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # target_scale = сколько долларов приходится на 1 единицу сырого таргета.
        # California Housing: таргет измеряется в $100k, поэтому множитель = 100_000.
        # Diamonds: таргет уже в $, поэтому множитель = 1.0 (см. load_diamonds).
        self.target_scale = 100_000.0

    def load_california_housing(self):
        """Загружает датасет California Housing"""
        print("\n📊 Loading the real California Housing dataset...")

        california_housing = fetch_california_housing()
        feature_names = california_housing.feature_names
        X = california_housing.data
        y = california_housing.target

        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        self.data = df
        self.target_scale = 100_000.0  # 1 единица таргета = $100 000

        return df

    def load_diamonds(self):
        """
        Загружает датасет Diamonds (из seaborn) — по аналогии с load_california_housing.
        """
        print("\n💎 Loading the Diamonds dataset...")

        df = sns.load_dataset('diamonds').dropna().copy()

        # В Diamonds есть 3 категориальных столбца (cut, color, clarity), которые по природе ПОРЯДКОВЫЕ.
        # Кодируем их упорядоченными кодами категорий, чтобы стандартный StandardScaler в prepare_data работал.
        cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
        color_order = ['J', 'I', 'H', 'G', 'F', 'E',
                       'D']            # J = худший → D = лучший
        clarity_order = ['I1', 'SI2', 'SI1',
                         'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        df['cut'] = pd.Categorical(
            df['cut'], categories=cut_order, ordered=True).codes
        df['color'] = pd.Categorical(
            df['color'], categories=color_order, ordered=True).codes
        df['clarity'] = pd.Categorical(
            df['clarity'], categories=clarity_order, ordered=True).codes

        # Переименовываем 'price' → 'target', чтобы остальной конвейер работал без изменений (drop('target', axis=1)).
        df = df.rename(columns={'price': 'target'})
        self.data = df
        self.target_scale = 1.0  # таргет уже в долларах — умножение на 100k не нужно

        return df

    def prepare_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """
        Готовит данные для обучения.

        Args:
            test_size: размер тестовой выборки
            val_size: размер валидационной выборки (от оставшихся данных)
            random_state: seed для воспроизводимости

        Returns:
            Кортеж с обучающими, валидационными и тестовыми данными
        """
        print("\n🔧 Preparing data for deep learning...")

        X = self.data.drop('target', axis=1).values
        y = self.data['target'].values

        # Нормализация
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Разбиваем данные
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )

        print(f"📊 Training set size: {X_train.shape}")
        print(f"📊 Validation set size: {X_val.shape}")
        print(f"📊 Test set size: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelBuilder:
    """Класс для построения и компиляции стандартной модели"""

    @staticmethod
    def create_deep_housing_model(input_dim):
        """
        Создаёт оптимизированную нейросеть для предсказания цен на жильё (стандартный MSE)
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model, "Standard MSE"


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self):
        self.models = {}
        self.histories = {}
        self.training_times = {}

    def setup_callbacks(self):
        """Настраивает колбэки для обучения"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        time_callback = TimeCallback(print_every=5)

        return [early_stopping, reduce_lr, time_callback]

    def train_model(self, model, model_name, X_train, y_train, X_val, y_val, epochs=20, batch_size=256):
        """
        Обучает модель.
        """
        print(f"\nTraining model {model_name}...")
        start_time = time.time()

        callbacks = self.setup_callbacks()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time

        self.models[model_name] = model
        self.histories[model_name] = history
        self.training_times[model_name] = training_time

        print(
            f"⏱️ Training time for {model_name}: {training_time:.2f} seconds")
        return history

    def diagnose_training_time(self):
        """Выводит отчёт о времени обучения"""
        total_time = sum(self.training_times.values())
        print(f"⏱️ Total training time: {total_time:.2f} seconds")
        if total_time > 300:
            print(
                f"⚠️ WARNING: {total_time:.1f}s - too slow. Check model size / data / TF.")
        elif total_time > 180:
            print(f"⚠️ {total_time:.1f}s - could be faster.")
        else:
            print(f"✅ Training time is good: {total_time:.1f}s")


class ModelEvaluator:
    """Класс для оценки модели"""

    def __init__(self, scaler_y, target_scale=100_000.0):
        # target_scale переводит сырой таргет обратно в доллары.
        # California Housing = 100_000 (по умолчанию), Diamonds = 1.0.
        self.scaler_y = scaler_y
        self.target_scale = target_scale
        self.metrics = {}

    def evaluate_model(self, model, model_name, X_test, y_test, training_time):
        """
        Оценивает модель и сохраняет метрики.
        """
        # Предсказания (в нормализованных единицах таргета)
        y_pred = model.predict(X_test, verbose=0)

        # Обратное преобразование (к исходной шкале датасета: «сотни тысяч $»)
        y_test_original_units = self.scaler_y.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        y_pred_original_units = self.scaler_y.inverse_transform(
            y_pred
        ).flatten()

        # ПЕРЕВОД В ДОЛЛАРЫ: множитель зависит от датасета
        # (California Housing: 100_000, Diamonds: 1.0 — см. DataLoader.target_scale).
        y_test_dollars = y_test_original_units * self.target_scale
        y_pred_dollars = y_pred_original_units * self.target_scale

        # Метрики в долларах
        mse = mean_squared_error(y_test_dollars, y_pred_dollars)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
        r2 = r2_score(y_test_dollars, y_pred_dollars)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time,
            # сохраняем массивы в долларах, чтобы все графики/выводы ниже были в $
            'y_test_original': y_test_dollars,
            'y_pred_original': y_pred_dollars
        }
        self.metrics[model_name] = metrics
        return metrics

    def print_metrics(self, model_name):
        """Выводит метрики модели (в реальных $)"""
        metrics = self.metrics[model_name]
        print(f"\n🔹 {model_name}:")
        print(f"   • MSE:  {metrics['mse']:.2f} $^2")
        print(f"   • RMSE: {metrics['rmse']:.2f} $")
        print(f"   • MAE:  {metrics['mae']:.2f} $")
        print(f"   • R²:   {metrics['r2']:.4f}")


class DataVisualizer:
    """Класс для визуализации результатов"""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def plot_results(self, trainer, evaluator):
        """Визуализирует результаты обучения и оценки"""
        plt.figure(figsize=(20, 12))

        # График 1: история обучения — Loss
        plt.subplot(2, 4, 1)
        for model_name, history in trainer.histories.items():
            plt.plot(history.history['loss'],
                     label=f'{model_name} (train)', linewidth=2)
            plt.plot(history.history['val_loss'],
                     label=f'{model_name} (val)', linewidth=2)
        plt.title('MSE Loss history')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 2: история обучения — MAE
        plt.subplot(2, 4, 2)
        for model_name, history in trainer.histories.items():
            plt.plot(history.history['mae'], label=model_name, linewidth=2)
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График 3: истинные значения vs предсказанные
        plt.subplot(2, 4, 3)
        (model_name, metrics), = evaluator.metrics.items()
        y_test = metrics['y_test_original']
        y_pred = metrics['y_pred_original']
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual price ($)')
        plt.ylabel('Predicted price ($)')
        plt.title(f'{model_name}\nR² = {metrics["r2"]:.4f}')
        plt.grid(True, alpha=0.3)

        # График 4: остатки vs предсказания
        plt.subplot(2, 4, 5)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted price ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residuals vs Predictions')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class MSEDemo:
    """Класс для демонстрации работы MSE"""

    @staticmethod
    def demonstrate_mse_on_batch(evaluator):
        """Демонстрирует MSE на реальном батче"""
        print("\n🔍 MSE DEMONSTRATION ON A REAL BATCH:")
        print("="*70)

        if not evaluator.metrics:
            print("No data to demonstrate")
            return

        # Берём метрики единственной модели
        (model_name, metrics), = evaluator.metrics.items()
        y_true_original = metrics['y_test_original']
        y_pred_original = metrics['y_pred_original']

        batch_size = min(10, len(y_true_original))
        batch_indices = np.random.choice(
            len(y_true_original), batch_size, replace=False)
        y_true_batch = y_true_original[batch_indices]
        y_pred_batch = y_pred_original[batch_indices]

        print(f"Model: {model_name}")
        print(f"Batch size: {batch_size}")
        print(f"{'#':<3} {'Actual price ($)':<20} {'Predicted ($)':<20} {'Error ($)':<15} {'Error^2 ($^2)':<18}")
        print("-" * 70)

        total_squared_error = 0.0
        for i in range(batch_size):
            true_price = y_true_batch[i]
            pred_price = y_pred_batch[i]
            error = true_price - pred_price
            squared_error = error ** 2
            total_squared_error += squared_error
            print(
                f"{i+1:<3} {true_price:<15.2f} {pred_price:<15.2f} {error:<12.2f} {squared_error:<15.2f}")

        mse_batch = total_squared_error / batch_size
        print(f"\nMSE for the batch: {mse_batch:.2f} $^2")


class HousingAnalyzer:
    """Главный класс для анализа данных о жилье"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.model_builder = ModelBuilder()
        self.trainer = ModelTrainer()
        self.evaluator = None
        self.visualizer = None
        self.mse_demo = MSEDemo()

    def run_analysis(self, dataset="california"):
        """
        Запускает полный анализ.

        dataset: "california" (по умолчанию) или "diamonds".
        Между датасетами различается только вызов загрузчика;
        весь остальной конвейер используется без изменений.
        """
        print("="*70)
        print("🏠 MSE FOR DEEP LEARNING - REAL HOUSING DATASET")
        print("="*70)

        # 1. Загрузка данных
        if dataset == "diamonds":
            self.data_loader.load_diamonds()
        else:
            self.data_loader.load_california_housing()

        # 2. Подготовка данных
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.prepare_data()

        # 3. Создание evaluator и visualizer
        # Передаём target_scale, чтобы метрики/графики выводились в реальных долларах для ОБОИХ датасетов.
        self.evaluator = ModelEvaluator(
            self.data_loader.scaler_y,
            target_scale=self.data_loader.target_scale,
        )
        self.visualizer = DataVisualizer(self.data_loader)

        # 4. Построение и обучение стандартной модели
        print("\n🧠 Building the deep neural network architecture (Standard MSE)...")
        model, loss_name = self.model_builder.create_deep_housing_model(
            X_train.shape[1])

        print(f"\n🏗️ Neural network architecture:")
        model.summary()

        print("\n🚀 Training the model on the real data...")
        history = self.trainer.train_model(
            model, "Standard MSE", X_train, y_train, X_val, y_val
        )

        # 5. Диагностика времени обучения
        self.trainer.diagnose_training_time()

        # 6. Оценка модели
        print("\n🎯 Model evaluation...")
        _ = self.evaluator.evaluate_model(
            model, "Standard MSE", X_test, y_test,
            self.trainer.training_times["Standard MSE"]
        )

        # 7. Вывод результатов
        print(f"\n📊 RESULTS ON THE REAL DATA:")
        print("="*60)
        self.evaluator.print_metrics("Standard MSE")

        # 8. Визуализация
        self.visualizer.plot_results(self.trainer, self.evaluator)

        # 9. Демонстрация MSE
        self.mse_demo.demonstrate_mse_on_batch(self.evaluator)


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    """Главная функция, запускающая анализ"""
    analyzer = HousingAnalyzer()
    # ДОМАШНЕЕ ЗАДАНИЕ: чтобы прогнать тот же конвейер на датасете Diamonds,
    # просто поменяйте аргумент ниже на "diamonds".
    analyzer.run_analysis(dataset="california")  # или: dataset="diamonds"

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
