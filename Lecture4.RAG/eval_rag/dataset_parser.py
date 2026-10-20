# dataset_parser.py
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class DatasetParser:
    """Простой парсер датасета для оценки RAG"""

    def __init__(self):
        self.required_columns = ['question']  # Только question обязательно
        self.optional_columns = ['expected_response']

    def load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Загрузить датасет из Excel или CSV файла"""
        try:
            # Определяем расширение файла
            file_extension = Path(file_path).suffix.lower()

            # Загружаем файл
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"Неподдерживаемый формат файла: {file_extension}")
                return None

            # Проверяем наличие обязательных колонок
            missing_columns = [
                col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                print(f"Отсутствуют обязательные колонки: {missing_columns}")
                print(f"Найденные колонки: {list(df.columns)}")
                return None

            # Удаляем строки с пустыми значениями в обязательных колонках
            df_clean = df.dropna(subset=self.required_columns)

            print(f"Загружен датасет с {len(df_clean)} вопросами")
            return df_clean

        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return None

    def get_questions(self, df: pd.DataFrame) -> List[str]:
        """Получить список всех вопросов"""
        return df['question'].tolist()

    def get_expected_responses(self, df: pd.DataFrame) -> List[str]:
        """Получить список ожидаемых ответов, заменяя nan на пустую строку"""
        return df['expected_response'].fillna("").astype(str).tolist()

    def get_question_response_pairs(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Получить список пар вопрос-ответ

        Returns:
            Список словарей с ключами 'question' и 'expected_response'
        """
        pairs = []
        for _, row in df.iterrows():
            pairs.append({
                'question': row['question'],
                'expected_response': row['expected_response']
            })
        return pairs

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Валидация датасета

        Returns:
            Словарь с информацией о датасете
        """
        info = {
            'total_rows': len(df),
            'valid_pairs': 0,
            'empty_questions': 0,
            'empty_responses': 0,
            'avg_question_length': 0,
            'avg_response_length': 0
        }

        valid_pairs = 0
        question_lengths = []
        response_lengths = []

        for _, row in df.iterrows():
            question = str(row['question']).strip()
            response = str(row['expected_response']).strip()

            if not question or question == 'nan':
                info['empty_questions'] += 1
                continue

            if not response or response == 'nan':
                info['empty_responses'] += 1
                continue

            valid_pairs += 1
            question_lengths.append(len(question))
            response_lengths.append(len(response))

        info['valid_pairs'] = valid_pairs
        if question_lengths:
            info['avg_question_length'] = sum(
                question_lengths) / len(question_lengths)
        if response_lengths:
            info['avg_response_length'] = sum(
                response_lengths) / len(response_lengths)

        return info

    def preview_dataset(self, df: pd.DataFrame, n: int = 3):
        """Показать первые n примеров из датасета"""
        print("\n=== Превью датасета ===")
        for i, row in df.head(n).iterrows():
            print(f"\nПример {i+1}:")
            print(f"Вопрос: {row['question']}")

            expected_response = row.get('expected_response')
            if expected_response is not None and not pd.isna(expected_response):
                print(f"Ожидаемый ответ: {expected_response}")
            else:
                print("Ожидаемый ответ: Не задан")
        print("=" * 50)
