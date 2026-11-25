"""Читання CSV з телеметрією AI4I 2020, нормалізація назв стовпців і перевірка даних."""
from typing import Tuple, Union, IO
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_URL, RAW_REQUIRED_COLUMNS, TEST_SIZE, RANDOM_STATE


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Прибирає дужки й пробіли з назв стовпців: 'Torque [Nm]' -> 'Torque_Nm'."""
    cleaned = df.copy()
    cleaned.columns = [
        col.replace("[", "").replace("]", "").replace(" ", "_")
        for col in cleaned.columns
    ]
    return cleaned


def validate_telemetry_schema(df: pd.DataFrame) -> None:
    """Кидає ValueError, якщо бракує стовпців датчиків або в них є NaN."""
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"У вхідному наборі відсутні обов'язкові стовпці: {missing}")

    if df[RAW_REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Виявлено пропущені значення (NaN) у показниках датчиків.")


def load_raw_dataset(source: Union[str, IO] = DATA_URL) -> pd.DataFrame:
    """Читає CSV за URL, шляхом або з файлу, завантаженого в Streamlit, і одразу його перевіряє."""
    df = pd.read_csv(source)
    df = normalize_column_names(df)
    validate_telemetry_schema(df)
    return df


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Стратифікований поділ 80/20. Ділимо до SMOTE, інакше синтетичні точки потраплять
    у тест і метрики будуть завищені через витік даних.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
