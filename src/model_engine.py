"""SMOTE, навчання бустингових моделей і переведення імовірності відмови в рівень ризику."""
from typing import Dict, Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from config import LGBM_PARAMS, RANDOM_STATE, TAU_WARN, TAU_CRIT


def balance_training_data(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE до співвідношення 1:1. Викликати тільки на train: тест має лишатися
    з реальним дисбалансом класів, інакше оцінка буде нечесною.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def train_primary_classifier(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LGBMClassifier:
    """Основна модель: LightGBM на train-частині після SMOTE."""
    X_bal, y_bal = balance_training_data(X_train, y_train)
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_bal, y_bal)
    return model


def classify_risk_level(
    probability: float,
    tau_warn: float = TAU_WARN,
    tau_crit: float = TAU_CRIT,
) -> Tuple[str, str]:
    """
    p >= tau_crit дає 'critical', p >= tau_warn дає 'warning', усе інше 'normal'.
    Межі включні; разом із кодом повертається текст повідомлення для інтерфейсу.
    """
    if probability >= tau_crit:
        return (
            "critical",
            "КРИТИЧНИЙ РИЗИК ВІДМОВИ! Рекомендовано негайну зупинку та технічне обслуговування.",
        )
    if probability >= tau_warn:
        return (
            "warning",
            "ПІДВИЩЕНИЙ РИЗИК. Потрібен посилений моніторинг вузлів.",
        )
    return (
        "normal",
        "ШТАТНИЙ РЕЖИМ РОБОТИ. Ознак передаварійного стану не виявлено.",
    )


def evaluate_ensemble_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Порівнює LightGBM, XGBoost і CatBoost. SMOTE робиться один раз на train, тому всі три
    моделі вчаться на однакових даних, а метрики рахуються на незбалансованому тесті.
    """
    X_bal, y_bal = balance_training_data(X_train, y_train)

    candidates = {
        "LightGBM (Обрана)": LGBMClassifier(**LGBM_PARAMS),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            verbose=0,
            random_state=RANDOM_STATE,
        ),
    }

    trained_models = {}
    metrics_rows = []

    for name, clf in candidates.items():
        clf.fit(X_bal, y_bal)
        trained_models[name] = clf

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics_rows.append(
            {
                "Модель": name,
                "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "Precision": round(float(precision_score(y_test, y_pred)), 4),
                "Recall": round(float(recall_score(y_test, y_pred)), 4),
                "F1-Score": round(float(f1_score(y_test, y_pred)), 4),
                "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
            }
        )

    return trained_models, pd.DataFrame(metrics_rows)
