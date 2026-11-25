"""
Відтворює експеримент із розділу 2: LightGBM проти XGBoost і CatBoost.
Таблицю метрик і графіки (ROC, матриця помилок, SHAP) зберігає в artifacts/.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, confusion_matrix

from config import TARGET_COLUMN
from src.data_loader import load_raw_dataset, split_dataset
from src.feature_engineering import compute_physical_features
from src.model_engine import evaluate_ensemble_suite


def main(output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_dataset()
    X = compute_physical_features(raw_df)
    y = raw_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    models, summary_df = evaluate_ensemble_suite(X_train, y_train, X_test, y_test)

    print("=== Зведена таблиця метрик на тестовій вибірці (2000 записів) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, "benchmark_metrics.csv"), index=False)

    plt.figure(figsize=(8, 6))
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = summary_df.loc[summary_df["Модель"] == name, "ROC-AUC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Порівняння ROC-кривих ансамблевих класифікаторів")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()

    best_model = models["LightGBM (Обрана)"]
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Фактичний клас")
    plt.title("Матриця помилок LightGBM")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    explainer = shap.TreeExplainer(best_model)
    sv = explainer.shap_values(X_test)
    shap_matrix = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(shap_matrix, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
