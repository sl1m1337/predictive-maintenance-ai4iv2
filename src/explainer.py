"""SHAP-пояснення прогнозів LightGBM: внески ознак для конкретного запису та графік до них."""
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from config import FEATURE_NAMES_UA


class DiagnosticExplainer:
    """Обгортка над shap.TreeExplainer для навченої моделі."""

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, input_row: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """SHAP-внески ознак для одного рядка (клас «відмова») і base value моделі в log-odds."""
        sv = self.explainer.shap_values(input_row)
        if isinstance(sv, list):
            shap_row = np.asarray(sv[1][0], dtype=float)
        elif len(np.asarray(sv).shape) == 3:
            shap_row = np.asarray(sv[0, :, 1], dtype=float)
        else:
            shap_row = np.asarray(sv[0], dtype=float)

        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[-1])
        else:
            base_val = float(base_val)

        return shap_row, base_val

    def check_additivity(self, input_row: pd.DataFrame, tol: float = 1e-3) -> bool:
        """Адитивність SHAP: base_value + sum(phi_i) має дорівнювати сирому виходу моделі в log-odds."""
        shap_row, base_val = self.explain_instance(input_row)
        raw_margin = float(self.model.predict(input_row, raw_score=True)[0])
        return abs((base_val + float(np.sum(shap_row))) - raw_margin) <= tol

    def plot_local_contributions(self, input_row: pd.DataFrame):
        """Горизонтальні стовпці внесків: червоні підвищують ризик відмови, сині знижують."""
        shap_row, _ = self.explain_instance(input_row)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        colors = ["#d9534f" if val > 0 else "#2b7bba" for val in shap_row]
        y_pos = np.arange(len(FEATURE_NAMES_UA))

        ax.barh(y_pos, shap_row, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(FEATURE_NAMES_UA)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Внесок ознаки у прогноз (SHAP value)")
        ax.set_title("Червоний = підвищує ризик відмови | Синій = знижує ризик")
        plt.tight_layout()
        return fig
