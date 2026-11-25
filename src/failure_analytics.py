"""
Вкладка 4: статистика типів відмов у AI4I 2020 і симулятор порогу τ від 0.10 до 0.80.
Робочі пороги з config.py тут не змінюються, лише перераховуються метрики для інших τ.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TARGET_COLUMN, TAU_CRIT, TAU_WARN

FAILURE_MODES: Dict[str, Dict[str, str]] = {
    "TWF": {
        "name": "Знос інструменту",
        "criterion": "заміна/відмова інструменту за зносу 200–240 хв",
    },
    "HDF": {
        "name": "Порушення тепловідведення",
        "criterion": "ΔT < 8.6 K за обертів < 1380 об/хв",
    },
    "PWF": {
        "name": "Відмова за потужністю",
        "criterion": "P < 3500 Вт або P > 9000 Вт",
    },
    "OSF": {
        "name": "Перевантаження",
        "criterion": "знос × момент > 11 000 / 12 000 / 13 000 хв·Нм (L / M / H)",
    },
    "RNF": {
        "name": "Випадкова відмова",
        "criterion": "0,1 % незалежно від параметрів процесу",
    },
}
FAILURE_MODE_COLUMNS = list(FAILURE_MODES)

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.01

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES_BLUE = "#2a78d6"
_SERIES_ORANGE = "#eb6834"
_SERIES_AQUA = "#1baf7a"
_SERIES_VIOLET = "#6250d6"
_SERIES_RED = "#e34948"


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"У наборі даних відсутні стовпці для аналітики відмов: {missing}")


def compute_failure_type_distribution(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Кількість і частка кожного типу відмови: від усіх записів і серед самих відмов."""
    _require_columns(raw_df, FAILURE_MODE_COLUMNS)
    total = len(raw_df)
    counts = {code: int(raw_df[code].astype(int).sum()) for code in FAILURE_MODE_COLUMNS}
    mode_total = sum(counts.values())

    rows = []
    for code in FAILURE_MODE_COLUMNS:
        rows.append(
            {
                "Код": code,
                "Тип відмови": FAILURE_MODES[code]["name"],
                "Критерій виникнення (AI4I 2020)": FAILURE_MODES[code]["criterion"],
                "Кількість": counts[code],
                "Частка від усіх записів (%)": round(counts[code] / total * 100.0, 2) if total else 0.0,
                "Частка серед типів відмов (%)": (
                    round(counts[code] / mode_total * 100.0, 2) if mode_total else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_failure_modes(raw_df: pd.DataFrame) -> Dict[str, object]:
    """
    Загальні цифри по відмовах. Типи відмов можуть перекриватися і не завжди збігаються
    з Machine_failure, тому ці розбіжності рахуються окремо.
    """
    _require_columns(raw_df, FAILURE_MODE_COLUMNS + [TARGET_COLUMN])
    modes = raw_df[FAILURE_MODE_COLUMNS].astype(int)
    modes_per_record = modes.sum(axis=1)
    failure = raw_df[TARGET_COLUMN].astype(int)
    total = len(raw_df)
    return {
        "total_records": int(total),
        "machine_failures": int(failure.sum()),
        "failure_rate_pct": round(float(failure.mean() * 100.0), 2) if total else 0.0,
        "mode_events": int(modes.values.sum()),
        "multi_mode_records": int((modes_per_record > 1).sum()),
        "failures_without_mode": int(((failure == 1) & (modes_per_record == 0)).sum()),
        "modes_without_failure": int(((failure == 0) & (modes_per_record > 0)).sum()),
    }


def failure_rate_by_equipment_type(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Частота відмов окремо для серій L, M і H."""
    _require_columns(raw_df, ["Type", TARGET_COLUMN])
    labels = {"L": "L (Базова)", "M": "M (Середня)", "H": "H (Висока)"}
    rows = []
    for letter, label in labels.items():
        subset = raw_df[raw_df["Type"] == letter]
        n = len(subset)
        k = int(subset[TARGET_COLUMN].astype(int).sum())
        rows.append(
            {
                "Серія": label,
                "Записів": n,
                "Відмов": k,
                "Частота відмов (%)": round(k / n * 100.0, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_threshold_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> Dict[str, float]:
    """
    Confusion matrix і метрики для правила p ≥ τ, того самого, що в classify_risk_level.
    FN тут означає пропущені аварії: відмова була, а система її не помітила.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("Розмірності y_true та y_prob не збігаються.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Поріг має бути в діапазоні [0, 1].")

    y_pred = (y_prob >= float(threshold)).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "missed_failures": fn,
        "false_alarms": fp,
        "total_failures": tp + fn,
        "alarms": tp + fp,
    }


def threshold_grid(
    start: float = THRESHOLD_MIN, stop: float = THRESHOLD_MAX, step: float = THRESHOLD_STEP
) -> np.ndarray:
    n_steps = int(round((stop - start) / step))
    return np.round(start + step * np.arange(n_steps + 1), 2)


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Метрики для кожного τ із сітки, за замовчуванням від 0.10 до 0.80 з кроком 0.01."""
    grid = threshold_grid() if thresholds is None else np.asarray(thresholds, dtype=float)
    return pd.DataFrame([compute_threshold_metrics(y_true, y_prob, t) for t in grid])


def threshold_comparison_table(
    y_true: Sequence[int], y_prob: Sequence[float], selected: float
) -> pd.DataFrame:
    """Робочі пороги 0.25 і 0.50 поруч із порогом, вибраним на повзунку."""
    points = [
        (f"τ_п = {TAU_WARN:.2f} (попередження)", TAU_WARN),
        (f"τ = {TAU_CRIT:.2f} (критичний, робочий)", TAU_CRIT),
        (f"τ = {selected:.2f} (обраний)", selected),
    ]
    rows = []
    for label, thr in points:
        m = compute_threshold_metrics(y_true, y_prob, thr)
        rows.append(
            {
                "Поріг": label,
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1']:.4f}",
                "Виявлено аварій (TP)": m["tp"],
                "Хибні тривоги (FP)": m["fp"],
                "Пропущені аварії (FN)": m["fn"],
            }
        )
    return pd.DataFrame(rows)


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9, length=0)
    ax.set_axisbelow(True)


def plot_failure_type_distribution(distribution_df: pd.DataFrame):
    """Кількість відмов кожного типу горизонтальними стовпцями, найчастіший тип угорі."""
    data = distribution_df.sort_values("Кількість", ascending=True)
    labels = [f"{c} · {n}" for c, n in zip(data["Код"], data["Тип відмови"])]
    counts = data["Кількість"].to_numpy()
    shares = data["Частка серед типів відмов (%)"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 3.4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, height=0.56, color=_SERIES_BLUE, edgecolor="white", linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=_INK_PRIMARY, fontsize=9.5)
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_xlim(0, max(counts.max() * 1.28, 1))
    ax.set_xlabel("Кількість записів із відмовою даного типу", color=_INK_SECONDARY, fontsize=9)

    for y, count, share in zip(y_pos, counts, shares):
        ax.text(count + counts.max() * 0.015, y, f"{count}  ({share:.1f} %)",
                va="center", ha="left", fontsize=9, color=_INK_PRIMARY)

    ax.set_title("Розподіл типів відмов у наборі AI4I 2020", loc="left",
                 fontsize=11, color=_INK_PRIMARY, pad=10)
    fig.tight_layout()
    return fig


def _mark_reference_thresholds(ax) -> None:
    """Малює робочі пороги τ_п і τ пунктиром, із підписами над графіком."""
    for thr, label in ((TAU_WARN, f"τ_п = {TAU_WARN:.2f}"), (TAU_CRIT, f"τ = {TAU_CRIT:.2f}")):
        ax.axvline(thr, color=_INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.text(thr, 1.015, label, ha="center", va="bottom", fontsize=8, color=_INK_MUTED,
                transform=ax.get_xaxis_transform())


def _dodge_label_positions(values: Sequence[float], min_gap: float) -> np.ndarray:
    """Розсуває підписи на кінцях ліній по вертикалі, щоб вони не накладалися."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    placed = values.copy()
    for prev, cur in zip(order[:-1], order[1:]):
        if placed[cur] - placed[prev] < min_gap:
            placed[cur] = placed[prev] + min_gap
    return placed


def plot_threshold_curves(sweep_df: pd.DataFrame, selected: float):
    """Precision, Recall і F1 залежно від τ; вибраний поріг показано вертикальною лінією."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = sweep_df["threshold"].to_numpy()
    series = (
        ("precision", "Precision", _SERIES_BLUE),
        ("recall", "Recall", _SERIES_ORANGE),
        ("f1", "F1-Score", _SERIES_AQUA),
    )
    current = sweep_df.iloc[int(np.argmin(np.abs(x - selected)))]

    _style_axes(ax)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    _mark_reference_thresholds(ax)
    ax.axvline(selected, color=_INK_PRIMARY, linewidth=1.2, zorder=2)

    end_positions = _dodge_label_positions([sweep_df[key].iloc[-1] for key, _, _ in series], 0.06)
    for (key, label, color), y_label in zip(series, end_positions):
        y = sweep_df[key].to_numpy()
        ax.plot(x, y, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter([selected], [current[key]], s=64, color=color, edgecolor="white",
                   linewidth=2, zorder=4)
        ax.text(x[-1] + 0.008, y_label, label, va="center", ha="left", fontsize=8.5,
                color=_INK_SECONDARY)

    ax.set_xlim(x[0], x[-1] + 0.075)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Поріг прийняття рішення τ", color=_INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Значення метрики", color=_INK_SECONDARY, fontsize=9)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=3)
    ax.set_title(f"Метрики якості залежно від порогу (обрано τ = {selected:.2f})",
                 loc="left", fontsize=11, color=_INK_PRIMARY, pad=22)
    fig.tight_layout()
    return fig


def plot_error_tradeoff(sweep_df: pd.DataFrame, selected: float):
    """FN і FP залежно від τ: що вищий поріг, то менше хибних тривог і більше пропущених аварій."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = sweep_df["threshold"].to_numpy()
    series = (
        ("fn", "Пропущені аварії (FN)", _SERIES_RED),
        ("fp", "Хибні тривоги (FP)", _SERIES_VIOLET),
    )
    current = sweep_df.iloc[int(np.argmin(np.abs(x - selected)))]
    y_max = float(max(sweep_df["fn"].max(), sweep_df["fp"].max(), 1))

    _style_axes(ax)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    _mark_reference_thresholds(ax)
    ax.axvline(selected, color=_INK_PRIMARY, linewidth=1.2, zorder=2)

    for key, label, color in series:
        y = sweep_df[key].to_numpy()
        ax.plot(x, y, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter([selected], [current[key]], s=64, color=color, edgecolor="white",
                   linewidth=2, zorder=4)
        ax.annotate(f"{int(current[key])}", (selected, current[key]), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, color=_INK_PRIMARY)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xlabel("Поріг прийняття рішення τ", color=_INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Кількість записів", color=_INK_SECONDARY, fontsize=9)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=1)
    ax.set_title("Компроміс помилок: пропущені аварії vs хибні тривоги",
                 loc="left", fontsize=11, color=_INK_PRIMARY, pad=22)
    fig.tight_layout()
    return fig
