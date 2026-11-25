"""
Протокол діагностики одного агрегату (TXT і JSON, вкладка 1) та зведений CSV-звіт для пакета (вкладка 2).
Власних порогів тут немає: рівень ризику завжди береться з classify_risk_level.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, FEATURE_NAMES_UA, TAU_CRIT, TAU_WARN, TYPE_MAPPING
from src.model_engine import classify_risk_level

SYSTEM_NAME = "Predictive Maintenance AI v2.0 (Enterprise Edition)"
MODEL_DESCRIPTION = "LightGBM + SMOTE (1:1) + SHAP TreeExplainer"
PROTOCOL_SCHEMA_VERSION = "2.0"

FEATURE_LABELS: Dict[str, str] = dict(zip(FEATURE_COLUMNS, FEATURE_NAMES_UA))

FEATURE_UNITS: Dict[str, str] = {
    "Type_encoded": "",
    "Air_temperature_K": "K",
    "Process_temperature_K": "K",
    "Rotational_speed_rpm": "об/хв",
    "Torque_Nm": "Нм",
    "Tool_wear_min": "хв",
    "Temp_diff_K": "K",
    "Power_W": "Вт",
}

FEATURE_DECIMALS: Dict[str, int] = {
    "Type_encoded": 0,
    "Air_temperature_K": 1,
    "Process_temperature_K": 1,
    "Rotational_speed_rpm": 0,
    "Torque_Nm": 1,
    "Tool_wear_min": 0,
    "Temp_diff_K": 2,
    "Power_W": 2,
}

TYPE_LETTERS: Dict[int, str] = {code: letter for letter, code in TYPE_MAPPING.items()}
TYPE_LABELS_UA: Dict[int, str] = {0: "L (Базова)", 1: "M (Середня)", 2: "H (Висока)"}

RISK_LEVEL_UA: Dict[str, str] = {
    "normal": "ШТАТНИЙ РЕЖИМ",
    "warning": "ПІДВИЩЕНИЙ РИЗИК",
    "critical": "КРИТИЧНИЙ РИЗИК ВІДМОВИ",
}
RISK_LEVEL_SHORT_UA: Dict[str, str] = {
    "normal": "Норма",
    "warning": "Підвищений ризик",
    "critical": "Критичний ризик",
}
BATCH_ACTIONS_UA: Dict[str, str] = {
    "normal": "Експлуатація за регламентом ТО",
    "warning": "Посилений моніторинг, ТО у найближче технологічне вікно",
    "critical": "Негайна зупинка та позапланове ТО",
}

# У вкладці 2 вердикт бінарний: рахується тільки за TAU_CRIT, без проміжного рівня
VERDICT_FAILURE = "ВІДМОВА (Ризик)"
VERDICT_NORMAL = "Норма"

# Межі такі самі, як у повзунків вкладки 1, і з невеликим запасом покривають
# діапазон значень у AI4I 2020. За ними модель уже екстраполює.
SENSOR_LIMITS: Dict[str, tuple] = {
    "Air_temperature_K": (295.0, 305.0),
    "Process_temperature_K": (305.0, 315.0),
    "Rotational_speed_rpm": (1150.0, 2900.0),
    "Torque_Nm": (3.0, 80.0),
    "Tool_wear_min": (0.0, 260.0),
}
SENSOR_COLUMNS: List[str] = list(SENSOR_LIMITS)

# Критерії відмов взято з опису датасету AI4I 2020 (S. Matzka, 2020)
TWF_WEAR_ZONE_MIN = (200.0, 240.0)       # хв, у цьому проміжку інструмент міняють або він ламається
HDF_TEMP_DIFF_MAX_K = 8.6                # HDF: ΔT нижче 8.6 K ...
HDF_RPM_MAX = 1380.0                     # ... і водночас оберти нижче 1380 об/хв
PWF_POWER_RANGE_W = (3500.0, 9000.0)     # Вт, поза цим діапазоном буде відмова за потужністю
OSF_STRAIN_LIMITS: Dict[int, float] = {0: 11000.0, 1: 12000.0, 2: 13000.0}  # знос * момент, хв·Нм, для L/M/H
PRE_ALERT_MARGIN = 0.10                  # коли до межі менше 10 %, ставимо статус «НАБЛИЖЕННЯ»

MECHANISM_STATUS_UA: Dict[str, str] = {
    "triggered": "ПЕРЕВИЩЕНО",
    "approaching": "НАБЛИЖЕННЯ",
    "ok": "НОРМА",
    "info": "ІНФО",
}

BASE_RECOMMENDATIONS: Dict[str, str] = {
    "critical": (
        "Негайно зупинити агрегат і провести позапланове технічне обслуговування; "
        "повторний пуск — лише після усунення причин і повторної діагностики."
    ),
    "warning": (
        "Посилити моніторинг: скоротити інтервал контролю параметрів і запланувати "
        "технічне обслуговування на найближче технологічне вікно."
    ),
    "normal": (
        "Продовжувати експлуатацію в штатному режимі; технічне обслуговування — "
        "згідно з чинним регламентом."
    ),
}

FACTOR_ACTIONS: Dict[str, str] = {
    "Tool_wear_min": "Перевірити стан ріжучого інструменту та підготувати його заміну.",
    "Torque_Nm": (
        "Перевірити механічне навантаження на шпиндель: режими різання, "
        "стан заготовки та підшипникових вузлів."
    ),
    "Rotational_speed_rpm": (
        "Перевірити режим обертання та привід шпинделя; уникати тривалої роботи "
        "на нетипових обертах."
    ),
    "Power_W": "Перевірити електропривод і систему живлення агрегату.",
    "Temp_diff_K": "Перевірити систему охолодження та тепловідведення (вентилятор, радіатор, подачу МОР).",
    "Air_temperature_K": "Перевірити мікроклімат цеху та роботу вентиляції.",
    "Process_temperature_K": "Перевірити температурний режим процесу та подачу мастильно-охолоджувальної рідини.",
    "Type_encoded": (
        "Врахувати серію обладнання: агрегати нижчої серії мають менший запас "
        "за перевантаженням (межа OSF для серії L — 11 000 хв·Нм)."
    ),
}

ObservationLike = Union[pd.DataFrame, pd.Series, Mapping[str, float]]


def _as_series(observation: ObservationLike) -> pd.Series:
    """Робить Series з DataFrame на один рядок, Series або dict і перевіряє, що є всі ознаки."""
    if isinstance(observation, pd.DataFrame):
        if len(observation) != 1:
            raise ValueError("Протокол формується для одного агрегату: очікується 1 рядок даних.")
        series = observation.iloc[0]
    elif isinstance(observation, pd.Series):
        series = observation
    else:
        series = pd.Series(dict(observation))

    missing = [col for col in FEATURE_COLUMNS if col not in series.index]
    if missing:
        raise ValueError(f"У спостереженні відсутні ознаки: {missing}")
    return series


def _type_code(value: float) -> int:
    code = int(round(float(value)))
    if code not in TYPE_LETTERS:
        raise ValueError(f"Невідомий код типу обладнання: {value}")
    return code


def _format_number(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def format_feature_value(feature: str, value: float) -> str:
    """Значення ознаки з одиницями для звітів; тип обладнання виводиться літерою."""
    if feature == "Type_encoded":
        return TYPE_LETTERS[_type_code(value)]
    text = _format_number(value, FEATURE_DECIMALS.get(feature, 2))
    unit = FEATURE_UNITS.get(feature, "")
    return f"{text} {unit}".strip()


def format_timestamp(moment: datetime) -> str:
    """Дата для протоколу у вигляді 19.11.2026 16:57:03 (UTC+02:00)."""
    text = moment.strftime("%d.%m.%Y %H:%M:%S")
    offset = moment.utcoffset()
    if offset is not None:
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        hours, minutes = divmod(abs(total_minutes), 60)
        text += f" (UTC{sign}{hours:02d}:{minutes:02d})"
    return text


def extract_positive_class_shap(shap_output) -> np.ndarray:
    """
    Зводить вихід TreeExplainer.shap_values до матриці (n_samples, n_features) для класу «відмова».
    Залежно від версії shap там може бути список, 2D- або 3D-масив.
    """
    if isinstance(shap_output, list):
        matrix = np.asarray(shap_output[1], dtype=float)
    else:
        matrix = np.asarray(shap_output, dtype=float)
        if matrix.ndim == 3:
            matrix = matrix[:, :, 1]
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def check_sensor_limits(observation: ObservationLike) -> List[Dict[str, object]]:
    """
    Список показників, які виходять за SENSOR_LIMITS, відсутні або не є числом.
    Межі включні, тож значення рівно на межі вважається нормальним.
    """
    series = observation if isinstance(observation, pd.Series) else (
        observation.iloc[0] if isinstance(observation, pd.DataFrame) else pd.Series(dict(observation))
    )
    violations: List[Dict[str, object]] = []
    for feature, (low, high) in SENSOR_LIMITS.items():
        if feature not in series.index:
            violations.append(
                {"feature": feature, "name": FEATURE_LABELS[feature], "value": None,
                 "min": low, "max": high, "unit": FEATURE_UNITS[feature], "reason": "відсутнє значення"}
            )
            continue
        value = float(series[feature])
        if not np.isfinite(value):
            reason = "некоректне значення"
        elif value < low:
            reason = "нижче допустимого діапазону"
        elif value > high:
            reason = "вище допустимого діапазону"
        else:
            continue
        violations.append(
            {"feature": feature, "name": FEATURE_LABELS[feature], "value": value,
             "min": low, "max": high, "unit": FEATURE_UNITS[feature], "reason": reason}
        )
    return violations


def format_limit_violations(violations: Sequence[Mapping[str, object]]) -> str:
    """Порушення діапазонів одним рядком для звіту або «—», якщо їх немає."""
    if not violations:
        return "—"
    parts = []
    for item in violations:
        if item["value"] is None or not np.isfinite(float(item["value"])):
            parts.append(f"{item['name']}: {item['reason']}")
        else:
            decimals = FEATURE_DECIMALS.get(str(item["feature"]), 2)
            parts.append(
                f"{item['name']} = {_format_number(item['value'], decimals)} {item['unit']} "
                f"(допустимо {item['min']:g}–{item['max']:g})"
            )
    return "; ".join(parts)


def evaluate_failure_mechanisms(observation: ObservationLike) -> List[Dict[str, object]]:
    """
    Перевіряє п'ять механізмів відмов AI4I 2020 за їхніми фізичними критеріями.
    Статус: 'triggered', 'approaching' (до межі менше 10 %), 'ok' або 'info' для RNF.
    """
    s = _as_series(observation)
    type_code = _type_code(s["Type_encoded"])
    wear = float(s["Tool_wear_min"])
    torque = float(s["Torque_Nm"])
    rpm = float(s["Rotational_speed_rpm"])
    temp_diff = float(s["Temp_diff_K"])
    power = float(s["Power_W"])
    margin = PRE_ALERT_MARGIN

    results: List[Dict[str, object]] = []

    # TWF: після 200 хв зносу інструмент у зоні, де його міняють або він ламається
    twf_low, twf_high = TWF_WEAR_ZONE_MIN
    if wear >= twf_low:
        status = "triggered"
        detail = f"знос {wear:.0f} хв у зоні заміни/відмови інструменту {twf_low:.0f}–{twf_high:.0f} хв"
    elif wear >= twf_low * (1.0 - margin):
        status = "approaching"
        detail = f"знос {wear:.0f} хв; до зони ризику {twf_low:.0f} хв залишилось {twf_low - wear:.0f} хв"
    else:
        status, detail = "ok", f"знос {wear:.0f} хв < {twf_low * (1.0 - margin):.0f} хв"
    results.append({"code": "TWF", "name": "Знос інструменту", "status": status, "detail": detail})

    # HDF: малий ΔT разом із низькими обертами означає, що тепло погано відводиться
    if temp_diff < HDF_TEMP_DIFF_MAX_K and rpm < HDF_RPM_MAX:
        status = "triggered"
        detail = f"ΔT = {temp_diff:.2f} K < {HDF_TEMP_DIFF_MAX_K} K за {rpm:.0f} < {HDF_RPM_MAX:.0f} об/хв"
    elif temp_diff < HDF_TEMP_DIFF_MAX_K * (1.0 + margin) and rpm < HDF_RPM_MAX * (1.0 + margin):
        status = "approaching"
        detail = f"ΔT = {temp_diff:.2f} K та {rpm:.0f} об/хв — у 10 % від межі критерію HDF"
    else:
        status = "ok"
        detail = f"ΔT = {temp_diff:.2f} K, {rpm:.0f} об/хв — критерій HDF не виконується"
    results.append({"code": "HDF", "name": "Порушення тепловідведення", "status": status, "detail": detail})

    # PWF: потужність P = M * ω має лишатися в межах 3500–9000 Вт
    p_low, p_high = PWF_POWER_RANGE_W
    if power < p_low or power > p_high:
        status = "triggered"
        detail = f"P = {power:.0f} Вт поза діапазоном {p_low:.0f}–{p_high:.0f} Вт"
    elif power < p_low * (1.0 + margin) or power > p_high * (1.0 - margin):
        status = "approaching"
        detail = f"P = {power:.0f} Вт — у 10 % від межі діапазону {p_low:.0f}–{p_high:.0f} Вт"
    else:
        status, detail = "ok", f"P = {power:.0f} Вт у діапазоні {p_low:.0f}–{p_high:.0f} Вт"
    results.append({"code": "PWF", "name": "Відмова за потужністю", "status": status, "detail": detail})

    # OSF: добуток зносу на момент показує перевантаження; ліміт залежить від серії
    strain = wear * torque
    limit = OSF_STRAIN_LIMITS[type_code]
    share = strain / limit * 100.0
    letter = TYPE_LETTERS[type_code]
    if strain > limit:
        status = "triggered"
        detail = f"знос × момент = {strain:,.0f} хв·Нм > {limit:,.0f} хв·Нм (серія {letter})"
    elif strain >= limit * (1.0 - margin):
        status = "approaching"
        detail = f"знос × момент = {strain:,.0f} хв·Нм — {share:.0f} % від межі {limit:,.0f} хв·Нм (серія {letter})"
    else:
        status = "ok"
        detail = f"знос × момент = {strain:,.0f} хв·Нм — {share:.0f} % від межі {limit:,.0f} хв·Нм (серія {letter})"
    results.append(
        {"code": "OSF", "name": "Перевантаження", "status": status, "detail": detail.replace(",", " ")}
    )

    # RNF: випадкова відмова, від параметрів процесу не залежить, тому лише інформуємо
    results.append(
        {
            "code": "RNF",
            "name": "Випадкова відмова",
            "status": "info",
            "detail": "≈0,1 % незалежно від параметрів; фізичного предиктора немає — контроль регламентним ТО",
        }
    )
    return results


def rank_shap_factors(
    shap_values: Sequence[float], observation: ObservationLike, top_n: int = 3
) -> List[Dict[str, object]]:
    """top_n ознак із найбільшим |φ| (у log-odds) і напрямок, у який вони тягнуть ризик."""
    s = _as_series(observation)
    phi = np.asarray(shap_values, dtype=float).ravel()
    if phi.shape[0] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Очікується {len(FEATURE_COLUMNS)} значень SHAP, отримано {phi.shape[0]}."
        )
    order = np.argsort(-np.abs(phi), kind="stable")[:top_n]
    factors = []
    for rank, idx in enumerate(order, start=1):
        feature = FEATURE_COLUMNS[idx]
        value = float(s[feature])
        factors.append(
            {
                "rank": rank,
                "feature": feature,
                "name": FEATURE_LABELS[feature],
                "value": value,
                "value_display": format_feature_value(feature, value),
                "shap": float(phi[idx]),
                "direction": (
                    "підвищує ризик" if phi[idx] > 0
                    else "знижує ризик" if phi[idx] < 0
                    else "нейтральний вплив"
                ),
            }
        )
    return factors


def generate_engineering_recommendations(
    risk_code: str,
    mechanisms: Sequence[Mapping[str, object]],
    top_factors: Sequence[Mapping[str, object]],
    observation: Optional[ObservationLike] = None,
) -> List[str]:
    """
    Збирає рекомендації: базову дію за рівнем ризику, дії за спрацьованими механізмами AI4I
    і перевірки для SHAP-факторів, що підвищують ризик. Дублікати прибираються.
    """
    if risk_code not in BASE_RECOMMENDATIONS:
        raise ValueError(f"Невідомий рівень критичності: {risk_code}")

    recs: List[str] = [BASE_RECOMMENDATIONS[risk_code]]
    s = _as_series(observation) if observation is not None else None
    by_code = {str(m["code"]): m for m in mechanisms}

    def status(code: str) -> str:
        return str(by_code.get(code, {}).get("status", "ok"))

    if status("TWF") == "triggered":
        recs.append(
            "Замінити ріжучий інструмент: знос перебуває в зоні 200–240 хв, "
            "де в AI4I 2020 відбувається заміна або відмова інструменту (TWF)."
        )
    elif status("TWF") == "approaching":
        recs.append("Підготувати заміну інструменту: знос наближається до зони ризику TWF (200 хв).")

    if status("HDF") == "triggered":
        recs.append(
            "Відновити тепловідведення (HDF): перевірити вентилятор, радіатор і подачу МОР; "
            "не експлуатувати агрегат на знижених обертах до нормалізації ΔT ≥ 8.6 K."
        )
    elif status("HDF") == "approaching":
        recs.append("Контролювати тепловідведення: ΔT та оберти наближаються до межі критерію HDF.")

    if status("PWF") == "triggered":
        recs.append(
            "Скоригувати режим (PWF): потужність процесу поза діапазоном 3500–9000 Вт — "
            "змінити поєднання обертів і моменту, перевірити електропривод."
        )
    elif status("PWF") == "approaching":
        recs.append("Потужність процесу наближається до межі діапазону 3500–9000 Вт — контролювати режим.")

    if status("OSF") == "triggered":
        recs.append(
            "Усунути перевантаження (OSF): зменшити крутний момент/подачу або замінити інструмент — "
            "добуток зносу на момент перевищує межу для цієї серії обладнання."
        )
    elif status("OSF") == "approaching":
        recs.append("Запланувати заміну інструменту до досягнення межі перевантаження OSF.")

    if risk_code != "normal":
        for factor in top_factors:
            if float(factor["shap"]) > 0:
                action = FACTOR_ACTIONS.get(str(factor["feature"]))
                if action:
                    recs.append(f"{action} [SHAP: «{factor['name']}», φ = {float(factor['shap']):+.3f}]")

    if s is not None and check_sensor_limits(s):
        recs.append(
            "Перевірити справність датчиків: показники поза допустимим діапазоном — "
            "прогноз моделі в цій області є екстраполяцією."
        )

    unique: List[str] = []
    for rec in recs:
        if rec not in unique:
            unique.append(rec)
    return unique


def build_diagnostic_protocol(
    observation: ObservationLike,
    probability: float,
    shap_values: Sequence[float],
    base_value: float,
    inspected_at: Optional[datetime] = None,
    equipment_id: Optional[str] = None,
) -> Dict[str, object]:
    """
    Протокол діагностики одного агрегату у вигляді dict: датчики, ΔT і P, імовірність,
    рівень ризику, топ-3 SHAP і рекомендації. shap_values і base_value беруться з explain_instance.
    """
    s = _as_series(observation)
    p = float(probability)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"Імовірність відмови має бути в діапазоні [0, 1], отримано {probability}.")

    moment = inspected_at or datetime.now().astimezone()
    type_code = _type_code(s["Type_encoded"])
    risk_code, risk_message = classify_risk_level(p)

    phi = np.asarray(shap_values, dtype=float).ravel()
    top_factors = rank_shap_factors(phi, s, top_n=3)
    mechanisms = evaluate_failure_mechanisms(s)
    violations = check_sensor_limits(s)
    recommendations = generate_engineering_recommendations(risk_code, mechanisms, top_factors, s)

    sensors = []
    for feature in SENSOR_COLUMNS:
        low, high = SENSOR_LIMITS[feature]
        value = float(s[feature])
        sensors.append(
            {
                "feature": feature,
                "name": FEATURE_LABELS[feature],
                "value": value,
                "value_display": format_feature_value(feature, value),
                "unit": FEATURE_UNITS[feature],
                "min": low,
                "max": high,
                "within_limits": bool(low <= value <= high),
            }
        )

    base = float(base_value)
    shap_sum = float(np.sum(phi))
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": f"PD-{moment:%Y%m%d-%H%M%S}",
        "system": SYSTEM_NAME,
        "model": MODEL_DESCRIPTION,
        "inspected_at": moment.isoformat(timespec="seconds"),
        "inspected_at_display": format_timestamp(moment),
        "equipment": {
            "id": (equipment_id or "").strip() or "—",
            "type_code": TYPE_LETTERS[type_code],
            "type_label": TYPE_LABELS_UA[type_code],
        },
        "sensors": sensors,
        "derived": {
            "temp_diff_k": float(s["Temp_diff_K"]),
            "power_w": float(s["Power_W"]),
        },
        "assessment": {
            "probability": p,
            "probability_pct": round(p * 100.0, 2),
            "risk_code": risk_code,
            "risk_level": RISK_LEVEL_UA[risk_code],
            "conclusion": risk_message,
            "tau_warn": float(TAU_WARN),
            "tau_crit": float(TAU_CRIT),
        },
        "explanation": {
            "method": "SHAP TreeExplainer (значення Шеплі, простір log-odds)",
            "base_value": base,
            "shap_sum": shap_sum,
            "model_output_log_odds": base + shap_sum,
            "top_factors": top_factors,
        },
        "mechanisms": mechanisms,
        "limit_violations": violations,
        "recommendations": recommendations,
    }


def _dotted(label: str, value: str, width: int = 40) -> str:
    dots = "." * max(2, width - len(label))
    return f"   {label} {dots} {value}"


def render_protocol_text(protocol: Mapping[str, object]) -> str:
    """Протокол звичайним текстом для завантаження в TXT."""
    rule = "=" * 78
    thin = "-" * 78
    a = protocol["assessment"]
    e = protocol["explanation"]
    eq = protocol["equipment"]
    d = protocol["derived"]

    lines = [
        rule,
        "ПРОТОКОЛ ТЕХНІЧНОЇ ДІАГНОСТИКИ АГРЕГАТУ",
        f"№ {protocol['protocol_id']}",
        rule,
        _dotted("Дата та час перевірки", str(protocol["inspected_at_display"])),
        _dotted("Система діагностики", str(protocol["system"])),
        _dotted("Модель", str(protocol["model"])),
        _dotted("Інвентарний номер агрегату", str(eq["id"])),
        _dotted("Тип обладнання (серія)", str(eq["type_label"])),
        "",
        "1. ПАРАМЕТРИ ДАТЧИКІВ",
        thin,
    ]
    for sensor in protocol["sensors"]:
        mark = "у межах" if sensor["within_limits"] else "ПОЗА МЕЖАМИ"
        lines.append(
            _dotted(sensor["name"], f"{sensor['value_display']:<14} [{mark}: "
                    f"{sensor['min']:g}–{sensor['max']:g} {sensor['unit']}]")
        )

    lines += [
        "",
        "2. РОЗРАХУНКОВІ ФІЗИЧНІ ПОКАЗНИКИ",
        thin,
        _dotted("Перепад температур ΔT = T_проц − T_пов", f"{d['temp_diff_k']:.2f} K"),
        _dotted("Механічна потужність P = M · ω", f"{d['power_w']:.2f} Вт"),
        "",
        "3. РЕЗУЛЬТАТ ОЦІНЮВАННЯ ТЕХНІЧНОГО СТАНУ",
        thin,
        _dotted("Імовірність відмови", f"{a['probability_pct']:.2f} %"),
        _dotted("Рівень критичності", str(a["risk_level"])),
        _dotted("Пороги класифікації", f"τ_п = {a['tau_warn']:.2f}; τ = {a['tau_crit']:.2f}"),
        f"   Висновок: {a['conclusion']}",
        "",
        "4. ТОП-3 ФАКТОРИ ВПЛИВУ НА ПРОГНОЗ (SHAP)",
        thin,
    ]
    for f in e["top_factors"]:
        lines.append(
            f"   {f['rank']}. {f['name']} = {f['value_display']}; "
            f"φ = {f['shap']:+.4f} ({f['direction']})"
        )
    lines += [
        f"   Базове значення E[f(x)] = {e['base_value']:+.4f}; Σφ = {e['shap_sum']:+.4f}; "
        f"f(x) = {e['model_output_log_odds']:+.4f} (log-odds)",
        "",
        "5. КОНТРОЛЬ МЕХАНІЗМІВ ВІДМОВ (критерії набору AI4I 2020)",
        thin,
    ]
    for m in protocol["mechanisms"]:
        status = MECHANISM_STATUS_UA[str(m["status"])]
        lines.append(f"   [{status:<10}] {m['code']} — {m['name']}: {m['detail']}")

    lines += ["", "6. ІНЖЕНЕРНІ РЕКОМЕНДАЦІЇ", thin]
    for i, rec in enumerate(protocol["recommendations"], start=1):
        lines.append(f"   {i}. {rec}")

    lines += [
        "",
        rule,
        "Протокол сформовано автоматично системою предиктивної діагностики.",
        "Остаточне рішення щодо експлуатації агрегату приймає відповідальний інженер.",
        "",
        "Відповідальний інженер: ______________________   Підпис: ____________",
        rule,
        "",
    ]
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Тип {type(obj)} не серіалізується у JSON")


def protocol_to_json(protocol: Mapping[str, object]) -> str:
    """Той самий протокол у JSON, наприклад для імпорту в CMMS/ERP."""
    return json.dumps(protocol, ensure_ascii=False, indent=2, default=_json_default)


def protocol_file_name(protocol: Mapping[str, object], extension: str = "txt") -> str:
    return f"protocol_{protocol['protocol_id']}.{extension.lstrip('.')}"


def build_batch_report(
    batch_features: pd.DataFrame,
    probabilities: Sequence[float],
    shap_matrix: Optional[np.ndarray] = None,
    analyzed_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Таблиця пакетного аналізу, по рядку на запис. Вердикт рахується як у вкладці 2
    (p ≥ TAU_CRIT), а рівень ризику за двома порогами з classify_risk_level.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in batch_features.columns]
    if missing:
        raise ValueError(f"У пакеті даних відсутні ознаки: {missing}")

    probs = np.asarray(probabilities, dtype=float).ravel()
    if probs.shape[0] != len(batch_features):
        raise ValueError("Кількість імовірностей не збігається з кількістю записів пакета.")
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Імовірності відмови мають бути в діапазоні [0, 1].")

    if shap_matrix is not None:
        shap_matrix = np.asarray(shap_matrix, dtype=float)
        if shap_matrix.shape != (len(batch_features), len(FEATURE_COLUMNS)):
            raise ValueError("Розмірність матриці SHAP не відповідає пакету даних.")

    moment = analyzed_at or datetime.now().astimezone()
    stamp = format_timestamp(moment)
    rows = []
    for pos, (idx, row) in enumerate(batch_features[FEATURE_COLUMNS].iterrows()):
        p = float(probs[pos])
        risk_code, _ = classify_risk_level(p)
        triggered = [
            str(m["code"]) for m in evaluate_failure_mechanisms(row) if m["status"] == "triggered"
        ]
        record = {
            "№": pos + 1,
            "Індекс запису": idx,
            "Тип обладнання": TYPE_LETTERS[_type_code(row["Type_encoded"])],
            "Темп. повітря (K)": round(float(row["Air_temperature_K"]), 2),
            "Темп. процесу (K)": round(float(row["Process_temperature_K"]), 2),
            "Оберти (об/хв)": round(float(row["Rotational_speed_rpm"]), 1),
            "Момент (Нм)": round(float(row["Torque_Nm"]), 2),
            "Знос (хв)": round(float(row["Tool_wear_min"]), 1),
            "ΔT (K)": round(float(row["Temp_diff_K"]), 2),
            "Потужність (Вт)": round(float(row["Power_W"]), 1),
            "Імовірність відмови (%)": round(p * 100.0, 2),
            "Рівень критичності": RISK_LEVEL_SHORT_UA[risk_code],
            "Вердикт системи": VERDICT_FAILURE if p >= TAU_CRIT else VERDICT_NORMAL,
        }
        if shap_matrix is not None:
            # головний фактор пишемо тільки для warning/critical і тільки якщо він підвищує ризик (φ > 0)
            phi = shap_matrix[pos]
            top_idx = int(np.argmax(phi))
            record["Головний фактор ризику (SHAP)"] = (
                f"{FEATURE_NAMES_UA[top_idx]} (φ = {phi[top_idx]:+.3f})"
                if risk_code != "normal" and phi[top_idx] > 0
                else "—"
            )
        record["Механізми відмов (критерії AI4I)"] = ", ".join(triggered) if triggered else "—"
        record["Поза допустимим діапазоном"] = format_limit_violations(check_sensor_limits(row))
        record["Рекомендована дія"] = BATCH_ACTIONS_UA[risk_code]
        record["Дата аналізу"] = stamp
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_batch_report(report_df: pd.DataFrame) -> Dict[str, object]:
    """Скільки записів на кожному рівні ризику, плюс середня й максимальна імовірність."""
    levels = report_df["Рівень критичності"] if len(report_df) else pd.Series(dtype=str)
    probs = report_df["Імовірність відмови (%)"] if len(report_df) else pd.Series(dtype=float)
    return {
        "total": int(len(report_df)),
        "critical": int((levels == RISK_LEVEL_SHORT_UA["critical"]).sum()),
        "warning": int((levels == RISK_LEVEL_SHORT_UA["warning"]).sum()),
        "normal": int((levels == RISK_LEVEL_SHORT_UA["normal"]).sum()),
        "failure_verdicts": int((report_df.get("Вердикт системи", pd.Series(dtype=str)) == VERDICT_FAILURE).sum()),
        "mean_probability_pct": round(float(probs.mean()), 2) if len(probs) else 0.0,
        "max_probability_pct": round(float(probs.max()), 2) if len(probs) else 0.0,
    }


def batch_report_to_csv(report_df: pd.DataFrame, sep: str = ",") -> bytes:
    """CSV у UTF-8 з BOM, бо без нього Excel ламає кирилицю."""
    return report_df.to_csv(index=False, sep=sep).encode("utf-8-sig")


def batch_report_file_name(analyzed_at: Optional[datetime] = None) -> str:
    moment = analyzed_at or datetime.now().astimezone()
    return f"batch_report_{moment:%Y%m%d_%H%M%S}.csv"
