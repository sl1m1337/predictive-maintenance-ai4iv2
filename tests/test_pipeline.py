"""
Тести конвеєра: фізичні ознаки, валідація, пороги ризику, протокол, SHAP і пакетний звіт.
Мережа не потрібна, бо модель вчиться на синтетичному журналі у форматі AI4I 2020.
"""
import codecs
import io
import itertools
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import FEATURE_COLUMNS, TARGET_COLUMN, TAU_CRIT, TAU_WARN
from src.data_loader import normalize_column_names, validate_telemetry_schema, split_dataset
from src.feature_engineering import compute_physical_features, build_single_observation
from src.model_engine import classify_risk_level, train_primary_classifier
from src.explainer import DiagnosticExplainer
from src.failure_analytics import compute_threshold_metrics
from src.report_generator import (
    BASE_RECOMMENDATIONS,
    RISK_LEVEL_SHORT_UA,
    SENSOR_COLUMNS,
    SENSOR_LIMITS,
    VERDICT_FAILURE,
    VERDICT_NORMAL,
    batch_report_to_csv,
    build_batch_report,
    build_diagnostic_protocol,
    check_sensor_limits,
    extract_positive_class_shap,
    protocol_to_json,
    render_protocol_text,
    summarize_batch_report,
)

KYIV_TZ = timezone(timedelta(hours=2))


def _synthetic_ai4i_frame(n_rows: int = 1600, seed: int = 2020) -> pd.DataFrame:
    """
    Синтетичні дані з оригінальними назвами стовпців AI4I 2020.
    Відмова ставиться за тими самими фізичними критеріями HDF, PWF, OSF і TWF, що в датасеті.
    """
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n_rows, p=[0.6, 0.3, 0.1])
    air = np.clip(rng.normal(300.0, 2.0, n_rows), 295.3, 304.5).round(1)
    proc = np.clip(air + 10.0 + rng.normal(0.0, 1.0, n_rows), 305.7, 313.8).round(1)
    rpm = np.clip(rng.normal(1540.0, 180.0, n_rows), 1168, 2886).round()
    torque = np.clip(rng.normal(40.0, 10.0, n_rows), 3.8, 76.6).round(1)
    wear = rng.integers(0, 254, n_rows)
    power = torque * rpm * 2.0 * np.pi / 60.0
    osf_limit = np.select([types == "L", types == "M"], [11000.0, 12000.0], 13000.0)
    failure = (
        (((proc - air) < 8.6) & (rpm < 1380))
        | (power < 3500.0)
        | (power > 9000.0)
        | (wear * torque > osf_limit)
        | (wear >= 240)
    )
    return pd.DataFrame(
        {
            "Type": types,
            "Air temperature [K]": air,
            "Process temperature [K]": proc,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure.astype(int),
        }
    )


@pytest.fixture(scope="module")
def trained_system():
    """SMOTE + LightGBM + SHAP на синтетиці, один раз на весь модуль."""
    raw = normalize_column_names(_synthetic_ai4i_frame())
    validate_telemetry_schema(raw)
    X = compute_physical_features(raw)
    y = raw[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_primary_classifier(X_train, y_train)
    return model, DiagnosticExplainer(model), X_test, y_test


def test_physical_features_calculation():
    """ΔT і P = M * ω для простих вхідних значень."""
    obs = build_single_observation(
        type_code=0,
        air_temp=300.0,
        proc_temp=310.0,
        rpm=1500,
        torque=40.0,
        tool_wear=100,
    )
    assert obs["Temp_diff_K"].iloc[0] == 10.0
    expected_power = round(40.0 * (1500 * 2.0 * np.pi / 60.0), 2)
    assert np.isclose(obs["Power_W"].iloc[0], expected_power, atol=1e-2)


def test_risk_classification_thresholds():
    """По одній імовірності на кожну зону: нижче 0.25, між 0.25 і 0.50, вище 0.50."""
    code_norm, _ = classify_risk_level(0.0011)
    code_warn, _ = classify_risk_level(0.3200)
    code_crit, _ = classify_risk_level(0.7964)

    assert code_norm == "normal"
    assert code_warn == "warning"
    assert code_crit == "critical"


def test_schema_validation_missing_column():
    """Без обов'язкових стовпців валідація має впасти з ValueError."""
    broken_df = pd.DataFrame({"Type": ["L"], "Air_temperature_K": [300.0]})
    with pytest.raises(ValueError):
        validate_telemetry_schema(broken_df)


def test_stratified_split_preserves_ratio():
    """Стратифікація 80/20 зберігає частку відмов: 4 % від 100 записів дає рівно одну відмову в тесті."""
    raw = pd.DataFrame(
        {
            "Type": ["L"] * 96 + ["M"] * 4,
            "Air_temperature_K": [300.0] * 100,
            "Process_temperature_K": [310.0] * 100,
            "Rotational_speed_rpm": [1500] * 100,
            "Torque_Nm": [40.0] * 100,
            "Tool_wear_min": [100] * 100,
            "Machine_failure": [0] * 96 + [1] * 4,
        }
    )
    norm = normalize_column_names(raw)
    X = compute_physical_features(norm)
    y = norm["Machine_failure"]

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_test.sum() == 1


def test_diagnostic_protocol_generation():
    """Вміст протоколу, експорт у TXT і JSON та помилки на некоректних вхідних даних."""
    obs = build_single_observation(
        type_code=0, air_temp=300.0, proc_temp=308.5, rpm=1350, torque=60.0, tool_wear=210
    )
    shap_values = np.array([0.05, -0.10, 0.20, 0.90, -1.40, 2.10, 0.60, -0.30])
    moment = datetime(2026, 11, 19, 16, 57, 3, tzinfo=KYIV_TZ)

    protocol = build_diagnostic_protocol(
        obs, 0.8731, shap_values, base_value=-2.5, inspected_at=moment, equipment_id="ВФ-07"
    )

    assert protocol["protocol_id"] == "PD-20261119-165703"
    assert protocol["inspected_at_display"] == "19.11.2026 16:57:03 (UTC+02:00)"
    assert protocol["equipment"] == {"id": "ВФ-07", "type_code": "L", "type_label": "L (Базова)"}

    assert [s["feature"] for s in protocol["sensors"]] == SENSOR_COLUMNS
    assert all(s["within_limits"] for s in protocol["sensors"])
    assert protocol["limit_violations"] == []
    assert protocol["derived"]["temp_diff_k"] == pytest.approx(8.5)
    assert protocol["derived"]["power_w"] == pytest.approx(60.0 * 1350 * 2.0 * np.pi / 60.0, abs=1e-2)

    # 0.8731 вище τ = 0.50, тож очікуємо critical
    assessment = protocol["assessment"]
    assert assessment["risk_code"] == "critical"
    assert assessment["probability_pct"] == pytest.approx(87.31)
    assert (assessment["tau_warn"], assessment["tau_crit"]) == (TAU_WARN, TAU_CRIT)

    top = protocol["explanation"]["top_factors"]
    assert [f["feature"] for f in top] == ["Tool_wear_min", "Torque_Nm", "Rotational_speed_rpm"]
    assert [f["rank"] for f in top] == [1, 2, 3]
    assert [f["direction"] for f in top] == ["підвищує ризик", "знижує ризик", "підвищує ризик"]
    assert protocol["explanation"]["model_output_log_odds"] == pytest.approx(-2.5 + shap_values.sum())

    # знос 210 хв дає TWF; ΔT 8.5 K при 1350 об/хв дає HDF; 210 * 60 = 12 600 > 11 000 хв·Нм для серії L дає OSF
    statuses = {m["code"]: m["status"] for m in protocol["mechanisms"]}
    assert {code for code, status in statuses.items() if status == "triggered"} == {"TWF", "HDF", "OSF"}
    assert statuses["RNF"] == "info"

    recs = protocol["recommendations"]
    assert recs[0] == BASE_RECOMMENDATIONS["critical"]
    assert len(recs) >= 4 and len(recs) == len(set(recs))
    assert any("інструмент" in r for r in recs)

    text = render_protocol_text(protocol)
    for fragment in (
        "ПРОТОКОЛ ТЕХНІЧНОЇ ДІАГНОСТИКИ",
        "19.11.2026 16:57:03 (UTC+02:00)",
        "ВФ-07",
        "1. ПАРАМЕТРИ ДАТЧИКІВ",
        "ΔT",
        "8.50 K",
        "8482.30 Вт",
        "87.31 %",
        "КРИТИЧНИЙ РИЗИК ВІДМОВИ",
        "4. ТОП-3 ФАКТОРИ ВПЛИВУ НА ПРОГНОЗ (SHAP)",
        "6. ІНЖЕНЕРНІ РЕКОМЕНДАЦІЇ",
    ):
        assert fragment in text
    assert text.encode("utf-8").decode("utf-8") == text

    restored = json.loads(protocol_to_json(protocol))
    assert restored["assessment"]["risk_code"] == "critical"
    assert len(restored["explanation"]["top_factors"]) == 3

    with pytest.raises(ValueError):
        build_diagnostic_protocol(obs, 1.2, shap_values, base_value=0.0)
    with pytest.raises(ValueError):
        build_diagnostic_protocol(obs, 0.5, shap_values[:5], base_value=0.0)


def test_shap_additivity(trained_system):
    """E[f(x)] + Σφ_i має дорівнювати виходу моделі, а сигмоїда від нього збігатися з predict_proba."""
    model, explainer, X_test, _ = trained_system
    rows = [X_test.iloc[[i]] for i in range(5)]
    rows.append(build_single_observation(2, 302.0, 311.5, 1420, 52.5, 180))

    for row in rows:
        assert explainer.check_additivity(row, tol=1e-3)

        shap_row, base_val = explainer.explain_instance(row)
        assert shap_row.shape == (len(FEATURE_COLUMNS),)
        log_odds = base_val + float(np.sum(shap_row))
        prob = float(model.predict_proba(row)[0, 1])
        assert 1.0 / (1.0 + np.exp(-log_odds)) == pytest.approx(prob, abs=1e-4)


def test_sensor_boundary_values(trained_system):
    """Усі комбінації меж повзунків, значення поза межами та граничні випадки порогів ризику."""
    model, _, _, _ = trained_system
    bounds = [SENSOR_LIMITS[col] for col in SENSOR_COLUMNS]

    observations = []
    for type_code, air, proc, rpm, torque, wear in itertools.product(range(3), *bounds):
        obs = build_single_observation(type_code, air, proc, rpm, torque, wear)
        assert obs["Temp_diff_K"].iloc[0] == pytest.approx(round(proc - air, 2))
        assert obs["Power_W"].iloc[0] == pytest.approx(torque * rpm * 2.0 * np.pi / 60.0, abs=1e-2)
        assert np.isfinite(obs.to_numpy(dtype=float)).all()
        assert check_sensor_limits(obs) == []
        observations.append(obs)

    batch = pd.concat(observations, ignore_index=True)
    assert len(batch) == 3 * 2 ** len(SENSOR_COLUMNS)
    assert batch["Temp_diff_K"].min() == pytest.approx(0.0)
    assert batch["Temp_diff_K"].max() == pytest.approx(20.0)
    assert batch["Power_W"].min() == pytest.approx(3.0 * 1150 * 2.0 * np.pi / 60.0, abs=1e-2)
    assert batch["Power_W"].max() == pytest.approx(80.0 * 2900 * 2.0 * np.pi / 60.0, abs=1e-2)

    probs = model.predict_proba(batch)[:, 1]
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    for i in (0, len(batch) - 1):
        protocol = build_diagnostic_protocol(batch.iloc[[i]], probs[i], np.zeros(8), 0.0)
        assert protocol["assessment"]["risk_code"] == classify_risk_level(probs[i])[0]

    outside = build_single_observation(0, 294.9, 315.1, 3000, 2.5, 261)
    assert {v["feature"] for v in check_sensor_limits(outside)} == set(SENSOR_COLUMNS)
    broken = observations[0].copy()
    broken.loc[0, "Torque_Nm"] = np.nan
    assert [v["feature"] for v in check_sensor_limits(broken)] == ["Torque_Nm"]

    # пороги включні: p рівно на τ уже належить до вищого рівня, а nextafter дає значення трохи нижче τ
    assert classify_risk_level(0.0)[0] == "normal"
    assert classify_risk_level(np.nextafter(TAU_WARN, 0.0))[0] == "normal"
    assert classify_risk_level(TAU_WARN)[0] == "warning"
    assert classify_risk_level(np.nextafter(TAU_CRIT, 0.0))[0] == "warning"
    assert classify_risk_level(TAU_CRIT)[0] == "critical"
    assert classify_risk_level(1.0)[0] == "critical"


def test_batch_prediction_correctness(trained_system):
    """Пакетний прогноз, метрики симулятора порогу проти sklearn і зведений CSV-звіт."""
    model, explainer, X_test, y_test = trained_system
    y_true = y_test.to_numpy()
    probs = model.predict_proba(X_test)[:, 1]

    assert probs.shape == (len(X_test),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    head = X_test.head(25)
    row_probs = np.array([model.predict_proba(head.iloc[[i]])[0, 1] for i in range(len(head))])
    assert np.allclose(row_probs, probs[:25], atol=1e-10)

    assert probs[y_true == 1].mean() > probs[y_true == 0].mean()
    assert roc_auc_score(y_true, probs) > 0.85

    m =compute_threshold_metrics(y_true, probs, TAU_CRIT)
    y_pred = (probs >= TAU_CRIT).astype(int)
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert m["tp"] + m["fn"] == int(y_true.sum())
    assert m["missed_failures"] == m["fn"]

    batch =X_test.head(40)
    batch_probs = probs[:40]
    shap_matrix = extract_positive_class_shap(explainer.explainer.shap_values(batch))
    report = build_batch_report(
        batch, batch_probs, shap_matrix=shap_matrix,
        analyzed_at=datetime(2026, 11, 19, 17, 0, 0, tzinfo=KYIV_TZ),
    )
    assert len(report) == 40
    assert report["Індекс запису"].tolist() == batch.index.tolist()
    expected_verdicts = np.where(batch_probs >= TAU_CRIT, VERDICT_FAILURE, VERDICT_NORMAL)
    assert report["Вердикт системи"].tolist() == expected_verdicts.tolist()
    expected_levels = [RISK_LEVEL_SHORT_UA[classify_risk_level(p)[0]] for p in batch_probs]
    assert report["Рівень критичності"].tolist() == expected_levels

    summary = summarize_batch_report(report)
    assert summary["critical"] + summary["warning"] + summary["normal"] == 40
    assert summary["failure_verdicts"] == summary["critical"]

    csv_bytes = batch_report_to_csv(report)
    assert csv_bytes.startswith(codecs.BOM_UTF8)
    restored = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
    assert restored.shape == report.shape
    assert restored.columns.tolist() == report.columns.tolist()
    assert np.allclose(restored["Імовірність відмови (%)"], np.round(batch_probs * 100.0, 2))
