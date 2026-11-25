"""
Streamlit-застосунок: діагностика одного агрегату з протоколом, пакетний аналіз CSV,
метрики моделей і аналітика відмов із симулятором порогу.
"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import TARGET_COLUMN, TAU_CRIT, TAU_WARN
from src.data_loader import load_raw_dataset, split_dataset
from src.feature_engineering import compute_physical_features, build_single_observation
from src.model_engine import train_primary_classifier, classify_risk_level
from src.explainer import DiagnosticExplainer
from src.report_generator import (
    build_diagnostic_protocol,
    render_protocol_text,
    protocol_to_json,
    protocol_file_name,
    extract_positive_class_shap,
    build_batch_report,
    summarize_batch_report,
    batch_report_to_csv,
    batch_report_file_name,
)
from src.failure_analytics import (
    THRESHOLD_MIN,
    THRESHOLD_MAX,
    THRESHOLD_STEP,
    compute_failure_type_distribution,
    summarize_failure_modes,
    failure_rate_by_equipment_type,
    compute_threshold_metrics,
    threshold_sweep,
    threshold_comparison_table,
    plot_failure_type_distribution,
    plot_threshold_curves,
    plot_error_tradeoff,
)

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")


def render_markdown_table(df: pd.DataFrame) -> str:
    """Перетворює DataFrame на просту Markdown-таблицю, без індексу і HTML."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)


@st.cache_resource
def initialize_system():
    """Дані, ознаки, split, LightGBM зі SMOTE лише на train і SHAP. Завдяки кешу виконується один раз."""
    raw_df = load_raw_dataset()
    X = compute_physical_features(raw_df)
    y = raw_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_primary_classifier(X_train, y_train)
    diagnostic_explainer = DiagnosticExplainer(model)
    return model, diagnostic_explainer, X_test


@st.cache_resource(show_spinner="Підготовка аналітики відмов…")
def initialize_analytics(_model, _X_test):
    """
    Дані для вкладки 4 на тій самій тестовій вибірці (2000 записів), що й у вкладці 3.
    y_test береться з raw_df за індексами X_test, тому split не робиться вдруге.
    """
    raw_df = load_raw_dataset()
    y_test = raw_df.loc[_X_test.index, TARGET_COLUMN].astype(int).to_numpy()
    y_prob = _model.predict_proba(_X_test)[:, 1]
    return {
        "distribution": compute_failure_type_distribution(raw_df),
        "summary": summarize_failure_modes(raw_df),
        "by_type": failure_rate_by_equipment_type(raw_df),
        "y_test": y_test,
        "y_prob": y_prob,
        "sweep": threshold_sweep(y_test, y_prob),
    }


@st.cache_data(show_spinner=False)
def compute_batch_shap(_diagnostic_explainer, batch_df: pd.DataFrame) -> np.ndarray:
    """SHAP-матриця для пакета; кешується за вмістом batch_df."""
    return extract_positive_class_shap(_diagnostic_explainer.explainer.shap_values(batch_df))


def _metric_delta(value: float, baseline: float, fmt: str):
    """Різниця з робочим τ = 0.50 для st.metric; None, коли різниці немає, щоб не малювати стрілку."""
    diff = value - baseline
    return None if abs(diff) < 1e-12 else format(diff, fmt)


def main():
    st.title("Інтелектуальна система предиктивної діагностики промислового обладнання")
    st.markdown(
        "Модель: **LightGBM + SMOTE + Explainable AI (SHAP)** | Дані телеметрії: **AI4I 2020**"
    )

    model, diagnostic_explainer, X_test = initialize_system()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Інтерактивна діагностика агрегату",
            "2. Пакетний аналіз телеметрії (CSV)",
            "3. Метрики якості моделі",
            "4. Аналітика відмов та симулятор порогів",
        ]
    )

    with tab1:
        st.sidebar.header("Параметри датчиків верстата")
        eq_type = st.sidebar.radio(
            "Тип обладнання (серія)", ["L (Базова)", "M (Середня)", "H (Висока)"]
        )
        type_val = {"L (Базова)": 0, "M (Середня)": 1, "H (Висока)": 2}[eq_type]

        air_temp = st.sidebar.slider("Температура повітря (K)", 295.0, 305.0, 300.0, 0.1)
        proc_temp = st.sidebar.slider("Температура процесу (K)", 305.0, 315.0, 310.0, 0.1)
        rpm = st.sidebar.slider("Швидкість обертів (об/хв)", 1150, 2900, 1500, 10)
        torque = st.sidebar.slider("Крутний момент (Нм)", 3.0, 80.0, 40.0, 0.5)
        tool_wear = st.sidebar.slider("Знос інструменту (хв)", 0, 260, 100, 1)

        input_df = build_single_observation(
            type_val, air_temp, proc_temp, rpm, torque, tool_wear
        )
        temp_diff = float(input_df["Temp_diff_K"].iloc[0])
        power_w = float(input_df["Power_W"].iloc[0])

        st.subheader("Поточні та розрахункові фізичні показники")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Оберти шпинделя", f"{rpm} об/хв")
        c2.metric("Крутний момент", f"{torque} Нм")
        c3.metric("Перепад температур (ΔT)", f"{temp_diff} K")
        c4.metric("Механічна потужність (P)", f"{power_w} Вт")

        prob = float(model.predict_proba(input_df)[0][1])
        status_code, status_msg = classify_risk_level(prob)

        st.divider()
        col_res, col_shap = st.columns([1, 1.4])
        with col_res:
            st.subheader("Результат діагностики")
            st.metric("Імовірність відмови обладнання", f"{prob * 100:.2f}%")
            if status_code == "critical":
                st.error(status_msg)
            elif status_code == "warning":
                st.warning(status_msg)
            else:
                st.success(status_msg)

        with col_shap:
            st.subheader("Локальна SHAP-інтерпретація рішення")
            fig = diagnostic_explainer.plot_local_contributions(input_df)
            st.pyplot(fig)

        st.divider()
        st.subheader("Протокол технічної діагностики")
        equipment_id = st.text_input(
            "Інвентарний номер агрегату (необов'язково)",
            value="",
            key="protocol_equipment_id",
        )
        shap_row, base_val = diagnostic_explainer.explain_instance(input_df)
        protocol = build_diagnostic_protocol(
            input_df,
            prob,
            shap_row,
            base_val,
            inspected_at=datetime.now().astimezone(),
            equipment_id=equipment_id or None,
        )
        protocol_text = render_protocol_text(protocol)
        top_names = ", ".join(f["name"] for f in protocol["explanation"]["top_factors"])
        st.caption(
            f"Протокол № {protocol['protocol_id']} · топ-3 фактори SHAP: {top_names} · "
            f"рекомендацій: {len(protocol['recommendations'])}"
        )
        dl_txt, dl_json = st.columns(2)
        dl_txt.download_button(
            "Завантажити протокол (TXT)",
            data=protocol_text.encode("utf-8"),
            file_name=protocol_file_name(protocol, "txt"),
            mime="text/plain",
            key="download_protocol_txt",
        )
        dl_json.download_button(
            "Завантажити протокол (JSON)",
            data=protocol_to_json(protocol).encode("utf-8"),
            file_name=protocol_file_name(protocol, "json"),
            mime="application/json",
            key="download_protocol_json",
        )
        with st.expander("Попередній перегляд протоколу", expanded=False):
            st.code(protocol_text, language=None)

    with tab2:
        st.subheader("Пакетний аналіз журналу телеметрії (вибірка 15 записів AI4I 2020)")
        uploaded_csv = st.file_uploader(
            "Завантажити власний журнал телеметрії у форматі CSV (опціонально)",
            type=["csv"],
        )

        if uploaded_csv is not None:
            try:
                user_raw = load_raw_dataset(uploaded_csv)
                batch_df = compute_physical_features(user_raw)
                st.info(f"Завантажено та перевірено {len(batch_df)} записів із файлу користувача.")
            except Exception as exc:
                st.error(f"Помилка валідації CSV-файлу: {exc}")
                batch_df = X_test.head(15).copy()
        else:
            batch_df = X_test.head(15).copy()

        probs = model.predict_proba(batch_df)[:, 1]
        out_df = pd.DataFrame(
            {
                "Оберти (об/хв)": batch_df["Rotational_speed_rpm"].values,
                "Момент (Нм)": batch_df["Torque_Nm"].values,
                "Знос (хв)": batch_df["Tool_wear_min"].values,
                "ΔT (K)": np.round(batch_df["Temp_diff_K"].values, 2),
                "Потужність (Вт)": np.round(batch_df["Power_W"].values, 1),
                "Імовірність відмови (%)": np.round(probs * 100, 2),
                "Вердикт системи": np.where(probs >= TAU_CRIT, "ВІДМОВА (Ризик)", "Норма"),
            }
        )
        st.markdown(render_markdown_table(out_df))

        analyzed_at = datetime.now().astimezone()
        batch_report = build_batch_report(
            batch_df,
            probs,
            shap_matrix=compute_batch_shap(diagnostic_explainer, batch_df),
            analyzed_at=analyzed_at,
        )
        batch_summary = summarize_batch_report(batch_report)
        st.caption(
            f"Зведений звіт: {batch_summary['total']} записів — критичний ризик: "
            f"{batch_summary['critical']}, підвищений ризик: {batch_summary['warning']}, "
            f"норма: {batch_summary['normal']}. Для кожного запису звіт містить рівень "
            "критичності, виконані критерії механізмів відмов AI4I, контроль діапазонів датчиків "
            "і рекомендовану дію, а для записів із ризиком — головний фактор за SHAP."
        )
        st.download_button(
            "Завантажити зведений CSV-звіт",
            data=batch_report_to_csv(batch_report),
            file_name=batch_report_file_name(analyzed_at),
            mime="text/csv",
            key="download_batch_report_csv",
        )

    with tab3:
        st.subheader("Порівняльні метрики ансамблевих моделей (тестова вибірка: 2000 записів)")
        metrics_data = pd.DataFrame(
            [
                {
                    "Модель": "LightGBM (Обрана)",
                    "Accuracy": "0.9825",
                    "Precision": "0.7037",
                    "Recall": "0.8382",
                    "F1-Score": "0.7651",
                    "ROC-AUC": "0.9734",
                },
                {
                    "Модель": "XGBoost",
                    "Accuracy": "0.9740",
                    "Precision": "0.5800",
                    "Recall": "0.8529",
                    "F1-Score": "0.6905",
                    "ROC-AUC": "0.9774",
                },
                {
                    "Модель": "CatBoost",
                    "Accuracy": "0.9655",
                    "Precision": "0.4958",
                    "Recall": "0.8676",
                    "F1-Score": "0.6310",
                    "ROC-AUC": "0.9773",
                },
            ]
        )
        st.markdown(render_markdown_table(metrics_data))

    with tab4:
        analytics = initialize_analytics(model, X_test)

        st.subheader("Розподіл типів відмов у наборі даних AI4I 2020")
        summary = analytics["summary"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Записів у наборі", f"{summary['total_records']:,}".replace(",", " "))
        s2.metric("Відмов (Machine failure)", f"{summary['machine_failures']}")
        s3.metric("Частка відмов", f"{summary['failure_rate_pct']:.2f}%")
        s4.metric("Записів із кількома типами відмов", f"{summary['multi_mode_records']}")

        col_dist, col_series = st.columns([1.4, 1])
        with col_dist:
            fig_dist = plot_failure_type_distribution(analytics["distribution"])
            st.pyplot(fig_dist)
            plt.close(fig_dist)
        with col_series:
            st.markdown("**Частота відмов за серіями обладнання**")
            st.markdown(render_markdown_table(analytics["by_type"]))
            st.caption(
                f"Сума подій за типами ({summary['mode_events']}) відрізняється від кількості "
                f"відмов ({summary['machine_failures']}): {summary['multi_mode_records']} записів "
                f"мають кілька типів одночасно, {summary['modes_without_failure']} записів мають "
                f"позначку типу без загальної відмови, {summary['failures_without_mode']} відмов "
                "не мають позначки типу."
            )
        st.markdown(render_markdown_table(analytics["distribution"]))

        st.divider()

        st.subheader("Симулятор порогу прийняття рішень (тестова вибірка)")
        y_test_sim = analytics["y_test"]
        y_prob_sim = analytics["y_prob"]
        st.caption(
            f"Тестова вибірка: {len(y_test_sim)} записів, фактичних відмов: {int(y_test_sim.sum())}. "
            f"Робочі пороги системи не змінюються: τ_п = {TAU_WARN:.2f}, τ = {TAU_CRIT:.2f}. "
            "Симулятор показує, як змінилися б метрики за іншого порогу."
        )
        threshold = st.slider(
            "Поріг прийняття рішення τ (запис класифікується як відмова, якщо p ≥ τ)",
            min_value=THRESHOLD_MIN,
            max_value=THRESHOLD_MAX,
            value=float(TAU_CRIT),
            step=THRESHOLD_STEP,
            format="%.2f",
            key="threshold_simulator",
        )
        threshold = round(float(threshold), 2)
        current = compute_threshold_metrics(y_test_sim, y_prob_sim, threshold)
        baseline = compute_threshold_metrics(y_test_sim, y_prob_sim, TAU_CRIT)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Precision",
            f"{current['precision']:.4f}",
            delta=_metric_delta(current["precision"], baseline["precision"], "+.4f"),
        )
        m2.metric(
            "Recall",
            f"{current['recall']:.4f}",
            delta=_metric_delta(current["recall"], baseline["recall"], "+.4f"),
        )
        m3.metric(
            "F1-Score",
            f"{current['f1']:.4f}",
            delta=_metric_delta(current["f1"], baseline["f1"], "+.4f"),
        )
        m4.metric(
            "Пропущені аварії (FN)",
            f"{current['fn']} з {current['total_failures']}",
            delta=_metric_delta(current["fn"], baseline["fn"], "+d"),
            delta_color="inverse",
        )
        st.caption(
            f"За τ = {threshold:.2f}: виявлено аварій (TP) — {current['tp']}, "
            f"хибних тривог (FP) — {current['fp']}, усього спрацювань — {current['alarms']}. "
            "Відхилення наведено відносно робочого порогу τ = 0.50."
        )

        col_curves, col_errors = st.columns(2)
        with col_curves:
            fig_curves = plot_threshold_curves(analytics["sweep"], threshold)
            st.pyplot(fig_curves)
            plt.close(fig_curves)
        with col_errors:
            fig_errors = plot_error_tradeoff(analytics["sweep"], threshold)
            st.pyplot(fig_errors)
            plt.close(fig_errors)

        st.markdown(
            render_markdown_table(threshold_comparison_table(y_test_sim, y_prob_sim, threshold))
        )


if __name__ == "__main__":
    main()
