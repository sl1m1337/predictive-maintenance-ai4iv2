"""Фізичні ознаки поверх сирих показів датчиків: перепад температур і механічна потужність."""
import numpy as np
import pandas as pd
from config import TYPE_MAPPING, FEATURE_COLUMNS


def compute_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Кодує тип обладнання, додає ΔT і P та повертає стовпці в порядку FEATURE_COLUMNS."""
    featured = df.copy()

    if "Type_encoded" not in featured.columns:
        featured["Type_encoded"] = featured["Type"].map(TYPE_MAPPING)
        if featured["Type_encoded"].isnull().any():
            raise ValueError("Невідоме значення категорії обладнання у стовпці 'Type'.")

    # ΔT = T_процесу - T_повітря. Малий перепад означає, що тепло погано відводиться (критерій HDF)
    featured["Temp_diff_K"] = (
        featured["Process_temperature_K"] - featured["Air_temperature_K"]
    )

    # P = M * ω, де ω = 2π * n / 60 переводить об/хв у рад/с, тому P виходить у ватах (критерій PWF)
    angular_velocity = featured["Rotational_speed_rpm"] * 2.0 * np.pi / 60.0
    featured["Power_W"] = featured["Torque_Nm"] * angular_velocity

    return featured[FEATURE_COLUMNS]


def build_single_observation(
    type_code: int,
    air_temp: float,
    proc_temp: float,
    rpm: int,
    torque: float,
    tool_wear: int,
) -> pd.DataFrame:
    """Один рядок ознак зі значень повзунків у вкладці 1; ΔT і P округлюються до сотих."""
    temp_diff = round(proc_temp - air_temp, 2)
    power_w = round(torque * (rpm * 2.0 * np.pi / 60.0), 2)
    return pd.DataFrame(
        [
            {
                "Type_encoded": type_code,
                "Air_temperature_K": air_temp,
                "Process_temperature_K": proc_temp,
                "Rotational_speed_rpm": rpm,
                "Torque_Nm": torque,
                "Tool_wear_min": tool_wear,
                "Temp_diff_K": temp_diff,
                "Power_W": power_w,
            }
        ]
    )
