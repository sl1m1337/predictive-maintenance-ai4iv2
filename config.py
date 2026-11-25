"""Налаштування проєкту: звідки брати дані, які ознаки використовуємо, параметри LightGBM і пороги ризику."""

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Пороги з підрозділу 3.4. TAU_CRIT = 0.50 збігається з межею, за якою працює predict(),
# тому за ним рахуються метрики у вкладці 3 і вердикт у вкладці 2.
# TAU_WARN дає проміжну зону, щоб попередити про ризик ще до критичного рівня.
TAU_WARN = 0.25
TAU_CRIT = 0.50

TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

RAW_REQUIRED_COLUMNS = [
    "Type",
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]

FEATURE_COLUMNS = [
    "Type_encoded",
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Temp_diff_K",
    "Power_W",
]

FEATURE_NAMES_UA = [
    "Тип обладнання",
    "Темп. повітря",
    "Темп. процесу",
    "Швидкість обертів",
    "Крутний момент",
    "Знос інструменту",
    "Перепад температур",
    "Мех. потужність",
]

TARGET_COLUMN = "Machine_failure"

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
