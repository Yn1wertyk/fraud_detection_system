import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Колонки нового датасету:
# transaction_id, user_id, amount, transaction_type, merchant_category,
# country, hour, device_risk_score, ip_risk_score, is_fraud

TRANSACTION_TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
MERCHANT_CATEGORIES = ["Clothing", "Electronics", "Food", "Gambling", "Grocery",
                        "Travel", "Utilities", "Other"]
HIGH_RISK_COUNTRIES = {"TR", "NG", "IN", "RU", "CN", "PK"}


def build_features(df: pd.DataFrame, single: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Побудова ознак для виявлення фінансових махінацій.

    Вхідні колонки датасету:
        transaction_id, user_id, amount, transaction_type, merchant_category,
        country, hour, device_risk_score, ip_risk_score, is_fraud (опційно)

    Args:
        df: DataFrame з транзакціями
        single: True якщо обробляємо одну транзакцію для інференсу

    Returns:
        Tuple[features_df, target_series | None]
    """
    df = df.copy()

    # --- Базові числові ознаки ---
    df["amount_log"] = np.log1p(df["amount"])
    df["high_amount_flag"] = (df["amount"] > 1000).astype(int)

    # Часові ознаки з колонки hour (0–23)
    df["hour"] = df["hour"].astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

    # --- Ризикові скори ---
    df["device_risk_score"] = df["device_risk_score"].astype(float)
    df["ip_risk_score"] = df["ip_risk_score"].astype(float)
    df["combined_risk_score"] = (df["device_risk_score"] + df["ip_risk_score"]) / 2
    df["high_device_risk"] = (df["device_risk_score"] > 0.7).astype(int)
    df["high_ip_risk"] = (df["ip_risk_score"] > 0.7).astype(int)
    df["both_high_risk"] = ((df["device_risk_score"] > 0.7) & (df["ip_risk_score"] > 0.7)).astype(int)

    # --- Географічні ознаки ---
    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # --- One-hot encoding для transaction_type ---
    for tx_type in TRANSACTION_TYPES:
        col = f"tx_type_{tx_type.lower()}"
        df[col] = (df["transaction_type"] == tx_type).astype(int)

    # --- One-hot encoding для merchant_category ---
    for cat in MERCHANT_CATEGORIES:
        col = f"merchant_{cat.lower()}"
        df[col] = (df["merchant_category"] == cat).astype(int)

    # --- Агрегати по user_id (rolling на основі рядків) ---
    df_sorted = df.sort_index()  # зберігаємо оригінальний порядок (датасет вже відсортовано)

    df["user_tx_count"] = df.groupby("user_id").cumcount()  # кількість попередніх транзакцій

    df["user_amount_mean"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["user_amount_std"] = df.groupby("user_id")["amount"].transform(
        lambda x: x.shift(1).expanding().std()
    )
    df["user_device_risk_mean"] = df.groupby("user_id")["device_risk_score"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Відхилення суми від середнього по користувачу
    df["amount_vs_user_mean"] = df["amount"] / (df["user_amount_mean"].fillna(df["amount"]) + 1e-8)

    # --- Агрегати по merchant_category ---
    df["merchant_tx_count"] = df.groupby("merchant_category").cumcount()
    df["merchant_amount_mean"] = df.groupby("merchant_category")["amount"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Вибір фічей для моделі
    feature_cols = [
        # Числові
        "amount", "amount_log", "hour",
        # Бінарні флаги
        "high_amount_flag", "is_night", "is_business_hours",
        "high_device_risk", "high_ip_risk", "both_high_risk",
        "is_high_risk_country",
        # Ризикові скори
        "device_risk_score", "ip_risk_score", "combined_risk_score",
        # Агрегати по юзеру
        "user_tx_count", "user_amount_mean", "user_amount_std",
        "user_device_risk_mean", "amount_vs_user_mean",
        # Агрегати по мерчанту
        "merchant_tx_count", "merchant_amount_mean",
        # One-hot: transaction_type
        *[f"tx_type_{t.lower()}" for t in TRANSACTION_TYPES],
        # One-hot: merchant_category
        *[f"merchant_{c.lower()}" for c in MERCHANT_CATEGORIES],
    ]

    # Заповнення NaN
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df["is_fraud"] if ("is_fraud" in df.columns and not single) else None

    return X, y


def prepare_single_transaction(tx_dict: dict, historical_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Підготовка однієї транзакції для інференсу.

    Args:
        tx_dict: словник з полями транзакції
        historical_data: DataFrame з попередніми транзакціями цього юзера (опційно)

    Returns:
        DataFrame з однією строкою готових ознак
    """
    tx_df = pd.DataFrame([tx_dict])

    if historical_data is not None and not historical_data.empty:
        combined_df = pd.concat([historical_data, tx_df], ignore_index=True)
        X, _ = build_features(combined_df, single=True)
        return X.iloc[-1:].copy()

    X, _ = build_features(tx_df, single=True)
    return X
