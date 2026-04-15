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

FEATURE_COLS = [
    "amount",
    "amount_log",
    "hour",
    "is_night",
    "is_business_hours",
    "is_high_risk_country",
    "device_risk_score",
    "ip_risk_score",
    "combined_risk_score",
    *[f"tx_type_{t.lower()}" for t in TRANSACTION_TYPES],
    *[f"merchant_{c.lower()}" for c in MERCHANT_CATEGORIES],
]


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базові ознаки, що не залежать від інших рядків.
    Безпечні для використання і під час тренування, і під час інференсу.
    """
    df = df.copy()

    # Числові
    df["amount_log"] = np.log1p(df["amount"].astype(float))

    # Часові
    df["hour"] = df["hour"].astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

    # Ризикові скори — залишаємо як є (вони вже несуть сигнал)
    df["device_risk_score"] = df["device_risk_score"].astype(float)
    df["ip_risk_score"] = df["ip_risk_score"].astype(float)
    # Добуток — нелінійна взаємодія, корисніша за просте середнє
    df["combined_risk_score"] = df["device_risk_score"] * df["ip_risk_score"]

    # Географія
    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # One-hot: transaction_type
    for tx_type in TRANSACTION_TYPES:
        df[f"tx_type_{tx_type.lower()}"] = (df["transaction_type"] == tx_type).astype(int)

    # One-hot: merchant_category
    for cat in MERCHANT_CATEGORIES:
        df[f"merchant_{cat.lower()}"] = (df["merchant_category"] == cat).astype(int)

    return df


def build_features(
    df: pd.DataFrame,
    single: bool = False,
    user_stats: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Побудова ознак для виявлення фінансових махінацій.

    Під час тренування user_stats=None — агрегати рахуються по train-фолду.
    Під час інференсу user_stats передається із збереженої статистики.

    Args:
        df: DataFrame з транзакціями
        single: True якщо обробляємо одну транзакцію для інференсу
        user_stats: DataFrame з агрегатами по user_id (user_id, user_amount_mean,
                    user_amount_std, user_tx_count, user_device_risk_mean).
                    Якщо None — рахується по df (тільки для тренування).

    Returns:
        Tuple[features_df, target_series | None]
    """
    df = _base_features(df).reset_index(drop=True)

    # Приводимо user_id до рядка в df, щоб merge завжди працював
    # незалежно від того, чи прийшов user_id як int чи str
    df["user_id"] = df["user_id"].astype(str)

    if user_stats is None:
        # Тренування: рахуємо по наявних рядках
        stats = (
            df.groupby("user_id")
            .agg(
                user_amount_mean=("amount", "mean"),
                user_amount_std=("amount", "std"),
                user_tx_count=("amount", "count"),
                user_device_risk_mean=("device_risk_score", "mean"),
            )
            .reset_index()
        )
        stats["user_amount_std"] = stats["user_amount_std"].fillna(0)
        stats["user_id"] = stats["user_id"].astype(str)
        df = df.merge(stats, on="user_id", how="left")
    else:
        user_stats = user_stats.copy()
        user_stats["user_id"] = user_stats["user_id"].astype(str)
        df = df.merge(user_stats, on="user_id", how="left")

    df["user_tx_count"] = df["user_tx_count"].fillna(1)
    df["user_amount_mean"] = df["user_amount_mean"].fillna(df["amount"])
    df["user_amount_std"] = df["user_amount_std"].fillna(0)
    df["user_device_risk_mean"] = df["user_device_risk_mean"].fillna(df["device_risk_score"])

    # Відхилення суми від середнього по юзеру (z-score-подібне)
    df["amount_vs_user_mean"] = (df["amount"] - df["user_amount_mean"]) / (
        df["user_amount_std"] + 1e-8
    )

    feature_cols = FEATURE_COLS + [
        "user_tx_count",
        "user_amount_mean",
        "user_amount_std",
        "user_device_risk_mean",
        "amount_vs_user_mean",
    ]

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df["is_fraud"] if ("is_fraud" in df.columns and not single) else None

    return X, y


def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рахує агрегати по user_id для збереження разом із моделлю.
    Використовується при інференсі для нових транзакцій.
    """
    stats = (
        df.groupby("user_id")
        .agg(
            user_amount_mean=("amount", "mean"),
            user_amount_std=("amount", "std"),
            user_tx_count=("amount", "count"),
            user_device_risk_mean=("device_risk_score", "mean"),
        )
        .reset_index()
    )
    stats["user_amount_std"] = stats["user_amount_std"].fillna(0)
    stats["user_id"] = stats["user_id"].astype(str)
    return stats


def prepare_single_transaction(tx_dict: dict) -> pd.DataFrame:
    """
    Підготовка однієї транзакції для інференсу (без історії юзера).
    Агрегати по юзеру заповнюються нейтральними значеннями.
    """
    tx_df = pd.DataFrame([tx_dict])
    X, _ = build_features(tx_df, single=True)
    return X
