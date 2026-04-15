import pandas as pd
import numpy as np
from typing import Tuple, Optional

TRANSACTION_TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
MERCHANT_CATEGORIES = [
    "Clothing", "Electronics", "Food", "Gambling",
    "Grocery", "Travel", "Utilities", "Other"
]

HIGH_RISK_COUNTRIES = {"TR", "NG", "IN", "RU", "CN", "PK"}

BASE_FEATURES = [
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

USER_FEATURES = [
    "user_tx_count",
    "user_amount_mean",
    "user_amount_std",
    "user_device_risk_mean",
    "amount_vs_user_mean",
]


# ------------------------
# БАЗОВІ ФІЧІ (безпечні)
# ------------------------
def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["amount"] = df["amount"].astype(float)
    df["amount_log"] = np.log1p(df["amount"])

    df["hour"] = df["hour"].astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

    df["device_risk_score"] = df["device_risk_score"].astype(float)
    df["ip_risk_score"] = df["ip_risk_score"].astype(float)

    # взаємодія
    df["combined_risk_score"] = (
        df["device_risk_score"] * df["ip_risk_score"]
    )

    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    # one-hot
    for t in TRANSACTION_TYPES:
        df[f"tx_type_{t.lower()}"] = (df["transaction_type"] == t).astype(int)

    for c in MERCHANT_CATEGORIES:
        df[f"merchant_{c.lower()}"] = (df["merchant_category"] == c).astype(int)

    return df


# ------------------------
# USER FEATURES (БЕЗ leakage)
# ------------------------
def _apply_user_stats(
    df: pd.DataFrame,
    user_stats: Optional[pd.DataFrame]
) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)

    if user_stats is not None:
        user_stats = user_stats.copy()
        user_stats["user_id"] = user_stats["user_id"].astype(str)

        df = df.merge(user_stats, on="user_id", how="left")

    # fallback (НОВА логіка — без leakage)
    df["user_tx_count"] = df.get("user_tx_count", 1).fillna(1)
    df["user_amount_mean"] = df.get("user_amount_mean", df["amount"]).fillna(df["amount"])
    df["user_amount_std"] = df.get("user_amount_std", 0).fillna(0)
    df["user_device_risk_mean"] = df.get(
        "user_device_risk_mean",
        df["device_risk_score"]
    ).fillna(df["device_risk_score"])

    # стабільний z-score
    df["amount_vs_user_mean"] = (
        df["amount"] - df["user_amount_mean"]
    ) / (df["user_amount_std"] + 1e-6)

    return df


# ------------------------
# MAIN FUNCTION
# ------------------------
def build_features(
    df: pd.DataFrame,
    user_stats: Optional[pd.DataFrame] = None,
    single: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    df = _base_features(df)
    df = _apply_user_stats(df, user_stats)

    feature_cols = BASE_FEATURES + USER_FEATURES

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]

    y = None
    if not single and "is_fraud" in df.columns:
        y = df["is_fraud"]

    return X, y


# ------------------------
# USER STATS (окремо!)
# ------------------------
def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)

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

    return stats


# ------------------------
# SINGLE TRANSACTION
# ------------------------
def prepare_single_transaction(tx_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([tx_dict])
    X, _ = build_features(df, user_stats=None, single=True)
    return X
