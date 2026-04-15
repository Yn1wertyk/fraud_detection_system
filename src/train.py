import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
)
from lightgbm import LGBMClassifier

from features import build_features, compute_user_stats

RANDOM_STATE = 42
N_FOLDS = 5

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SRC_DIR, "..")
DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, "data", "synthetic_fraud_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


def _best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Повертає поріг, що максимізує F1 на precision-recall кривій."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1[:-1]))  # останній елемент не має відповідного порогу
    return float(thresholds[idx])


def train_model(data_path: str = DEFAULT_DATA_PATH, target_col: str = "is_fraud"):
    """
    Тренування моделі виявлення фінансових махінацій.

    Очікуваний формат CSV:
        transaction_id, user_id, amount, transaction_type, merchant_category,
        country, hour, device_risk_score, ip_risk_score, is_fraud
    """
    print("Завантаження даних...")
    df = pd.read_csv(data_path)
    print(f"Рядків: {len(df)}, колонки: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(
            f"Цільова колонка '{target_col}' відсутня. "
            f"Доступні: {list(df.columns)}"
        )

    y_full = df[target_col].values
    fraud_rate = y_full.mean()
    print(f"Частка фроду: {fraud_rate:.4f}  ({y_full.sum()} / {len(y_full)})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    ap_scores: list[float] = []
    thresholds: list[float] = []
    models: list[LGBMClassifier] = []
    feature_names: list[str] = []

    # LightGBM параметри
    lgbm_params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        # Замість is_unbalance використовуємо scale_pos_weight для більш
        # контрольованого балансування
        scale_pos_weight=int((1 - fraud_rate) / (fraud_rate + 1e-8)),
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    print(f"\nscale_pos_weight = {lgbm_params['scale_pos_weight']}")

    print(f"\nКрос-валідація (StratifiedKFold, {N_FOLDS} фолдів)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_full)):
        print(f"\n--- Фолд {fold + 1}/{N_FOLDS} ---")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # Агрегати по user_id рахуємо ТІЛЬКИ по train-частині фолду
        fold_user_stats = compute_user_stats(df_train)

        X_train, y_train = build_features(df_train)
        # Для валідації передаємо статистику з train-частини (без витоку)
        X_val, y_val = build_features(df_val, user_stats=fold_user_stats)

        if fold == 0:
            feature_names = list(X_train.columns)
            print(f"Кількість ознак: {len(feature_names)}")
            print(f"Ознаки: {feature_names}")

        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(200),
            ],
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_proba)
        thresh = _best_threshold(y_val.values, y_proba)

        ap_scores.append(ap)
        thresholds.append(thresh)
        models.append(model)

        y_pred = (y_proba >= thresh).astype(int)
        print(f"PR-AUC: {ap:.4f}  |  Оптимальний поріг: {thresh:.4f}")
        print(classification_report(y_val, y_pred, target_names=["Normal", "Fraud"]))

    best_idx = int(np.argmax(ap_scores))
    best_model = models[best_idx]
    best_threshold = thresholds[best_idx]

    print(f"\nНайкращий фолд: {best_idx + 1}")
    print(f"PR-AUC: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
    print(f"Збережений поріг: {best_threshold:.4f}")

    # Зберігаємо user_stats по ВСЬОМУ датасету — для інференсу
    full_user_stats = compute_user_stats(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_out_path = os.path.join(MODELS_DIR, "fraud_model.pkl")

    joblib.dump(
        {
            "model": best_model,
            "features": feature_names,
            "threshold": best_threshold,
            "user_stats": full_user_stats,
            "pr_auc_scores": ap_scores,
            "best_fold": best_idx,
            "feature_importance": dict(
                zip(feature_names, best_model.feature_importances_)
            ),
        },
        model_out_path,
    )
    print(f"\nМодель збережена: {model_out_path}")

    plot_feature_importance(best_model, feature_names)

    return best_model, ap_scores


def plot_feature_importance(
    model: LGBMClassifier,
    feature_names,
    top_n: int = 20,
    out_dir: str = MODELS_DIR,
):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][: min(top_n, len(feature_names))]

    plt.figure(figsize=(12, 8))
    plt.title(f"Топ-{len(indices)} найважливіших ознак")
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(
        range(len(indices)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Графік збережено: {plot_path}")


if __name__ == "__main__":
    train_model()
