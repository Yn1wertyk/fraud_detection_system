import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
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
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx])


def train_model(data_path: str = DEFAULT_DATA_PATH, target_col: str = "is_fraud"):
    print("Завантаження даних...")
    df = pd.read_csv(data_path)
    print(f"Рядків: {len(df)}, колонки: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(f"Цільова колонка '{target_col}' відсутня.")

    # 🔥 ВАЖЛИВО: user_id як string
    df["user_id"] = df["user_id"].astype(str)

    y_full = df[target_col].values
    groups = df["user_id"].values

    fraud_rate = y_full.mean()
    print(f"Частка фроду: {fraud_rate:.4f}")

    # ✅ GroupKFold замість StratifiedKFold
    gkf = GroupKFold(n_splits=N_FOLDS)

    ap_scores = []
    thresholds = []
    models = []
    feature_names = []

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
        scale_pos_weight=int((1 - fraud_rate) / (fraud_rate + 1e-8)),
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )

    print(f"\nscale_pos_weight = {lgbm_params['scale_pos_weight']}")
    print(f"\nКрос-валідація (GroupKFold, {N_FOLDS} фолдів)...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y_full, groups)):
        print(f"\n--- Фолд {fold + 1}/{N_FOLDS} ---")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # ✅ агрегати тільки з train
        fold_user_stats = compute_user_stats(df_train)

        # ✅ І train, і val використовують ОДНІ І ТІ САМІ stats
        X_train, y_train = build_features(df_train, user_stats=fold_user_stats)
        X_val, y_val = build_features(df_val, user_stats=fold_user_stats)

        if fold == 0:
            feature_names = list(X_train.columns)
            print(f"Кількість ознак: {len(feature_names)}")

        model = LGBMClassifier(**lgbm_params)

        model.fit(
            X_train,
            y_train,
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

        print(f"PR-AUC: {ap:.4f} | Поріг: {thresh:.4f}")
        print(classification_report(y_val, y_pred))

    best_idx = int(np.argmax(ap_scores))
    best_model = models[best_idx]
    best_threshold = thresholds[best_idx]

    print(f"\nНайкращий фолд: {best_idx + 1}")
    print(f"PR-AUC: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")

    # ✅ для продакшену — агрегати по ВСЬОМУ train датасету
    full_user_stats = compute_user_stats(df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "fraud_model.pkl")

    joblib.dump(
        {
            "model": best_model,
            "features": feature_names,
            "threshold": best_threshold,
            "user_stats": full_user_stats,
            "pr_auc_scores": ap_scores,
        },
        model_path,
    )

    print(f"\nМодель збережена: {model_path}")

    plot_feature_importance(best_model, feature_names)

    return best_model, ap_scores


def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(path)
    plt.close()

    print(f"Графік збережено: {path}")


if __name__ == "__main__":
    train_model()
