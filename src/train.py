import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import os

from features import build_features

RANDOM_STATE = 45

# Шлях до CSV датасету за замовчуванням
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_fraud_dataset.csv")


def train_model(data_path: str = DEFAULT_DATA_PATH, target_col: str = "is_fraud"):
    """
    Тренування моделі виявлення фінансових махінацій.

    Очікуваний формат CSV:
        transaction_id, user_id, amount, transaction_type, merchant_category,
        country, hour, device_risk_score, ip_risk_score, is_fraud

    Args:
        data_path: шлях до CSV-файлу з датасетом
        target_col: назва цільової колонки
    """
    print("Завантаження даних...")
    df = pd.read_csv(data_path)
    print(f"Рядків завантажено: {len(df)}")
    print(f"Колонки: {list(df.columns)}")

    # Перевірка наявності цільової колонки
    if target_col not in df.columns:
        raise ValueError(
            f"Цільова колонка '{target_col}' відсутня у датасеті.\n"
            f"Доступні колонки: {list(df.columns)}"
        )

    print("\nПобудова ознак...")
    X, y = build_features(df)

    print(f"Розмір датасету: {X.shape}")
    print(f"Частка махінацій: {y.mean():.4f} ({y.sum()} / {len(y)})")
    print(f"Ознаки: {list(X.columns)}")

    # Стратифікована крос-валідація (замість TimeSeriesSplit, бо немає tx_time)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ap_scores = []
    models = []

    print("\nПочинаємо крос-валідацію (StratifiedKFold, 5 фолдів)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nФолд {fold + 1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            is_unbalance=True,
            verbose=-1
        )

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]

        ap_score = average_precision_score(y_val, y_pred_proba)
        ap_scores.append(ap_score)
        models.append(model)

        print(f"PR-AUC: {ap_score:.4f}")

        # Оптимальний поріг за F1
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]

        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        print(f"Оптимальний поріг: {optimal_threshold:.4f}")
        print(classification_report(y_val, y_pred_binary, target_names=["Normal", "Fraud"]))

    # Найкраща модель
    best_idx = int(np.argmax(ap_scores))
    best_model = models[best_idx]

    print(f"\nНайкраща модель: Фолд {best_idx + 1}")
    print(f"Середній PR-AUC: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")

    # Збереження
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": best_model,
        "features": list(X.columns),
        "pr_auc_scores": ap_scores,
        "best_fold": best_idx,
        "feature_importance": dict(zip(X.columns, best_model.feature_importances_))
    }

    joblib.dump(model_data, "models/fraud_model.pkl")
    print("\nМодель збережена в models/fraud_model.pkl")

    # Візуалізація важливості ознак
    plot_feature_importance(best_model, X.columns)

    return best_model, ap_scores


def plot_feature_importance(model, feature_names, top_n: int = 20):
    """
    Візуалізація топ-N найважливіших ознак.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    actual_top_n = min(top_n, len(feature_names))
    indices = indices[:actual_top_n]

    plt.figure(figsize=(12, 8))
    plt.title(f"Топ-{actual_top_n} найважливіших ознак")
    plt.bar(range(actual_top_n), importance[indices])
    plt.xticks(
        range(actual_top_n),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right"
    )
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Графік важливості ознак збережено в models/feature_importance.png")


if __name__ == "__main__":
    train_model()
