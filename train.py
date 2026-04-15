import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from features import build_features
import os

RANDOM_STATE = 45

def train_model(data_path: str, time_col: str = "tx_time", target_col: str = "is_fraud"):
    """
    Тренування моделі для виявлення фінансових махінацій
    """
    print("Завантаження даних...")
    df = pd.read_csv(data_path, parse_dates=[time_col])
    df = df.sort_values(time_col)
    
    print("Побудова ознак...")
    X, y = build_features(df)
    
    print(f"Розмір датасету: {X.shape}")
    print(f"Частка махінацій: {y.mean():.4f}")
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    ap_scores = []
    models = []
    
    print("\nПочинаємо крос-валідацію...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nФолд {fold + 1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Розрахунок ваги для дизбалансу класів
        pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        
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
        
        # Прогнозування
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Метрики
        ap_score = average_precision_score(y_val, y_pred_proba)
        ap_scores.append(ap_score)
        models.append(model)
        
        print(f"PR-AUC: {ap_score:.4f}")
        
        # Звіт класифікації для оптимального порогу
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        print(f"Оптимальний поріг: {optimal_threshold:.4f}")
        print(classification_report(y_val, y_pred_binary, target_names=['Normal', 'Fraud']))
    
    # Вибираємо найкращу модель
    best_idx = np.argmax(ap_scores)
    best_model = models[best_idx]
    
    print(f"\nНайкраща модель: Фолд {best_idx + 1}")
    print(f"Середній PR-AUC: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
    
    # Збереження моделі
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": best_model,
        "features": list(X.columns),
        "pr_auc_scores": ap_scores,
        "best_fold": best_idx,
        "feature_importance": dict(zip(X.columns, best_model.feature_importances_))
    }
    
    joblib.dump(model_data, "models/fraud_model.pkl")
    print("Модель збережена в models/fraud_model.pkl")
    
    # Візуалізація важливості ознак
    plot_feature_importance(best_model, X.columns)
    
    return best_model, ap_scores

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Візуалізація важливості ознак
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    actual_top_n = min(top_n, len(feature_names))
    indices = indices[:actual_top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Топ-{actual_top_n} найважливіших ознак')
    plt.bar(range(actual_top_n), importance[indices])
    plt.xticks(range(actual_top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    train_model(r'C:\Yn1wertyk\dyploma\data\fraud_dataset.csv')
