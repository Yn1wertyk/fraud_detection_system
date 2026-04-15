import joblib
import pandas as pd
import numpy as np
import shap
import os
from typing import Dict, Any, Optional

from features import prepare_single_transaction

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")


class FraudDetector:
    """
    Детектор фінансових махінацій на основі навченої LightGBM-моделі.

    Очікувані поля транзакції:
        user_id, amount, transaction_type, merchant_category,
        country, hour, device_risk_score, ip_risk_score
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не знайдена: {model_path}\n"
                f"Спочатку запустіть тренування: python src/train.py"
            )

        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data["model"]
            self.features = self.model_data["features"]
            self.explainer = shap.TreeExplainer(self.model)
            print(f"Модель завантажена: {model_path}")
            print(f"Кількість ознак: {len(self.features)}")
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження моделі: {e}")

    def predict_single(
        self,
        transaction: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Прогнозування для однієї транзакції.

        Args:
            transaction: словник з полями транзакції
            historical_data: DataFrame з попередніми транзакціями цього юзера (опційно)

        Returns:
            Словник: fraud_probability, decision, risk_level, top_features, explanation
        """
        try:
            X = prepare_single_transaction(transaction, historical_data)

            # Вирівнюємо колонки з тими, що очікує модель
            for col in self.features:
                if col not in X.columns:
                    X[col] = 0
            X_features = X[self.features]

            # Прогноз
            proba_array = self.model.predict_proba(X_features)
            fraud_proba = float(proba_array[0, 1]) if proba_array.shape[1] > 1 else float(proba_array[0, 0])

            # SHAP-пояснення
            shap_values = self.explainer.shap_values(X_features)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                feature_shap = shap_values[1][0]
            elif isinstance(shap_values, list):
                feature_shap = shap_values[0][0]
            else:
                feature_shap = shap_values[0]

            feature_contributions = dict(zip(self.features, feature_shap))
            top_contributions = dict(
                sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )

        except Exception as e:
            print(f"Помилка прогнозування: {e}")
            return {
                "fraud_probability": 0.0,
                "decision": "ERROR",
                "risk_level": "UNKNOWN",
                "top_features": {},
                "explanation": f"Помилка обробки транзакції: {e}"
            }

        # Бізнес-рішення
        if fraud_proba >= 0.8:
            decision, risk_level = "BLOCK", "HIGH"
        elif fraud_proba >= 0.5:
            decision, risk_level = "REVIEW", "MEDIUM"
        else:
            decision, risk_level = "ALLOW", "LOW"

        return {
            "fraud_probability": fraud_proba,
            "decision": decision,
            "risk_level": risk_level,
            "top_features": top_contributions,
            "explanation": self._generate_explanation(top_contributions, fraud_proba)
        }

    def _generate_explanation(self, contributions: Dict[str, float], fraud_proba: float) -> str:
        if fraud_proba < 0.3:
            return "Транзакція виглядає нормально. Всі показники в межах норми."

        explanations = []
        if fraud_proba >= 0.8:
            explanations.append("ВИСОКА ЙМОВІРНІСТЬ МАХІНАЦІЇ!")
        elif fraud_proba >= 0.5:
            explanations.append("Потребує додаткової перевірки.")

        for feature, contribution in list(contributions.items())[:3]:
            if abs(contribution) > 0.01:
                label = "Підозрілий" if contribution > 0 else "Нормальний"
                explanations.append(f"{label} показник: {feature}")

        return " ".join(explanations)


# --- Singleton для FastAPI ---
_detector: Optional[FraudDetector] = None


def get_detector() -> FraudDetector:
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector
