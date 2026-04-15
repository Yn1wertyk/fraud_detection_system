import joblib
import pandas as pd
import numpy as np
import shap
import os
from typing import Dict, Any
from features import prepare_single_transaction

class FraudDetector:
    """
    Клас для виявлення фінансових махінацій
    """
    
    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        """
        Ініціалізація детектора
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не знайдена за шляхом: {model_path}\n"
                f"Спочатку запустіть тренування моделі: python src/train.py"
            )
        
        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data["model"]
            self.features = self.model_data["features"]
            self.explainer = shap.TreeExplainer(self.model)
            print(f"✅ Модель успішно завантажена з {model_path}")
            print(f"📊 Кількість ознак: {len(self.features)}")
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження моделі: {str(e)}")
        
    def predict_single(self, transaction: Dict[str, Any], 
                      historical_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Прогнозування для однієї транзакції
        
        Args:
            transaction: Словник з даними транзакції
            historical_data: Історичні дані для розрахунку агрегатів
            
        Returns:
            Словник з результатами прогнозування
        """
        try:
            # Підготовка ознак
            X = prepare_single_transaction(transaction, historical_data)
            X_features = X[self.features]
            
            print(f"🔍 Підготовлені ознаки: {X_features.iloc[0].to_dict()}")
            
            # Прогнозування
            fraud_proba_array = self.model.predict_proba(X_features)
            
            # Перевірка розмірності масиву
            if fraud_proba_array.shape[1] > 1:
                fraud_proba = fraud_proba_array[0, 1]  # Клас 1 (махінація)
            else:
                fraud_proba = fraud_proba_array[0, 0]  # Якщо тільки один клас
            
            print(f"🎯 Ймовірність махінації: {fraud_proba:.4f}")
            
            # SHAP пояснення
            shap_values = self.explainer.shap_values(X_features)
            
            # Перевірка структури SHAP values
            if isinstance(shap_values, list) and len(shap_values) > 1:
                feature_shap = shap_values[1][0]  # Клас 1
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                feature_shap = shap_values[0][0]  # Єдиний клас
            else:
                feature_shap = shap_values[0]  # Numpy array
            
            # Топ-10 найважливіших ознак
            feature_contributions = dict(zip(self.features, feature_shap))
            top_contributions = dict(
                sorted(feature_contributions.items(), 
                      key=lambda x: abs(x[1]), reverse=True)[:10]
            )
            
        except Exception as e:
            print(f"❌ Помилка прогнозування: {e}")
            return {
                "fraud_probability": 0.0,
                "decision": "ERROR",
                "risk_level": "UNKNOWN",
                "top_features": {},
                "explanation": f"Помилка обробки транзакції: {str(e)}"
            }
        
        # Бізнес-рішення
        if fraud_proba >= 0.8:
            decision = "BLOCK"
            risk_level = "HIGH"
        elif fraud_proba >= 0.5:
            decision = "REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "ALLOW"
            risk_level = "LOW"
        
        return {
            "fraud_probability": float(fraud_proba),
            "decision": decision,
            "risk_level": risk_level,
            "top_features": top_contributions,
            "explanation": self._generate_explanation(top_contributions, fraud_proba)
        }
    
    def _generate_explanation(self, contributions: Dict[str, float], 
                            fraud_proba: float) -> str:
        """
        Генерація текстового пояснення
        """
        if fraud_proba < 0.3:
            return "Транзакція виглядає нормально. Всі показники в межах норми."
        
        explanations = []
        
        for feature, contribution in list(contributions.items())[:3]:
            if abs(contribution) > 0.01:  # Значущий внесок
                if contribution > 0:
                    explanations.append(f"Підозрілий показник: {feature}")
                else:
                    explanations.append(f"Нормальний показник: {feature}")
        
        if fraud_proba >= 0.8:
            explanations.insert(0, "ВИСОКА ЙМОВІРНІСТЬ МАХІНАЦІЇ!")
        elif fraud_proba >= 0.5:
            explanations.insert(0, "Потребує додаткової перевірки.")
        
        return " ".join(explanations)

def load_detector() -> FraudDetector:
    """
    Завантаження детектора (singleton pattern)
    """
    try:
        return FraudDetector()
    except FileNotFoundError as e:
        print(f"❌ {str(e)}")
        raise
    except Exception as e:
        print(f"❌ Критична помилка завантаження моделі: {str(e)}")
        raise

# Глобальний екземпляр для FastAPI
detector = None

def get_detector() -> FraudDetector:
    """
    Отримання глобального екземпляра детектора
    """
    global detector
    if detector is None:
        detector = load_detector()
    return detector
