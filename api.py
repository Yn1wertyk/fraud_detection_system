from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import uvicorn
from inference import get_detector

app = FastAPI(
    title="Fraud Detection API",
    description="API для виявлення фінансових махінацій",
    version="1.0.0"
)

class Transaction(BaseModel):
    """
    Модель транзакції
    """
    user_id: str = Field(..., description="ID користувача")
    amount: float = Field(..., ge=0, description="Сума транзакції")
    currency: str = Field(..., description="Валюта")
    merchant: str = Field(..., description="Мерчант")
    device_id: Optional[str] = Field(None, description="ID пристрою")
    country: Optional[str] = Field(None, description="Країна")
    tx_time: str = Field(..., description="Час транзакції (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "amount": 150.50,
                "currency": "UAH",
                "merchant": "merchant_abc",
                "device_id": "device_456",
                "country": "UA",
                "tx_time": "2024-01-15T14:30:00"
            }
        }

class FraudResponse(BaseModel):
    """
    Відповідь системи виявлення махінацій
    """
    fraud_probability: float
    decision: str
    risk_level: str
    top_features: Dict[str, float]
    explanation: str

@app.on_event("startup")
async def startup_event():
    """
    Перевірка наявності моделі при запуску API
    """
    try:
        detector = get_detector()
        print("✅ API запущено успішно. Модель завантажена.")
    except FileNotFoundError:
        print("❌ УВАГА: Модель не знайдена!")
        print("📋 Для налаштування системи запустіть:")
        print("   python scripts/setup_model.py")
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")

@app.get("/")
async def root():
    """
    Головна сторінка API
    """
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": ["/score", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    """
    Перевірка здоров'я сервісу
    """
    try:
        detector = get_detector()
        return {
            "status": "healthy", 
            "model_loaded": True,
            "message": "Система готова до роботи"
        }
    except FileNotFoundError:
        return {
            "status": "model_missing", 
            "model_loaded": False,
            "message": "Модель не знайдена. Запустіть: python scripts/setup_model.py"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "model_loaded": False,
            "error": str(e)
        }

@app.post("/score", response_model=FraudResponse)
async def score_transaction(transaction: Transaction):
    """
    Оцінка транзакції на предмет махінацій
    """
    try:
        detector = get_detector()
        
        # Конвертуємо в словник
        tx_dict = transaction.model_dump()
        
        # Прогнозування
        result = detector.predict_single(tx_dict)
        
        return FraudResponse(**result)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="Модель не знайдена. Спочатку запустіть: python scripts/setup_model.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")

@app.post("/batch_score")
async def batch_score(transactions: list[Transaction]):
    """
    Пакетна оцінка транзакцій
    """
    try:
        detector = get_detector()
        results = []
        
        for tx in transactions:
            tx_dict = tx.model_dump()
            result = detector.predict_single(tx_dict)
            results.append(result)
        
        return {"results": results, "count": len(results)}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="Модель не знайдена. Спочатку запустіть: python scripts/setup_model.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка пакетної обробки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
