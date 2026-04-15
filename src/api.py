from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn

from inference import get_detector

app = FastAPI(
    title="Fraud Detection API",
    description="API для виявлення фінансових махінацій",
    version="2.0.0"
)


# ---------------------------------------------------------------------------
# Pydantic-схеми
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """
    Поля відповідають колонкам датасету synthetic_fraud_dataset.csv
    (без transaction_id та is_fraud — вони не потрібні для інференсу).
    """
    user_id: str = Field(..., description="ID користувача")
    amount: float = Field(..., ge=0, description="Сума транзакції")
    transaction_type: str = Field(..., description="Тип транзакції: ATM | Online | POS | QR | Transfer")
    merchant_category: str = Field(..., description="Категорія мерчанта: Food | Electronics | Travel | ...")
    country: str = Field(..., description="Код країни (2 літери), наприклад UA, US, TR")
    hour: int = Field(..., ge=0, le=23, description="Година транзакції (0–23)")
    device_risk_score: float = Field(..., ge=0.0, le=1.0, description="Ризик-скор пристрою (0–1)")
    ip_risk_score: float = Field(..., ge=0.0, le=1.0, description="Ризик-скор IP-адреси (0–1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "363",
                "amount": 4922.59,
                "transaction_type": "ATM",
                "merchant_category": "Travel",
                "country": "TR",
                "hour": 12,
                "device_risk_score": 0.99,
                "ip_risk_score": 0.95
            }
        }
    }


class FraudResponse(BaseModel):
    fraud_probability: float
    decision: str
    risk_level: str
    top_features: Dict[str, float]
    explanation: str


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    try:
        get_detector()
        print("API запущено. Модель завантажена.")
    except FileNotFoundError:
        print("УВАГА: Модель не знайдена! Запустіть: python src/train.py")
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API v2",
        "endpoints": ["/score", "/batch_score", "/health", "/docs"]
    }


@app.get("/health")
async def health_check():
    try:
        get_detector()
        return {"status": "healthy", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "model_missing", "model_loaded": False,
                "message": "Модель не знайдена. Запустіть: python src/train.py"}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}


@app.post("/score", response_model=FraudResponse)
async def score_transaction(transaction: Transaction):
    """Оцінка однієї транзакції."""
    try:
        detector = get_detector()
        result = detector.predict_single(transaction.model_dump())
        return FraudResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Модель не знайдена. Запустіть: python src/train.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {e}")


@app.post("/batch_score")
async def batch_score(transactions: List[Transaction]):
    """Пакетна оцінка транзакцій."""
    try:
        detector = get_detector()
        results = [detector.predict_single(tx.model_dump()) for tx in transactions]
        return {"results": results, "count": len(results)}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Модель не знайдена. Запустіть: python src/train.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка пакетної обробки: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
