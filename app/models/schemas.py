from datetime import datetime
from pydantic import BaseModel, field_validator
from typing import List, Optional


class Order(BaseModel):
    date: str
    value: float

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Data inválida: '{v}'. Use o formato YYYY-MM-DD.")
        return v


class CustomerRequest(BaseModel):
    customer_id: str
    orders: List[Order]


class CustomerIntelligence(BaseModel):
    customer_id: str
    churn_risk: float
    segment: str
    purchase_trend: str
    customer_value: str
    recommended_action: str
    reasons: List[str]
    confidence: str  # "low" | "medium" | "high" — baseado na quantidade de eventos


class BatchItemResult(BaseModel):
    customer_id: str
    analysis: Optional[CustomerIntelligence] = None
    error: Optional[str] = None
