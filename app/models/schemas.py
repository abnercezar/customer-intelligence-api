from pydantic import BaseModel
from typing import List, Optional


class Order(BaseModel):
    date: str
    value: float


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


class BatchItemResult(BaseModel):
    customer_id: str
    analysis: Optional[CustomerIntelligence] = None
    error: Optional[str] = None
