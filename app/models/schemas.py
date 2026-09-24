from datetime import date as date_type, datetime, timedelta
from pydantic import BaseModel, field_validator
from typing import List, Optional

MAX_ORDERS_PER_CUSTOMER = 5000


class Order(BaseModel):
    date: str
    value: float

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            parsed = datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Data inválida: '{v}'. Use o formato YYYY-MM-DD.")
        # 1 dia de folga para diferença de fuso entre a loja e o servidor.
        if parsed.date() > date_type.today() + timedelta(days=1):
            raise ValueError(f"Data no futuro: '{v}'. Envie só compras que já aconteceram.")
        return v

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"Valor negativo: {v}. Devoluções e estornos não são compras — remova essas linhas."
            )
        return v


class CustomerRequest(BaseModel):
    customer_id: str
    orders: List[Order]

    @field_validator("orders")
    @classmethod
    def validate_orders_size(cls, v: List[Order]) -> List[Order]:
        if len(v) > MAX_ORDERS_PER_CUSTOMER:
            raise ValueError(f"Máximo de {MAX_ORDERS_PER_CUSTOMER} pedidos por cliente.")
        return v


class CustomerIntelligence(BaseModel):
    customer_id: str
    # Score do modelo (0–1) para ordenar clientes. Não é probabilidade calibrada:
    # para mostrar a pessoas, use risk_level.
    churn_risk: float
    risk_level: str  # "low" | "medium" | "high"
    segment: str
    purchase_trend: str
    customer_value: str
    recommended_action: str
    reasons: List[str]
    confidence: str  # "low" | "medium" | "high" — baseado na quantidade de eventos
    # Cortes de valor medidos na base que treinou o artefato. Opcionais para clientes antigos.
    value_high_from: Optional[float] = None
    value_medium_from: Optional[float] = None


class BatchItemResult(BaseModel):
    customer_id: str
    analysis: Optional[CustomerIntelligence] = None
    error: Optional[str] = None
