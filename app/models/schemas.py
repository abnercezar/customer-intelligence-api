from datetime import date as date_type, datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional

MAX_ORDERS_PER_CUSTOMER = 5000


class Order(BaseModel):
    date: str = Field(description="Dia da compra, assim: 2026-03-10. Só compra que já aconteceu.")
    value: float = Field(description="Quanto a pessoa pagou. Maior que zero. Devolução não entra.")

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
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "ana",
            "orders": [
                {"date": "2026-03-10", "value": 300},
                {"date": "2026-04-15", "value": 350},
                {"date": "2026-05-20", "value": 400},
            ],
        }
    })

    customer_id: str = Field(description="Nome ou código. Volta na resposta e não entra no cálculo.")
    orders: List[Order] = Field(description="As compras, uma por uma.")

    @field_validator("orders")
    @classmethod
    def validate_orders_size(cls, v: List[Order]) -> List[Order]:
        if len(v) > MAX_ORDERS_PER_CUSTOMER:
            raise ValueError(f"Máximo de {MAX_ORDERS_PER_CUSTOMER} pedidos por cliente.")
        return v


class CustomerIntelligence(BaseModel):
    """As três leituras valem juntas: atenção, grupo e se o valor sobe ou desce."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "ana",
            "churn_risk": 0.22,
            "risk_level": "low",
            "segment": "loyal",
            "purchase_trend": "growing",
            "customer_value": "medium",
            "recommended_action": "maintain_relationship",
            "reasons": ["comportamento dentro do padrão esperado"],
            "confidence": "medium",
            "value_high_from": 3000,
            "value_medium_from": 800,
        }
    })

    customer_id: str = Field(description="O mesmo nome enviado. Não muda o cálculo.")
    churn_risk: float = Field(description="De 0 a 1, só para ordenar. Não é a chance de perder o cliente.")
    risk_level: str = Field(description="Atenção: low pode esperar, medium olhe, high vale chamar.")
    segment: str = Field(description="Grupo: new, potential, loyal, champion ou at_risk.")
    purchase_trend: str = Field(description="growing sobe, stable parado, declining caindo.")
    customer_value: str = Field(description="Quanto já gastou nesta base: low, medium ou high.")
    recommended_action: str = Field(description="O que fazer: chamar, dar um oi, ou deixar quieto.")
    reasons: List[str] = Field(description="Por que a leitura ficou assim, em frases.")
    confidence: str = Field(description="low = 1 ou 2 compras. medium = 3 ou 4. high = 5 ou mais.")
    value_high_from: Optional[float] = Field(default=None, description="A partir deste total, o gasto é alto.")
    value_medium_from: Optional[float] = Field(default=None, description="A partir deste total, o gasto é médio.")


class BatchItemResult(BaseModel):
    customer_id: str = Field(description="Nome deste cliente.")
    analysis: Optional[CustomerIntelligence] = Field(default=None, description="A leitura, se deu certo.")
    error: Optional[str] = Field(default=None, description="O que falhou neste cliente. Os outros seguem.")
