from typing import List

from fastapi import Depends, FastAPI, HTTPException
from app.models.schemas import BatchItemResult, CustomerRequest, CustomerIntelligence
from app.models.predictor import predict
from app.security import require_api_key

MAX_BATCH_SIZE = 500
MAX_BATCH_ORDERS = 100_000

app = FastAPI(
    title="Customer Intelligence API",
    description="Analisa comportamento de clientes e retorna previsão de churn, segmento e ação recomendada.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze", response_model=CustomerIntelligence, dependencies=[Depends(require_api_key)])
def analyze(request: CustomerRequest):
    if not request.orders:
        raise HTTPException(status_code=400, detail="Envie pelo menos 1 pedido.")

    orders = [{"date": o.date, "value": o.value} for o in request.orders]

    try:
        result = predict(orders, request.customer_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao processar análise.")

    return result


@app.post("/batch", response_model=List[BatchItemResult], dependencies=[Depends(require_api_key)])
def batch(requests: List[CustomerRequest]):
    if not requests:
        raise HTTPException(status_code=400, detail="Envie pelo menos 1 cliente.")
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Máximo de {MAX_BATCH_SIZE} clientes por chamada.")
    if sum(len(request.orders) for request in requests) > MAX_BATCH_ORDERS:
        raise HTTPException(status_code=400, detail=f"Máximo de {MAX_BATCH_ORDERS} pedidos somados por chamada.")

    results = []
    for request in requests:
        # Um cliente com erro não derruba o lote inteiro.
        if not request.orders:
            results.append({"customer_id": request.customer_id, "error": "Envie pelo menos 1 pedido."})
            continue

        orders = [{"date": o.date, "value": o.value} for o in request.orders]
        try:
            results.append({"customer_id": request.customer_id, "analysis": predict(orders, request.customer_id)})
        except Exception as e:
            results.append({"customer_id": request.customer_id, "error": str(e)})

    return results
