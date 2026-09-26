import math
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.models.schemas import BatchItemResult, CustomerRequest, CustomerIntelligence, HealthStatus, ServiceStatus
from app.models.predictor import predict
from app.protection import BodyLimitMiddleware, SecurityHeadersMiddleware
from app.security import require_api_key

MAX_BATCH_SIZE = 500
MAX_BATCH_ORDERS = 100_000

API_DESCRIPTION = """
Coleção para ler o histórico de compras de um cliente. Cada chamada é independente. A API não guarda o cliente.

## Autenticação

| | |
| --- | --- |
| Tipo | API Key |
| Header | `X-API-Key` |
| Valor | a chave da empresa |

`GET /` e `GET /health` não pedem chave. Nas outras rotas, clique em **Authorize** e cole a chave.

## Pedidos

### `GET /health`

Confere se o processo responde. Sem header.

```json
{"status": "healthy"}
```

### `POST /analyze`

Um cliente. Header `X-API-Key`.

```json
{
  "customer_id": "ana",
  "orders": [
    {"date": "2026-03-10", "value": 300},
    {"date": "2026-04-15", "value": 350}
  ]
}
```

`200` devolve a leitura: `risk_level`, `segment`, `purchase_trend`, `recommended_action`, `reasons`, `confidence`. `customer_id` volta igual e não entra no cálculo.

### `POST /batch`

Vários clientes, no mesmo formato, dentro de uma lista. Header `X-API-Key`. Até 500 clientes e 100.000 pedidos. Um erro não cancela o lote: o item volta com `error` e os outros seguem.

## Respostas de erro

| Código | Quando |
| --- | --- |
| 400 | Lista vazia ou acima do teto |
| 403 | Chave ausente ou diferente |
| 413 | Corpo maior que 10 MB |
| 422 | Data, valor ou customer_id inválido |
| 429 | Muitas leituras na mesma chave |
| 500 | Servidor sem chave configurada, ou falha interna sem detalhe |
"""

ANALYZE_DESCRIPTION = """
Uma pessoa por vez.

1. **Authorize**: cole a chave. Sem ela a resposta é recusada.
2. **Try it out**: o exemplo da Ana já vem preenchido. A data fica assim: 2026-03-10. O valor é maior que zero.
3. **Execute**.

Olhe estes três na resposta. Eles valem juntos:

- risk_level: low = pode esperar. medium = olhe (a partir de 0,35). high = vale chamar (a partir de 0,65).
- segment: o grupo mais parecido.
- purchase_trend: growing = subindo, stable = parado, declining = caindo.
"""

BATCH_DESCRIPTION = """
A mesma leitura, para várias pessoas de uma vez. O corpo é uma lista, não um cliente só.

Até 500 pessoas. Se uma falhar, ela volta com error e as outras seguem: um erro não cancela o lote.

Authorize, Try it out, Execute. Use a mesma chave.
"""

app = FastAPI(
    title="Customer Intelligence API",
    description=API_DESCRIPTION,
    version="1.0.0",
    swagger_ui_parameters={"docExpansion": "list", "defaultModelsExpandDepth": 0},
    openapi_tags=[
        {
            "name": "Situação",
            "description": "Diz se o processo está respondendo. Estas rotas não pedem chave.",
        },
        {
            "name": "Leitura",
            "description": "Pede a chave em Authorize. Lê um cliente ou vários.",
        },
    ],
)

# Cabeçalhos por fora, para também cobrir a recusa de corpo grande.
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "não finito"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Um valor infinito não pode quebrar a própria resposta de erro.
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(_json_safe(exc.errors()))})


@app.get("/", tags=["Situação"], summary="A API está no ar", response_model=ServiceStatus)
def root():
    """Sem chave. Só confirma que este endereço abre."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["Situação"], summary="Está no ar?", response_model=HealthStatus)
def health():
    """Sem chave. Se voltar healthy, o programa está no ar. Comece por aqui."""
    return {"status": "healthy"}


@app.post(
    "/analyze",
    response_model=CustomerIntelligence,
    dependencies=[Depends(require_api_key)],
    tags=["Leitura"],
    summary="Ler um cliente",
    description=ANALYZE_DESCRIPTION,
)
def analyze(request: CustomerRequest):
    if not request.orders:
        raise HTTPException(status_code=400, detail="Envie pelo menos 1 pedido.")

    orders = [{"date": o.date, "value": o.value} for o in request.orders]

    try:
        result = predict(orders, request.customer_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao processar análise.")

    return result


@app.post(
    "/batch",
    response_model=List[BatchItemResult],
    dependencies=[Depends(require_api_key)],
    tags=["Leitura"],
    summary="Ler vários clientes",
    description=BATCH_DESCRIPTION,
)
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
        except Exception:
            results.append({"customer_id": request.customer_id, "error": "Erro interno ao processar análise."})

    return results
