from typing import List

from fastapi import Depends, FastAPI, HTTPException
from app.models.schemas import BatchItemResult, CustomerRequest, CustomerIntelligence
from app.models.predictor import predict
from app.security import require_api_key

MAX_BATCH_SIZE = 500
MAX_BATCH_ORDERS = 100_000

API_DESCRIPTION = """
Você manda as compras de uma pessoa. A resposta traz três coisas juntas:

- Precisa de atenção? Baixo, médio ou alto.
- Com qual grupo se parece? Novo, potencial, leal, campeão ou em risco.
- O valor das compras está subindo, parado ou caindo?

O nome não entra no cálculo. A API não guarda o cliente.

Para testar: abra **Está no ar?** (sem chave). Depois clique em **Authorize**, cole a chave, abra **Ler um cliente** e execute o exemplo.
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
    swagger_ui_parameters={"docExpansion": "full", "defaultModelsExpandDepth": 1},
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


@app.get("/", tags=["Situação"], summary="A API está no ar")
def root():
    """Sem chave. Só confirma que este endereço abre."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["Situação"], summary="Está no ar?")
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
        except Exception as e:
            results.append({"customer_id": request.customer_id, "error": str(e)})

    return results
