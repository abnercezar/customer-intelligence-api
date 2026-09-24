# Customer Behavioral Intelligence API

## O que é este projeto

Uma camada de inteligência comportamental que analisa o histórico de eventos de clientes e devolve sinais acionáveis para qualquer tipo de negócio. Não é um CRM.

**Pergunta central que o produto responde:**
> "O que está acontecendo com meus clientes e quais precisam da minha atenção?"

## Localização

```
C:\Users\abner\customer-intelligence-api\
```

## Stack

- Python 3.11
- FastAPI + Pydantic
- scikit-learn (RandomForest + KMeans)
- pandas / numpy
- Streamlit (dashboard)
- pytest

## Estrutura de arquivos

```
app/
  main.py              — endpoints FastAPI (/analyze, /batch, /health)
  security.py          — autenticação via API Key (X-API-Key header)
  ml/
    features.py        — extração de features RFM (recency, frequency, monetary, trend, interval)
    train.py           — treino: RandomForest (churn) + KMeans (segmentação)
    segmenter.py       — K-Means: agrupa e rotula clusters automaticamente
    model.joblib       — modelo de churn treinado
    segmenter.joblib   — pipeline K-Means treinado
    segment_map.joblib — mapeamento cluster_id → nome do segmento
  models/
    schemas.py         — Pydantic: CustomerRequest, CustomerIntelligence, BatchItemResult
    predictor.py       — lógica completa: features → churn → segmento → razões → resposta
dashboard.py           — interface Streamlit local
tests/
  test_api.py          — 12 testes cobrindo auth, analyze, batch, edge cases
data/
  sample_customers.json
requirements.txt
```

## Fluxo de dados

```
POST /analyze
  └── CustomerRequest (customer_id + orders[])
        └── predictor.predict()
              ├── features.extract_features()     → recency, frequency, monetary, trend, interval
              ├── model.predict_proba()            → behavior_score (0–1)
              ├── segmenter.predict_segment()      → champion | loyal | potential | at_risk | new
              ├── _trend()                         → growing | stable | declining
              ├── _value()                         → high | medium | low
              ├── _action()                        → ação recomendada
              └── _reasons()                       → lista de razões explicáveis
```

## Sinais do produto

| Sinal | Ação |
|-------|------|
| new | onboarding |
| healthy | maintain_engagement |
| loyal | maintain_relationship |
| potential | nurture |
| at_risk | retention |
| reactivation | reactivation |
| growth_opportunity | expansion |

## Estado atual do modelo

- **Churn (RandomForest): treinado com dados reais** do Online Retail II (UCI, loja do Reino Unido, 2009–2011, valores em GBP, muitos atacadistas) — `app/ml/online_retail.py`
  - Rótulo por data de corte: features só com pedidos antes do corte; churn = não comprou nos 90 dias seguintes
  - Avaliação temporal (treino em cortes 2010-06 a 2011-06, teste em 2011-09): acurácia 72,2% vs. baseline 57,3%, ROC AUC 0,786
  - Validado nesse dataset, não em clientes do produto — o score não é probabilidade calibrada para outros negócios
- **Segmentação (K-Means): ainda treinada com dados sintéticos** (`python -m app.ml.train --synthetic`). Com os dados reais os clusters ficam muito desbalanceados (valores de atacado dominam)
- Separação clara: feature engineering → model → business rules → recommendation

## Regras de desenvolvimento

1. Ler o código antes de modificar
2. Não recriar o que já existe
3. Não refatorar sem necessidade clara
4. Não alterar contratos de API sem documentar impacto
5. Não apagar funcionalidade sem confirmação
6. Todo código novo tem teste
7. Não inventar dados nem apresentar sintéticos como reais
8. Não chamar heurística de IA nem score de probabilidade calibrada sem validação
9. Priorizar explicabilidade
10. Preferir a solução mais simples

## Critério de sucesso

> Dado o histórico de um cliente, o sistema identifica uma mudança relevante no comportamento e explica claramente por que ela merece atenção.

## Rodar localmente

```bash
# Treinar modelo de churn com Online Retail II (baixa o dataset para data/raw/ na 1ª vez)
python -m app.ml.train

# Treinar churn + segmentador com dados sintéticos
python -m app.ml.train --synthetic

# API
uvicorn app.main:app --reload

# Dashboard
streamlit run dashboard.py --browser.gatherUsageStats false

# Testes
python -m pytest tests/ -v
```

## Variáveis de ambiente

```
API_KEY=sua-chave-aqui
API_URL=http://127.0.0.1:8000/analyze  (dashboard)
```
