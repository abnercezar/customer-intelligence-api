# Como colocar o projeto em funcionamento

O projeto usa **Python 3.11**. Neste Ubuntu o `python3` é o 3.14, e as dependências de `requirements.txt` não instalam nele. Use o interpretador `python3.11` e o ambiente `.venv` desta pasta.

No Windows com PowerShell, o equivalente local está em `commands.md`.

Os modelos já gravados em `app/ml/` bastam para a primeira leitura. Treinar de novo é opcional e só vale no computador local.

---

# Local

A API e o dashboard rodam na sua máquina. A chave local é `teste123`. Ela não vale no Railway.

## 1. Entrar na pasta do projeto

```bash
cd /home/abnercezar/dev/customer-intelligence-api
```

## 2. Instalar o Python 3.11 (só na primeira vez)

Pule esta etapa se `python3.11 --version` mostrar `Python 3.11.x`.

O apt do Ubuntu 26.04 não traz o 3.11. Ele vem do PPA deadsnakes, ao lado do Python do sistema. Não troque o `python3` padrão da máquina. Não instale o `uvicorn` pelo `apt`.

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

## 3. Criar o ambiente e instalar as dependências

Pule a criação do `.venv` se a pasta já existir. Rode o `pip install` de novo sempre que `requirements.txt` mudar.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Com o ambiente ativo, `python` e `pip` são os do `.venv`. Para sair: `deactivate`.

Sem esta etapa, `source .venv/bin/activate` responde `No such file or directory` e o `uvicorn` não é o do projeto.

## 4. Conferir os testes

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Todos devem passar. Os testes sobem a API em memória e não precisam do servidor no ar.

## 5. Subir a API

Terminal 1. Deixe aberto. Fechar a janela ou apertar Ctrl+C derruba a API.

```bash
cd /home/abnercezar/dev/customer-intelligence-api
source .venv/bin/activate
export API_KEY="teste123"
uvicorn app.main:app --reload
```

Espere a linha `Uvicorn running on http://127.0.0.1:8000`.

`export` vale só para este terminal. Em um terminal novo, ative o `.venv` e exporte a chave de novo.

Sem `API_KEY`, `/health` continua abrindo e `/analyze` responde 500 com "API_KEY não configurada".

### Conferir que está no ar

Em outro terminal:

```bash
curl http://127.0.0.1:8000/health
```

A resposta é `{"status":"healthy"}`. Essa rota não pede chave.

No navegador, abra http://127.0.0.1:8000/docs.

1. Clique em **Authorize** e cole `teste123`.
2. Abra **Ler um cliente** (`POST /analyze`) → **Try it out** → **Execute**. O exemplo da Ana já vem preenchido.
3. Para vários clientes, use **Ler vários clientes** (`POST /batch`) com o conteúdo de `data/sample_customers.json`.

Pelo terminal, um cliente:

```bash
curl -s http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: teste123" \
  -d '{"customer_id":"ana","orders":[{"date":"2026-03-10","value":300},{"date":"2026-04-15","value":350},{"date":"2026-05-20","value":400}]}'
```

O lote de exemplo:

```bash
curl -s http://127.0.0.1:8000/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: teste123" \
  --data-binary @data/sample_customers.json
```

## 6. Subir o dashboard

Terminal 2. A API do terminal 1 precisa continuar no ar. A `API_KEY` é a mesma.

```bash
cd /home/abnercezar/dev/customer-intelligence-api
source .venv/bin/activate
export API_KEY="teste123"
streamlit run dashboard.py --browser.gatherUsageStats false
```

Acesse http://localhost:8501. O dashboard chama `http://127.0.0.1:8000/analyze` por padrão.

## 7. Parar

Em cada terminal da API e do dashboard, aperte Ctrl+C.

## Treinar em uma base própria (opcional)

Os arquivos `model.joblib`, `segmenter.joblib`, `segment_map.joblib` e `thresholds.joblib` já estão em `app/ml/`. A API lê esses arquivos. Um treino novo só os substitui se a medição do corte de teste passar.

O CSV precisa de `customer_id`, `date` e `value` (ou `cliente`, `data` e `valor`), uma linha por pedido, datas em `YYYY-MM-DD` e valores maiores que zero. O histórico precisa cobrir bem mais que 90 dias.

```bash
source .venv/bin/activate

# Só grava os .joblib se o corte de teste passar.
python -m app.ml.train --orders data/pedidos.csv

# Online Retail II, só como comparação. Na primeira vez baixa o dataset para data/raw/.
python -m app.ml.train --benchmark
```

Se a medição falhar, o comando termina com erro e os arquivos atuais continuam valendo. Depois de um treino que grave o artefato, pare a API e suba de novo: modelo e limiares ficam em memória depois da primeira leitura.

`--synthetic` gera uma fórmula artificial, sem corte temporal, e sobrescreve o artefato. Não use isso como modelo de clientes reais.

## Problemas no local

| O que aparece | O que fazer |
| --- | --- |
| `.venv/bin/activate: No such file or directory` | A pasta `.venv` não existe. Refaça a etapa 3. Não instale o `uvicorn` pelo `apt`. |
| `python3.11: command not found` | Refaça a etapa 2. |
| `No module named ...` ou erro ao instalar numpy/scikit-learn | O `.venv` foi criado com o Python 3.14. Apague a pasta `.venv` e refaça a etapa 3 com `python3.11`. |
| Navegador não abre `http://127.0.0.1:8000` | A API não está no ar. Refaça a etapa 5 e deixe o terminal aberto. |
| `403` com "API key inválida ou ausente" | A chave do Authorize, do `curl` ou do dashboard é outra. Use `teste123` nos dois terminais. |
| `500` com "API_KEY não configurada" | A API subiu sem `export API_KEY`. Pare com Ctrl+C e refaça a etapa 5. |
| Dashboard mostra a API offline | Suba a API (etapa 5) antes do dashboard, com a mesma chave. |
| `address already in use` | Já existe uma API na porta 8000. Feche o terminal antigo ou suba com `--port 8002`. Se mudar a porta, ajuste `API_URL` no dashboard. |

## Variáveis no local

| Variável | Onde | Valor |
| --- | --- | --- |
| `API_KEY` | API e dashboard | `teste123` |
| `API_URL` | dashboard | `http://127.0.0.1:8000/analyze` (já é o padrão) |

Para uma chave mais forte no lugar de `teste123`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use a mesma string no `export API_KEY` da API e no do dashboard.

---

# Produção

O Railway sobe só a API. Nada de `.venv`, `uvicorn` nem `teste123`. A chave é a variável `API_KEY` em **Variables** do serviço.

## Ler pela documentação

1. Abra https://web-production-1a7e53.up.railway.app/docs
2. Clique em **Authorize** e cole a `API_KEY` do Railway.
3. Abra **Ler um cliente** (`POST /analyze`), clique em **Try it out** e depois em **Execute**.
4. Para vários clientes, use **Ler vários clientes** (`POST /batch`).

https://web-production-1a7e53.up.railway.app/health responde `{"status":"healthy"}` e não pede chave.

## Ler pelo terminal

Um cliente:

```bash
curl -s https://web-production-1a7e53.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: a-chave-do-railway" \
  -d '{"customer_id":"ana","orders":[{"date":"2026-03-10","value":300},{"date":"2026-04-15","value":350},{"date":"2026-05-20","value":400}]}'
```

Vários clientes, com o arquivo de exemplo:

```bash
curl -s https://web-production-1a7e53.up.railway.app/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: a-chave-do-railway" \
  --data-binary @data/sample_customers.json
```

## Dashboard local apontando para a produção

O dashboard continua na sua máquina. A API que ele chama é a publicada. Use a chave do Railway, não `teste123`.

```bash
cd /home/abnercezar/dev/customer-intelligence-api
source .venv/bin/activate
export API_KEY="a-chave-do-railway"
export API_URL="https://web-production-1a7e53.up.railway.app/analyze"
streamlit run dashboard.py --browser.gatherUsageStats false
```

Acesse http://localhost:8501.

## Problemas na produção

| Resposta | Significado |
| --- | --- |
| `403` | A chave está ausente ou é diferente da `API_KEY` do Railway. |
| `500` com "API_KEY não configurada" | A variável não está em **Variables**. Defina e faça um novo deploy. |
| `422` | Data inválida, data no futuro ou valor negativo. |

## Variáveis na produção

| Variável | Onde | Valor |
| --- | --- | --- |
| `API_KEY` | **Variables** do serviço no Railway | A chave do serviço, não `teste123` |
| `API_URL` | Só se o dashboard local falar com a API publicada | `https://web-production-1a7e53.up.railway.app/analyze` |
