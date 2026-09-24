import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="A mesma chave com que a API foi ligada. Está no ar? não pede.",
)


def require_api_key(x_api_key: Optional[str] = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    if not expected:
        # Sem chave configurada a API fica fechada, nunca aberta.
        raise HTTPException(status_code=500, detail="API_KEY não configurada no servidor.")

    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="API key inválida ou ausente.")
