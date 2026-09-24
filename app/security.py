import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("API_KEY")
    if not expected:
        # Sem chave configurada a API fica fechada, nunca aberta.
        raise HTTPException(status_code=500, detail="API_KEY não configurada no servidor.")

    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="API key inválida ou ausente.")
