import hashlib
import json
import os
import secrets
import threading
import time
from collections import deque
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="A chave da empresa que integra a API. As rotas de situação não pedem chave.",
)


def _configured_keys() -> list[str]:
    """API_KEY segue como cliente default. API_KEYS é um JSON nome → chave."""
    keys: list[str] = []
    single = os.getenv("API_KEY")
    if single:
        keys.append(single)

    raw = os.getenv("API_KEYS")
    if not raw:
        return keys

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="API_KEYS inválido no servidor.")

    if not isinstance(parsed, dict) or any(not isinstance(value, str) or not value for value in parsed.values()):
        raise HTTPException(status_code=500, detail="API_KEYS inválido no servidor.")

    keys.extend(parsed.values())
    return keys


def _key_matches(presented: str, known: list[str]) -> bool:
    presented_bytes = presented.encode()
    matched = False
    for key in known:
        matched = secrets.compare_digest(presented_bytes, key.encode()) or matched
    return matched


_WINDOW_SECONDS = 60
_lock = threading.Lock()
_hits: dict[str, deque] = {}
_failures: deque = deque()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _prune(hits: deque, now: float) -> None:
    while hits and now - hits[0] > _WINDOW_SECONDS:
        hits.popleft()


def _rate_limited(presented: str) -> bool:
    """Conta leituras da chave certa. A chave em si não fica na memória, só o hash."""
    bucket = hashlib.sha256(presented.encode()).hexdigest()
    limit = _int_env("RATE_LIMIT_PER_MINUTE", 120)
    now = time.monotonic()
    with _lock:
        hits = _hits.setdefault(bucket, deque())
        _prune(hits, now)
        if len(hits) >= limit:
            return True
        hits.append(now)
        return False


def _auth_failures_exceeded() -> bool:
    limit = _int_env("AUTH_FAILURE_LIMIT_PER_MINUTE", 60)
    now = time.monotonic()
    with _lock:
        _prune(_failures, now)
        if len(_failures) >= limit:
            return True
        _failures.append(now)
        return False


def require_api_key(x_api_key: Optional[str] = Security(api_key_header)):
    known = _configured_keys()
    if not known:
        # Sem chave configurada a API fica fechada, nunca aberta.
        raise HTTPException(status_code=500, detail="API_KEY não configurada no servidor.")

    if x_api_key is not None and _key_matches(x_api_key, known):
        if _rate_limited(x_api_key):
            raise HTTPException(
                status_code=429,
                detail="Muitas leituras nesta chave. Espere um minuto.",
                headers={"Retry-After": "60"},
            )
        return

    if _auth_failures_exceeded():
        raise HTTPException(
            status_code=429,
            detail="Muitas tentativas. Espere um minuto.",
            headers={"Retry-After": "60"},
        )
    raise HTTPException(status_code=403, detail="API key inválida ou ausente.")
