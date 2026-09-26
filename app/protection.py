from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

# Um lote no teto (100 mil pedidos) cabe aqui. Acima disso a chamada é recusada antes do cálculo.
MAX_BODY_BYTES = 10 * 1024 * 1024

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS.items():
                    headers[name] = value
                path = scope.get("path", "")
                if not path.startswith("/docs") and path != "/redoc":
                    headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                chunks.append(message)
                break
            if message["type"] != "http.request":
                chunks.append(message)
                break
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Corpo da requisição grande demais."},
                )
                await response(scope, _closed, send)
                return
            chunks.append(message)
            if not message.get("more_body", False):
                break

        index = 0

        async def replay():
            nonlocal index
            if index < len(chunks):
                current = chunks[index]
                index += 1
                return current
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


async def _closed():
    return {"type": "http.disconnect"}
