import base64
from typing import Any

try:
    from security import SecurityHandler  # type: ignore  # reuse existing module at repo root
except Exception:  # pragma: no cover
    SecurityHandler = None  # type: ignore

class SecurityService:
    def __init__(self):
        self.available = SecurityHandler is not None
        if self.available:
            try:
                self.handler = SecurityHandler()  # type: ignore[operator]
            except Exception:  # pragma: no cover
                self.handler = None
                self.available = False
        else:
            self.handler = None

    def encrypt(self, data: Any) -> str:
        if not self.available or not self.handler:
            return "SECURITY_MODULE_UNAVAILABLE"
        enc = self.handler.encrypt(data)
        return base64.b64encode(enc).decode()

    def decrypt(self, token: str) -> str:
        if not self.available or not self.handler:
            return "SECURITY_MODULE_UNAVAILABLE"
        try:
            raw = base64.b64decode(token.encode())
            dec = self.handler.decrypt(raw)
            return dec
        except Exception as e:  # pragma: no cover
            return f"DECRYPT_ERROR:{e}"

security_service = SecurityService()
