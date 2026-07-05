from fastapi import Request, HTTPException, status, Depends
from omni_memory.config import settings

def require_admin_enabled():
    if not settings.enable_admin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin disabled")

def admin_api_key_guard(request: Request, _: None = Depends(require_admin_enabled)):
    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key")
    if key != settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid X-API-Key")
    return True
