from typing import Optional
from starlette.requests import HTTPConnection
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param
import firebase_admin
from firebase_admin import credentials, auth

try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
        print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize Firebase Admin SDK automatically. Error: {e}")

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    connection: HTTPConnection,
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
):
    """
    Universal dependency to get the current user from Firebase JWT token.
    """
    token = None
    if bearer:
        token = bearer.credentials
    else:
        # Fallback for manual check if HTTPBearer didn't catch it
        authorization = connection.headers.get("Authorization")
        if authorization:
            scheme, auth_cred = get_authorization_scheme_param(authorization)
            if scheme.lower() == "bearer":
                token = auth_cred
    
    # 2. Check Query Parameters (fallback for WebSockets)
    if not token:
        token = connection.query_params.get("token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        decoded_token = auth.verify_id_token(token)
        # Store user in connection state for easy access in routes
        connection.state.user = decoded_token
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_websocket_token(websocket) -> Optional[dict]:
    """
    Manually verify a Firebase JWT token for WebSocket connections.
    Extracts the token from query params (?token=...) or the Authorization header.
    Returns the decoded token dict on success, or None on failure.
    """
    token = websocket.query_params.get("token")

    if not token:
        authorization = websocket.headers.get("Authorization")
        if authorization:
            scheme, auth_cred = get_authorization_scheme_param(authorization)
            if scheme.lower() == "bearer":
                token = auth_cred

    if not token:
        return None

    try:
        decoded_token = auth.verify_id_token(token)
        websocket.state.user = decoded_token
        return decoded_token
    except Exception:
        return None
