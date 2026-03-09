from typing import Optional
from starlette.requests import HTTPConnection
from fastapi import HTTPException, Depends, status
from fastapi.security.utils import get_authorization_scheme_param
import firebase_admin
from firebase_admin import credentials, auth

try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
except Exception as e:
    print(f"Warning: Could not initialize Firebase Admin SDK automatically. Error: {e}")

async def get_current_user(connection: HTTPConnection):
    """
    Universal dependency to get the current user from Firebase JWT token.
    """
    token = None
    
    # 1. Check Authorization Header
    authorization = connection.headers.get("Authorization")
    if authorization:
        scheme, credentials = get_authorization_scheme_param(authorization)
        if scheme.lower() == "bearer":
            token = credentials
    
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
