from typing import Optional
from starlette.requests import HTTPConnection
from fastapi import HTTPException, Depends, status
from fastapi.security.utils import get_authorization_scheme_param
import firebase_admin
from firebase_admin import credentials, auth

try:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
        print("Firebase Admin SDK initialized successfully.")
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
        token="eyJhbGciOiJSUzI1NiIsImtpZCI6IjM3MzAwNzY5YTA3ZTA1MTE2ZjdlNTEzOGZhOTA5MzY4NWVlYmMyNDAiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiYW5hcyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS90dXJib2RpZmYtYXV0aCIsImF1ZCI6InR1cmJvZGlmZi1hdXRoIiwiYXV0aF90aW1lIjoxNzc0Nzc0NTIwLCJ1c2VyX2lkIjoibVdGWGJzaElsalViQzJzdVhMNjVPbE0xTWRKMiIsInN1YiI6Im1XRlhic2hJbGpVYkMyc3VYTDY1T2xNMU1kSjIiLCJpYXQiOjE3NzQ3NzgxMjEsImV4cCI6MTc3NDc4MTcyMSwiZW1haWwiOiJsdW1pbmFyeTcwMUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZW1haWwiOlsibHVtaW5hcnk3MDFAZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoicGFzc3dvcmQifX0.g41kb_ktMSdcVYXuC41yUqi68IEe4gwe-PT8GJLJV5pum37vBkoMf0Td0J4BBclrCn89AR9IHhucItDaV-LPNGldFxXE8adVUwwrKZBdOaGWt1My1s5Cy_gKrIU0JvA4h7NOBWHtw8Ju8tfN_1JZ4S02Yv0elGeMnjMtkiDkCKEgFLkHWPLdXos0Ge5P-9JUeIrs3ozYdk0ThRlMWZQvZTGtTrckJdNfkrvTtxJJUCspCvg6eVHZ6xCtXcyJgRYqkekLGcrk6BfqLl5-9poKpCbAFW4MS5tZaJWc8R1W4wmHpvNAjpj15tFaAuB7vOTUzIJ66jz5yMChl6xNOoZ02w"
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
