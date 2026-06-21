import base64
import hashlib
import hmac
import json
from typing import Any


def _get_secret() -> str:
    import os
    import streamlit as st

    if secret := os.getenv("DOWNLOAD_TOKEN_SECRET"):
        return secret
    try:
        return st.secrets["DOWNLOAD_TOKEN_SECRET"]
    except Exception:
        return "mnis-kolokwium-dev-secret"


def create_signed_download_token(payload: dict[str, Any]) -> str:
    """Token zawiera dane wyniku — działa na Streamlit Cloud bez zapisu na dysk."""
    data = base64.urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    sig = hmac.new(
        _get_secret().encode("utf-8"),
        data.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:20]
    return f"{data}.{sig}"


def verify_signed_download_token(token: str) -> dict[str, Any] | None:
    try:
        data, sig = token.rsplit(".", 1)
        expected = hmac.new(
            _get_secret().encode("utf-8"),
            data.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()[:20]
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
