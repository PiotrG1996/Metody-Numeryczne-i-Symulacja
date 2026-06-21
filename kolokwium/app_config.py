import os
import socket
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).parent.resolve()
INTERNAL_QUESTIONS_PATH = APP_DIR / "data" / "kolokwium.json"
LOCAL_QUESTIONS_PATH = APP_DIR.parent / "kolokwium.json"
RECIPIENT_EMAIL = "piotr.gapski@doctorate.put.poznan.pl"
WEB3FORMS_URL = "https://api.web3forms.com/submit"
RESULTS_DIR = Path("wyniki")
EXAM_DURATION_SECONDS = 20 * 60
EXAM_TIMER_ENABLED = False
DEFAULT_PORT = os.getenv("STREAMLIT_SERVER_PORT", "8501")

STREAMLIT_CLOUD_URL = "https://metody-numeryczne-i-symulacja.streamlit.app"
_AUTO_URL_VALUES = {"", "auto", "local"}

__all__ = [
    "APP_DIR",
    "INTERNAL_QUESTIONS_PATH",
    "LOCAL_QUESTIONS_PATH",
    "get_questions_path",
    "RECIPIENT_EMAIL",
    "WEB3FORMS_URL",
    "RESULTS_DIR",
    "EXAM_DURATION_SECONDS",
    "EXAM_TIMER_ENABLED",
    "STREAMLIT_CLOUD_URL",
    "get_app_base_url",
    "get_upload_page_url",
    "get_web3forms_access_key",
    "is_production",
]


def get_questions_path() -> Path:
    """Lokalnie: root kolokwium.json (jeśli istnieje); na Cloud: data/kolokwium.json."""
    if override := os.getenv("QUESTIONS_PATH", "").strip():
        return Path(override)

    if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
        return INTERNAL_QUESTIONS_PATH

    if LOCAL_QUESTIONS_PATH.is_file():
        return LOCAL_QUESTIONS_PATH

    return INTERNAL_QUESTIONS_PATH


def _from_secrets(key: str, default: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return default


def _get_local_lan_ip() -> str | None:
    """Adres IPv4 komputera w sieci lokalnej (Wi‑Fi / LAN)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _port_from_host(host: str) -> str:
    return host.split(":")[-1] if ":" in host else DEFAULT_PORT


def _get_url_from_request() -> str | None:
    """URL na podstawie nagłówka Host (LAN, localhost lub Streamlit Cloud)."""
    try:
        host = st.context.headers.get("Host")
        if not host:
            return None

        hostname = host.split(":")[0]
        scheme = st.context.headers.get("X-Forwarded-Proto", "http")

        if hostname.endswith(".streamlit.app"):
            return f"https://{hostname}"

        port = _port_from_host(host)
        if "localhost" in host or host.startswith("127.0.0.1"):
            lan_ip = _get_local_lan_ip()
            if lan_ip:
                return f"http://{lan_ip}:{port}"
            return f"http://{host}"

        return f"{scheme}://{host}"
    except Exception:
        return None


def is_production() -> bool:
    """True na Streamlit Cloud (produkcja)."""
    try:
        host = st.context.headers.get("Host", "")
        if ".streamlit.app" in host:
            return True
    except Exception:
        pass
    configured = (os.getenv("APP_BASE_URL") or _from_secrets("APP_BASE_URL", "")).strip()
    return configured.rstrip("/") == STREAMLIT_CLOUD_URL


def get_app_base_url() -> str:
    configured = (os.getenv("APP_BASE_URL") or _from_secrets("APP_BASE_URL", "")).strip().rstrip("/")

    if configured.lower() not in _AUTO_URL_VALUES:
        return configured

    if request_url := _get_url_from_request():
        return request_url

    # Fallback: produkcja → Cloud, lokalnie → LAN / localhost
    if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
        return STREAMLIT_CLOUD_URL

    if lan_ip := _get_local_lan_ip():
        return f"http://{lan_ip}:{DEFAULT_PORT}"

    return f"http://localhost:{DEFAULT_PORT}"


def get_web3forms_access_key() -> str:
    return os.getenv("WEB3FORMS_ACCESS_KEY") or _from_secrets("WEB3FORMS_ACCESS_KEY")


def get_upload_page_url() -> str:
    return (os.getenv("UPLOAD_PAGE_URL") or _from_secrets("UPLOAD_PAGE_URL", "")).strip()
