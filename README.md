# Laboratorium MNiS

Aplikacja kolokwium: **Termin 1 - Laboratorium MNiS 21.06.2026**

## Streamlit Cloud

- **URL:** https://metody-numeryczne-i-symulacja.streamlit.app/
- **Main file:** `app.py` (lub `kolokwium/app.py`)
- **Secrets:** skopiuj z `.streamlit/secrets.toml.example` do Streamlit Cloud → Settings → Secrets

### Wymagane pliki w repozytorium

- `kolokwium/data/kolokwium.json` — pytania (nie udostępniaj root `kolokwium.json` studentom)
- `kolokwium/fonts/` — czcionki do PDF (DejaVuSans lub ArialUnicode)

### Ważne na Cloud

- **Produkcja:** https://metody-numeryczne-i-symulacja.streamlit.app/
- W Secrets ustaw `APP_BASE_URL = "https://metody-numeryczne-i-symulacja.streamlit.app"` (lub `auto` — wykryje domenę `.streamlit.app` automatycznie).
- Wyniki **nie są trwale zapisywane** na dysku serwera — student pobiera CSV/PDF od razu lub przez **QR** (link z tokenem).
- Lokalnie ustaw `APP_BASE_URL=auto` w `.env` — aplikacja użyje adresu LAN (np. `http://192.168.x.x:8501`).
- Ustaw `DOWNLOAD_TOKEN_SECRET` na losowy długi ciąg (ten sam w Secrets i lokalnie).

## Lokalnie

```bash
cd kolokwium
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
