# Laboratorium Metody Numeryczne i Symulacja (MNiS)

Kompleksowe repozytorium zawierające aplikację kolokwium Streamlit, materiały dydaktyczne (zadania laboratoryjne) oraz przykładowe symulacje z przedmiotu **Metody Numeryczne i Symulacja**.

---

## 📋 Spis treści

- [Struktura repozytorium](#-struktura-repozytorium)
- [Aplikacja kolokwium](#-aplikacja-kolokwium)
  - [Wymagania](#wymagania)
  - [Uruchomienie lokalne](#uruchomienie-lokalne)
  - [Konfiguracja](#konfiguracja)
  - [Zmienne środowiskowe / Secrets](#zmienne-środowiskowe--secrets)
- [Streamlit Cloud (rozwiązanie produkcyjne)](#-streamlit-cloud-produkcja)
  - [Wdrożenie](#wdrożenie)
  - [Wskazówki](#wskazówki)
- [Wysyłanie wyników (Web3Forms)](#-wysyłanie-wyników-web3forms)
- [Zadania laboratoryjne](#-zadania-laboratoryjne)
- [Przykłady i symulacje](#-przykłady-i-symulacje)
- [Bezpieczeństwo](#-bezpieczeństwo)

---

## 📁 Struktura repozytorium

```
├── app.py                          # Główny plik uruchomieniowy (bootstrapper)
├── requirements.txt                # Zależności Python (root)
├── README.md                       # Ten plik
├── .gitignore
│
├── kolokwium/                      # Aplikacja kolokwium (Streamlit)
│   ├── app.py                      # Główna aplikacja Streamlit
│   ├── app_config.py               # Konfiguracja (URL, klucze, ścieżki)
│   ├── pdf_export.py               # Eksport wyników do PDF
│   ├── questions_loader.py         # Wczytywanie pytań z JSON
│   ├── send_results.py             # Wysyłanie wyników (Web3Forms)
│   ├── token_utils.py              # Generowanie/weryfikacja tokenów
│   ├── utils.py                    # Funkcje pomocnicze
│   ├── code_snippet.json           # Fragment kodu (do zadań)
│   ├── pytania.json                # Pytania (root, lokalnie)
│   ├── requirements.txt            # Zależności aplikacji
│   ├── setup_and_run.sh            # Skrypt uruchomieniowy (Linux/macOS)
│   ├── setup_and_run.ps1           # Skrypt uruchomieniowy (Windows)
│   │
│   ├── data/
│   │   └── kolokwium.json          # Pytania (wersja dla Streamlit Cloud)
│   │
│   └── fonts/
│       ├── ArialBold.ttf           # Czcionka do PDF (bold)
│       └── ArialUnicode.ttf        # Czcionka do PDF (Unicode)
│
├── zadania/                        # Materiały dydaktyczne (PDF)
│   ├── MNiS_lab_01.pdf             # Laboratorium 1
│   ├── MNiS_lab_02.pdf             # Laboratorium 2
│   ├── ...                         # (aż do 10)
│   └── MNiS_lab_10.pdf
│
├── examples/                       # Przykłady i instrukcje dodatkowe
│   ├── instrukcja_fala_trojkatna.pdf
│   └── Symulacje_extra.pdf
│
└── images/                         # Wykorzystywane obrazy
    ├── blad_sredni.png
    ├── calkowanie_numeryczne_1.png
    ├── odchylenie.png
    └── wartosc_skuteczna.png
```

---

## 📝 Aplikacja kolokwium

Aplikacja Streamlit do przeprowadzania kolokwiów z **Metod Numerycznych i Symulacji**. Umożliwia:

- Rozwiązywanie zadań przez studentów w czasie rzeczywistym
- Eksport wyników do plików CSV/PDF
- Wysyłanie wyników e-mailem przez Web3Forms
- Generowanie kodów QR z linkami do pobrania wyników

### Wymagania

- Python 3.10+
- `pip` (Python package manager)

### Uruchomienie lokalne

```bash
# Sklonuj repozytorium
git clone https://github.com/PiotrG1996/Metody-Numeryczne-i-Symulacja.git
cd Metody-Numeryczne-i-Symulacja/kolokwium

# Utwórz i aktywuj wirtualne środowisko
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom aplikację
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem `http://localhost:8501`.

### Konfiguracja

Plik `.env` (w katalogu `kolokwium/`):

```env
APP_BASE_URL=auto
DOWNLOAD_TOKEN_SECRET=your-long-random-secret
WEB3FORMS_ACCESS_KEY=your-web3forms-access-key
UPLOAD_PAGE_URL=                    # opcjonalnie
QUESTIONS_PATH=                     # opcjonalnie, ścieżka do pliku z pytaniami
```

Parametry:

| Zmienna             | Opis                                                               | Wymagana |
|---------------------|--------------------------------------------------------------------|----------|
| `APP_BASE_URL`      | Bazowy URL aplikacji (`auto` = wykryj automatycznie)               | ✅       |
| `DOWNLOAD_TOKEN_SECRET` | Tajny klucz do generowania tokenów pobierania wyników           | ✅       |
| `WEB3FORMS_ACCESS_KEY` | Klucz API Web3Forms do wysyłki e-mail                            | ✅       |
| `UPLOAD_PAGE_URL`   | URL strony przesyłania wyników (opcjonalnie)                       | ❌       |
| `QUESTIONS_PATH`    | Ścieżka do niestandardowego pliku z pytaniami                      | ❌       |

### Zmienne środowiskowe / Secrets

W aplikacji priorytet jest następujący (od najwyższego):
1. `st.secrets` (Streamlit Cloud Secrets)
2. Zmienna środowiskowa (np. `os.getenv`)
3. Plik `.env` (wczytywany przez `python-dotenv`)
4. Wartość domyślna w kodzie

#### Ustawianie `.env` lokalnie

```bash
cd kolokwium
cat > .env << EOF
APP_BASE_URL=auto
DOWNLOAD_TOKEN_SECRET=my-strong-random-secret-123
WEB3FORMS_ACCESS_KEY=32f40995-9191-4b45-99ec-61b53751ec0b
EOF
```

> **UWAGA:** Plik `.env` został dodany do `.gitignore` — nie commituj go do repozytorium.

---

## ☁️ Streamlit Cloud (rozwiązanie produkcyjne)

- **URL:** [https://metody-numeryczne-i-symulacja.streamlit.app/](https://metody-numeryczne-i-symulacja.streamlit.app/)
- **Main file:** `app.py` (bootstrapper uruchamiający `kolokwium/app.py`)
- **Python version:** 3.10+

### Wdrożenie

1. Podłącz repozytorium GitHub do [Streamlit Cloud](https://streamlit.io/cloud)
2. Ustaw **Main file** na `app.py`
3. Skonfiguruj **Secrets** (patrz poniżej)
4. Wdróż — aplikacja uruchomi się automatycznie

### Secrets (Streamlit Cloud)

Skopiuj poniższy blok do **Streamlit Cloud → Settings → Secrets**:

```toml
APP_BASE_URL = "https://metody-numeryczne-i-symulacja.streamlit.app"
DOWNLOAD_TOKEN_SECRET = "your-long-random-secret"
WEB3FORMS_ACCESS_KEY = "32f40995-9191-4b45-99ec-61b53751ec0b"
```

### Wskazówki dotyczące Streamlit Cloud

- **Produkcja:** https://metody-numeryczne-i-symulacja.streamlit.app/
- W Secrets ustaw `APP_BASE_URL = "https://metody-numeryczne-i-symulacja.streamlit.app"` — lub `auto`, które wykryje domenę `.streamlit.app` automatycznie.
- Wyniki **nie są trwale zapisywane** na dysku serwera — student pobiera CSV/PDF od razu lub przez **QR** (link z tokenem).
- Lokalnie ustaw `APP_BASE_URL=auto` w `.env` — aplikacja użyje adresu LAN (np. `http://192.168.x.x:8501`).
- Ustaw `DOWNLOAD_TOKEN_SECRET` na losowy długi ciąg (ten sam w Secrets i lokalnie).
- Plik `kolokwium/data/kolokwium.json` zawiera pytania — nie udostępniaj go studentom.
- Wymagane czcionki w `kolokwium/fonts/` do generowania PDF (ArialUnicode.ttf / ArialBold.ttf).

---

## 📬 Wysyłanie wyników (Web3Forms)

Aplikacja wykorzystuje [Web3Forms](https://web3forms.com/) jako zewnętrzną usługę do wysyłania powiadomień e-mail z wynikami kolokwium.

### Konfiguracja w panelu Web3Forms

1. Zaloguj się do [dashboardu Web3Forms](https://app.web3forms.com/dashboard)
2. Utwórz nowy formularz lub skonfiguruj istniejący
3. Uzyskaj **Access Key**
4. Skonfiguruj adresy e-mail odbiorców (wyniki będą wysyłane na adres prowadzącego):
   `piotr.gapski@doctorate.put.poznan.pl`
5. Opcjonalnie ustaw temat wiadomości i powiadomienia (potwierdzenie wysyłki, błędy, status)

### Klucz API

```env
WEB3FORMS_ACCESS_KEY=32f40995-9191-4b45-99ec-61b53751ec0b
```

> **Uwaga:** Klucz w powyższym przykładzie jest przykładowy. W środowisku produkcyjnym należy użyć własnego klucza.

### Działanie

- `send_results.py` — moduł odpowiedzialny za wysyłanie wyników przez Web3Forms
- Wyniki są wysyłane jako załączniki lub w treści wiadomości
- Web3Forms nie wymaga własnego backendu — działa jako usługa zewnętrzna (API)

---

## 📚 Zadania laboratoryjne

W katalogu `zadania/` znajdują się instrukcje do laboratoriów w formacie PDF (łącznie 10 zestawów):

| Plik                        | Temat                                         |
|-----------------------------|-----------------------------------------------|
| `MNiS_lab_01.pdf`           | Laboratorium 1                                |
| `MNiS_lab_02.pdf`           | Laboratorium 2                                |
| `MNiS_lab_03.pdf`           | Laboratorium 3                                |
| `MNiS_lab_04.pdf`           | Laboratorium 4                                |
| `MNiS_lab_05.pdf`           | Laboratorium 5                                |
| `MNiS_lab_06.pdf`           | Laboratorium 6                                |
| `MNiS_lab_07.pdf`           | Laboratorium 7                                |
| `MNiS_lab_08.pdf`           | Laboratorium 8                                |
| `MNiS_lab_09.pdf`           | Laboratorium 9                                |
| `MNiS_lab_09_v2.pdf`        | Laboratorium 9 (wersja 2)                     |
| `MNiS_lab_10.pdf`           | Laboratorium 10                               |

---

## 🧪 Przykłady i symulacje

| Plik                                        | Opis                                                    |
|---------------------------------------------|---------------------------------------------------------|
| `examples/instrukcja_fala_trojkatna.pdf`    | Instrukcja dotycząca generowania fali trójkątnej        |
| `examples/Symulacje_extra.pdf`              | Dodatkowe materiały z symulacji                         |

### Obrazy

W katalogu `images/` znajdują się grafiki wykorzystywane w materiałach dydaktycznych:

- `blad_sredni.png` — błąd średni
- `calkowanie_numeryczne_1.png` — całkowanie numeryczne
- `odchylenie.png` — odchylenie
- `wartosc_skuteczna.png` — wartość skuteczna

---

## 🔒 Bezpieczeństwo

| ✅ Dozwolone                                       | ❌ Zabronione                         |
|----------------------------------------------------|---------------------------------------|
| Używanie Streamlit Secrets                         | Commitowanie kluczy API do repozytorium |
| Używanie zmiennych środowiskowych                  | Przechowywanie sekretów w kodzie źródłowym |
| Regularna rotacja kluczy API (jeśli możliwe)       | Udostępnianie pliku `kolokwium.json` studentom |
| Używanie Web3Forms (brak potrzeby własnego backendu) |                                       |

### Pliki ignorowane przez Git (`.gitignore`)

```
kolokwium/venv/
kolokwium/__pycache__/
kolokwium/.env
kolokwium/wyniki/
/kolokwium.json
__pycache__/
*.pyc
.DS_Store
```

> Przed pierwszym commitem upewnij się, że żadne poufne dane nie znajdują się w plikach śledzonych przez Git.

---

## 🚀 Skrypty uruchomieniowe

W katalogu `kolokwium/` dostępne są skrypty ułatwiające pierwszą konfigurację:

```bash
# Linux/macOS
chmod +x setup_and_run.sh
./setup_and_run.sh

# Windows (PowerShell)
.\setup_and_run.ps1
```

Skrypty automatycznie tworzą wirtualne środowisko, instalują zależności i uruchamiają aplikację.

---

## 👤 Autor

**Piotr Gapski**  
E-mail: piotr.gapski@doctorate.put.poznan.pl

---

## 📄 Licencja

Projekt edukacyjny — Politechnika Poznańska.
