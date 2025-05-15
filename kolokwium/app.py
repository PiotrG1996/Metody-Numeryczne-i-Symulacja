import streamlit as st
import pandas as pd
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pathlib import Path
from questions_loader import QuestionLoader
from utils import evaluate_answers, sanitize_filename

# --- Inicjalizacja modułów ---
load_dotenv()
loader = QuestionLoader()

# --- Stałe aplikacji ---
RESULTS_DIR = Path("wyniki")
EMAIL_CONFIG = {
    "host": os.getenv("EMAIL_HOST"),
    "port": int(os.getenv("EMAIL_PORT")),
    "user": os.getenv("EMAIL_USER"),
    "password": os.getenv("EMAIL_PASS")
}

def initialize_session_state():
    """Inicjalizuje stan sesji z wartościami domyślnymi"""
    session_defaults = {
        "questions": [],
        "results": None,
        "user_answers": {},
        "uploaded_file": None,
        "selected_file": "pytania.json",
        "student_name": "",
        "recipient_email": "piotr.gapski@doctorate.put.poznan.pl"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            if key == "user_answers":  # Inicjalizuj jako słownik z pustymi wartościami
                st.session_state[key] = {str(i): "" for i in range(len(st.session_state.questions))}
            else:
                st.session_state[key] = value

def handle_file_loading():
    """Obsługa procesu ładowania plików"""
    try:
        if st.session_state.uploaded_file:
            file_ext = Path(st.session_state.uploaded_file.name).suffix.lower()
            if file_ext == ".json":
                st.session_state.questions = loader.from_json(st.session_state.uploaded_file)
            elif file_ext == ".csv":
                st.session_state.questions = loader.from_csv(st.session_state.uploaded_file)
        elif st.session_state.selected_file:
            file_path = Path(st.session_state.selected_file)
            if file_path.suffix == ".json":
                st.session_state.questions = loader.from_json(file_path)
            elif file_path.suffix == ".csv":
                st.session_state.questions = loader.from_csv(file_path)
        
        st.session_state.results = None
        st.session_state.user_answers = {}
        st.sidebar.success(f"✅ Załadowano {len(st.session_state.questions)} pytań!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Błąd ładowania: {str(e)}")
        st.session_state.questions = []

def save_results(results: dict) -> Path:
    """Zapisuje wyniki do plików CSV i JSON"""
    try:
        sanitized_name = sanitize_filename(st.session_state.student_name)
        RESULTS_DIR.mkdir(exist_ok=True)

        df = pd.DataFrame([{
            "Pytanie": q["question"],
            "Twoja odpowiedź": a["user_answer"],
            "Poprawna odpowiedź": q["correct"],
            "Wynik": "Poprawnie" if a["is_correct"] else "Błędnie"
        } for q, a in zip(st.session_state.questions, results["results"])])

        csv_path = RESULTS_DIR / f"wynik_{sanitized_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        return csv_path
        
    except Exception as e:
        st.error(f"❌ Błąd zapisu wyników: {str(e)}")
        st.stop()

def send_email(results: dict, attachment_path: Path):
    """Wysyła wyniki emailem z załącznikiem"""
    try:
        # Tworzenie treści emaila
        body = f"""
        Wyniki zaliczenia - Metody Numeryczne i Symulacja
        -----------------------------------------------
        Student: {st.session_state.student_name}
        Data: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        Wynik: {results['score']}/{len(st.session_state.questions)}
        """
        
        # Konfiguracja wiadomości
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG["user"]
        msg["To"] = st.session_state.recipient_email
        msg["Subject"] = f"Wyniki testu - {sanitize_filename(st.session_state.student_name)}"
        
        # Dodaj treść i załącznik
        msg.attach(MIMEText(body, "plain"))
        
        with open(attachment_path, "rb") as f:
            attachment = MIMEText(f.read(), _subtype="csv")
            attachment.add_header("Content-Disposition", "attachment", filename=attachment_path.name)
            msg.attach(attachment)

        # Wysyłanie emaila
        with smtplib.SMTP(EMAIL_CONFIG["host"], EMAIL_CONFIG["port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["user"], EMAIL_CONFIG["password"])
            server.send_message(msg)
            
        st.success(f"📧 Wyniki wysłane do: {st.session_state.recipient_email}")
        
    except Exception as e:
        st.error(f"❌ Błąd wysyłania emaila: {str(e)}")

def main():
    # --- Konfiguracja strony ---
    st.set_page_config(
        page_title="System Zaliczeniowy MNiS",
        page_icon="🧪",
        layout="centered"
    )
    st.title("🧪 Termin 1 - Laboratorium MNiS")
    
    initialize_session_state()

    # --- Panel boczny ---
    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        
        # Ładowanie plików
        st.subheader("📁 Źródło pytań")
        st.session_state.uploaded_file = st.file_uploader(
            "Wgraj plik (JSON/CSV)",
            type=["json", "csv"],
            accept_multiple_files=False
        )
        st.session_state.selected_file = st.text_input(
            "Lub podaj ścieżkę do pliku",
            value="pytania.json"
        )
        
        if st.button("🔄 Załaduj pytania"):
            handle_file_loading()

        # Dane studenta
        st.subheader("🎓 Dane studenta")
        st.session_state.student_name = st.text_input("Imię i nazwisko")
        st.session_state.recipient_email = st.text_input(
            "Email odbiorcy",
            value="piotr.gapski@doctorate.put.poznan.pl"
        )

    # --- Główny interfejs ---
    if not st.session_state.questions:
        st.info("⏳ Najpierw wgraj plik z pytaniami w panelu bocznym")
        return

    with st.form("quiz_form"):
        st.subheader("📝 Test zaliczeniowy")
        
        for idx, question in enumerate(st.session_state.questions):
            with st.container():
                st.markdown(f"### Pytanie {idx+1}")
                st.markdown(f"**{question['question']}**")
                
                if "code" in question:
                    st.code(question["code"], language="python")
                
                options = question["options"]
                
                # Użyj klucza jako stringa i zainicjuj wartość domyślną
                answer_key = f"q_{idx}"
                selected = st.radio(
                    label="Wybierz odpowiedź:",
                    options=list(options.keys()),
                    format_func=lambda k: f"{k}) {options[k]}",
                    key=answer_key,
                    index=None  # Wymuś wybór odpowiedzi
                )
                
                # Zapisz odpowiedź jako string
                st.session_state.user_answers[str(idx)] = selected if selected else ""
                st.divider()

        if st.form_submit_button("✅ Zakończ test"):
            if not st.session_state.student_name:
                st.error("❗ Wprowadź imię i nazwisko")
                st.stop()
                
            if len(st.session_state.user_answers) != len(st.session_state.questions):
                st.error("❗ Odpowiedz na wszystkie pytania!")
                st.stop()
                
            st.session_state.results = evaluate_answers(
                st.session_state.questions,
                st.session_state.user_answers
            )
            csv_path = save_results(st.session_state.results)
            send_email(st.session_state.results, csv_path)
            st.rerun()


    if st.session_state.results:
        st.subheader("📊 Wyniki")
        
        for idx, result in enumerate(st.session_state.results["results"]):
            question = st.session_state.questions[idx]
            user_answer = result["user_answer"] if result["user_answer"] else "BRAK ODPOWIEDZI"
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Pytanie {idx+1}**")
                st.markdown("✅" if result["is_correct"] else "❌")
                
            with col2:
                st.markdown(f"**{question['question']}**")
                st.markdown(f"Twoja odpowiedź: `{user_answer}`")
                st.markdown(f"Poprawna odpowiedź: `{question['correct']}`")
                if "explanation" in question:
                    st.info(f"**Wyjaśnienie:** {question['explanation']}")
                st.divider()

        st.metric("Wynik końcowy", 
                f"{st.session_state.results['score']}/{len(st.session_state.questions)}")
        
if __name__ == "__main__":
    main()
