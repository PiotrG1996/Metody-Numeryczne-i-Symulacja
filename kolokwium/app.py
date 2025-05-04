import streamlit as st
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from dotenv import load_dotenv
from questions_loader import load_questions_from_json, load_questions_from_csv
from utils import evaluate_answers, sanitize_filename

# === Wczytaj dane z .env ===
load_dotenv()
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# === Konfiguracja aplikacji ===
st.set_page_config(page_title="Zaliczenie", layout="centered")
st.title("🧪 Zaliczenie - Metody Numeryczne i Symulacja")

# === PANEL BOCZNY: Wczytywanie pytań i danych użytkownika ===
st.sidebar.header("📁 Wczytaj pytania")
source = st.sidebar.selectbox("Format pliku", ["JSON", "CSV"])
filename = st.sidebar.text_input("Nazwa pliku", value="pytania.json")

st.sidebar.header("🧑‍🎓 Dane studenta")
student_name = st.sidebar.text_input("Imię i nazwisko")
recipient_email = st.sidebar.text_input("Adres e-mail odbiorcy", value="piotr.gapski@doctorate.put.poznan.pl")

# === Inicjalizacja sesji ===
if "questions" not in st.session_state:
    st.session_state.questions = []
if "results" not in st.session_state:
    st.session_state.results = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

# === Przycisk ładowania pytań ===
if st.sidebar.button("📥 Załaduj pytania"):
    try:
        if source == "JSON":
            questions = load_questions_from_json(filename)
        else:
            questions = load_questions_from_csv(filename)

        if not questions:
            st.warning("⚠️ Nie znaleziono żadnych pytań w pliku.")
            st.session_state.questions = []
        else:
            st.session_state.questions = questions
            st.session_state.results = None
            st.session_state.user_answers = {}
            st.success(f"✅ Załadowano {len(questions)} pytań.")

    except FileNotFoundError:
        st.error(f"❌ Nie znaleziono pliku: `{filename}`")
    except Exception as e:
        st.error(f"❌ Błąd wczytywania: {e}")

# === Wyświetlanie pytań ===
questions = st.session_state.get("questions", [])

if questions:
    with st.form("quiz_form"):
        st.subheader("📝 Odpowiedz na pytania")
        for idx, q in enumerate(questions):
            options = q['options']
            
            # Prefill the answer if already stored in session state
            selected_answer = st.session_state.user_answers.get(str(idx))
            selected = st.radio(
                f"{idx+1}. {q['question']}",
                options=list(options.keys()),
                format_func=lambda k: f"{k}) {options[k]}",
                key=f"q_{idx}"
            )
            st.session_state.user_answers[str(idx)] = selected

            if "code" in q:
                st.code(q["code"], language="python")

        submitted = st.form_submit_button("✅ Sprawdź odpowiedzi")

    if submitted:
        if not student_name:
            st.error("❗ Wprowadź imię i nazwisko przed wysłaniem wyników.")
        elif not recipient_email:
            st.error("❗ Wprowadź adres e-mail odbiorcy przed wysłaniem wyników.")
        else:
            if len(st.session_state.user_answers) != len(questions):
                st.error("❌ Wszystkie pytania muszą być odpowiedziane!")
            else:
                results_data = evaluate_answers(questions, st.session_state.user_answers)
                st.session_state.results = results_data

                # Zapis wyników
                try:
                    sanitized_name = sanitize_filename(student_name)
                    os.makedirs("wyniki", exist_ok=True)

                    df = pd.DataFrame([{
                        "Pytanie": q["question"],
                        "Twoja odpowiedź": a["user_answer"],
                        "Poprawna odpowiedź": q["correct"],
                        "Wynik": "Poprawnie" if a["is_correct"] else "Błędnie"
                    } for q, a in zip(questions, results_data["results"])])
                    
                    # Zapisz pliki z sanitizowaną nazwą
                    csv_path = f"wyniki/wynik_{sanitized_name}.csv"
                    json_path = f"wyniki/wynik_{sanitized_name}.json"
                    
                    df.to_csv(csv_path, index=False)
                    df.to_json(json_path, orient="records")
                    st.success(f"📁 Pliki zapisano w: {os.path.abspath(csv_path)}")

                except Exception as e:
                    st.error(f"❌ Błąd zapisu wyników: {str(e)}")

                # Wysyłanie e-maila
                try:
                    body = f"""
                    Wyniki zaliczenia - Metody Numeryczne i Symulacja
                    -----------------------------------------------
                    Student: {student_name}
                    Data: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                    Wynik: {results_data['score']}/{len(questions)}

                    Szczegółowe wyniki:
                    {df.to_string(index=False)}
                    """

                    msg = MIMEMultipart()
                    msg["From"] = EMAIL_USER
                    msg["To"] = recipient_email
                    msg["Subject"] = f"Wyniki testu - {sanitized_name}"
                    msg.attach(MIMEText(body, "plain"))

                    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                        server.starttls()
                        server.login(EMAIL_USER, EMAIL_PASS)
                        server.send_message(msg)

                    st.success(f"📧 Wyniki wysłane do: {recipient_email}")
                except Exception as e:
                    st.error(f"❌ Błąd wysyłania e-mail: {str(e)}")


# === Wyświetlanie wyników ===
if st.session_state.results:
    results = st.session_state.results
    st.subheader("📊 Wyniki")
    
    # Oblicz sanitized_name na nowo na podstawie aktualnej nazwy studenta
    sanitized_name = sanitize_filename(student_name) or "anonymous"  # Dodane tutaj
    
    for idx, result in enumerate(results["results"]):
        q = questions[idx]
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"**Pytanie {idx+1}**")
            st.markdown("✅ Poprawnie" if result["is_correct"] else "❌ Błędnie")
            
        with col2:
            st.markdown(f"**{q['question']}**")
            st.markdown(f"Twoja odpowiedź: `{result['user_answer']}`")
            st.markdown(f"Poprawna odpowiedź: `{q['correct']}`")
            if "explanation" in q:
                st.info(f"**Wyjaśnienie:** {q['explanation']}")
        
        st.divider()

    st.markdown(f"## Podsumowanie: **{results['score']}/{len(questions)}**")
    
    # Przyciski pobierania
    df = pd.DataFrame(results["results"])
    
    st.download_button(
        "📥 Pobierz wyniki jako CSV",
        df.to_csv(index=False),
        file_name=f"wyniki/wynik_{sanitized_name}.csv"
    )

    st.download_button(
        "📥 Pobierz wyniki jako JSON",
        df.to_json(orient="records"),
        file_name=f"wyniki/wynik_{sanitized_name}.json"
    )