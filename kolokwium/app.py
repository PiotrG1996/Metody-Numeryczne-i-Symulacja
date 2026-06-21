import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import qrcode
import time
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

from app_config import (
    RECIPIENT_EMAIL,
    WEB3FORMS_URL,
    RESULTS_DIR,
    EXAM_DURATION_SECONDS,
    EXAM_TIMER_ENABLED,
    get_questions_path,
    get_web3forms_access_key,
    get_app_base_url,
    get_upload_page_url,
)
from questions_loader import QuestionLoader
from utils import evaluate_answers, sanitize_filename, format_options_markdown, get_answer_text
from pdf_export import (
    build_results_pdf_bytes,
    build_results_pdf_bytes_from_csv,
    save_results_pdf,
)
from token_utils import create_signed_download_token, verify_signed_download_token

loader = QuestionLoader()


def get_query_param(key: str) -> str | None:
    value = st.query_params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def initialize_session_state():
    """Inicjalizuje stan sesji z wartościami domyślnymi"""
    session_defaults = {
        "questions": [],
        "results": None,
        "user_answers": {},
        "student_name": "",
        "load_error": None,
        "results_sent": False,
        "web3forms_payload": None,
        "result_csv_path": None,
        "result_download_token": None,
        "exam_started_at": None,
        "timed_out": False,
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            if key == "user_answers":
                st.session_state[key] = {str(i): "" for i in range(len(st.session_state.questions))}
            else:
                st.session_state[key] = value


def load_questions_internal() -> None:
    """Ładuje pytania z kolokwium.json przy starcie aplikacji."""
    if st.session_state.questions:
        return

    try:
        questions_path = get_questions_path()
        if not questions_path.is_file():
            raise FileNotFoundError(f"Brak pliku pytań: {questions_path}")
        st.session_state.questions = loader.from_json(questions_path)
        st.session_state.user_answers = {str(i): "" for i in range(len(st.session_state.questions))}
        st.session_state.load_error = None
    except Exception as e:
        st.session_state.load_error = str(e)
        st.session_state.questions = []


def ensure_exam_started() -> None:
    """Uruchamia sesję testu po wpisaniu imienia i nazwiska."""
    if st.session_state.exam_started_at is not None:
        return
    if not st.session_state.student_name.strip() or not st.session_state.questions:
        return
    st.session_state.exam_started_at = time.time()
    st.session_state.results = None
    st.session_state.results_sent = False
    st.session_state.web3forms_payload = None
    st.session_state.timed_out = False


def get_remaining_seconds() -> int:
    if not EXAM_TIMER_ENABLED:
        return EXAM_DURATION_SECONDS
    started = st.session_state.get("exam_started_at")
    if started is None:
        return EXAM_DURATION_SECONDS
    elapsed = time.time() - started
    return max(0, int(EXAM_DURATION_SECONDS - elapsed))


def format_remaining_time(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def finalize_exam(*, timed_out: bool = False) -> None:
    """Ocenia odpowiedzi, zapisuje wyniki i przygotowuje wysyłkę."""
    if st.session_state.results is not None:
        return

    for idx in range(len(st.session_state.questions)):
        st.session_state.user_answers.setdefault(str(idx), "")

    st.session_state.results = evaluate_answers(
        st.session_state.questions,
        st.session_state.user_answers,
    )
    csv_path = save_results(st.session_state.results)
    st.session_state.result_csv_path = str(csv_path)
    st.session_state.result_download_token = create_download_token(
        st.session_state.results, csv_path
    )
    st.session_state.web3forms_payload = build_web3forms_payload(
        st.session_state.results, csv_path
    )
    st.session_state.results_sent = False
    st.session_state.timed_out = timed_out and EXAM_TIMER_ENABLED
    st.rerun()


def exam_timer_widget():
    """Wyświetla licznik czasu (tylko gdy EXAM_TIMER_ENABLED)."""
    if st.session_state.results or not st.session_state.questions:
        return

    remaining = get_remaining_seconds()
    time_label = format_remaining_time(remaining)

    if EXAM_TIMER_ENABLED:
        if remaining <= 0:
            st.error("⏱️ Czas minął! Test zostaje automatycznie zakończony.")
            finalize_exam(timed_out=True)
            return

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("⏱️ Czas", time_label)
        with col2:
            if remaining <= 300:
                st.warning(
                    "Zostało mniej niż 5 minut. Zakończ test lub odpowiedzi zostaną wysłane automatycznie."
                )
            else:
                st.caption(
                    f"Masz {EXAM_DURATION_SECONDS // 60} minut od wpisania imienia i nazwiska."
                )
    else:
        st.caption(f"⏱️ Limit czasu: {EXAM_DURATION_SECONDS // 60} min (obecnie wyłączony)")


@st.fragment(run_every=timedelta(seconds=1))
def live_exam_timer():
    exam_timer_widget()


def render_exam_timer():
    if EXAM_TIMER_ENABLED:
        live_exam_timer()
    else:
        exam_timer_widget()


def create_download_token(results: dict, csv_path: Path) -> str:
    """Krótki token (same odpowiedzi) — mieści się w kodzie QR."""
    return create_signed_download_token({
        "v": 1,
        "student": st.session_state.student_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "score": results["score"],
        "answers": [r["user_answer"] for r in results["results"]],
        "filename": csv_path.name,
    })


def rebuild_csv_from_token(entry: dict) -> str | None:
    """Odtwarza CSV z krótkiego tokena + pliku pytań na serwerze."""
    if entry.get("csv"):
        return entry["csv"]

    answers = entry.get("answers")
    questions_path = get_questions_path()
    if not answers or not questions_path.is_file():
        return None

    try:
        questions = loader.from_json(questions_path)
    except Exception:
        return None

    if len(answers) != len(questions):
        return None

    df = pd.DataFrame([{
        "Pytanie": q["question"],
        "Twoja odpowiedź": ua,
        "Poprawna odpowiedź": q["correct"],
        "Wynik": "Poprawnie" if ua.lower() == q["correct"].lower() else "Błędnie",
    } for q, ua in zip(questions, answers)])

    return df.to_csv(index=False, encoding="utf-8")


def build_download_url(token: str) -> str:
    return f"{get_app_base_url()}/?dl={token}"


def generate_qr_image(url: str) -> bytes:
    qr = qrcode.QRCode(box_size=6, border=2, version=None)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def handle_result_download_page() -> bool:
    """Obsługuje ?dl=TOKEN — strona pobierania wyniku (link z QR)"""
    token = get_query_param("dl")
    if not token:
        return False

    st.title("📥 Pobieranie wyniku")
    entry = verify_signed_download_token(token)

    if not entry:
        st.error("❌ Nieprawidłowy lub wygasły link do pobrania.")
        return True

    student = entry.get("student", "—")
    created = entry.get("created", "")
    filename = entry.get("filename", "wynik.csv")
    csv_content = rebuild_csv_from_token(entry)

    if not csv_content:
        st.error("❌ Brak danych wyniku w linku.")
        return True

    st.info(f"Student: **{student}**")

    col_csv, col_pdf = st.columns(2)
    with col_csv:
        st.download_button(
            label="⬇️ Pobierz CSV",
            data=csv_content.encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )
    with col_pdf:
        try:
            pdf_bytes = build_results_pdf_bytes_from_csv(csv_content, student, created)
            st.download_button(
                label="📄 Pobierz PDF",
                data=pdf_bytes,
                file_name=Path(filename).with_suffix(".pdf").name,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ Nie udało się wygenerować PDF: {e}")
    return True


def save_results(results: dict) -> Path:
    """Zapisuje wyniki do pliku CSV"""
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
        df.to_csv(csv_path, index=False, encoding="utf-8")
        try:
            save_results_pdf(csv_path, st.session_state.student_name)
        except Exception:
            pass
        return csv_path

    except Exception as e:
        st.error(f"❌ Błąd zapisu wyników: {str(e)}")
        st.stop()


def build_web3forms_payload(results: dict, attachment_path: Path) -> dict:
    """Buduje payload do wysyłki Web3Forms"""
    message_lines = [
        "Wyniki zaliczenia - Metody Numeryczne i Symulacja",
        "-----------------------------------------------",
        f"Student: {st.session_state.student_name}",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Wynik: {results['score']}/{len(st.session_state.questions)}",
        "",
        "Szczegóły odpowiedzi:",
    ]
    for idx, (question, result) in enumerate(
        zip(st.session_state.questions, results["results"])
    ):
        status = "Poprawnie" if result["is_correct"] else "Błędnie"
        message_lines.append(
            f"{idx + 1}. {status} | odpowiedź: {result['user_answer'] or '-'} "
            f"| poprawna: {question['correct']}"
        )

    return {
        "access_key": get_web3forms_access_key(),
        "subject": f"Wyniki testu - {sanitize_filename(st.session_state.student_name)}",
        "name": st.session_state.student_name,
        "message": "\n".join(message_lines),
    }


def send_results_via_browser(payload: dict):
    """Wysyła wyniki z przeglądarki użytkownika (Web3Forms wymaga client-side)"""
    if not get_web3forms_access_key():
        return

    payload_json = json.dumps(payload)
    components.html(
        f"""
        <div id="w3f-status" style="font-family:sans-serif;font-size:14px;padding:4px 0;">
            Wysyłanie wyników na {RECIPIENT_EMAIL}...
        </div>
        <script>
        (async () => {{
            const status = document.getElementById("w3f-status");
            try {{
                const response = await fetch("{WEB3FORMS_URL}", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }},
                    body: JSON.stringify({payload_json}),
                }});
                const data = await response.json();
                const errMsg = (data.body && data.body.message) || data.message;
                if (data.success) {{
                    status.textContent = "✅ Wyniki wysłane na {RECIPIENT_EMAIL}";
                    status.style.color = "#0a7";
                }} else {{
                    status.textContent = "❌ Błąd wysyłki: " + (errMsg || response.status);
                    status.style.color = "#c00";
                }}
            }} catch (err) {{
                status.textContent = "❌ Błąd połączenia: " + err.message;
                status.style.color = "#c00";
            }}
        }})();
        </script>
        """,
        height=40,
    )


def show_result_sharing(csv_path: Path, token: str):
    """Wyświetla QR z linkiem do pobrania wyniku i opcjonalnie stronę uploadu"""
    download_url = build_download_url(token)

    st.subheader("📲 Udostępnij wynik")
    col_qr, col_info = st.columns([1, 2])

    with col_qr:
        st.image(generate_qr_image(download_url), caption="QR — pobierz wynik", width=180)

    with col_info:
        st.markdown("**Link do pobrania pliku CSV:**")
        st.code(download_url, language=None)
        st.caption(
            "Zeskanuj QR — link działa na Streamlit Cloud. "
            f"Adres aplikacji: `{get_app_base_url()}`"
        )

        with open(csv_path, "rb") as f:
            st.download_button(
                label="⬇️ Pobierz CSV na tym komputerze",
                data=f.read(),
                file_name=csv_path.name,
                mime="text/csv",
            )

        pdf_path = csv_path.with_suffix(".pdf")
        try:
            if pdf_path.is_file():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📄 Pobierz PDF na tym komputerze",
                        data=f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                    )
            else:
                pdf_bytes = build_results_pdf_bytes(
                    csv_path, st.session_state.student_name
                )
                st.download_button(
                    label="📄 Pobierz PDF",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
        except Exception as e:
            st.warning(f"PDF niedostępny: {e}")

    upload_url = get_upload_page_url()
    if upload_url:
        st.markdown("**Strona do przesłania pliku (opcjonalnie):**")
        up_col1, up_col2 = st.columns([1, 2])
        with up_col1:
            st.image(generate_qr_image(upload_url), caption="QR — upload", width=140)
        with up_col2:
            st.link_button("Otwórz stronę uploadu", upload_url)


def main():
    st.set_page_config(
        page_title="System Zaliczeniowy MNiS",
        page_icon="🧪",
        layout="centered",
    )

    initialize_session_state()
    load_questions_internal()

    if handle_result_download_page():
        return

    st.title("Termin 1 - Laboratorium MNiS 21.06.2026")

    with st.sidebar:
        st.header("⚙️ Konfiguracja")
        st.subheader("🎓 Dane studenta")
        st.session_state.student_name = st.text_input("Imię i nazwisko")
        st.caption(f"Wyniki zostaną wysłane na: {RECIPIENT_EMAIL}")
        if st.session_state.exam_started_at and st.session_state.questions and not st.session_state.results:
            if EXAM_TIMER_ENABLED:
                remaining = get_remaining_seconds()
                st.metric("⏱️ Pozostały czas", format_remaining_time(remaining))

    if not st.session_state.questions:
        if st.session_state.load_error:
            st.error(f"❌ Nie udało się załadować pytań: {st.session_state.load_error}")
        else:
            st.error("❌ Brak pytań w pliku kolokwium.json")
        return

    if not st.session_state.student_name.strip():
        st.info("⏳ Wprowadź imię i nazwisko w panelu bocznym, aby rozpocząć test")
        return

    ensure_exam_started()
    render_exam_timer()

    st.subheader("📝 Test zaliczeniowy")

    for idx, question in enumerate(st.session_state.questions):
        with st.container():
            st.markdown(f"### Pytanie {idx + 1}")
            st.markdown(question["question"])

            if question.get("code"):
                st.code(question["code"], language="python")

            options = question["options"]
            st.markdown(format_options_markdown(options))

            answer_key = f"q_{idx}"
            selected = st.radio(
                label="Wybierz literę odpowiedzi:",
                options=list(options.keys()),
                format_func=lambda k: k.upper(),
                key=answer_key,
                index=None,
                horizontal=True,
            )

            st.session_state.user_answers[str(idx)] = selected if selected else ""
            st.divider()

    if st.button("✅ Zakończ test", type="primary"):
        if not st.session_state.student_name:
            st.error("❗ Wprowadź imię i nazwisko")
            st.stop()

        if EXAM_TIMER_ENABLED and get_remaining_seconds() <= 0:
            finalize_exam(timed_out=True)
            st.stop()

        if any(not answer for answer in st.session_state.user_answers.values()):
            st.error("❗ Odpowiedz na wszystkie pytania!")
            st.stop()

        finalize_exam(timed_out=False)

    if st.session_state.results:
        st.subheader("📊 Wyniki")
        if st.session_state.timed_out:
            st.warning("⏱️ Test zakończono automatycznie po upływie limitu czasu (20 min).")

        if not st.session_state.results_sent and st.session_state.web3forms_payload:
            send_results_via_browser(st.session_state.web3forms_payload)
            st.session_state.results_sent = True

        if st.session_state.result_download_token and st.session_state.result_csv_path:
            show_result_sharing(
                Path(st.session_state.result_csv_path),
                st.session_state.result_download_token,
            )

        for idx, result in enumerate(st.session_state.results["results"]):
            question = st.session_state.questions[idx]
            user_answer = result["user_answer"] if result["user_answer"] else "BRAK ODPOWIEDZI"
            explanation = question["explanation"]
            if not explanation:
                explanation = "Brak wyjaśnienia dla tego pytania."

            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Pytanie {idx+1}**")
                st.markdown("✅" if result["is_correct"] else "❌")

            with col2:
                st.markdown(question["question"])
                user_text = get_answer_text(user_answer, question["options"], user_answer)
                correct_text = get_answer_text(question["correct"], question["options"])
                st.markdown(f"Twoja odpowiedź: **{user_answer})** {user_text}")
                st.markdown(f"Poprawna odpowiedź: **{question['correct']})** {correct_text}")
                st.info(f"**Wyjaśnienie:** {explanation}")
                st.divider()

        st.metric(
            "Wynik końcowy",
            f"{st.session_state.results['score']}/{len(st.session_state.questions)}",
        )


if __name__ == "__main__":
    main()
