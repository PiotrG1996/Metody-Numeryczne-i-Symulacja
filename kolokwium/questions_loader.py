import json
import csv
import streamlit as st
from typing import List, Dict

# --- Funkcje do wczytywania pytań ---
def load_questions_from_json(filename: str) -> List[Dict]:
    """Wczytuje pytania z pliku JSON z walidacją struktury"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Walidacja struktury
        required_keys = ["question", "options", "correct"]
        for i, question in enumerate(data):
            if not all(key in question for key in required_keys):
                raise ValueError(f"Brak wymaganych kluczy w pytaniu {i+1}")
            
            # Normalizacja odpowiedzi
            question["correct"] = question["correct"].lower()
            
            # Sprawdzenie istnienia poprawnej odpowiedzi w opcjach
            if question["correct"] not in question["options"]:
                raise ValueError(f"Błędna poprawna odpowiedź w pytaniu {i+1}")
        
        return data
    
    except Exception as e:
        st.error(f"Błąd wczytywania pliku JSON: {str(e)}")
        st.stop()

def load_questions_from_csv(filename: str) -> List[Dict]:
    """Wczytuje pytania z pliku CSV"""
    try:
        questions = []
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                options = {k.lower(): v for k, v in row.items() if k.lower() in ['a', 'b', 'c', 'd']}
                questions.append({
                    "question": row["question"],
                    "options": options,
                    "correct": row["correct"].lower()
                })
        return questions
    except Exception as e:
        st.error(f"Błąd wczytywania pliku CSV: {str(e)}")
        st.stop()

# --- Główna funkcja aplikacji ---
def main():
    st.set_page_config(page_title="Quiz", layout="centered")
    st.title("📘 Interaktywny Quiz NumPy")

    # Inicjalizacja stanu sesji
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # Wybór i ładowanie pliku
    questions_file = st.selectbox(
        "📂 Wybierz źródło pytań:",
        ["pytania.json", "pytania.csv"],
        index=0
    )

    # Wczytaj pytania jeśli nie są załadowane
    if not st.session_state.questions:
        if questions_file.endswith(".json"):
            st.session_state.questions = load_questions_from_json(questions_file)
        else:
            st.session_state.questions = load_questions_from_csv(questions_file)

    # Wyświetlanie pytań
    for idx, question in enumerate(st.session_state.questions):
        st.markdown(f"### Pytanie {idx+1}")
        st.markdown(f"**{question['question']}**")
        
        # Wyświetl opcje odpowiedzi
        options = question["options"]
        selected = st.radio(
            label="Wybierz odpowiedź:",
            options=list(options.keys()),
            format_func=lambda k: f"{k}) {options[k]}",
            key=f"q{idx}",
            index=None
        )
        st.session_state.user_answers[idx] = selected.lower() if selected else None
        st.markdown("---")

    # Przycisk sprawdzania
    if st.button("🎯 Sprawdź odpowiedzi", type="primary"):
        st.session_state.submitted = True

    # Wyświetl wyniki
    if st.session_state.submitted:
        st.subheader("📊 Wyniki")
        correct = 0
        
        for idx, question in enumerate(st.session_state.questions):
            st.markdown(f"**Pytanie {idx+1}:** {question['question']}")
            
            user_answer = st.session_state.user_answers.get(idx)
            correct_answer = question["correct"]
            options = question["options"]

            if user_answer == correct_answer:
                st.success(f"✅ Poprawnie! Twoja odpowiedź: {user_answer}) {options[user_answer]}")
                correct += 1
            else:
                st.error(f"❌ Błędnie! Twoja odpowiedź: {user_answer or 'brak'}) {options.get(user_answer, '-')}")
                st.info(f"Poprawna odpowiedź: {correct_answer}) {options[correct_answer]}")
            
            st.markdown("---")

        st.success(f"## Twój końcowy wynik: {correct}/{len(st.session_state.questions)}")
        st.balloons()

if __name__ == "__main__":
    main()