import json
import csv
from pathlib import Path
from typing import Dict, List, Union
from streamlit.runtime.uploaded_file_manager import UploadedFile

class QuestionLoader:
    """Klasa obsługująca ładowanie pytań z różnych źródeł"""

    def __init__(self):
        self.base_dir = Path(__file__).parent

    def _validate_question(self, question: Dict) -> None:
        """Waliduje strukturę pojedynczego pytania"""
        required = ["question", "options", "correct"]
        for key in required:
            if key not in question:
                raise ValueError(f"Brak wymaganego klucza: {key}")
        if not isinstance(question["options"], dict) or len(question["options"]) < 2:
            raise ValueError("Opcje odpowiedzi muszą być słownikiem z co najmniej 2 pozycjami")
        if question["correct"].lower() not in [k.lower() for k in question["options"].keys()]:
            raise ValueError(f"Poprawna odpowiedź '{question['correct']}' nie istnieje w opcjach")
        for opt in ["explanation", "code"]:
            if opt in question and not isinstance(question[opt], str):
                raise ValueError(f"'{opt}' musi być typu string")

    def _ensure_explanation(self, question: Dict) -> None:
        """Zapewnia obecność klucza 'explanation'"""
        if "explanation" not in question or not question["explanation"].strip():
            question["explanation"] = "Brak wyjaśnienia dla tego pytania."

    def from_json(self, source: Union[Path, UploadedFile]) -> List[Dict]:
        """Ładuje pytania z pliku JSON"""
        try:
            if isinstance(source, Path):
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = json.loads(source.getvalue().decode('utf-8'))

            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and "questions" in data:
                questions = data["questions"]
            else:
                raise ValueError("Nieprawidłowy format JSON. Oczekiwano listy pytań lub obiektu z kluczem 'questions'")

            for q in questions:
                # Nie nadpisuj explanation jeśli już jest poprawne
                if "explanation" in q and isinstance(q["explanation"], str):
                    q["explanation"] = q["explanation"].strip()
                if "code" in q and isinstance(q["code"], str):
                    q["code"] = q["code"].strip()
                self._validate_question(q)
                self._ensure_explanation(q)
            return questions

        except Exception as e:
            raise RuntimeError(f"Błąd ładowania JSON: {str(e)}") from e

    def from_csv(self, source: Union[Path, UploadedFile]) -> List[Dict]:
        """Ładuje pytania z pliku CSV"""
        try:
            if isinstance(source, Path):
                with open(source, 'r', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
            else:
                content = source.getvalue().decode('utf-8').splitlines()
                rows = list(csv.DictReader(content))

            questions = []
            for row in rows:
                options = {k: v for k, v in row.items() if k.lower() in ['a', 'b', 'c', 'd']}
                question = {
                    "question": row["question"],
                    "options": options,
                    "correct": row["correct"].lower(),
                    "explanation": row.get("explanation", "").strip(),
                    "code": row.get("code", "").strip()
                }
                self._validate_question(question)
                self._ensure_explanation(question)
                questions.append(question)
            return questions

        except Exception as e:
            raise RuntimeError(f"Błąd ładowania CSV: {str(e)}") from e

    def load_default(self) -> List[Dict]:
        """Ładuje domyślne pytania z pakietu"""
        default_path = self.base_dir / "data" / "default_questions.json"
        return self.from_json(default_path)