import json
import csv
from pathlib import Path
from typing import Dict, List, Union
from io import TextIOWrapper
from streamlit.runtime.uploaded_file_manager import UploadedFile

class QuestionLoader:
    """Klasa obsługująca ładowanie pytań z różnych źródeł"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent

    def _validate_question(self, question: Dict) -> None:
        """Waliduje strukturę pojedynczego pytania"""
        required_keys = ["question", "options", "correct"]
        if not all(key in question for key in required_keys):
            raise ValueError(f"Brak wymaganych kluczy w pytaniu: {required_keys}")
            
        if not isinstance(question["options"], dict) or len(question["options"]) < 2:
            raise ValueError("Opcje odpowiedzi muszą być słownikiem z co najmniej 2 pozycjami")
            
        if question["correct"].lower() not in [k.lower() for k in question["options"].keys()]:
            raise ValueError(f"Poprawna odpowiedź '{question['correct']}' nie istnieje w opcjach")

    def from_json(self, source: Union[Path, UploadedFile]) -> List[Dict]:
        """Ładuje pytania z pliku JSON w dwóch formatach"""
        try:
            # Wczytaj dane
            if isinstance(source, Path):
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = json.loads(source.getvalue().decode('utf-8'))
            
            # Rozpoznaj format
            if isinstance(data, list):
                questions = data
            elif "questions" in data and isinstance(data["questions"], list):
                questions = data["questions"]
            else:
                raise ValueError("Nieprawidłowy format JSON. Oczekiwano listy pytań lub obiektu z kluczem 'questions'")
            
            # Walidacja wszystkich pytań
            for question in questions:
                self._validate_question(question)
                
            return questions
            
        except Exception as e:
            raise RuntimeError(f"Błąd ładowania JSON: {str(e)}") from e

    def from_csv(self, source: Union[Path, UploadedFile]) -> List[Dict]:
        """Ładuje pytania z pliku CSV z dysku lub uploadu"""
        try:
            questions = []
            
            # Wczytaj dane CSV
            if isinstance(source, Path):
                with open(source, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            else:
                content = source.getvalue().decode('utf-8').splitlines()
                reader = csv.DictReader(content)
                rows = list(reader)
            
            # Przetwórz wiersze
            for row in rows:
                question = {
                    "question": row["question"],
                    "options": {
                        k: v 
                        for k, v in row.items() 
                        if k.lower() in ['a', 'b', 'c', 'd']
                    },
                    "correct": row["correct"].lower()
                }
                self._validate_question(question)
                questions.append(question)
                
            return questions
            
        except Exception as e:
            raise RuntimeError(f"Błąd ładowania CSV: {str(e)}") from e

    def load_default(self) -> List[Dict]:
        """Ładuje domyślne pytania z pakietu"""
        default_path = self.base_dir / "data" / "default_questions.json"
        return self.from_json(default_path)