from typing import List, Dict, Any
import re

def check_answer(user_key: str, correct_key: str) -> bool:
    """
    Compares the user's answer key with the correct one (case-insensitive).

    Parameters:
        user_key (str): User's answer key (e.g., 'a').
        correct_key (str): Correct answer key (e.g., 'b').

    Returns:
        bool: True if the keys match, False otherwise.
    """
    return user_key.strip().lower() == correct_key.strip().lower()


def extract_user_key(raw_input: str) -> str:
    """
    Extracts the letter key from raw user input (e.g., 'b) answer text' → 'b').

    Parameters:
        raw_input (str): Raw input string.

    Returns:
        str: Extracted lowercase answer key, or empty string if invalid.
    """
    return raw_input.strip()[:1].lower() if raw_input else ""


def get_answer_text(answer_key: str, options: Dict[str, str], default: str = "Brak odpowiedzi") -> str:
    """
    Retrieves the answer text corresponding to the given key.

    Parameters:
        answer_key (str): The answer key (e.g., 'a').
        options (Dict[str, str]): Available answer options.
        default (str): Default text if the key is invalid or not present.

    Returns:
        str: The answer text if the key is valid, else the default text.
    """
    return options.get(answer_key, default)


def evaluate_answers(questions, user_answers):
    results = []
    score = 0
    
    for idx, question in enumerate(questions):
        user_answer = user_answers.get(str(idx))
        is_correct = (user_answer == question["correct"])
        
        if is_correct:
            score += 1
            
        results.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": question["correct"],
            "is_correct": is_correct
        })
    
    return {
        "score": score,
        "total": len(questions),
        "results": results
    }
    
def sanitize_filename(name):
    if not name:    
        return "anonymous"
    # Usuń znaki specjalne i zastąp spacje podkreśleniami
    name = re.sub(r'[^\w\s-]', '', name.strip())
    return re.sub(r'[-\s]+', '_', name)[:50]    