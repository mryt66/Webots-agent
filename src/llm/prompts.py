from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "Jeśli prośba użytkownika nie dotyczy symulacji Webots ani ruchu robota, odpowiedz tekstem bez narzędzi. "
    "Dostajesz zrzut ekranu z góry. Używaj narzędzi tylko gdy trzeba wykonać akcję w symulacji. "
    "Puszka stoi na którymś ze stolików; w widoku z góry jest szara i okrągła. "
    "Dostępne narzędzia: move_forward, move_backward, rotate_right_90, rotate_left_90, rotate_back, grab_right, release_right. "
    "Gdy narzędzia są potrzebne, zwróć w jednej odpowiedzi pełną sekwencję wywołań narzędzi (w odpowiedniej kolejności), bez dodatkowego tekstu w tej samej wiadomości. "
    "Gdy narzędzia nie są potrzebne, odpowiedz samym tekstem. "
    "Kierunek, w który patrzy robot, wynika z kierunku dwóch rąk/ramion wystających z robota na obrazku (na starcie zwykle patrzy w prawo). "
    "Zasada: obracanie (rotate_*) wykonuj tylko w centrum (home). Jeśli robot nie jest w centrum, a potrzebujesz obrotu, najpierw wróć do centrum. "
)


PLANNER_PROMPT = (
    "Zwróć TYLKO obiekt JSON z dokładnie dwoma kluczami: comment (string) oraz sequence (tablica stringów). "
    "sequence może być puste, jeśli nie trzeba użyć narzędzi. "
    "Elementy sequence muszą być jedną z nazw narzędzi: move_forward, move_backward, rotate_right_90, rotate_left_90, rotate_back, grab_right, release_right. "
    "Bez markdown, bez dodatkowych kluczy, bez tekstu przed/po JSON. "
    "Ustal kolejność na podstawie zrzutu ekranu. Kierunek robota wynika z kierunku dwóch rąk/ramion wystających z robota. "
    "Zasada: obracanie (rotate_*) wykonuj tylko w centrum (home); jeśli potrzeba obrotu poza centrum, wróć najpierw do centrum. "
)


def prompt_source_path() -> str:
    return str(Path(__file__).resolve())


def tool_declarations() -> List[Dict[str, Any]]:
    return [
        {
            "name": "move_forward",
            "description": "Moves the robot base forward by the default distance.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "move_backward",
            "description": "Moves the robot base backward by the default distance.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "rotate_right_90",
            "description": "Rotates the robot base 90 degrees to the right (in place).",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "rotate_left_90",
            "description": "Rotates the robot base 90 degrees to the left (in place).",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "rotate_back",
            "description": "Rotates the robot base 180 degrees (in place).",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "grab_right",
            "description": "Closes the right gripper using default parameters (tries to grab the can).",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "release_right",
            "description": "Opens the right gripper (releases).",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
