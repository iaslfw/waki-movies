"""Static Telegram reply texts."""

from src.telegram_bot.language import ResponseLanguage, normalize_response_language

INVALID_REQUEST_REPLY = (
    "Please send a movie genre or short description, "
    'for example: "funny sci-fi movie" or "dark horror thriller".'
)
HELP_REPLY = (
    "You can write something like:\n\n"
    "- I want a funny movie\n"
    "- Recommend a sci-fi movie\n"
    "- I like movies like The Matrix"
)
START_REPLY = (
    "Hi! I'm WaKi-Movies. "
    "Send me a movie request, genre, or short description, "
    "and I'll recommend matching movies."
)

_RECOMMENDATION_ERROR_REPLIES: dict[ResponseLanguage, str] = {
    "en": (
        "Sorry, I couldn't create recommendations right now. Please try again later."
    ),
    "de": (
        "Entschuldigung, ich konnte gerade keine Empfehlungen erstellen. "
        "Bitte versuche es später noch einmal."
    ),
}

RECOMMENDATION_ERROR_REPLY = _RECOMMENDATION_ERROR_REPLIES["en"]


def get_recommendation_error_reply(response_language: ResponseLanguage | str) -> str:
    language = normalize_response_language(response_language)
    return _RECOMMENDATION_ERROR_REPLIES[language]
