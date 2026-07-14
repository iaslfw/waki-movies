"""Local Telegram request normalization helpers."""


def normalize_user_message(message: str) -> str:
    return " ".join(message.strip().split())
