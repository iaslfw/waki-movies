from src.telegram_bot.request_validation import normalize_user_message


def test_normalize_user_message_strips_and_collapses_whitespace() -> None:
    assert (
        normalize_user_message("  sci-fi\t  drama\nplease  ") == "sci-fi drama please"
    )


def test_normalize_user_message_returns_empty_string_for_blank_input() -> None:
    assert normalize_user_message(" \t\n ") == ""
