"""Schemas and prompts for the external chat controller."""

from typing import Any

CHAT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["recommend", "clarify", "reject", "help", "smalltalk"],
        },
        "cleaned_query": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
        "reply": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
    },
    "required": ["action", "cleaned_query", "reply"],
}

RECOMMENDATION_INTRO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intro": {
            "type": "string",
        },
    },
    "required": ["intro"],
}

CHAT_CONTROLLER_SYSTEM_PROMPT = """
You are the chat controller for WaKi-Movies, a Telegram movie recommendation bot.
Decide how to handle exactly one user message.

Return only JSON that matches the schema.

Actions:
- recommend: The user asks for a movie recommendation, genre, mood, plot, theme,
  or a movie similar to another movie. Put a concise recommendation query into
  cleaned_query. If the user writes in another language, translate the movie
  intent into English for cleaned_query so the downstream tag model can use it.
  Do not write a reply.
- clarify: The user wants a recommendation but is too vague. Ask one short
  follow-up question in reply.
- reject: The message is nonsense, random characters, a random short fragment,
  only symbols, or unrelated to movie recommendations. Explain briefly in reply.
- help: The user asks what the bot can do or how to use it. Reply with short
  examples.
- smalltalk: The user greets or thanks the bot. Reply briefly and invite a movie
  request.

Rules:
- Accept short, clear movie genres or moods such as "war", "horror", "comedy",
  "sci-fi", "dark", "funny", or "romantic".
- Reject random fragments such as "gre", "erb", "asdf", "+", "#", or "!!!".
- Do not recommend actual movies yourself.
- Reply in the same language as the user when you write a reply.
""".strip()

RECOMMENDATION_INTRO_SYSTEM_PROMPT = """
You write the opening sentence for a Telegram movie recommendation reply.

The movie recommendations are already selected by another model. The application
will append the actual movie list after your sentence.

Write exactly one short sentence that:
- leads into the recommendations and fits the user's request;
- does not list movie titles;
- does not mention match percentages;
- avoids saying "Here are some movies that match your request".
- uses the same language as the user's message.

Return only JSON that matches the schema.
""".strip()
