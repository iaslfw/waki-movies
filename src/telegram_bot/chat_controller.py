import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.telegram_bot.chat_prompts import (
    CHAT_CONTROLLER_SYSTEM_PROMPT,
    CHAT_DECISION_SCHEMA,
    RECOMMENDATION_INTRO_SCHEMA,
    RECOMMENDATION_INTRO_SYSTEM_PROMPT,
)

ChatAction = Literal["recommend", "clarify", "reject", "help", "smalltalk"]

VALID_ACTIONS: set[str] = {"recommend", "clarify", "reject", "help", "smalltalk"}


class ChatControllerError(RuntimeError):
    """Raised when the external chat controller cannot return a valid decision."""


@dataclass(frozen=True)
class ChatDecision:
    action: ChatAction
    cleaned_query: str | None
    reply: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatDecision":
        action = str(data.get("action", "")).strip()
        if action not in VALID_ACTIONS:
            raise ChatControllerError(f"Invalid chat action: {action!r}")

        cleaned_query = cls._optional_clean_string(data.get("cleaned_query"))
        reply = cls._optional_clean_string(data.get("reply"))

        return cls(
            action=action,  # type: ignore[arg-type]
            cleaned_query=cleaned_query,
            reply=reply,
        )

    @staticmethod
    def _optional_clean_string(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        normalized = " ".join(value.strip().split())
        return normalized or None


class MistralChatController:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def decide(self, message: str) -> ChatDecision:
        if not self.is_configured:
            raise ChatControllerError("MISTRAL_API_KEY is missing.")

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
            "messages": [
                {"role": "system", "content": CHAT_CONTROLLER_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "chat_decision",
                    "description": "Routing decision for the WaKi-Movies bot.",
                    "schema": CHAT_DECISION_SCHEMA,
                    "strict": True,
                },
            },
        }

        response_data = self._post_json(payload)
        content = self._extract_message_content(response_data)
        decision_data = self._parse_json_content(content)
        return ChatDecision.from_dict(decision_data)

    def write_recommendation_intro(
        self,
        user_message: str,
        recommendation_query: str,
        recommendations: list[dict[str, Any]],
    ) -> str:
        if not self.is_configured:
            raise ChatControllerError("MISTRAL_API_KEY is missing.")

        formatted_recommendations = []
        for index, recommendation in enumerate(recommendations, 1):
            score_percent = float(recommendation["similarity_score"]) * 100
            formatted_recommendations.append({
                "rank": index,
                "title": str(recommendation["title"]),
                "match_percent": f"{score_percent:.1f}%",
                "overview": str(recommendation["overview"]),
            })

        user_payload = {
            "user_message": user_message,
            "recommendation_query": recommendation_query,
            "recommendations": formatted_recommendations,
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 120,
            "stream": False,
            "messages": [
                {"role": "system", "content": RECOMMENDATION_INTRO_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=True),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_intro",
                    "description": "Opening sentence for selected movie recommendations.",
                    "schema": RECOMMENDATION_INTRO_SCHEMA,
                    "strict": True,
                },
            },
        }

        response_data = self._post_json(payload)
        content = self._extract_message_content(response_data)
        intro_data = self._parse_json_content(content)
        intro = intro_data.get("intro")
        if not isinstance(intro, str) or not intro.strip():
            raise ChatControllerError("Mistral recommendation intro is invalid.")

        return intro.strip()

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ChatControllerError(
                f"Mistral API returned HTTP {exc.code}: {body}"
            ) from exc
        except (TimeoutError, URLError) as exc:
            raise ChatControllerError("Mistral API request failed.") from exc

        try:
            loaded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ChatControllerError("Mistral API returned invalid JSON.") from exc

        if not isinstance(loaded, dict):
            raise ChatControllerError("Mistral API returned an unexpected payload.")

        return loaded

    @staticmethod
    def _extract_message_content(response_data: dict[str, Any]) -> str | dict[str, Any]:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatControllerError("Mistral API response has no message.") from exc

        if not isinstance(content, (str, dict)):
            raise ChatControllerError("Mistral API response message is invalid.")

        return content

    @staticmethod
    def _parse_json_content(content: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(content, dict):
            return content

        try:
            loaded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ChatControllerError("Mistral decision is not valid JSON.") from exc

        if not isinstance(loaded, dict):
            raise ChatControllerError("Mistral decision is not a JSON object.")

        return loaded
