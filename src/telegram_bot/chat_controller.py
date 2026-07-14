import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.telegram_bot.chat_prompts import (
    CHAT_CONTROLLER_SYSTEM_PROMPT,
    CHAT_DECISION_SCHEMA,
    RECOMMENDATION_REPLY_SCHEMA,
    RECOMMENDATION_REPLY_SYSTEM_PROMPT,
)
from src.telegram_bot.language import ResponseLanguage, normalize_response_language

ChatAction = Literal["recommend", "clarify", "reject", "help", "smalltalk"]

VALID_ACTIONS: set[str] = {"recommend", "clarify", "reject", "help", "smalltalk"}


class ChatControllerError(RuntimeError):
    """Raised when the external chat controller cannot return a valid decision."""


@dataclass(frozen=True)
class ChatDecision:
    action: ChatAction
    cleaned_query: str | None
    reply: str | None
    response_language: ResponseLanguage = "en"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatDecision":
        action = str(data.get("action", "")).strip()
        if action not in VALID_ACTIONS:
            raise ChatControllerError(f"Invalid chat action: {action!r}")

        cleaned_query = cls._optional_clean_string(data.get("cleaned_query"))
        response_language = normalize_response_language(data.get("response_language"))
        reply = cls._optional_clean_string(data.get("reply"))

        return cls(
            action=action,  # type: ignore[arg-type]
            cleaned_query=cleaned_query,
            reply=reply,
            response_language=response_language,
        )

    @staticmethod
    def _optional_clean_string(value: Any) -> str | None:
        if not isinstance(value, str):
            return None

        normalized = " ".join(value.strip().split())
        return normalized or None


@dataclass(frozen=True)
class RecommendationReplyParts:
    intro: str
    overviews: list[str | None]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        expected_item_count: int,
    ) -> "RecommendationReplyParts":
        intro = cls._required_clean_string(data.get("intro"), "intro")
        items = data.get("items")
        if not isinstance(items, list):
            raise ChatControllerError("Mistral recommendation reply has no items.")

        overviews: list[str | None] = [None] * expected_item_count
        for item in items:
            if not isinstance(item, dict):
                raise ChatControllerError("Mistral recommendation item is invalid.")

            rank = item.get("rank")
            if not isinstance(rank, int):
                raise ChatControllerError(
                    "Mistral recommendation item rank is invalid."
                )
            if rank < 1 or rank > expected_item_count:
                raise ChatControllerError(
                    "Mistral recommendation item rank is out of range."
                )

            overview = cls._required_clean_string(item.get("overview"), "overview")
            overviews[rank - 1] = overview

        if any(overview is None for overview in overviews):
            raise ChatControllerError(
                "Mistral recommendation reply is missing item overviews."
            )

        return cls(intro=intro, overviews=overviews)

    @staticmethod
    def _required_clean_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise ChatControllerError(
                f"Mistral recommendation {field_name} is invalid."
            )

        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ChatControllerError(f"Mistral recommendation {field_name} is empty.")

        return normalized


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

    def write_recommendation_reply(
        self,
        user_message: str,
        recommendation_query: str,
        recommendations: list[dict[str, Any]],
        response_language: ResponseLanguage | str,
    ) -> RecommendationReplyParts:
        if not self.is_configured:
            raise ChatControllerError("MISTRAL_API_KEY is missing.")

        formatted_recommendations = []
        for index, recommendation in enumerate(recommendations, 1):
            formatted_recommendations.append({
                "rank": index,
                "title": str(recommendation["title"]),
                "overview": str(recommendation["overview"]),
            })

        user_payload = {
            "user_message": user_message,
            "recommendation_query": recommendation_query,
            "response_language": normalize_response_language(response_language),
            "recommendations": formatted_recommendations,
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": False,
            "messages": [
                {"role": "system", "content": RECOMMENDATION_REPLY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_reply",
                    "description": "Localized text for selected movie recommendations.",
                    "schema": RECOMMENDATION_REPLY_SCHEMA,
                    "strict": True,
                },
            },
        }

        response_data = self._post_json(payload)
        content = self._extract_message_content(response_data)
        reply_data = self._parse_json_content(content)
        return RecommendationReplyParts.from_dict(
            reply_data,
            expected_item_count=len(recommendations),
        )

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
