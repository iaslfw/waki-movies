import asyncio
import logging
from threading import Thread
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.inference.inference import MoodPredictor
from src.inference.recommender import MovieRecommender
from src.settings import Settings
from src.telegram_bot.chat_controller import (
    ChatControllerError,
    ChatDecision,
    MistralChatController,
)

logger = logging.getLogger(__name__)

INVALID_REQUEST_REPLY = (
    "Please send a movie genre or short description, "
    'for example: "funny sci-fi movie" or "dark horror thriller".'
)
CLARIFY_REQUEST_REPLY = (
    "Sure. What kind of movie are you in the mood for: funny, scary, "
    "romantic, action, or something else?"
)
HELP_REPLY = (
    "You can write something like:\n\n"
    "- I want a funny movie\n"
    "- Recommend a sci-fi movie\n"
    "- I like movies like The Matrix"
)
SMALLTALK_REPLY = "Hi! Send me a genre, movie title, or short description."
KNOWN_SINGLE_WORD_REQUESTS = {
    "action",
    "adventure",
    "anime",
    "comedy",
    "crime",
    "dark",
    "drama",
    "family",
    "fantasy",
    "funny",
    "happy",
    "horror",
    "kids",
    "romance",
    "romantic",
    "sad",
    "scary",
    "sci-fi",
    "scifi",
    "thriller",
    "war",
    "western",
}
VAGUE_SINGLE_WORD_REQUESTS = {"film", "movie", "recommendation", "something"}


class TelegramBot:
    """Run the Telegram application without blocking the caller."""

    def __init__(
        self,
        predictor: MoodPredictor,
        recommender: MovieRecommender,
    ) -> None:
        Settings.validate()

        self.predictor = predictor
        self.recommender = recommender
        self.chat_controller = MistralChatController(
            api_key=Settings.MISTRAL_API_KEY,
            model=Settings.MISTRAL_MODEL,
            api_url=Settings.MISTRAL_API_URL,
            timeout_seconds=Settings.MISTRAL_TIMEOUT_SECONDS,
        )
        self.recommendation_lock = asyncio.Lock()
        self.thread: Thread | None = None
        self.application = (
            Application.builder().token(Settings.TELEGRAM_API_TOKEN).build()
        )
        self.application.add_handler(CommandHandler("start", self.start_handler))
        self.application.add_handler(CommandHandler("help", self.help_handler))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler)
        )

    def start(self) -> None:
        """Start polling in a background thread."""
        self.thread = Thread(
            target=self.application.run_polling,
            kwargs={"stop_signals": None},
            name="telegram-bot",
        )
        self.thread.start()

    def wait(self) -> None:
        """Wait for the background bot thread."""
        if self.thread is not None:
            self.thread.join()

    async def start_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None:
            return

        await update.message.reply_text(
            "Hi! I'm WaKi-Movies. "
            "Send me a movie request, genre, or short description, "
            "and I'll recommend matching movies."
        )

    async def help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None:
            return

        await update.message.reply_text(HELP_REPLY)

    async def message_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None or update.message.text is None:
            return

        user_message = self.normalize_user_message(update.message.text)
        if not user_message:
            await update.message.reply_text(INVALID_REQUEST_REPLY)
            return

        decision = await self.get_chat_decision(user_message)
        logger.info(
            "Chat decision action=%s cleaned_query=%r",
            decision.action,
            decision.cleaned_query,
        )

        if decision.action != "recommend":
            await update.message.reply_text(self.get_reply_for_decision(decision))
            return

        recommendation_query = decision.cleaned_query or user_message

        try:
            async with self.recommendation_lock:
                recommendations = await asyncio.to_thread(
                    self.get_recommendations_for_message,
                    recommendation_query,
                )

            recommendation_reply = await self.format_recommendation_reply(
                user_message=user_message,
                recommendation_query=recommendation_query,
                recommendations=recommendations,
            )
            await update.message.reply_text(recommendation_reply)
        except Exception:
            logger.exception("Failed to create movie recommendations.")
            await update.message.reply_text(
                "Sorry, I couldn't create recommendations right now. "
                "Please try again later."
            )

    def get_recommendations_for_message(
        self,
        message: str,
    ) -> list[dict[str, Any]]:
        prediction = self.predictor.predict(message)
        return self.recommender.get_recommendations(prediction)

    async def format_recommendation_reply(
        self,
        user_message: str,
        recommendation_query: str,
        recommendations: list[dict[str, Any]],
    ) -> str:
        if not self.chat_controller.is_configured:
            return self.format_recommendations(recommendations)

        try:
            intro = await asyncio.to_thread(
                self.chat_controller.write_recommendation_intro,
                user_message,
                recommendation_query,
                recommendations,
            )
            return f"{intro}\n\n{self.format_recommendation_items(recommendations)}"
        except ChatControllerError:
            logger.exception(
                "Mistral recommendation formatter failed. Using local fallback."
            )
            return self.format_recommendations(recommendations)

    async def get_chat_decision(self, message: str) -> ChatDecision:
        if not self.chat_controller.is_configured:
            return self.get_fallback_chat_decision(message)

        try:
            return await asyncio.to_thread(self.chat_controller.decide, message)
        except ChatControllerError:
            logger.exception("Mistral chat controller failed. Using local fallback.")
            return self.get_fallback_chat_decision(message)

    def get_fallback_chat_decision(self, message: str) -> ChatDecision:
        normalized_message = self.normalize_user_message(message)
        if self.is_vague_movie_request(normalized_message):
            return ChatDecision(
                action="clarify",
                cleaned_query=None,
                reply=CLARIFY_REQUEST_REPLY,
            )
        if self.is_valid_movie_request(normalized_message):
            return ChatDecision(
                action="recommend",
                cleaned_query=normalized_message,
                reply=None,
            )
        return ChatDecision(
            action="reject", cleaned_query=None, reply=INVALID_REQUEST_REPLY
        )

    @staticmethod
    def get_reply_for_decision(decision: ChatDecision) -> str:
        if decision.reply:
            return decision.reply

        fallback_replies = {
            "clarify": CLARIFY_REQUEST_REPLY,
            "reject": INVALID_REQUEST_REPLY,
            "help": HELP_REPLY,
            "smalltalk": SMALLTALK_REPLY,
        }
        return fallback_replies.get(decision.action, INVALID_REQUEST_REPLY)

    @staticmethod
    def normalize_user_message(message: str) -> str:
        return " ".join(message.strip().split())

    @staticmethod
    def is_valid_movie_request(message: str) -> bool:
        lower_message = message.lower()
        tokens = lower_message.split()
        meaningful_character_count = sum(char.isalnum() for char in message)
        letter_count = sum(char.isalpha() for char in message)

        if meaningful_character_count < 2 or letter_count < 2:
            return False

        if len(tokens) == 1:
            return tokens[0] in KNOWN_SINGLE_WORD_REQUESTS

        return True

    @staticmethod
    def is_vague_movie_request(message: str) -> bool:
        tokens = message.lower().split()
        return len(tokens) == 1 and tokens[0] in VAGUE_SINGLE_WORD_REQUESTS

    def format_recommendations(self, recommendations: list[dict[str, Any]]) -> str:
        lines = ["I found these movie picks for you:\n"]
        lines.append(self.format_recommendation_items(recommendations))
        return "\n".join(lines).strip()

    def format_recommendation_items(self, recommendations: list[dict[str, Any]]) -> str:
        lines = []
        for i, recommendation in enumerate(recommendations, 1):
            score_percent = float(recommendation["similarity_score"]) * 100
            title = str(recommendation["title"])
            overview = str(recommendation["overview"])

            lines.append(f"{i}. {title} ({score_percent:.1f}% Match)")
            lines.append(overview)
            lines.append("")

        return "\n".join(lines).strip()
