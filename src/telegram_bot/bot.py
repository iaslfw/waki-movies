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

from src.inference.inference import MovieTagPredictor
from src.inference.recommender import MovieRecommender
from src.settings import Settings
from src.telegram_bot.chat_controller import (
    ChatControllerError,
    ChatDecision,
    MistralChatController,
)
from src.telegram_bot.recommendation_formatter import (
    format_recommendation_items,
    format_recommendations,
)
from src.telegram_bot.replies import (
    CLARIFY_REQUEST_REPLY,
    HELP_REPLY,
    INVALID_REQUEST_REPLY,
    RECOMMENDATION_ERROR_REPLY,
    SMALLTALK_REPLY,
    START_REPLY,
)
from src.telegram_bot.request_validation import (
    contains_known_movie_tag,
    get_known_request_terms,
    is_known_single_word_request,
    is_vague_movie_request,
    is_valid_movie_request,
    normalize_for_matching,
    normalize_user_message,
)

logger = logging.getLogger(__name__)


class TelegramBot:
    """Run the Telegram application without blocking the caller."""

    def __init__(
        self,
        predictor: MovieTagPredictor,
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

        await update.message.reply_text(START_REPLY)

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
                    user_message,
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
            await update.message.reply_text(RECOMMENDATION_ERROR_REPLY)

    def get_recommendations_for_message(
        self,
        user_message: str,
        recommendation_query: str,
    ) -> list[dict[str, Any]]:
        prediction = self.predictor.predict(recommendation_query)
        title_match_query = f"{user_message} {recommendation_query}"
        return self.recommender.get_recommendations(
            prediction,
            query=title_match_query,
        )

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
            decision = await asyncio.to_thread(self.chat_controller.decide, message)
        except ChatControllerError:
            logger.exception("Mistral chat controller failed. Using local fallback.")
            return self.get_fallback_chat_decision(message)

        return self.normalize_chat_decision(decision, message)

    def normalize_chat_decision(
        self, decision: ChatDecision, message: str
    ) -> ChatDecision:
        if decision.action != "clarify":
            return decision

        normalized_message = self.normalize_user_message(message)
        if not self.contains_known_movie_tag(normalized_message):
            return decision

        return ChatDecision(
            action="recommend",
            cleaned_query=decision.cleaned_query or normalized_message,
            reply=None,
        )

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
        return normalize_user_message(message)

    @classmethod
    def is_valid_movie_request(cls, message: str) -> bool:
        return is_valid_movie_request(message, cls.get_known_request_terms())

    @classmethod
    def is_known_single_word_request(cls, token: str) -> bool:
        return is_known_single_word_request(token, cls.get_known_request_terms())

    @classmethod
    def contains_known_movie_tag(cls, message: str) -> bool:
        return contains_known_movie_tag(message, cls.get_known_request_terms())

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        return normalize_for_matching(text)

    @classmethod
    def get_known_request_terms(cls) -> set[str]:
        return get_known_request_terms(Settings.create_movie_tag_list())

    @staticmethod
    def is_vague_movie_request(message: str) -> bool:
        return is_vague_movie_request(message)

    def format_recommendations(self, recommendations: list[dict[str, Any]]) -> str:
        return format_recommendations(recommendations)

    def format_recommendation_items(self, recommendations: list[dict[str, Any]]) -> str:
        return format_recommendation_items(recommendations)
