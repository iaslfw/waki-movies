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
from src.telegram_bot.language import ResponseLanguage, normalize_response_language
from src.telegram_bot.recommendation_formatter import format_recommendation_items
from src.telegram_bot.replies import (
    HELP_REPLY,
    INVALID_REQUEST_REPLY,
    RECOMMENDATION_ERROR_REPLY,
    START_REPLY,
    get_recommendation_error_reply,
)
from src.telegram_bot.request_validation import normalize_user_message

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

        try:
            decision = await self.get_chat_decision(user_message)
        except ChatControllerError:
            logger.exception("Mistral chat controller failed.")
            await update.message.reply_text(RECOMMENDATION_ERROR_REPLY)
            return

        logger.info(
            "Chat decision action=%s response_language=%s cleaned_query=%r",
            decision.action,
            decision.response_language,
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
                response_language=decision.response_language,
                recommendations=recommendations,
            )
            await update.message.reply_text(recommendation_reply)
        except Exception:
            logger.exception("Failed to create movie recommendations.")
            await update.message.reply_text(
                get_recommendation_error_reply(decision.response_language)
            )

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
        response_language: ResponseLanguage | str,
        recommendations: list[dict[str, Any]],
    ) -> str:
        reply_parts = await asyncio.to_thread(
            self.chat_controller.write_recommendation_reply,
            user_message,
            recommendation_query,
            recommendations,
            response_language,
        )
        recommendation_items = format_recommendation_items(
            recommendations,
            response_language,
            reply_parts.overviews,
        )
        return f"{reply_parts.intro}\n\n{recommendation_items}"

    async def get_chat_decision(self, message: str) -> ChatDecision:
        decision = await asyncio.to_thread(self.chat_controller.decide, message)
        return self.normalize_chat_decision(decision)

    @staticmethod
    def normalize_chat_decision(decision: ChatDecision) -> ChatDecision:
        response_language = normalize_response_language(decision.response_language)

        if decision.action == "recommend":
            if not decision.cleaned_query:
                raise ChatControllerError("Mistral recommendation query is missing.")

            return ChatDecision(
                action="recommend",
                cleaned_query=decision.cleaned_query,
                reply=None,
                response_language=response_language,
            )

        if not decision.reply:
            raise ChatControllerError("Mistral reply is missing.")

        return ChatDecision(
            action=decision.action,
            cleaned_query=decision.cleaned_query,
            reply=decision.reply,
            response_language=response_language,
        )

    @staticmethod
    def get_reply_for_decision(decision: ChatDecision) -> str:
        if decision.reply:
            return decision.reply

        raise ChatControllerError("Mistral reply is missing.")

    @staticmethod
    def normalize_user_message(message: str) -> str:
        return normalize_user_message(message)
