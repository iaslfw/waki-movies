import asyncio
import logging
from threading import Thread
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.inference.inference import MoodPredictor
from src.inference.recommender import MovieRecommender
from src.settings import Settings

logger = logging.getLogger(__name__)


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
        self.recommendation_lock = asyncio.Lock()
        self.thread: Thread | None = None
        self.application = Application.builder().token(Settings.TELEGRAM_API_TOKEN).build()
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

        await update.message.reply_text(
            "You can write something like:\n\n"
            "- I want a funny movie\n"
            "- Recommend a sci-fi movie\n"
            "- I like movies like The Matrix"
        )

    async def message_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None or update.message.text is None:
            return

        user_message = update.message.text

        try:
            async with self.recommendation_lock:
                recommendations = await asyncio.to_thread(
                    self.get_recommendations_for_message,
                    user_message,
                )

            await update.message.reply_text(self.format_recommendations(recommendations))
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

    def format_recommendations(self, recommendations: list[dict[str, Any]]) -> str:
        lines = ["Here are some movies that match your request:\n"]

        for i, recommendation in enumerate(recommendations, 1):
            score_percent = float(recommendation["similarity_score"]) * 100
            title = str(recommendation["title"])
            overview = str(recommendation["overview"])

            lines.append(f"{i}. {title} ({score_percent:.1f}% Match)")
            lines.append(overview)
            lines.append("")

        return "\n".join(lines).strip()
