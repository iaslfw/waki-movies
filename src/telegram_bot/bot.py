from threading import Thread

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.settings import Settings


class TelegramBot:
    """Run the Telegram application without blocking the caller."""

    def __init__(self) -> None:
        Settings.validate()

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
            "Hallo! Ich bin WaKi-Movies. "
            "Schreib mir einen Filmwunsch, ein Genre oder eine Beschreibung, "
            "und ich gebe dir spaeter passende Filmempfehlungen."
        )

    async def help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None:
            return

        await update.message.reply_text(
            "Du kannst mir zum Beispiel schreiben:\n\n"
            "- Ich suche einen lustigen Film\n"
            "- Empfiehl mir einen Sci-Fi-Film\n"
            "- Ich mag Filme wie Matrix\n\n"
            "Aktuell ist die Empfehlungslogik noch ein Platzhalter."
        )

    async def message_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None or update.message.text is None:
            return

        user_message = update.message.text

        await update.message.reply_text(
            f"Du hast geschrieben: {user_message}\n\n"
            "Spaeter wird daraus eine echte Filmempfehlung erzeugt. "
            "Aktuell ist der Telegram Bot technisch angebunden."
        )
