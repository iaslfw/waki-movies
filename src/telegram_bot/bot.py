import asyncio
from threading import Event, Thread

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.settings import Settings


class TelegramBot:
    """Run the Telegram application without blocking the caller."""

    def __init__(self) -> None:
        Settings.validate()

        token = Settings.TELEGRAM_API_TOKEN
        if token is None:
            raise RuntimeError("Telegram token validation did not provide a token.")

        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: Thread | None = None
        self.started = Event()
        self.error: BaseException | None = None

        self.application = (
            Application.builder().token(token).post_init(self.post_init).build()
        )
        self.application.add_handler(CommandHandler("start", self.start_handler))
        self.application.add_handler(CommandHandler("help", self.help_handler))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler)
        )

    def start(self) -> None:
        """Start polling in a background thread and return after initialization."""
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Telegram bot is already running.")

        self.started.clear()
        self.error = None
        self.thread = Thread(target=self.run, name="telegram-bot")
        self.thread.start()

        while not self.started.wait(timeout=0.1):
            if self.application.running:
                self.started.set()
                break

            if not self.thread.is_alive():
                if self.error is not None:
                    raise RuntimeError("Telegram bot could not be started.") from self.error
                raise RuntimeError("Telegram bot stopped before initialization.")

        if not self.application.running:
            if self.error is not None:
                raise RuntimeError("Telegram bot could not be started.") from self.error
            raise RuntimeError("Telegram bot stopped before initialization.")

    def stop(self) -> None:
        """Stop polling and wait until the background thread exits."""
        if self.thread is None or not self.thread.is_alive():
            return

        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.application.stop_running)

        self.wait()

    def wait(self) -> None:
        """Wait for the background bot thread."""
        if self.thread is not None:
            self.thread.join()

        if self.error is not None:
            raise RuntimeError("Telegram bot stopped with an error.") from self.error

    def run(self) -> None:
        """Run polling in the bot thread."""
        try:
            self.application.run_polling(stop_signals=None)
        except BaseException as error:
            self.error = error
            self.started.set()

    async def post_init(self, application: Application) -> None:
        self.loop = asyncio.get_running_loop()

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
