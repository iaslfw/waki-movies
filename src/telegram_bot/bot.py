import asyncio
from threading import Event, Thread
from typing import Final

from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, MessageHandler, Updater, filters

from src.settings import Settings
from src.telegram_bot.handlers import help_handler, message_handler, start_handler


class TelegramBot:
    """Run the Telegram application without blocking the caller."""

    _THREAD_NAME: Final = "telegram-bot"

    def __init__(self) -> None:
        Settings.validate()

        token = Settings.TELEGRAM_API_TOKEN
        if token is None:
            raise RuntimeError("Telegram token validation did not provide a token.")

        self._application = self._build_application(token)
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._started = Event()
        self._startup_error: BaseException | None = None
        self._runtime_error: BaseException | None = None

    def start(self) -> None:
        """Start polling in a background thread and return when it is ready."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Telegram bot is already running.")

        self._started.clear()
        self._startup_error = None
        self._runtime_error = None
        self._thread = Thread(target=self._run, name=self._THREAD_NAME)
        self._thread.start()
        self._started.wait()

        if self._startup_error is not None:
            self._join()
            raise RuntimeError("Telegram bot could not be started.") from self._startup_error

    def stop(self) -> None:
        """Stop polling and wait until the background thread exits."""
        if self._thread is None or not self._thread.is_alive():
            return

        if self._loop is None or self._stop_event is None:
            return

        self._loop.call_soon_threadsafe(self._stop_event.set)
        self.wait()

    def wait(self) -> None:
        """Wait for the background bot thread."""
        self._join()
        self._raise_runtime_error()

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    @staticmethod
    def _build_application(token: str) -> Application:
        application = Application.builder().token(token).build()

        application.add_handler(CommandHandler("start", start_handler))
        application.add_handler(CommandHandler("help", help_handler))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
        )

        return application

    def _polling_error_callback(self, error: TelegramError) -> None:
        self._application.create_task(
            self._application.process_error(error=error, update=None)
        )

    def _raise_runtime_error(self) -> None:
        if self._runtime_error is not None:
            raise RuntimeError("Telegram bot stopped with an error.") from self._runtime_error

    def _run(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(self._run_application())
        except BaseException as error:
            if not self._started.is_set():
                self._startup_error = error
                self._started.set()
            else:
                self._runtime_error = error
        finally:
            if loop is not None:
                loop.close()

    async def _run_application(self) -> None:
        updater = self._application.updater
        if updater is None:
            raise RuntimeError("Telegram polling requires an application updater.")

        self._stop_event = asyncio.Event()

        try:
            await self._application.initialize()

            if self._application.post_init is not None:
                await self._application.post_init(self._application)

            await updater.start_polling(error_callback=self._polling_error_callback)
            await self._application.start()
            self._started.set()
            await self._stop_event.wait()
        finally:
            await self._shutdown(updater)

    async def _shutdown(self, updater: Updater) -> None:
        if updater.running:
            await updater.stop()

        if self._application.running:
            await self._application.stop()

            if self._application.post_stop is not None:
                await self._application.post_stop(self._application)

        await self._application.shutdown()

        if self._application.post_shutdown is not None:
            await self._application.post_shutdown(self._application)
