from threading import Event, Thread
from unittest import TestCase
from unittest.mock import AsyncMock

from src.telegram_bot.bot import TelegramBot


class TelegramBotErrorTest(TestCase):
    def test_start_reports_startup_error(self) -> None:
        bot = self._bot_with_run_error(ValueError("startup failed"))

        with self.assertRaisesRegex(
            RuntimeError, "Telegram bot could not be started."
        ) as raised:
            bot.start()

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_wait_reports_runtime_error(self) -> None:
        bot = self._bot_with_run_error(ValueError("polling failed"))
        bot._started.set()
        bot._thread = Thread(target=bot._run, name="telegram-bot-test")
        bot._thread.start()

        with self.assertRaisesRegex(
            RuntimeError, "Telegram bot stopped with an error."
        ) as raised:
            bot.wait()

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    @staticmethod
    def _bot_with_run_error(error: BaseException) -> TelegramBot:
        bot = object.__new__(TelegramBot)
        bot._thread = None
        bot._loop = None
        bot._stop_event = None
        bot._started = Event()
        bot._startup_error = None
        bot._runtime_error = None
        bot._run_application = AsyncMock(side_effect=error)
        return bot
