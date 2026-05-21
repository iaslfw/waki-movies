from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.settings import Settings
from src.telegram_bot.handlers import help_handler, message_handler, start_handler


def create_bot_application() -> Application:
    Settings.validate()

    application = Application.builder().token(Settings.TELEGRAM_API_TOKEN).build()

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    return application


def run_bot() -> None:
    application = create_bot_application()
    application.run_polling()
