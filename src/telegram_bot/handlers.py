from telegram import Update
from telegram.ext import ContextTypes


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "Hallo! Ich bin WaKi-Movies. "
        "Schreib mir einen Filmwunsch, ein Genre oder eine Beschreibung, "
        "und ich gebe dir später passende Filmempfehlungen."
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    await update.message.reply_text(
        "Du kannst mir zum Beispiel schreiben:\n\n"
        "- Ich suche einen lustigen Film\n"
        "- Empfiehl mir einen Sci-Fi-Film\n"
        "- Ich mag Filme wie Matrix\n\n"
        "Aktuell ist die Empfehlungslogik noch ein Platzhalter."
    )


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return

    user_message = update.message.text

    await update.message.reply_text(
        f"Du hast geschrieben: {user_message}\n\n"
        "Später wird daraus eine echte Filmempfehlung erzeugt. "
        "Aktuell ist der Telegram Bot technisch angebunden."
    )
