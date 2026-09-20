import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explain the single purpose of the bot."""
    if update.message:
        await update.message.reply_text(
            "Отправьте мне голосовое сообщение — я верну его текстом."
        )


async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download a Telegram voice note and return its transcription."""
    message = update.message
    if not message or not message.voice:
        return

    await message.chat.send_action(ChatAction.TYPING)
    status_message = await message.reply_text("Распознаю голосовое сообщение…")
    temp_path: Path | None = None

    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        await voice_file.download_to_drive(custom_path=temp_path)

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        with temp_path.open("rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="text",
            )

        text = str(transcription).strip()
        if not text:
            text = "Не удалось распознать речь. Попробуйте записать сообщение ещё раз."
        await status_message.edit_text(text)
    except Exception:
        logger.exception("Voice transcription failed")
        await status_message.edit_text(
            "Не удалось обработать голосовое сообщение. Попробуйте ещё раз позже."
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


async def unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt the user to send a voice note instead of another message type."""
    if update.message:
        await update.message.reply_text("Пожалуйста, отправьте голосовое сообщение.")


def validate_config() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Не заданы переменные окружения: {', '.join(missing)}")


def main() -> None:
    validate_config()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, transcribe_voice))
    application.add_handler(MessageHandler(filters.ALL, unsupported_message))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
