import asyncio
import logging
import os
import tempfile
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import numpy as np
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", encoding="utf-8-sig")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(BASE_DIR / "bot.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
whisper_model: WhisperModel | None = None
MAX_VOICE_BYTES = 20_000_000
MAX_VOICE_SECONDS = 600


def transcribe_locally(audio_path: Path) -> str:
    """Run Whisper on the local CPU and join all recognized segments."""
    global whisper_model
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    if audio.size == 0 or not np.any(audio):
        logger.info("Audio contains no signal")
        return ""
    if whisper_model is None:
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )

    segments, _ = whisper_model.transcribe(
        audio,
        task="transcribe",
        language="ru",
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    logger.info("Local transcription completed with %s characters", len(text))
    return text


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
    if (message.voice.file_size or 0) > MAX_VOICE_BYTES or message.voice.duration > MAX_VOICE_SECONDS:
        await message.reply_text("Отправьте запись длительностью до 10 минут и размером до 20 МБ.")
        return

    await message.chat.send_action(ChatAction.TYPING)
    status_message = await message.reply_text("Распознаю голосовое сообщение…")
    temp_path: Path | None = None

    try:
        voice_file = await context.bot.get_file(message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        await voice_file.download_to_drive(custom_path=temp_path)

        text = await asyncio.to_thread(transcribe_locally, temp_path)
        if not text:
            text = "В записи не обнаружена речь. Прослушайте её в Telegram и проверьте микрофон."
        await status_message.edit_text(text[:2000])
        for offset in range(2000, len(text), 2000):
            await message.reply_text(text[offset:offset + 2000])
        logger.info("Voice response delivered")
    except Exception as error:
        logger.error("Voice transcription failed (%s)", type(error).__name__)
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
    if not TELEGRAM_BOT_TOKEN or not re.fullmatch(r"\d+:[A-Za-z0-9_-]{30,}", TELEGRAM_BOT_TOKEN):
        raise RuntimeError("Укажите действительный TELEGRAM_BOT_TOKEN в .env (токен от @BotFather).")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Exception strings and tracebacks can contain Telegram download URLs with tokens.
    logger.error("Telegram update failed (%s)", type(context.error).__name__)


def main() -> None:
    validate_config()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(False).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(MessageHandler(filters.VOICE, transcribe_voice))
    application.add_handler(MessageHandler(filters.ALL, unsupported_message))
    application.add_error_handler(on_error)
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.error("Bot stopped (%s). Check .env, network and that only one instance is running.", type(error).__name__)
        raise SystemExit(1) from None
