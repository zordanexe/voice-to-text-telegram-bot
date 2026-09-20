import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import bot


class RecognitionTests(unittest.TestCase):
    def test_silence_does_not_load_model(self):
        for audio in (np.array([], dtype=np.float32), np.zeros(16000, dtype=np.float32)):
            with patch.object(bot, 'decode_audio', return_value=audio), patch.object(bot, 'WhisperModel') as model:
                self.assertEqual(bot.transcribe_locally(Path('silence.ogg')), '')
                model.assert_not_called()

    def test_speech_uses_vad_and_russian(self):
        model = Mock()
        model.transcribe.return_value = ([SimpleNamespace(text=' Привет '), SimpleNamespace(text=' мир. ')], None)
        with patch.object(bot, 'decode_audio', return_value=np.ones(16000)), patch.object(bot, 'whisper_model', model):
            self.assertEqual(bot.transcribe_locally(Path('voice.ogg')), 'Привет мир.')
            self.assertTrue(model.transcribe.call_args.kwargs['vad_filter'])
            self.assertEqual(model.transcribe.call_args.kwargs['language'], 'ru')

    def test_placeholder_token_is_rejected_without_echoing_it(self):
        with patch.object(bot, 'TELEGRAM_BOT_TOKEN', 'replace_with_botfather_token'):
            with self.assertRaises(RuntimeError) as caught:
                bot.validate_config()
            self.assertNotIn('replace_with_botfather_token', str(caught.exception))


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def make_update(self):
        message = SimpleNamespace(
            voice=SimpleNamespace(file_id='test', file_size=1000, duration=10),
            chat=SimpleNamespace(send_action=AsyncMock()), reply_text=AsyncMock(),
        )
        status = SimpleNamespace(edit_text=AsyncMock())
        message.reply_text.return_value = status
        voice_file = SimpleNamespace(download_to_drive=AsyncMock())
        context = SimpleNamespace(bot=SimpleNamespace(get_file=AsyncMock(return_value=voice_file)))
        return SimpleNamespace(message=message), context, status, voice_file

    async def test_long_text_is_split_and_audio_deleted(self):
        update, context, status, voice_file = self.make_update()
        text = 'Текст ' * 900
        with patch.object(bot, 'transcribe_locally', return_value=text):
            await bot.transcribe_voice(update, context)
        chunks = [status.edit_text.call_args.args[0]] + [c.args[0] for c in update.message.reply_text.call_args_list[1:]]
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        self.assertFalse(voice_file.download_to_drive.call_args.kwargs['custom_path'].exists())

    async def test_failure_deletes_audio_and_hides_exception(self):
        update, context, status, voice_file = self.make_update()
        with patch.object(bot, 'transcribe_locally', side_effect=RuntimeError('secret-value')):
            await bot.transcribe_voice(update, context)
        self.assertIn('Не удалось', status.edit_text.call_args.args[0])
        self.assertNotIn('secret-value', status.edit_text.call_args.args[0])
        self.assertFalse(voice_file.download_to_drive.call_args.kwargs['custom_path'].exists())

    async def test_silence_explained(self):
        update, context, status, _ = self.make_update()
        with patch.object(bot, 'transcribe_locally', return_value=''):
            await bot.transcribe_voice(update, context)
        self.assertIn('не обнаружена речь', status.edit_text.call_args.args[0])

    async def test_oversized_voice_not_downloaded(self):
        for size, duration in ((bot.MAX_VOICE_BYTES + 1, 10), (1000, bot.MAX_VOICE_SECONDS + 1)):
            update, context, _, _ = self.make_update()
            update.message.voice.file_size = size
            update.message.voice.duration = duration
            await bot.transcribe_voice(update, context)
            context.bot.get_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
