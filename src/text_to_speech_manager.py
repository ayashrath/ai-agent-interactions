import os
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv


class TextToSpeechManager:
    """
    Manages text to speech conversion using Azure Cognitive Services.
    Look at https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support,
    https://speech.microsoft.com/portal/voicegallery
    """
    def __init__(self):
        load_dotenv()

    @staticmethod
    def convert_text_to_speech(
            text: str, action: str = 'speak', speech_voice: str = "en-US-Ava:DragonHDLatestNeural",
            rate : float = 1.0
    )-> speechsdk.SpeechSynthesisResult:
        """
        Converts text to speech and plays it.
        :param text: Text to convert to speech.
        :param action: Action to perform with the speech ('speak'/'save'/stream).
        :param speech_voice: Voice to use for speech synthesis.
        :param rate: Speech rate (1.0 is normal speed).
        :return: Speech synthesis result.
        """

        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get('SPEECH_KEY'),
            endpoint=os.environ.get('SPEECH_ENDPOINT')
        )
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)  # how to output audio

        speech_config.speech_synthesis_voice_name = speech_voice  # name of the voice

        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config = speech_config,
            audio_config = audio_config
        )  # synthesiser object

        # rate using SSML
        rate_percent = int((rate - 1.0) * 100)
        ssml_text = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
          <voice name="{speech_voice}">
            <prosody rate="{rate_percent:+}%">
              {text}
            </prosody>
          </voice>
        </speak>
        """

        cancellation_details = "Cancelled before running"
        for i in range(1, 4):
            speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml_text).get()
            if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return speech_synthesis_result
            cancellation_details = speech_synthesis_result.cancellation_details
            print(f"Error Details: {cancellation_details.error_details}")
            print("Retrying...")
            time.sleep(20 * 2 ** i)
        return cancellation_details.error_details