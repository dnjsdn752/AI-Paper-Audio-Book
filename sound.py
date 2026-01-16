from google.cloud import texttospeech
from google.oauth2 import service_account
import os
from config import GOOGLE_APPLICATION_CREDENTIALS

# Creates a client tts
if GOOGLE_APPLICATION_CREDENTIALS and isinstance(GOOGLE_APPLICATION_CREDENTIALS, dict):
    credentials = service_account.Credentials.from_service_account_info(GOOGLE_APPLICATION_CREDENTIALS)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
else:
    client = texttospeech.TextToSpeechClient()

# Select the language and SSML Voice Gender (optional)
voice = texttospeech.VoiceSelectionParams(
    language_code='ko-KR',
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    name='ko-KR-Wavenet-B'
)

# Select the type of audio encoding
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

def gc_speak(txt):
    if txt is not None:

        # The text to synthesize
        text = txt

        # Construct the request
        input_text = texttospeech.SynthesisInput(text=text)

        # Performs the Text-to-Speech request
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )

        if isinstance(txt, str):  # 문자열인 경우에만 음성 출력
            # Save the audio response to a temporary file
            temp_file = './mp3/book_error_skip.mp3'
            if os.path.exists(temp_file):
                os.remove(temp_file)
            with open(temp_file, 'wb') as out:
                out.write(response.audio_content)
                
            playsound(temp_file)

            '''
            # Play the audio using default player
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['afplay', temp_file])
            elif platform.system() == 'Windows':  # Windows
                os.startfile(temp_file)
            elif platform.system() == 'Linux':  # Linux
                subprocess.run(['aplay', temp_file])
            else:
                print("Unsupported operating system. Cannot play audio automatically.")
            '''
gc_speak("죄송합니다. 현재 페이지는 인식이 불가합니다. 다음페이지로 넘겨주세요. ")