# Revised google_api.py using Gemini GenAI SDK
# Maintains STT, TTS, and image support with conversation support

import os
import time
import base64
import logging
import numpy as np
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import json

import pyaudio
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech

from dotenv import load_dotenv

# Language settings
default_language_code = "en-US"
default_language_name = "en-US-Standard-E"
language_code = default_language_code
language_name = default_language_name

# Initialize credentials and configure GenAI

def init_credentials(api_key: str):
    genai.configure(api_key=api_key)

def set_language(code, name):
    global language_code, language_name
    language_code = code
    language_name = name

# Initialize models

def init_text_model():
    return genai.GenerativeModel("gemini-2.0-flash-001")

def init_vision_model():
    return genai.GenerativeModel("gemini-2.0-flash-001")


# AI response functions
def ai15_text_response(model, prompt: str, conversation=None) -> str:
    if not hasattr(model, 'generate_content'):
        raise TypeError("Expected a GenAI model as the first argument.")

    contents = []

    if conversation and "history" in conversation and len(conversation["history"]) > 0:
        for entry in conversation["history"]:
            contents.append({"role": "user", "parts": [entry["user"]]})
            contents.append({"role": "model", "parts": [entry["ai"]]})

    # Always add the new prompt from user
    contents.append({"role": "user", "parts": [prompt]})

    # 👇 Correct call for Gemini 1.5 Pro
    response = model.generate_content(contents=contents)

    reply = response.candidates[0].content.parts[0].text.strip()

    # Save history for future turns
    if conversation and "history" in conversation:
        conversation["history"].append({"user": prompt, "ai": reply})

    return reply

def ai_text_response(model, prompt: str, conversation=None) -> str:
    if not hasattr(model, 'generate_content'):
        raise TypeError("Expected a GenAI model as the first argument.")

    # Build conversation text manually
    history_text = ""
    if conversation and "history" in conversation and len(conversation["history"]) > 0:
        for entry in conversation["history"]:
            history_text += f"User: {entry['user']}\nAI: {entry['ai']}\n"

    # Add new user prompt
    full_prompt = f"{history_text}User: {prompt}\nAI:"

    # Gemini 2.0 Flash expects simple prompt
    response = model.generate_content([full_prompt])
    reply = response.text.strip()

    # Save this turn
    if conversation and "history" in conversation:
        conversation["history"].append({"user": prompt, "ai": reply})

    return reply


def ai_image_response(model, image: Image.Image, text: str) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    response = model.generate_content([text, image_bytes])
    return response.text.strip()

# Conversation support using simple in-memory history


def create_conversation(history_file_path=''):
    conversation = {"history": []}

    if history_file_path and os.path.exists(history_file_path):
        with open(history_file_path, 'r') as f:
            history = json.load(f)
            for message in history:
                if 'user' in message and 'ai' in message:
                    conversation["history"].append(message)

    return conversation

def save_conversation(conversation, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(conversation["history"], f, indent=2)
        logging.info(f"Conversation saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save conversation: {e}")

# PyAudio and Speech-to-Text functions

def init_pyaudio():
    return pyaudio.PyAudio()

def init_speech_to_text():
    return speech.SpeechClient()

def start_speech_to_text(speech_client, py_audio):
    RATE = 48000
    CHUNK = int(RATE / 10)

    def audio_generator(stream, chunk):
        try:
            while True:
                data = stream.read(chunk)
                if not data:
                    break
                yield data
        except Exception as e:
            logging.error(f"Error reading from audio stream: {e}")
            yield None

    stream = py_audio.open(format=pyaudio.paInt16,
                           channels=2,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK)

    requests = (speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator(stream, CHUNK) if content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        use_enhanced=True,
        model="phone_call",
        audio_channel_count=2,
        enable_separate_recognition_per_channel=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True)

    responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)

    user_input = ""
    for response in responses:
        for result in response.results:
            if result.is_final and result.alternatives:
                user_input = result.alternatives[0].transcript
                return user_input, stream

    return user_input, stream

def stop_speech_to_text(stream):
    try:
        stream.stop_stream()
        stream.close()
    except Exception as e:
        logging.error(f"stop_speech_to_text error: {e}")

# Text-to-Speech setup and playback

def init_text_to_speech():
    client = texttospeech.TextToSpeechClient()
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=language_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    return client, voice, audio_config

def text_to_speech(text, client, voice, audio_config):
    input_text = texttospeech.SynthesisInput(text=text)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)

    try:
        audio_device = sd.query_devices("headphone")
        sd.default.device[1] = audio_device.get("index", sd.default.device[1])
    except Exception as e:
        logging.error(e)
    try:
        sd.play(audio_data, 24000)
        sd.wait()
    except Exception as e:
        logging.error(f"tts play error: {e}")

# Main interaction loop

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    load_dotenv(dotenv_path='../.env')
    api_key = os.getenv("GENAI_API_KEY", "")
    init_credentials(api_key)

    set_language(
        os.getenv("LANGUAGE_CODE", default_language_code),
        os.getenv("LANGUAGE_NAME", default_language_name)
    )

    py_audio = init_pyaudio()
    speech_client = init_speech_to_text()
    tts_client, voice, audio_config = init_text_to_speech()

    text_model = init_text_model()
    vision_model = init_vision_model()

    while True:
        user_input = input("Enter 'text', 'image', 'stt', 'tts', 'chat' or 'exit': ").strip().lower()
        if user_input == "exit":
            break

        elif user_input.startswith("text"):
            text = user_input[5:] if len(user_input) > 5 else input("Enter text: ")
            print(ai_text_response(text_model, text))

        elif user_input.startswith("image"):
            try:
                import media_api
                image = media_api.take_photo()
                if image:
                    text = user_input[6:] if len(user_input) > 6 else input("Enter prompt: ")
                    print(ai_image_response(vision_model, image, text))
            except Exception as e:
                logging.error(f"Image error: {e}")

        elif user_input == "stt":
            input_text, stream = start_speech_to_text(speech_client, py_audio)
            print(f"Recognized: {input_text}")
            time.sleep(1)
            stop_speech_to_text(stream)

        elif user_input.startswith("tts"):
            text = user_input[4:] if len(user_input) > 4 else input("Enter text to speak: ")
            text_to_speech(text, tts_client, voice, audio_config)

        elif user_input.startswith("chat"):
            history_file_path= "../res/ece_history.json"
            conversation = create_conversation(history_file_path)
            init_input = (
                "From now on, always answer as if a human being is speaking naturally, "
                "with concise, relevant, and conversational tone. "
                "Only respond in one-breath answers. "
                "If the input uses a different language, like language A, "
                "please respond in that same language."
            )
            text = ai_text_response(text_model, init_input, conversation)
            print(f"Recognized: {text}")
            text = user_input[5:] if len(user_input) > 5 else input("Enter text: ")
            response = ai_text_response(text_model, text, conversation)

            print(f"Recognized: {response}")


if __name__ == '__main__':
    main()
