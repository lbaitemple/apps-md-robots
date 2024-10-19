#
# Copyright 2024 MangDang (www.mangdang.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: This Python script is designed to handle various AI-driven tasks, including text-based conversations,
# speech recognition (speech-to-text), and speech synthesis (text-to-speech). It integrates Google Cloud services and other libraries to accomplish these tasks.
#
# Gemini Test Method: type 'text' followed by a ' ' (space), and the text you want to type, then press enter.
# Gemini Visio Pro Test method: type 'image' followed by a ' ' (space), and the text you want to type, then press enter.
# Speech-T-Text Test method: type 'text'. After pressing enter, start speaking, then press enter.
# Text-To-Speech Test method: type 'text' followed by a ' ' (space), and the text you want to type, then press enter.
#
# References: https://python.langchain.com/v0.1/docs/integrations/llms/google_vertex_ai_palm/
#             https://cloud.google.com/speech-to-text/docs
#             https://cloud.google.com/text-to-speech/docs
#

import logging
import os
import base64
import time
import numpy as np
import google.auth
from PIL import Image
from vertexai.preview.generative_models import Image as VertexImage
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
import pyaudio
import sounddevice as sd
import soundfile as sf
from google.cloud import speech
from google.cloud import texttospeech
from io import BytesIO
import asyncio
import json


# language code and name, default is en-US
language_code = "en-US"
language_name = "en-US-Standard-E"

def init_credentials(key_json_path):
    """
    Initializes Google Cloud credentials by setting the environment variable.

    Parameters:
    - key_json_path (str): The file path to the Google Cloud credentials JSON file.

    Returns:
    - credentials: The credentials object for Google Cloud authentication.
    - project_id: The project ID associated with the provided credentials.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_json_path
    credentials, project_id = google.auth.default()
    return credentials, project_id

def load_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []
    
def create_conversation(history_file_path=''):
    """
    Creates an instance of ConversationChain for AI interactions.

    Returns:
    - conversation (ConversationChain): The conversation object initialized with the AI model and prompt template.
    """

    hist_txt =""
    if (history_file_path != ''):
        history = load_history(history_file_path)
        # Populate memory with loaded history
        for message in history:
            if message['role'] == 'user':
                hist_txt = message['content']
                

    model = ChatVertexAI(
        model_name='gemini-pro',
        convert_system_message_to_human=True,
    )

    
    logging.debug(f"{hist_txt}")
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are a small female robo puppy, your name is Puppy. {}. You will be a helpful AI assistant.
                Your LLM api is connected to STT and several TTS models so you are able to hear the user
                and change your voice and accents whenever asked.
                After being asked to change voice, the TTS handles the process, so ALWAYS assume the voice has changed, so asnwer appropriately.
                ---
                ONLY use text and avoid any other kinds of characters from generating.
                MUST generate a reponse for 35 words or less. Your MUST condense a list to 20 words if you have one.
                ONLY give one breathe response.
                """.format(hist_txt)
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    conversation = ConversationChain(llm=model, prompt=prompt, verbose=False, memory=memory)
    logging.debug("conversation create end!")
    return conversation

def ai_text_response(conversation, input_text):
    """
    Generates a text response from the AI model based on the input text.

    Parameters:
    - conversation (ConversationChain): The conversation object containing the AI model state.
    - input_text (str): The text input to be processed by the AI model.

    Returns:
    - result (str): The text response generated by the AI model.
    """
    logging.debug("ai_text_response start!")
    ms_start = int(time.time() * 1000)

    result = conversation.invoke(input=input_text)

    logging.debug(f"ai_text_response response: {result}")
    result = result['response']
    logging.debug(f"text response: {result}")
    ms_end = int(time.time() * 1000)
    logging.debug(f"ai_text_response end, delay = {ms_end - ms_start}ms")
    return result

def ai_image_response(llm, image, text):
    """
    Generates a response from the AI model based on the provided image and text.

    Parameters:
    - llm (ChatVertexAI): The AI model instance for processing images.
    - image (PIL.Image): The image object to be processed.
    - text (str): The accompanying text for the image.

    Returns:
    - result (str): The text response generated by the AI model.
    """
    logging.debug("ai_image_response start!")
    ms_start = int(time.time() * 1000)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    image_message = {
        "type": "image_url",
        "image_url": {
            "url": image_data_url
        }
    }
    text_message = {"type": "text", "text": text}

    message = HumanMessage(content=[text_message, image_message])

    output = llm.invoke([message])

    #logging.debug(f"ai_image_response response: {output}")
    result = output.content
    logging.debug(f"text response: {result}")
    ms_end = int(time.time() * 1000)
    logging.debug(f"ai_image_response end, delay = {ms_end - ms_start}ms")
    return result

def set_language(code, name):
    """
    set the default languange for stt and tts.

    Parameters:
    - code (String): The language code.
    - name (String): The language name.

    Reference: https://cloud.google.com/text-to-speech/docs/voices
    """

    global language_code, language_name
    language_code = code
    language_name = name

def init_pyaudio():
    """
    Initializes the PyAudio library for handling audio streams.

    Returns:
    - p (pyaudio.PyAudio): The PyAudio instance ready for audio I/O operations.
    """
    p = pyaudio.PyAudio()
    return p

def init_speech_to_text():
    """
    Initializes the Google Cloud Speech-to-Text client for audio transcription.

    Returns:
    - speech_client (speech.SpeechClient): The initialized Speech-to-Text client.
    """
    speech_client = speech.SpeechClient()
    return speech_client

# Note:  Very important!!!
# After you called this function start_speech_to_text(), You need to call stop_speech_to_text() some time later, not immediately because it will crash
# Function to detect voice and transribe speech
def start_speech_to_text(speech_client, py_audio):
    """
    Starts the speech-to-text process to transcribe audio input to text.

    Parameters:
    - speech_client (speech.SpeechClient): The initialized Speech-to-Text client.
    - py_audio (pyaudio.PyAudio): The PyAudio instance for handling audio streams.

    Returns:
    - user_input (str): The transcribed text from the audio input.
    - stream: The audio stream used for the transcription process.
    """

    RATE = 48000
    CHUNK = int(RATE / 10)

    def audio_generator(stream, chunk):
        try:
            while True:
                data = stream.read(chunk)
                if not data:
                    # End of stream, break out of the loop
                    break
                yield data
        except Exception as e:
            # Handle any exceptions that may occur while reading from the stream
            logging.error(f"Error reading from audio stream: {e}")
            # You may want to close the stream or return a signal to indicate an error
            yield None


    stream = py_audio.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator(stream, CHUNK) if content)

    streaming_config = speech.StreamingRecognitionConfig(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
            use_enhanced=True,
            model="phone_call",
            audio_channel_count=2,
            enable_separate_recognition_per_channel=True,
        ),
        interim_results=True
    )
    responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)

    user_input = ""
    should_break=False
    for response in responses:
        for result in response.results:
            if result.is_final:
                if result.alternatives:
                    user_input = result.alternatives[0].transcript
                    logging.debug(user_input)
                    should_break = True
                    break
        if should_break:
            break
    logging.debug(f"voice text:{user_input}")
    #stream.stop_stream()
    #stream.close()
    return user_input, stream

# Note: Very Important!!!!
# You need to call stop_speech_to_text() some time after start_speech_to_text(), not immediately because it will crash
def stop_speech_to_text(stream):
    """
    Stops the speech-to-text stream to prevent potential crashes.

    Parameters:
    - stream: The audio stream object to be stopped and closed.
    """
    try:
        stream.stop_stream()
        stream.close()
    except Exception as e:
        logging.error(f"stop_speech_to_text error: {e}")
        pass


def init_text_to_speech():
    """
    Initializes the Google Cloud Text-to-Speech client and sets up voice and audio configurations.

    Returns:
    - tts_client (texttospeech.TextToSpeechClient): The initialized Text-to-Speech client.
    - voice (texttospeech.VoiceSelectionParams): The default voice instance.
    - audio_config (texttospeech.AudioConfig): The audio configuration instance.
    """
    # Create TextToSpeechClient instance
    tts_client = texttospeech.TextToSpeechClient()

    # Create voice instance
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=language_name)

    # Create audio configuration instance
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    return tts_client, voice, audio_config


def text_to_speech(text, tts_client, voice, audio_config):
    """
    Converts the provided text to speech using the Google Cloud Text-to-Speech service.

    Parameters:
    - text (str): The text to be converted to speech.
    - tts_client (texttospeech.TextToSpeechClient): The initialized Text-to-Speech client.
    - voice (texttospeech.VoiceSelectionParams): The voice instance to be used for speech synthesis.
    - audio_config (texttospeech.AudioConfig): The audio configuration instance.

    Returns:
    - None, but plays the synthesized speech to the audio output.
    """
    ms_start = int(time.time() * 1000)
    synthesis_input = texttospeech.SynthesisInput(text=text)

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
    ms_end = int(time.time() * 1000)
    logging.debug(f"google tts end, delay = {ms_end - ms_start}ms")


    logging.debug(sd.default.device)
    # specfiy the innner audio play device "bcm2835 Headphones" on mini pupper
    try:
        audio_device = sd.query_devices("headphone")
        logging.info(audio_device)
        sd.default.device[1] = audio_device.get("index", sd.default.device[1])
        logging.debug(sd.default.device)
    except Exception as e:
        logging.error(e)
    try:
        sd.play(audio_data, 24000)
        sd.wait()
    except Exception as e:
        logging.error(f"tts play error:{e}")


def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s',
        level=logging.DEBUG
    )

    current_file_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(current_file_path))
    logging.debug(f"init chdir: {os.path.dirname(current_file_path)}")

    from dotenv import load_dotenv
    load_dotenv(dotenv_path='../.env')
    api_path = os.environ.get('API_KEY_PATH', '')
    if os.path.exists(api_path):
        init_credentials(api_path)

    lang_code = os.environ.get('LANGUAGE_CODE', language_code)
    lang_name = os.environ.get('LANGUAGE_NAME', language_name)
    set_language(lang_code, lang_name)

    py_audio = init_pyaudio()
    speech_client = init_speech_to_text()
    tts_client, voice, audio_config = init_text_to_speech()
    conversation = create_conversation()
    multi_model = ChatVertexAI(model="gemini-pro-vision")

    while True:
        user_input = input("Enter function apis -- 'text'/'image'/'stt'/'tts' or 'exit' to quit: ").strip().lower()
        if not user_input:
            continue
        inputs = user_input.split()
        first_word = inputs[0]

        if "exit" == first_word:
            logging.debug("Exit!")
            break

        elif "text" == first_word:
            input_text = ' '.join(inputs[1:])
            if not input_text:
                logging.debug("No input text!")
            else:
                logging.debug(f"input text: {input_text}")
                response = ai_text_response(conversation=conversation, input_text=input_text)
                print(response)

        elif "image" == first_word:
            import media_api
            image = media_api.take_photo()

            if image is None:
                logging.debug("No image captured!")
            else:
                text_prompt = ' '.join(inputs[1:])
                if not text_prompt:
                    logging.debug("No text prompt provided!")
                else:
                    logging.debug(f"text prompt: {text_prompt}")
                    response = ai_image_response(multi_model, image=image, text=text_prompt)
        elif "stt" == first_word:
            input_text, stream = start_speech_to_text(speech_client, py_audio)

            if not input_text:
                logging.debug("No speech detected!")
            else:
                logging.debug(f"input text: {input_text}")
            time.sleep(1)
            stop_speech_to_text(stream)

        elif "tts" == first_word:
            input_text = ' '.join(inputs[1:])
            if not input_text:
                logging.debug("No input text provided!")
            else:
                logging.debug(f"TTS is speaking: {input_text}")
                text_to_speech(input_text, tts_client, voice, audio_config)

if __name__ == '__main__':
    main()
