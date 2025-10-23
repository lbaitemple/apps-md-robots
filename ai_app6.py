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
# Description: Enhanced AI app with TTS interrupt capability and visual feedback.
# Users can say "that is enough" or "stop" to interrupt long responses.
# Includes noise-robust speech-to-text using WebRTC VAD with visual status indicators.
# Visual feedback: hello_y.png (calibration), hello_r.png (listening), hello_g.png (completed)
#

import logging
import os
import time
import re
import numpy as np
from PIL import Image
import pyaudio
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import asyncio
import threading
import queue as queue_module
from google.cloud import texttospeech
from langchain_google_vertexai import ChatVertexAI
import random
import getpass
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import noisereduce as nr
import webrtcvad


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_queue import input_text_queue, output_text_queue, gif_queue, image_queue, movement_queue, stt_queue, heads_up_queue
from api import media_api, google_api, move_api, shell_api


RES_DIR = "cartoons"

# Game text for the rock-paper-scissors game
GAME_TEXT = "Let's play! Rock! Paper! Scissor! Shoot!"
ai_on = True

# TTS interrupt control
tts_interrupt_flag = threading.Event()  # Flag to signal TTS to stop
tts_active = False  # Track if TTS is currently speaking

# Define voice parameters for different languages and a default voice
voice0 = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-E")
voice_man = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Neural2-D")
voice_JP = texttospeech.VoiceSelectionParams(language_code="ja-JP", name="ja-JP-Neural2-B")
voice_CN = texttospeech.VoiceSelectionParams(language_code="cmn-CN", name="cmn-CN-Wavenet-A")
voice_IT = texttospeech.VoiceSelectionParams(language_code="it-IT", name="it-IT-Standard-B")
voice_DE = texttospeech.VoiceSelectionParams(language_code="de-DE", name="de-DE-Neural2-D")
voice_FR = texttospeech.VoiceSelectionParams(language_code="fr-FR", name="fr-FR-Standard-C")
voice_HK = texttospeech.VoiceSelectionParams(language_code="yue-HK", name="yue-HK-Standard-C")
voice_ES = texttospeech.VoiceSelectionParams(language_code="es-US", name="es-US-Wavenet-A")
voice_IL = texttospeech.VoiceSelectionParams(language_code="es-US", name="he-IL-Standard-A")

lang_voices = {
    "Japanese": voice_JP,
    "Chinese": voice_CN,
    "Italian": voice_IT,
    "German": voice_DE,
    "French": voice_FR,
    "Cantonese": voice_HK,
    "Spanish": voice_ES,
    "Hebrew": voice_IL,    
}
cur_voice = voice0

#heads up variables
playing_heads_up = False
heads_up_word = ""
heads_up_questions = 0

# Track last response for translation
last_response = ""


def show_status_image(status):
    """
    Display status images for different speech recognition stages.
    Only displays images when AI is active (ai_on=True).
    When AI is off, keeps the logo2.png image displayed.
    
    Args:
        status: 'calibrating', 'listening', 'completed', or 'ready'
    """
    global ai_on
    
    # Don't change image when AI is off (close_ai mode)
    if not ai_on:
        logging.debug(f"AI is off - keeping logo2.png, ignoring status '{status}'")
        return
    
    try:
        if status == 'calibrating':
            image = Image.open(f"{RES_DIR}/hello_y.png")  # Yellow for calibration
            image_queue.put(image)
            logging.info("🟡 Displaying calibration status (hello_y.png)")
        elif status == 'listening':
            image = Image.open(f"{RES_DIR}/hello_r.png")  # Red for active listening
            image_queue.put(image)
            logging.info("🔴 Displaying listening status (hello_r.png)")
        elif status == 'completed':
            image = Image.open(f"{RES_DIR}/hello_g.png")  # Green for completed
            image_queue.put(image)
            logging.info("🟢 Displaying completion status (hello_g.png)")
        elif status == 'ready':
            image = Image.open(f"{RES_DIR}/hello.png")    # Default ready state
            image_queue.put(image)
            logging.info("⚪ Displaying ready status (hello.png)")
    except Exception as e:
        logging.error(f"Failed to display status image for '{status}': {e}")


class NoiseRobustSTT:
    """
    Noise-robust speech-to-text for noisy environments.
    Uses WebRTC VAD and advanced noise reduction with visual feedback.
    """
    
    def __init__(self, speech_client, py_audio, sample_rate=16000, chunk_size=320, 
                 vad_aggressiveness=1, language_code="en-US"):
        """
        Initialize noise-robust STT.
        
        Args:
            speech_client: Google Speech client
            py_audio: PyAudio instance
            sample_rate: Audio sample rate (16000 recommended)
            chunk_size: Chunk size for VAD (320 = 20ms at 16kHz)
            vad_aggressiveness: VAD sensitivity (0-3, lower = more sensitive)
            language_code: Language code for speech recognition
        """
        self.speech_client = speech_client
        self.py_audio = py_audio
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = 1
        self.language_code = language_code
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_duration_ms = 20
        
        # VAD parameters for speech detection
        self.num_silent_frames_threshold = 25  # ~0.5 seconds of silence (more responsive)
        self.num_speech_frames_threshold = 3  # ~60ms to confirm speech (very sensitive)
        self.padding_frames = 10  # Extra frames before/after speech
        
                # Noise reduction settings (optimized for complex noise)
        self.noise_profile = None
        self.calibration_time = 2.0
        self.noise_reduction_strength = 0.5  # Moderate - preserve speech clarity
        self.use_stationary_noise = False
        self.silence_threshold = 500
        self.apply_noise_reduction = False  # Disabled by default - can cause audio to be too quiet
        self.enable_noise_reduction = False  # Disabled - VAD + raw audio works better
        
        # Speech detection state
        self.accumulated_audio = []
        self.ring_buffer = []
        self.ring_buffer_size = 30  # Keep 30 frames (~0.6 seconds) before speech
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False
        
        logging.info(f"Noise-Robust STT initialized (VAD aggressiveness: {vad_aggressiveness})")
    
    def calibrate_noise(self, stream):
        """
        Calibrate noise profile from ambient sound with visual feedback.
        """
        # Show calibration status image
        show_status_image('calibrating')
        
        print("\n" + "="*60)
        print("🎤 Calibrating noise profile for complex noise environment...")
        print("="*60)
        print("Please remain silent for 2 seconds to capture background noise...")
        print("(This will help filter TV noise, dog barking, people talking, lawn mowers, etc.)")
        print()
        
        logging.info("Calibrating noise profile for noisy environment...")
        logging.info("Capturing background noise (TV, dogs, people, lawn mowers, etc.)...")
        
        noise_samples = []
        frames_needed = int(self.sample_rate / self.chunk_size * self.calibration_time)
        
        for i in range(frames_needed):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            noise_samples.append(audio_data)
            
            # Show progress
            if (i + 1) % 8 == 0:
                progress = (i + 1) / frames_needed * 100
                print(f"  Capturing noise... {progress:.0f}%")
        
        self.noise_profile = np.concatenate(noise_samples)
        
        # Calculate adaptive threshold based on noise level
        noise_rms = np.sqrt(np.mean(self.noise_profile.astype(np.float32) ** 2))
        self.silence_threshold = max(500, noise_rms * 3.0)
        
        print(f"\n✓ Noise calibration complete!")
        print(f"  Detected noise level: {noise_rms:.0f}")
        print(f"  Adaptive silence threshold: {self.silence_threshold:.0f}")
        print("="*60 + "\n")
        
        logging.info(f"Noise calibration complete! Noise level: {noise_rms:.0f}, Threshold: {self.silence_threshold:.0f}")
        
        # Show ready status after calibration
        show_status_image('ready')
    
    def reduce_noise(self, audio_data):
        """
        Apply advanced noise reduction for complex noise environments.
        """
        if not self.enable_noise_reduction:
            return audio_data
            
        if self.noise_profile is not None and len(audio_data) > 0:
            try:
                reduced = nr.reduce_noise(
                    y=audio_data.astype(np.float32),
                    sr=self.sample_rate,
                    y_noise=self.noise_profile.astype(np.float32),
                    stationary=self.use_stationary_noise,
                    prop_decrease=self.noise_reduction_strength,
                    freq_mask_smooth_hz=500,  # Lower = preserve more speech
                    time_mask_smooth_ms=50,   # Lower = preserve more speech transients
                    n_fft=2048,
                    clip_noise_stationary=True
                )
                return reduced.astype(np.int16)
            except Exception as e:
                logging.warning(f"Noise reduction failed: {e}")
                return audio_data
        return audio_data
    
    def is_speech_vad(self, audio_data):
        """
        Use WebRTC VAD to detect if audio chunk contains speech.
        """
        try:
            # Ensure correct frame size for VAD
            expected_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000)
            
            if len(audio_data) != expected_size:
                if len(audio_data) < expected_size:
                    audio_data = np.pad(audio_data, (0, expected_size - len(audio_data)), 'constant')
                else:
                    audio_data = audio_data[:expected_size]
            
            audio_bytes = audio_data.tobytes()
            return self.vad.is_speech(audio_bytes, self.sample_rate)
            
        except Exception as e:
            # Fallback to amplitude-based detection
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            return rms >= self.silence_threshold
    
    def transcribe_audio(self, audio_bytes):
        """
        Transcribe audio using Google Speech-to-Text API.
        """
        try:
            from google.cloud import speech
            
            # Log audio info
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            duration = len(audio_array) / self.sample_rate
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            logging.info(f"Transcribing audio: duration={duration:.2f}s, RMS={rms:.0f}, size={len(audio_bytes)} bytes")
            
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Choose model based on audio duration
            # Use 'default' model for shorter audio, 'latest_long' for longer
            model_type = "default" if duration < 3.0 else "latest_long"
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model=model_type,
                use_enhanced=True,
                audio_channel_count=self.channels,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                alternative_language_codes=[
                    "zh-CN", "zh-TW", "es-ES", "fr-FR", "de-DE",
                    "ja-JP", "ko-KR", "pt-BR", "ru-RU", "it-IT"
                ]
            )
            logging.debug(f"Using model: {model_type} for {duration:.2f}s audio")
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    logging.info(f"Transcription: '{transcript}' (confidence: {confidence:.2%})")
                    return transcript
                else:
                    logging.warning("No alternatives in transcription result")
            else:
                logging.warning("No results from Speech API - audio may be too quiet, too short, or not contain clear speech")
            
            return None
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def listen_once(self, stream):
        """
        Listen for one complete speech utterance using VAD with visual feedback.
        Returns the transcribed text or None.
        """
        # Reset state
        self.accumulated_audio = []
        self.ring_buffer = []
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False
        speech_detected = False
        
        logging.debug("Listening for speech with noise-robust VAD...")
        
        # Listen loop with timeout (max 30 seconds)
        max_iterations = int(30 * self.sample_rate / self.chunk_size)
        
        for iteration in range(max_iterations):
            try:
                # Read audio chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Apply noise reduction
                cleaned_audio = self.reduce_noise(audio_data)
                
                # Use VAD to detect speech
                is_speech = self.is_speech_vad(cleaned_audio)
                
                if is_speech:
                    self.consecutive_speech_frames += 1
                    self.consecutive_silent_frames = 0
                    
                    # Confirm speech start after enough consecutive speech frames
                    if not self.is_currently_speaking and self.consecutive_speech_frames >= self.num_speech_frames_threshold:
                        self.is_currently_speaking = True
                        if not speech_detected:
                            ring_buffer_duration = len(self.ring_buffer) * 0.02
                            logging.info(f"🎤 Speech detected! Recording... (captured {ring_buffer_duration:.2f}s pre-speech buffer)")
                            speech_detected = True
                            
                            # Show listening status image
                            show_status_image('listening')
                            
                            # Add ring buffer (captures the first word!)
                            if self.ring_buffer:
                                logging.debug(f"Adding {len(self.ring_buffer)} ring buffer frames to audio")
                                self.accumulated_audio.extend(self.ring_buffer)
                                self.ring_buffer = []
                    
                    # Accumulate speech audio
                    if self.is_currently_speaking:
                        self.accumulated_audio.append(cleaned_audio)
                    else:
                        # Not speaking yet, maintain ring buffer
                        self.ring_buffer.append(cleaned_audio)
                        if len(self.ring_buffer) > self.ring_buffer_size:
                            self.ring_buffer.pop(0)
                else:
                    self.consecutive_silent_frames += 1
                    self.consecutive_speech_frames = 0
                    
                    # Show progress when detecting silence after speech
                    if self.is_currently_speaking and self.consecutive_silent_frames % 10 == 0 and self.consecutive_silent_frames > 0:
                        remaining = self.num_silent_frames_threshold - self.consecutive_silent_frames
                        if remaining > 0:
                            logging.debug(f"Silence detected: {self.consecutive_silent_frames}/{self.num_silent_frames_threshold} frames...")
                    
                    # Continue accumulating for padding
                    if self.is_currently_speaking and self.consecutive_silent_frames <= self.padding_frames:
                        self.accumulated_audio.append(cleaned_audio)
                    elif not self.is_currently_speaking:
                        # Maintain ring buffer
                        self.ring_buffer.append(cleaned_audio)
                        if len(self.ring_buffer) > self.ring_buffer_size:
                            self.ring_buffer.pop(0)
                
                # Check if speech ended
                if self.is_currently_speaking and self.consecutive_silent_frames >= self.num_silent_frames_threshold:
                    logging.info(f"Speech ended after {self.consecutive_silent_frames} silent frames (~{self.consecutive_silent_frames * 0.02:.1f}s). Processing...")
                    
                    # Show completion status image
                    show_status_image('completed')
                    
                    # Convert accumulated audio to bytes
                    if self.accumulated_audio:
                        full_audio = np.concatenate(self.accumulated_audio)
                        duration = len(full_audio) / self.sample_rate
                        
                        # Check minimum duration (Google requires at least 0.1s, but 0.5s is more reliable)
                        if duration < 0.5:
                            logging.warning(f"Audio too short ({duration:.2f}s), need at least 0.5s - ignoring")
                            # Reset and continue listening
                            self.accumulated_audio = []
                            self.speech_frames_buffer = []
                            self.consecutive_silent_frames = 0
                            self.consecutive_speech_frames = 0
                            self.is_currently_speaking = False
                            self.ring_buffer = []
                            show_status_image('ready')  # Return to ready state
                            return None
                        
                        audio_bytes = full_audio.tobytes()
                        
                        # Transcribe
                        transcript = self.transcribe_audio(audio_bytes)
                        
                        # Show ready status after processing
                        show_status_image('ready')
                        
                        return transcript
                    
                    # Show ready status if no audio accumulated
                    show_status_image('ready')
                    return None
                    
            except Exception as e:
                logging.error(f"Error during listening: {e}")
                show_status_image('ready')  # Return to ready state on error
                return None
        
        # Timeout - no speech detected
        logging.debug("Listening timeout - no speech detected")
        show_status_image('ready')  # Return to ready state on timeout
        return None


# Track last response for translation
last_response = ""

def send_response(text):
    """
    Send response to output queue and track it for potential translation.
    """
    global last_response
    last_response = text
    output_text_queue.put(text)


#heads up ai prompt
def create_heads_up_prompt(secret_word):
    prompt = f"""
You are an AI assistant playing a guessing game similar to "Heads-Up" or "20 Questions."
I have a secret word in mind, and I will tell it to you now.
Your role is to be the "Knower" or "Answerer." I will be the "Guesser."

**The Secret Word is: {secret_word}**

Your task is to help me guess this secret word by answering my questions.

Here are the rules for how you must behave:

1.  **Acknowledge:** After I give you this prompt (including the secret word), simply respond with "Okay, I have the secret word. I'm ready for your first question." .
2.  **Answer Questions:** I will ask you questions, or questions that can be answered with short, factual clarifications.
    *   If the question can be answered with yes or no try to stick to that, you may answer open-ended questions, but do not give too much information away
    *   DO NOT use any direct, significant parts, or obvious roots of the `[SECRET_WORD]` in your clues.** (e.g., if the word is "rainbow," do not say "it's a bow in the sky" or mention "rain").
3.  **DO NOT Reveal the Word:** Under NO circumstances should you say the secret word, spell it out, or give clues that directly lead to the word (e.g., "It rhymes with X," or "It starts with Y").
4.  **Be Truthful:** Your answers must be truthful based on the secret word I've given you.
5.  **Be Concise:** Keep your answers as short as possible while still being helpful.
6.  **Provide varied clues:** This includes descriptions, associations, actions related to it, sounds it might make (described in text), things it's similar to or different from, its purpose, common contexts, etc.
7.  **Answer the player's questions truthfully but cleverly, always steering them towards the `[SECRET_WORD]` without giving it away too easily.**
8.  **Listen to specific hint requests:** If the player asks "What does it sound like?" or "Give me an action," try to fulfill that type of hint.
9.  **Adjust difficulty:** If the player requests an "easier" or "harder" hint, try to adjust the directness or obscurity of your next clues accordingly.
10.  **Maintain a friendly, engaging, and encouraging tone.**
11.  **Your clue responses should be concise and directly address the player's query as a clue.** Avoid unnecessary conversational filler.
12.  **Handling Guesses:** If I say "Is the word [GUESS]?", you must respond with:
    *   "Yes, that's it! The word was {secret_word}." if I am correct.
    *   "No, that's not the word. Keep trying!" if I am incorrect.
13.  **Goal:** Your ultimate goal is to help me guess the secret word by accurately and concisely answering my questions within these rules.
14.  **DO NOT ASK Questions** it is your job to answer questions, not ask them

Let's begin. I have provided the secret word above. Await my first question after your acknowledgment. My next message will be a question.
"""
    return prompt

def detect_lang_usage(prompt, lang):
    adjectives = ['food', 'culture', 'characters', 'novel', 'history']
    language_phrases = [f'in {lang}', f'to {lang}', f'say in {lang}', f'translate to {lang}']
   
    for phrase in language_phrases:
        if phrase in prompt:
            return "Language choice"
   
    for adj in adjectives:
        if f'{lang} {adj}' in prompt:
            return "Adjective"
   
    return "Unknown"
   
def get_voice(prompt=None):
    """
    Determine the voice to be used based on the input prompt.
    """
    if not prompt:
        logging.debug(f"select key voice: None,default is voice0")
        return None, voice0
    for key, value in lang_voices.items():
        if key in prompt:
            if detect_lang_usage(prompt, key) == "Language choice":
                logging.info(f"select key: {key}")
                return key, value
    logging.info(f"no mapping, default is voice0")
    return None, voice0

move_cmd_functions = {
                 "action": move_api.init_movement,
                 "sit": move_api.squat,
                 "move forwards": move_api.move_forward,
                 "move backwards": move_api.move_backward,
                 "move left": move_api.move_left,
                 "move right": move_api.move_right,
                 "look up": move_api.look_up,
                 "look down": move_api.look_down,
                 "look left": move_api.look_left,
                 "look upper left": move_api.look_upperleft,
                 "look lower left": move_api.look_leftlower,
                 "look right": move_api.look_right,
                 "look upper right": move_api.look_upperright,
                 "look lower right": move_api.look_rightlower,
             }

def get_move_cmd(input_text, command_dict):
    """
    Find the command key in the input text based on the command dictionary.
    """
    if not input_text:
        return None
    for command_key in command_dict.keys():
        if re.search(r'\b' + re.escape(command_key) + r'\b', input_text):
            return command_key
    return None

def close_ai():
    global ai_on
    ai_on = False
    stt_queue.put(True)
    image = Image.open(f"{RES_DIR}/logo2.png")
    image_queue.put(image)

def open_ai():
    global ai_on
    ai_on = True
    stt_queue.put(True)
    show_status_image('ready')
    output_text_queue.put("OK, my friend.")

def reboot():
    command = "sudo reboot"
    shell_api.execute_command(command)

def power_off():
    command = "sudo poweroff"
    shell_api.execute_command(command)

sys_cmds_functions = {
        "shut up": close_ai,
        "speak please": open_ai,
        "reboot": reboot,
        "power off": power_off,
        }

def get_sys_cmd(input_text, command_dict):
    normalized_text = re.sub(r'[^\w\s]', '', input_text.lower())

    for command_key in command_dict.keys():
        if normalized_text == command_key.lower():
            return command_key, command_dict[command_key]

    return None, None


def cut_text_by_last_period(text, max_words_before_period=15):
    """
    Cut the text by the last period within a specified number of words.
    """
    words = text.split()

    last_period_index = -1
    for i, word in enumerate(words[:max_words_before_period]):
        if '.' in word:
            last_period_index = i

    if last_period_index != -1:
        return ' '.join(words[:last_period_index+1])

    first_period_index = -1
    for i, word in enumerate(words):
        if '.' in word:
            first_period_index = i
            break

    return ' '.join(words[:first_period_index+1]) if first_period_index != -1 else text

def remove_emojis(text):
    """
    Remove emojis from the text.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U00000200-\U00000250"
        "\U00000260-\U00002B55"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_asterisk_text(text):
    """
    Remove text between asterisks (e.g., *breathes*, *laughs*).
    """
    asterisk_pattern = re.compile(r'\*[^*]*\*')
    return asterisk_pattern.sub('', text)


def stt_task():
    """
    Enhanced task for noise-robust speech-to-text conversion with visual feedback.
    Uses WebRTC VAD and advanced noise reduction for noisy environments.
    """
    logging.debug("Enhanced noise-robust STT task start.")
    py_audio = google_api.init_pyaudio()
    speech_client = google_api.init_speech_to_text()
    logging.debug("init stt.")
    
    # Get language settings
    lang_code = os.environ.get('LANGUAGE_CODE', 'en-US')
    
    # Initialize noise-robust STT
    noise_robust_stt = NoiseRobustSTT(
        speech_client=speech_client,
        py_audio=py_audio,
        sample_rate=16000,
        chunk_size=320,  # 20ms frames for VAD
        vad_aggressiveness=2,  # 0-3: 2 = balanced between sensitivity and noise rejection
        language_code=lang_code
    )
    
    # Open audio stream
    stream = py_audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=320
    )
    
    # Calibrate noise profile once at startup with visual feedback
    print("\n" + "="*60)
    print("🎤 NOISE-ROBUST SPEECH-TO-TEXT WITH VISUAL FEEDBACK")
    print("="*60)
    noise_robust_stt.calibrate_noise(stream)
    print("✅ System ready for speech recognition!")
    print("🎙️  Using: WebRTC VAD (Voice Activity Detection)")
    print("⏱️  Auto-stops after ~0.5s of silence")
    print("🎯 Captures first word with 0.6s pre-speech buffer")
    print("🟡 Yellow: Calibrating noise profile")
    print("🔴 Red: Actively listening to speech")
    print("🟢 Green: Speech processing completed")
    print("⚪ White: Ready for next command")
    print("="*60 + "\n")
    logging.info("Calibration complete! Ready for speech recognition with visual feedback.")

    while True:
        logging.debug("stt wait for all init task has done / tts work has finished ... ...")
        should_stt = stt_queue.get()
        stt_queue.task_done()
        logging.info(f"should_stt: {should_stt}, ai_on:{ai_on}")
        while not stt_queue.empty():
            should_stt = stt_queue.get()
            logging.info(f"should_stt: {should_stt}")
            stt_queue.task_done()

        if not should_stt:
            continue
        
        logging.debug("stt task start loop, listening with noise-robust VAD...")
        
        # Use noise-robust STT with VAD and visual feedback
        user_input = noise_robust_stt.listen_once(stream)
        logging.debug(f"voice input: {user_input}")

        # Handle None input
        if not user_input:
            logging.debug(f"no input!")
            stt_queue.put(True)
            continue

        # Check for TTS interrupt commands
        global tts_interrupt_flag, tts_active
        interrupt_phrases = ["that is enough", "that's enough", "stop", "enough", "be quiet", "shut up"]
        if tts_active and any(phrase in user_input.lower() for phrase in interrupt_phrases):
            logging.info(f"🛑 TTS interrupt detected: '{user_input}'")
            tts_interrupt_flag.set()  # Signal TTS to stop
            stt_queue.put(True)
            continue

        move_key = get_move_cmd(user_input, move_cmd_functions)
        sys_cmd_key, sys_cmd_func = get_sys_cmd(user_input, sys_cmds_functions)
        global cur_voice, last_response
        if ai_on:
            lang, cur_voice = get_voice(user_input)

        if playing_heads_up:
            logging.debug(f"put voice text to input queue, heads up: {user_input}")
            input_text_queue.put(user_input)
            time.sleep(0.5)
            continue
        elif sys_cmd_key:
            logging.debug(f"sys cmd: {sys_cmd_key}")
            sys_cmd_func()
        elif "sit" == move_key or "action" == move_key:
            movement_queue.put(move_key)
            output_text_queue.put("OK, my friend.")
        elif "walk" in user_input or "come" in user_input:
            movement_queue.put("move forwards")
            output_text_queue.put("My friend, here I come.")
        elif move_key:
            movement_queue.put(move_key)
            output_text_queue.put(f"OK, my friend, {move_key} immediatly.")
        elif not ai_on:
            logging.info(f"ai is not on, do not use gemini")
            stt_queue.put(True)
            time.sleep(0.5)
            continue
        elif "heads up" in user_input and "play" in user_input:
            input_text_queue.put(user_input)
            stt_queue.put(False)
        elif ("don't want" in user_input.lower() and "play" in user_input.lower()) or \
             ("do not want" in user_input.lower() and "play" in user_input.lower()) or \
             ("exit" in user_input.lower()) or \
             ("quit" in user_input.lower()) or \
             ("stop" in user_input.lower() and ("game" in user_input.lower() or "playing" in user_input.lower())):
            input_text_queue.put(user_input)
            stt_queue.put(False)
        elif ("rock" in user_input or "paper" in user_input or "scissors" in user_input) or ("game" in user_input and "play" in user_input) or "play" in user_input:
            output_text_queue.put(GAME_TEXT)
        elif lang:
            # Check if user wants to translate the last response
            if last_response and any(phrase in user_input.lower() for phrase in ["say that", "repeat that", "translate that", "say it", "repeat it"]):
                logging.info(f"Translating last response to {lang}")
                translation_request = f"Translate this to {lang}: {last_response}"
                input_text_queue.put(translation_request)
                stt_queue.put(False)
            else:
                # Regular language switching for new question
                logging.debug(f"switch language: {lang}")
                user_input += f", Please reply in {lang}."
                input_text_queue.put(user_input)
                stt_queue.put(False)
        elif "test" in user_input:
            output_text_queue.put("test")
        else:
            logging.debug(f"put voice text to input queue: {user_input}")
            input_text_queue.put(user_input)
            stt_queue.put(False)
        
        time.sleep(0.5)


def gemini_task():
    """
    Task for handling Gemini AI interactions.
    """
    global last_response
    
    logging.debug("gemini task start.")
    history_file_path = "res/ece_history.json"
    conversation = google_api.create_conversation(history_file_path)

    init_input =  "From here on, always answer as if a human being is saying things off the top of his head which is always concise, relevant and contains a good conversational tone. so you will only and only answer in one breath responses. If the input contains a language other than English, for example, language A, please answer the question in language A."
    response = google_api.ai_text_response(conversation, init_input)
    logging.debug(f"init llm and first response: {response}")

    multi_model = ChatVertexAI(
        model_name='gemini-2.0-flash',
        convert_system_message_to_human=True,
    )
    with Image.open(f"{RES_DIR}/Trot.jpg") as image:
        logging.debug(f"Opened image: 320p")
        if image is None:
            logging.debug("No image captured!")
        else:
            text_prompt = "what is this?"
            response = google_api.ai_image_response(multi_model, image=image, text=text_prompt)
    logging.debug(f"init vision model and first response: {response}")
    stt_queue.put(True)
    show_status_image('ready')

    while True:
        logging.debug("tts wait for gemini responese text... ...")
        input_text = input_text_queue.get()
        input_text_queue.task_done()
        if not ai_on:
            continue

        logging.debug(f"user input from voice: {input_text}")
        stt_queue.put(False)
        user_input = input_text
        response = ""
        if not user_input:
            logging.debug(f"no input!")
        elif "clear history" in user_input:
            conversation.memory.clear()
        elif "photo" in user_input or "picture" in user_input or "xpression" in user_input:
            ms_start = int(time.time() * 1000)
            logging.debug(f"detect pic start!")
            image = media_api.take_photo()
            logging.debug(f"take photo finish!")

            if image:
                image = media_api.resize_image_to_width(image, 320)
                logging.debug(f"resize photo finish!")
                response = google_api.ai_image_response(multi_model, image=image, text=user_input)
                image_queue.put(image)
            else:
                response = google_api.ai_text_response(conversation, user_input)

            logging.debug(f"detect pic end!")
            ms_end = int(time.time() * 1000)
            logging.debug(f"ai_response end, delay = {ms_end - ms_start}ms")
            logging.debug("picture response end: {response}")
            output_text_queue.put(response)
        elif "rock paper scissors" in user_input:
            ms_start = int(time.time() * 1000)
            logging.debug(f"play game take photo")
            human_image = media_api.take_photo()
            logging.debug(f"play game take photo finish")

            gestures = ["rock", "paper", "scissors"]
            random.seed(int(time.time()))
            puppy_gesture = random.choice(gestures)
            logging.debug(f"puppy_gesture is: {puppy_gesture}")
            puppy_image = Image.open(f"{RES_DIR}/{puppy_gesture}.jpg")
            image_queue.put(puppy_image)

            human_gesture = google_api.ai_image_response(multi_model, image=human_image, text=user_input)
            human_gesture = human_gesture.replace(' ', '')
            logging.debug(f"human_gesture is: {human_gesture}")

            win_conditions = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
            result = "You win!" if win_conditions.get(human_gesture) == puppy_gesture else ("It's a tie!" if human_gesture == puppy_gesture else "You lose!")
            response = result
            output_text_queue.put(response)
            image = Image.open(f"{RES_DIR}/logo.png")
            image_queue.put(image)
                 
        elif "what is this" in user_input or ("what" and "holding") in user_input:
            ms_start = int(time.time() * 1000)
            logging.debug(f"identify take photo")
            input_image = media_api.take_photo()
            logging.debug(f"identify take photo finish")

            shown_object = google_api.ai_image_response(multi_model,image=input_image, text=user_input)
            logging.debug(f"shown_object is: {shown_object}")

            response = shown_object
            output_text_queue.put(response)
           
        elif "read this" in user_input:
            ms_start = int(time.time() * 1000)
            logging.debug(f"read take photo")
            input_image = media_api.take_photo()
            logging.debug(f"read take photo finish")

            shown_text = google_api.ai_image_response(multi_model,image=input_image, text=user_input)
            logging.debug(f"shown_text is: {shown_text}")

            response = shown_text
            output_text_queue.put(response)
           
        elif "play" in user_input and "heads up" in user_input:
            conversation.memory.clear()
            init_input =  "From here on, always answer as if a human being is saying things off the top of his head which is always concise, relevant and contains a good conversational tone. so you will only and only answer in one breath responses, figuratively. If the input contains a language other than English, for example, language A, please answer the question in language A."
            response = google_api.ai_text_response(conversation, init_input)
           
            ms_start = int(time.time() * 200)
            logging.debug(f"read take photo")
            input_image = media_api.take_photo()
            logging.debug(f"read take photo finish")

            shown_text = google_api.ai_image_response(multi_model,image=input_image, text="Tell me what the word on the paper is. Respond only with what is on the paper all in lowercase. Do not begin with a space. If there is no word on a card respond with 'no word' ")
            logging.debug(f"shown_text is: '{shown_text}'")
               
            heads_up_word = shown_text
            playing_heads_up = True
           
            if "no word" in heads_up_word.lower():
                playing_heads_up = False
                logging.debug("no word on heads up card, ending heads up sequence")
                output_text_queue.put("No word was provided, ending heads up sequence")
                continue
           
            conversation.memory.clear()

            heads_up_prompt = create_heads_up_prompt(heads_up_word)
            conversation_history = [
                HumanMessage(content=heads_up_prompt)
            ]
           
            response = multi_model.invoke(conversation_history)
            ai_acknowledgement = response.content
            logging.debug(f"prompt creation response: {ai_acknowledgement}")
            output_text_queue.put(ai_acknowledgement)

            conversation_history.append(AIMessage(content=ai_acknowledgement))

            guess_count = 0
           
            while playing_heads_up:
                input_text = input_text_queue.get()
                input_text_queue.task_done()
                if not ai_on:
                    continue
       
                logging.debug(f"user input from voice: {input_text}")
                stt_queue.put(False)
                user_input = input_text

                if ("don't want" in user_input.lower() and "play" in user_input.lower()) or \
                   ("do not want" in user_input.lower() and "play" in user_input.lower()) or \
                   ("exit" in user_input.lower()) or \
                   ("quit" in user_input.lower()) or \
                   ("stop" in user_input.lower() and ("game" in user_input.lower() or "playing" in user_input.lower())):
                    playing_heads_up = False
                    output_text_queue.put("Okay, exiting the heads up game. Thanks for playing!")
                    logging.debug("User requested to exit heads up game")
                    continue

                conversation_history.append(HumanMessage(content=user_input))
                response = multi_model.invoke(conversation_history)
                ai_answer = response.content
                logging.debug(f"ai answer: {ai_answer}")

                guess_count+=1
               
                conversation_history.append(AIMessage(content=ai_answer))

                if "that's it!" in ai_answer.lower() and heads_up_word.lower() in ai_answer.lower():
                    output_text_queue.put(f"Congratulations! You guessed the word: {heads_up_word}, in {guess_count} guesses!")
                    playing_heads_up = False
                else:
                    output_text_queue.put(ai_answer)
        else:
            logging.debug("text response start!")
            response = google_api.ai_text_response(conversation, user_input)
            logging.debug("text response end: {response}")
            send_response(response)
        time.sleep(0.05)


def tts_task():
    """
    Task for text-to-speech conversion and audio output with interrupt support.
    """
    global tts_interrupt_flag, tts_active
    
    logging.debug("tts task start.")
    os.system("amixer -c 0 sset 'Headphone' 100%")
    tts_client, voice, audio_config = google_api.init_text_to_speech()
    global voice0, cur_voice
    voice0 = voice
    cur_voice = voice
    logging.debug("init tts end.")
    while True:
        logging.debug("tts wait for gemini responese text... ...")
        out_text = output_text_queue.get()
        output_text_queue.task_done()
        out_text = remove_asterisk_text(remove_emojis(out_text))
        if not out_text or not ai_on:
            stt_queue.put(True)
            continue

        # Clear interrupt flag and mark TTS as active
        tts_interrupt_flag.clear()
        tts_active = True
        stt_queue.put(False)
        
        # Start TTS in a separate thread so we can monitor for interrupts
        tts_thread = threading.Thread(
            target=google_api.text_to_speech,
            args=(out_text, tts_client, cur_voice, audio_config)
        )
        tts_thread.start()
        
        # Monitor for interrupts while TTS is running
        while tts_thread.is_alive():
            if tts_interrupt_flag.is_set():
                logging.info("🛑 Interrupting TTS playback")
                # Stop the audio playback
                try:
                    os.system("killall -9 aplay 2>/dev/null")  # Stop audio playback
                except:
                    pass
                break
            time.sleep(0.05)
        
        tts_thread.join(timeout=0.1)
        tts_active = False
        
        if tts_interrupt_flag.is_set():
            logging.info("TTS interrupted by user")
            tts_interrupt_flag.clear()
        
        if GAME_TEXT == out_text:
            text = "I am playing rock paper scissors. Tell me what is this? rock paper or scissors? Only in one word, no punctuation and all in lowercase."
            input_text_queue.put(text)
        else:
            time.sleep(0.02)
            stt_queue.put(True)


def gif_task():
    """
    Task for handling GIF display.
    """
    logging.debug("gif task start.")
    gif_player = media_api.init_gifplayer(f"{RES_DIR}/")
    logging.debug("init gif end.")
    while True:
        logging.debug("wait for gif show... ...")
        should_show_gif = gif_queue.get()
        gif_queue.task_done()
        if should_show_gif:
            media_api.show_gif(gif_player)
        time.sleep(0.02)

def image_task():
    """
    Task for handling image display.
    """
    logging.debug("image task start.")
    logging.debug("init image end.")
    while True:
        logging.debug("wait for image show... ...")
        image = image_queue.get()
        image_queue.task_done()
        media_api.show_image(image)
        time.sleep(0.02)

def move_task():
    """
    Task for handling movement commands.
    """
    logging.debug("move task start.")
    logging.debug("init move end.")
    while True:
        logging.debug("wait for movement command ... ...")
        move_command = movement_queue.get()
        logging.debug(f"movement command is: {move_command}")
        movement_queue.task_done()
        if move_command in move_cmd_functions:
            move_cmd_functions[move_command]()
        else:
            logging.debug("No this command")
        time.sleep(1)

def heads_up_task():
    """
    Task for heads up
    """
    logging.debug("heads up task start.")
    while True:
        logging.debug("wait for heads up command ... ...")
        heads_up_queue.get()
       

def main():
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s',
        level=logging.DEBUG
    )
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(os.path.dirname(current_dir))
    logging.debug(f"init chdir: {current_dir}")

    from dotenv import load_dotenv
    load_dotenv(dotenv_path='./.env')
    api_path = os.environ.get('API_KEY_PATH', '')
    logging.debug(f"api key path: {api_path}")
    if os.path.exists(api_path):
        logging.debug("init credentials start.")
        google_api.init_credentials(api_path)
        logging.debug("init credentials end.")
    else:
        logging.debug("credentials file not exist.")

    lang_code = os.environ.get('LANGUAGE_CODE', 'en-US')
    lang_name = os.environ.get('LANGUAGE_NAME', 'en-US-Standard-E')
    google_api.set_language(lang_code, lang_name)

    logging.info("="*60)
    logging.info("AI APP 6 - VISUAL FEEDBACK SPEECH RECOGNITION")
    logging.info("Enhanced with visual status indicators:")
    logging.info("  🟡 Yellow: Calibrating noise profile")
    logging.info("  🔴 Red: Actively listening to speech")
    logging.info("  🟢 Green: Speech processing completed")
    logging.info("  ⚪ White: Ready for next command")
    logging.info("Optimized for noisy environments:")
    logging.info("  - TV noise, dog barking, people talking")
    logging.info("  - Lawn mowers, traffic, background music")
    logging.info("Uses: WebRTC VAD + Advanced Noise Reduction")
    logging.info("="*60)

    stt_thread = threading.Thread(target=stt_task)
    stt_thread.start()
    logging.debug("stt thread start.")

    gemini_thread = threading.Thread(target=gemini_task)
    gemini_thread.start()
    logging.debug("gemini thread start.")

    tts_thread = threading.Thread(target=tts_task)
    tts_thread.start()

    gif_thread = threading.Thread(target=gif_task)
    gif_thread.start()

    image_thread = threading.Thread(target=image_task)
    image_thread.start()

    move_thread = threading.Thread(target=move_task)
    move_thread.start()

    heads_up_thread = threading.Thread(target=heads_up_task)
    heads_up_thread.start()
   
    stt_thread.join()
    gemini_thread.join()
    tts_thread.join()
    gif_thread.join()
    image_thread.join()
    move_thread.join()
    heads_up_thread.join()


if __name__ == '__main__':
    main()