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
# Description: Enhanced AI app with immediate VAD-based barge-in and visual feedback.
# Speech onset cancels the current response before speech recognition completes.
# Includes noise-robust speech-to-text using WebRTC VAD with visual status indicators.
# Visual feedback: hello_y.png (calibration), hello_r.png (listening), hello_g.png (completed)
# When speech is detected in a foreign language, TTS automatically responds in that language.
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
import subprocess
import json
import urllib.error
import urllib.request
from collections import deque
import google.auth
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
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
GAME_TEXTS = {
    "Chinese":  "我们来玩石头剪刀布！石头！剪刀！布！出！",
    "Japanese": "じゃんけんぽん！グー！チョキ！パー！",
    "Korean":   "가위바위보！바위！가위！보！",
    "Spanish":  "¡Jugamos! ¡Piedra! ¡Papel! ¡Tijeras! ¡Ya!",
    "French":   "On joue ! Pierre ! Feuille ! Ciseaux ! Allez !",
    "German":   "Wir spielen! Stein! Schere! Papier! Los!",
}
RPS_WIN  = {"Chinese": "你赢了！", "Japanese": "あなたの勝ちです！", "Korean": "당신이 이겼습니다！",
            "Spanish": "¡Ganaste!", "French": "Vous avez gagné !", "German": "Du hast gewonnen!"}
RPS_TIE  = {"Chinese": "平局！",   "Japanese": "引き分けです！",    "Korean": "비겼습니다！",
            "Spanish": "¡Empate!", "French": "Égalité !",          "German": "Unentschieden!"}
RPS_LOSE = {"Chinese": "你输了！", "Japanese": "あなたの負けです！","Korean": "당신이 졌습니다！",
            "Spanish": "¡Perdiste!", "French": "Vous avez perdu !", "German": "Du hast verloren!"}

rps_game_lang = None  # language name of the current/last RPS game

# Keywords that trigger the RPS game in each non-English language.
# Any match fires the game; strings are checked as substrings of the transcript.
RPS_TRIGGERS = {
    "Chinese":  ["石头剪刀布", "石头", "剪刀"],
    "Japanese": ["じゃんけん", "グー", "チョキ", "パー"],
    "Korean":   ["가위바위보", "바위", "가위"],
    "Spanish":  ["piedra", "papel", "tijeras"],
    "French":   ["pierre", "feuille", "ciseaux"],
    "German":   ["stein", "schere", "papier"],
}
ai_on = True

# Barge-in settings.  MIN_SPEECH_MS validates the completed utterance; it does
# not delay the speech-start event that interrupts playback.
VAD_THRESH = float(os.environ.get("VAD_THRESH", "0.6"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "500"))
MIN_SILENCE_MS = int(os.environ.get("MIN_SILENCE_MS", "1200"))

# Response/audio generation control (the local equivalent of audioGeneration).
# A speech-start edge increments it, making older queued/in-flight work stale.
tts_interrupt_flag = threading.Event()
response_state_lock = threading.Lock()
response_generation = 0
tts_active = False
llm_active = False
echo_cancellation_enabled = False


def current_response_generation():
    with response_state_lock:
        return response_generation


def generation_is_current(generation):
    return generation == current_response_generation()


def clear_queue(work_queue):
    """Remove queued work while keeping Queue.join() accounting correct."""
    cleared = 0
    while True:
        try:
            work_queue.get_nowait()
            work_queue.task_done()
            cleared += 1
        except queue_module.Empty:
            return cleared


def handle_speech_started():
    """Handle the local equivalent of input_audio_buffer.speech_started."""
    global response_generation

    with response_state_lock:
        response_generation += 1
        generation = response_generation

    tts_interrupt_flag.set()
    try:
        sd.stop()
    except Exception as exc:
        logging.debug(f"Audio stop during barge-in failed: {exc}")

    cleared = clear_queue(output_text_queue)
    cleared_prompts = clear_queue(input_text_queue)
    logging.info(
        "input_audio_buffer.speech_started: cancelled response generation %s; "
        "cleared %s queued audio item(s) and %s stale prompt(s)",
        generation - 1,
        cleared,
        cleared_prompts,
    )
    threading.Thread(target=interrupt_livetalking, daemon=True).start()
    return generation


def queue_tts(text, generation=None):
    """Queue speech only if it belongs to the current response generation."""
    if not text:
        return False
    if generation is None:
        generation = current_response_generation()
    if not generation_is_current(generation):
        logging.info("Discarding stale TTS response from generation %s", generation)
        return False
    output_text_queue.put((generation, text))
    return True


def queue_llm(text, generation=None):
    """Queue a recognized user turn with its speech-start generation."""
    if not text:
        return False
    if generation is None:
        generation = current_response_generation()
    input_text_queue.put((generation, text))
    return True


def unpack_generation_item(item):
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
        return item
    return current_response_generation(), item


def interrupt_livetalking():
    """Tell a local/remote LiveTalking avatar to drop unsent speech."""
    base_url = os.environ.get(
        "LIVETALKING_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
    if not base_url:
        return
    session_id = os.environ.get("LIVETALKING_SESSION_ID", "0")
    try:
        session_id = int(session_id)
    except ValueError:
        pass
    payload = json.dumps({"sessionid": session_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/interrupt_talk",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            response.read()
        logging.info("LiveTalking /interrupt_talk sent for session %s", session_id)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logging.debug("LiveTalking interrupt unavailable: %s", exc)


def configure_echo_cancellation():
    """Enable PulseAudio/PipeWire's WebRTC echo-cancel source on Ubuntu."""
    if os.environ.get("ENABLE_ECHO_CANCELLATION", "1").lower() in {"0", "false", "no"}:
        logging.warning("Echo cancellation is disabled by ENABLE_ECHO_CANCELLATION")
        return False

    source_name = os.environ.get("AEC_SOURCE_NAME", "ai_app8_echo_cancel")
    sink_name = os.environ.get("AEC_SINK_NAME", "ai_app8_echo_cancel_sink")
    try:
        modules = subprocess.run(
            ["pactl", "list", "short", "modules"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        if f"source_name={source_name}" not in modules:
            command = [
                "pactl", "load-module", "module-echo-cancel",
                "aec_method=webrtc",
                f"source_name={source_name}",
                f"sink_name={sink_name}",
                "source_properties=device.description=AI_App8_Echo_Cancel",
                "sink_properties=device.description=AI_App8_Echo_Cancel_Sink",
            ]
            master_source = os.environ.get("AEC_MASTER_SOURCE")
            master_sink = os.environ.get("AEC_MASTER_SINK")
            if master_source:
                command.append(f"source_master={master_source}")
            if master_sink:
                command.append(f"sink_master={master_sink}")
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)

        # PortAudio's Pulse device honors these per-process routing variables.
        os.environ["PULSE_SOURCE"] = source_name
        os.environ["PULSE_SINK"] = sink_name
        logging.info("WebRTC acoustic echo cancellation enabled: %s / %s", source_name, sink_name)
        return True
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logging.warning(
            "Could not enable PulseAudio/PipeWire echo cancellation (%s). "
            "Barge-in remains enabled; install pactl and module-echo-cancel or set "
            "ENABLE_ECHO_CANCELLATION=0 to acknowledge the fallback.",
            exc,
        )
        return False


def choose_input_device(py_audio, prefer_pulse):
    """Resolve an optional input device index/name, preferring Pulse for AEC."""
    requested = os.environ.get("MIC_INPUT_DEVICE", "").strip()
    if requested.isdigit():
        return int(requested)

    search = requested.lower()
    fallback = None
    for index in range(py_audio.get_device_count()):
        info = py_audio.get_device_info_by_index(index)
        if int(info.get("maxInputChannels", 0)) < 1:
            continue
        name = str(info.get("name", "")).lower()
        if search and search in name:
            return index
        if prefer_pulse and ("pulse" in name or "pipewire" in name):
            fallback = index

    if requested:
        raise RuntimeError(f"MIC_INPUT_DEVICE did not match an input device: {requested}")
    return fallback


def choose_output_device(prefer_pulse):
    """Choose the Pulse/PipeWire output so its AEC sink receives TTS audio."""
    requested = os.environ.get("TTS_OUTPUT_DEVICE", "").strip()
    devices = sd.query_devices()
    fallback = None
    for index, info in enumerate(devices):
        if int(info.get("max_output_channels", 0)) < 1:
            continue
        name = str(info.get("name", "")).lower()
        if requested and (requested == str(index) or requested.lower() in name):
            return index
        if prefer_pulse and ("pulse" in name or "pipewire" in name):
            fallback = index
        elif not prefer_pulse and fallback is None and "headphone" in name:
            fallback = index

    if requested:
        raise RuntimeError(f"TTS_OUTPUT_DEVICE did not match an output device: {requested}")
    return fallback

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
voice_KR = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A")

lang_voices = {
    "Japanese": voice_JP,
    "Chinese": voice_CN,
    "Italian": voice_IT,
    "German": voice_DE,
    "French": voice_FR,
    "Cantonese": voice_HK,
    "Spanish": voice_ES,
    "Hebrew": voice_IL,
    "Korean": voice_KR,
}

# Mapping from Google STT language codes to lang_voices keys for auto-detection.
# Only languages present in lang_voices are listed; all others fall back to default.
lang_code_to_name = {
    "ja-JP":   "Japanese",
    "cmn-CN":  "Chinese",
    "cmn-Hans-CN": "Chinese",
    "cmn-Hant-TW": "Chinese",
    "zh-CN":   "Chinese",
    "zh-TW":   "Chinese",
    "it-IT":   "Italian",
    "de-DE":   "German",
    "fr-FR":   "French",
    "yue-HK":  "Cantonese",
    "yue-Hant-HK": "Cantonese",
    "es-US":   "Spanish",
    "es-ES":   "Spanish",
    "he-IL":   "Hebrew",
    "ko-KR":   "Korean",
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
                 vad_aggressiveness=1, language_code="en-US",
                 vad_threshold=VAD_THRESH, min_speech_ms=MIN_SPEECH_MS,
                 min_silence_ms=MIN_SILENCE_MS, on_speech_started=None,
                 project_id=None, location="us", recognizer="_",
                 language_codes=None, final_timeout=8.0):
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
        self.project_id = project_id
        self.location = location
        self.recognizer = recognizer
        self.language_codes = language_codes or ["auto"]
        self.final_timeout = final_timeout
        self.vad_threshold = min(1.0, max(0.0, vad_threshold))
        self.min_speech_ms = max(20, min_speech_ms)
        self.min_silence_ms = max(20, min_silence_ms)
        self.on_speech_started = on_speech_started

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.vad_frame_duration_ms = 20
        self.vad_window_size = 5
        self.vad_decisions = deque([0.0] * self.vad_window_size,
                                   maxlen=self.vad_window_size)
        self.last_vad_score = 0.0

        # VAD parameters for speech detection
        self.num_silent_frames_threshold = max(
            1, int(np.ceil(self.min_silence_ms / self.vad_frame_duration_ms)))
        self.num_speech_frames_threshold = 2  # ~40ms to confirm speech

                # Noise reduction settings (optimized for complex noise)
        self.noise_profile = None
        self.calibration_time = 2.0
        self.noise_reduction_strength = 0.5  # Moderate - preserve speech clarity
        self.use_stationary_noise = False
        self.silence_threshold = 500
        self.apply_noise_reduction = False  # Disabled by default - can cause audio to be too quiet
        self.enable_noise_reduction = False  # Disabled - VAD + raw audio works better

        # Speech detection state
        self.ring_buffer = []
        self.ring_buffer_size = 30  # Keep 30 frames (~0.6 seconds) before speech
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False

        self.recognizer_path = (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"recognizers/{self.recognizer}"
        )
        decoding_config = cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=self.channels,
        )
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=decoding_config,
            language_codes=self.language_codes,
            model="chirp_3",
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )

        logging.info(
            "Noise-Robust STT initialized (VAD threshold: %.2f, min speech: %sms, "
            "min silence: %sms, WebRTC aggressiveness: %s, model: chirp_3, "
            "languages: %s)",
            self.vad_threshold,
            self.min_speech_ms,
            self.min_silence_ms,
            vad_aggressiveness,
            ",".join(self.language_codes),
        )

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

        # Threshold = 2× noise floor, capped at 1000 so startup noise can't
        # inflate it above normal speech levels (typical speech RMS > 1000).
        noise_rms = np.sqrt(np.mean(self.noise_profile.astype(np.float32) ** 2))
        self.silence_threshold = min(1000, max(200, noise_rms * 2.0))

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

    _VAD_RATE = 16000  # WebRTC VAD works reliably at 16 kHz

    def _resample_to_vad_rate(self, audio_data):
        """Resample audio_data to _VAD_RATE using simple decimation (no scipy needed)."""
        if self.sample_rate == self._VAD_RATE:
            return audio_data
        from math import gcd
        g = gcd(int(self.sample_rate), self._VAD_RATE)
        up = self._VAD_RATE // g
        down = int(self.sample_rate) // g
        # Upsample then downsample via integer index to avoid scipy dependency
        n_out = int(len(audio_data) * up / down)
        indices = (np.arange(n_out) * down / up).astype(int)
        indices = np.clip(indices, 0, len(audio_data) - 1)
        return audio_data[indices]

    def is_speech_vad(self, audio_data):
        """
        Use WebRTC VAD to detect if audio chunk contains speech.
        Audio is resampled to 16 kHz so the VAD always receives a supported rate.
        VAD_THRESH is applied to a rolling probability made from the last five
        WebRTC decisions, so lowering it makes speech onset easier to trigger.
        """
        try:
            resampled = self._resample_to_vad_rate(audio_data)
            expected_size = int(self._VAD_RATE * self.vad_frame_duration_ms / 1000)
            if len(resampled) < expected_size:
                resampled = np.pad(resampled, (0, expected_size - len(resampled)), 'constant')
            else:
                resampled = resampled[:expected_size]
            raw_speech = self.vad.is_speech(resampled.tobytes(), self._VAD_RATE)

        except Exception as e:
            # Amplitude fallback — use a fixed low threshold so normal speech is detected
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            raw_speech = rms >= self.silence_threshold
            logging.warning(f"VAD exception ({e}), amplitude fallback: RMS={rms:.0f} threshold={self.silence_threshold:.0f} speech={raw_speech}")

        self.vad_decisions.append(1.0 if raw_speech else 0.0)
        self.last_vad_score = sum(self.vad_decisions) / len(self.vad_decisions)
        return self.last_vad_score >= self.vad_threshold

    def streaming_requests(self, audio_queue):
        """Yield the Chirp 3 config first, followed by raw microphone chunks."""
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.recognizer_path,
            streaming_config=self.streaming_config,
        )
        while True:
            audio_bytes = audio_queue.get()
            if audio_bytes is None:
                return
            # Speech-to-Text V2 limits each streaming audio request to 15 KB.
            for offset in range(0, len(audio_bytes), 15000):
                yield cloud_speech.StreamingRecognizeRequest(
                    audio=audio_bytes[offset:offset + 15000]
                )

    def transcribe_stream(self, audio_queue, result_queue):
        """Consume one Chirp 3 stream and return its final transcript."""
        final_parts = []
        latest_interim = ""
        detected_language = self.language_code
        confidence = 0.0
        try:
            responses = self.speech_client.streaming_recognize(
                requests=self.streaming_requests(audio_queue)
            )
            for response in responses:
                interim_parts = []
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript.strip()
                    if not transcript:
                        continue
                    if result.language_code:
                        detected_language = result.language_code
                    if result.is_final:
                        final_parts.append(transcript)
                        confidence = result.alternatives[0].confidence
                    else:
                        interim_parts.append(transcript)
                if interim_parts:
                    latest_interim = " ".join(interim_parts)
                    logging.debug("Chirp 3 interim transcript: %s", latest_interim)

            transcript = " ".join(final_parts).strip() or latest_interim.strip()
            result_queue.put((transcript or None, detected_language, confidence, None))
        except Exception as exc:
            result_queue.put((None, None, 0.0, exc))

    def finish_transcription(self, audio_queue, worker, result_queue, wait=True):
        """Close a microphone stream and optionally wait for Chirp's final result."""
        audio_queue.put(None)
        if not wait:
            return None, None

        worker.join(timeout=self.final_timeout)
        if worker.is_alive():
            logging.error(
                "Timed out after %.1fs waiting for the Chirp 3 final transcript",
                self.final_timeout,
            )
            return None, None
        try:
            transcript, detected_language, confidence, error = result_queue.get_nowait()
        except queue_module.Empty:
            logging.error("Chirp 3 stream ended without a transcription result")
            return None, None
        if error:
            logging.error("Chirp 3 streaming transcription error: %s", error)
            return None, None
        if not transcript:
            logging.warning("Chirp 3 returned no recognizable speech")
            return None, None

        logging.info(
            "Chirp 3 transcription: '%s' (confidence: %.2f, detected lang: %s)",
            transcript,
            confidence,
            detected_language,
        )
        return transcript, detected_language

    def listen_once(self, stream):
        """
        Listen for one complete speech utterance using VAD with visual feedback.
        Returns a tuple of (transcribed_text, detected_language_code), or (None, None).
        """
        # Reset state
        self.ring_buffer = []
        self.consecutive_silent_frames = 0
        self.consecutive_speech_frames = 0
        self.is_currently_speaking = False
        self.vad_decisions = deque([0.0] * self.vad_window_size,
                                   maxlen=self.vad_window_size)
        self.last_vad_score = 0.0
        speech_frame_count = 0
        speech_detected = False
        audio_queue = None
        result_queue = None
        transcription_worker = None
        transcription_closed = False

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

                # Periodic diagnostic log every ~5 seconds
                frames_per_5s = int(5 * self.sample_rate / self.chunk_size)
                if iteration > 0 and iteration % frames_per_5s == 0:
                    rms = np.sqrt(np.mean(cleaned_audio.astype(np.float32) ** 2))
                    logging.info(f"Listening... RMS={rms:.0f} threshold={self.silence_threshold:.0f} vad={is_speech} speaking={self.is_currently_speaking}")

                if is_speech:
                    self.consecutive_speech_frames += 1
                    self.consecutive_silent_frames = 0

                    # Confirm speech start after enough consecutive speech frames
                    if not self.is_currently_speaking and self.consecutive_speech_frames >= self.num_speech_frames_threshold:
                        self.is_currently_speaking = True
                        if not speech_detected:
                            ring_buffer_duration = len(self.ring_buffer) * 0.02
                            logging.info(f"🎤 Speech detected! Streaming to Chirp 3... (captured {ring_buffer_duration:.2f}s pre-speech buffer)")
                            speech_detected = True

                            # Fire barge-in immediately at the VAD speech-start edge.
                            # Chirp transcription starts now and runs while the
                            # user is speaking; VAD still owns turn boundaries.
                            if self.on_speech_started:
                                self.on_speech_started()

                            # Show listening status image
                            show_status_image('listening')

                            audio_queue = queue_module.Queue()
                            result_queue = queue_module.Queue(maxsize=1)
                            transcription_worker = threading.Thread(
                                target=self.transcribe_stream,
                                args=(audio_queue, result_queue),
                                daemon=True,
                            )
                            transcription_worker.start()

                            # Send the pre-speech buffer first to preserve the
                            # beginning of the user's first word.
                            if self.ring_buffer:
                                logging.debug(f"Streaming {len(self.ring_buffer)} ring buffer frames")
                                for buffered_audio in self.ring_buffer:
                                    audio_queue.put(buffered_audio.tobytes())
                                self.ring_buffer = []

                    if self.is_currently_speaking:
                        speech_frame_count += 1
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

                    if not self.is_currently_speaking:
                        # Maintain ring buffer
                        self.ring_buffer.append(cleaned_audio)
                        if len(self.ring_buffer) > self.ring_buffer_size:
                            self.ring_buffer.pop(0)

                # Once speech starts, stream every frame—including trailing
                # silence—while local VAD independently finds the turn end.
                if audio_queue is not None:
                    audio_queue.put(cleaned_audio.tobytes())

                # Check if speech ended
                if self.is_currently_speaking and self.consecutive_silent_frames >= self.num_silent_frames_threshold:
                    logging.info(f"Speech ended after {self.consecutive_silent_frames} silent frames (~{self.consecutive_silent_frames * 0.02:.1f}s). Processing...")

                    # Show completion status image
                    show_status_image('completed')

                    speech_duration_ms = speech_frame_count * self.vad_frame_duration_ms
                    if speech_duration_ms < self.min_speech_ms:
                        logging.warning(
                            "Speech too short (%sms, need %sms) - ignoring transcription",
                            speech_duration_ms,
                            self.min_speech_ms,
                        )
                        self.finish_transcription(
                            audio_queue, transcription_worker, result_queue, wait=False)
                        transcription_closed = True
                        show_status_image('ready')
                        return None, None

                    transcription_closed = True
                    transcript, detected_lang = self.finish_transcription(
                        audio_queue, transcription_worker, result_queue)
                    show_status_image('ready')
                    return transcript, detected_lang

            except Exception as e:
                logging.error(f"Error during listening: {e}")
                if audio_queue is not None and not transcription_closed:
                    self.finish_transcription(
                        audio_queue, transcription_worker, result_queue, wait=False)
                show_status_image('ready')  # Return to ready state on error
                return None, None

        # Timeout - no speech detected
        logging.debug("Listening timeout - no speech detected")
        if audio_queue is not None and not transcription_closed:
            self.finish_transcription(
                audio_queue, transcription_worker, result_queue, wait=False)
        show_status_image('ready')  # Return to ready state on timeout
        return None, None


# Track last response for translation
last_response = ""

def send_response(text, generation=None):
    """
    Send response to output queue and track it for potential translation.
    """
    global last_response
    last_response = text
    queue_tts(text, generation)


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

def detect_voice_from_transcript(transcript, stt_lang_code=None):
    """
    Detect the appropriate TTS voice from lang_voices based on the STT-reported
    language code and Unicode script ranges (reliable for CJK/Hebrew scripts).
    Returns the matching voice from lang_voices, or None to use the default voice.
    """
    def _any_in_range(text, lo, hi):
        return any(lo <= ord(c) <= hi for c in text)

    # STT lang code is the most reliable signal — check it first.
    if stt_lang_code:
        lang_name = lang_code_to_name.get(stt_lang_code)
        if lang_name:
            voice = lang_voices.get(lang_name)
            if voice:
                logging.info(f"Detected language '{lang_name}' via STT code '{stt_lang_code}'")
                return voice

    # Fallback: Unicode script detection for transcripts where the STT code
    # was missing or didn't match (e.g. mixed-code responses).
    if transcript:
        has_hiragana = _any_in_range(transcript, 0x3040, 0x309F)
        has_katakana = _any_in_range(transcript, 0x30A0, 0x30FF)
        has_cjk      = (_any_in_range(transcript, 0x4E00, 0x9FFF) or
                        _any_in_range(transcript, 0x3400, 0x4DBF))
        has_hebrew   = _any_in_range(transcript, 0x05D0, 0x05EA)

        if has_hiragana or has_katakana:
            logging.info("Detected Japanese script in transcript")
            return lang_voices.get("Japanese")
        if has_cjk:
            logging.info("Detected Chinese script in transcript")
            return lang_voices.get("Chinese")
        if has_hebrew:
            logging.info("Detected Hebrew script in transcript")
            return lang_voices.get("Hebrew")

        has_hangul = _any_in_range(transcript, 0xAC00, 0xD7AF)
        if has_hangul:
            logging.info("Detected Korean script in transcript")
            return lang_voices.get("Korean")

    return None  # Language not in lang_voices — caller uses default voice

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
    queue_tts("OK, my friend.")

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
    Automatically switches TTS voice to match foreign language detected in speech.
    """
    logging.debug("Enhanced noise-robust STT task start.")
    py_audio = google_api.init_pyaudio()
    credentials, detected_project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or detected_project_id
    if not project_id:
        raise RuntimeError(
            "Chirp 3 requires GOOGLE_CLOUD_PROJECT or a project_id in Google credentials"
        )
    stt_location = os.environ.get("GOOGLE_STT_LOCATION", "us")
    speech_client = SpeechClient(
        credentials=credentials,
        client_options=ClientOptions(
            api_endpoint=f"{stt_location}-speech.googleapis.com",
        ),
    )
    stt_languages = [
        code.strip()
        for code in os.environ.get("GOOGLE_STT_LANGUAGE_CODES", "auto").split(",")
        if code.strip()
    ]
    logging.info(
        "Initialized Google STT V2 Chirp 3 (project=%s, location=%s, languages=%s)",
        project_id,
        stt_location,
        ",".join(stt_languages),
    )

    # Get language settings
    lang_code = os.environ.get('LANGUAGE_CODE', 'en-US')

    # Detect the hardware's native sample rate to avoid paInvalidSampleRate
    input_device_index = choose_input_device(py_audio, echo_cancellation_enabled)
    if input_device_index is None:
        device_info = py_audio.get_default_input_device_info()
    else:
        device_info = py_audio.get_device_info_by_index(input_device_index)
    logging.info("Microphone input device: %s", device_info.get("name", input_device_index))
    native_rate = int(device_info['defaultSampleRate'])  # e.g. 48000 on Google Voice HAT
    # VAD requires exactly 10/20/30 ms frames; 20 ms at native_rate samples
    native_chunk = int(native_rate * 0.020)

    # Initialize noise-robust STT
    noise_robust_stt = NoiseRobustSTT(
        speech_client=speech_client,
        py_audio=py_audio,
        sample_rate=native_rate,
        chunk_size=native_chunk,
        vad_aggressiveness=0,  # 0-3: 0 = most sensitive to speech
        language_code=lang_code,
        vad_threshold=VAD_THRESH,
        min_speech_ms=MIN_SPEECH_MS,
        min_silence_ms=MIN_SILENCE_MS,
        on_speech_started=handle_speech_started,
        project_id=project_id,
        location=stt_location,
        recognizer=os.environ.get("GOOGLE_STT_RECOGNIZER", "_"),
        language_codes=stt_languages,
        final_timeout=float(os.environ.get("GOOGLE_STT_FINAL_TIMEOUT", "8")),
    )

    # Open audio stream
    stream = py_audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=native_rate,
        input=True,
        frames_per_buffer=native_chunk,
        input_device_index=input_device_index,
    )

    # Calibrate noise profile once at startup with visual feedback
    print("\n" + "="*60)
    print("🎤 NOISE-ROBUST SPEECH-TO-TEXT WITH VISUAL FEEDBACK")
    print("="*60)
    noise_robust_stt.calibrate_noise(stream)
    print("✅ System ready for speech recognition!")
    print("🎙️  Using: WebRTC VAD (Voice Activity Detection)")
    print("☁️  STT: Google Speech-to-Text V2 streaming with Chirp 3")
    print(f"⏱️  Auto-stops after {MIN_SILENCE_MS / 1000:.1f}s of silence")
    print(f"🛑 VAD barge-in threshold: {VAD_THRESH:.1f} (before transcription)")
    print(f"🗣️  Minimum valid speech: {MIN_SPEECH_MS}ms")
    print("🎯 Captures first word with 0.6s pre-speech buffer")
    print("🟡 Yellow: Calibrating noise profile")
    print("🔴 Red: Actively listening to speech")
    print("🟢 Green: Speech processing completed")
    print("⚪ White: Ready for next command")
    print("="*60 + "\n")
    logging.info("Calibration complete! Ready for speech recognition with visual feedback.")

    while True:
        # Listening stays active during both LLM generation and TTS playback.
        # Legacy queue signals are drained for compatibility with the command
        # handlers, but they no longer gate microphone capture.
        while not stt_queue.empty():
            stt_queue.get()
            stt_queue.task_done()

        logging.debug("stt task start loop, listening with noise-robust VAD...")

        # Use noise-robust STT with VAD and visual feedback; returns (transcript, lang_code)
        user_input, detected_lang_code = noise_robust_stt.listen_once(stream)
        logging.debug(f"voice input: {user_input}, detected lang: {detected_lang_code}")

        # Handle None input
        if not user_input:
            logging.debug(f"no input!")
            continue

        move_key = get_move_cmd(user_input, move_cmd_functions)
        sys_cmd_key, sys_cmd_func = get_sys_cmd(user_input, sys_cmds_functions)
        global cur_voice, last_response
        if ai_on:
            lang, cur_voice = get_voice(user_input)
            # Auto-detect foreign language from speech when no explicit language was requested.
            # Uses Unicode script ranges (reliable for CJK) + STT lang code (Latin scripts).
            if not lang:
                auto_voice = detect_voice_from_transcript(user_input, detected_lang_code)
                if auto_voice:
                    logging.info(f"Auto-detected foreign language voice from transcript, switching TTS voice")
                    cur_voice = auto_voice
                else:
                    cur_voice = voice0

        if playing_heads_up:
            logging.debug(f"put voice text to input queue, heads up: {user_input}")
            queue_llm(user_input)
            time.sleep(0.5)
            continue
        elif sys_cmd_key:
            logging.debug(f"sys cmd: {sys_cmd_key}")
            sys_cmd_func()
        elif "sit" == move_key or "action" == move_key:
            movement_queue.put(move_key)
            queue_tts("OK, my friend.")
        elif "walk" in user_input or "come" in user_input:
            movement_queue.put("move forwards")
            queue_tts("My friend, here I come.")
        elif move_key:
            movement_queue.put(move_key)
            queue_tts(f"OK, my friend, {move_key} immediatly.")
        elif not ai_on:
            logging.info(f"ai is not on, do not use gemini")
            stt_queue.put(True)
            time.sleep(0.5)
            continue
        elif ("heads up" in user_input and "play" in user_input) or \
             ("玩" in user_input and ("猜词" in user_input or "举牌" in user_input or "抬头" in user_input)):
            queue_llm(user_input)
            stt_queue.put(False)
        elif ("don't want" in user_input.lower() and "play" in user_input.lower()) or \
             ("do not want" in user_input.lower() and "play" in user_input.lower()) or \
             ("exit" in user_input.lower()) or \
             ("quit" in user_input.lower()) or \
             ("stop" in user_input.lower() and ("game" in user_input.lower() or "playing" in user_input.lower())) or \
             "退出" in user_input or "不玩了" in user_input or \
             ("停止" in user_input and "游戏" in user_input) or "结束游戏" in user_input:
            queue_llm(user_input)
            stt_queue.put(False)
        elif ("rock" in user_input or "paper" in user_input or "scissors" in user_input) or \
             ("game" in user_input and "play" in user_input) or "play" in user_input or \
             any(kw in user_input for kws in RPS_TRIGGERS.values() for kw in kws):
            global rps_game_lang
            rps_game_lang = next((n for n, v in lang_voices.items() if v == cur_voice), None)
            queue_tts(GAME_TEXTS.get(rps_game_lang, GAME_TEXT))
        elif lang:
            # Check if user wants to translate the last response
            if last_response and any(phrase in user_input.lower() for phrase in ["say that", "repeat that", "translate that", "say it", "repeat it"]):
                logging.info(f"Translating last response to {lang}")
                translation_request = f"Translate this to {lang}: {last_response}"
                queue_llm(translation_request)
                stt_queue.put(False)
            else:
                # Regular language switching for new question
                logging.debug(f"switch language: {lang}")
                user_input += f", Please reply in {lang}."
                queue_llm(user_input)
                stt_queue.put(False)
        elif "test" in user_input:
            queue_tts("test")
        else:
            logging.debug(f"put voice text to input queue: {user_input}")
            queue_llm(user_input)
            stt_queue.put(False)

        time.sleep(0.5)


def chunk_text(chunk):
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def cancellable_chat_response(model, messages, generation):
    """Stream a model call so a VAD generation change can close it early."""
    global llm_active

    if not generation_is_current(generation):
        return None
    stream = None
    parts = []
    llm_active = True
    try:
        stream = model.stream(messages)
        for chunk in stream:
            if not generation_is_current(generation):
                logging.info("Cancelling in-progress LLM stream for generation %s", generation)
                return None
            parts.append(chunk_text(chunk))
    finally:
        llm_active = False
        close = getattr(stream, "close", None)
        if close:
            close()

    if not generation_is_current(generation):
        return None
    return "".join(parts).strip()


def cancellable_ai_text_response(conversation, input_text, generation):
    """Run the existing ConversationChain prompt/memory through a cancellable stream."""
    inputs = {"input": input_text}
    memory_values = conversation.memory.load_memory_variables(inputs)
    messages = conversation.prompt.format_messages(**inputs, **memory_values)
    result = cancellable_chat_response(conversation.llm, messages, generation)
    if result is not None and generation_is_current(generation):
        conversation.memory.save_context(inputs, {"response": result})
        return result
    return None


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
        model_name=os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
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
        queued_input = input_text_queue.get()
        input_text_queue.task_done()
        request_generation, input_text = unpack_generation_item(queued_input)
        if not generation_is_current(request_generation):
            logging.info("Discarding stale prompt from generation %s", request_generation)
            continue
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
            queue_tts(response, request_generation)
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
            if win_conditions.get(human_gesture) == puppy_gesture:
                result = RPS_WIN.get(rps_game_lang, "You win!")
            elif human_gesture == puppy_gesture:
                result = RPS_TIE.get(rps_game_lang, "It's a tie!")
            else:
                result = RPS_LOSE.get(rps_game_lang, "You lose!")
            queue_tts(result, request_generation)
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
            queue_tts(response, request_generation)

        elif "read this" in user_input:
            ms_start = int(time.time() * 1000)
            logging.debug(f"read take photo")
            input_image = media_api.take_photo()
            logging.debug(f"read take photo finish")

            shown_text = google_api.ai_image_response(multi_model,image=input_image, text=user_input)
            logging.debug(f"shown_text is: {shown_text}")

            response = shown_text
            queue_tts(response, request_generation)

        elif ("play" in user_input and "heads up" in user_input) or \
             ("玩" in user_input and ("猜词" in user_input or "举牌" in user_input or "抬头" in user_input)):
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
                queue_tts("No word was provided, ending heads up sequence", request_generation)
                continue

            conversation.memory.clear()

            heads_up_prompt = create_heads_up_prompt(heads_up_word)
            conversation_history = [
                HumanMessage(content=heads_up_prompt)
            ]

            ai_acknowledgement = cancellable_chat_response(
                multi_model, conversation_history, request_generation)
            if ai_acknowledgement is None:
                continue
            logging.debug(f"prompt creation response: {ai_acknowledgement}")
            queue_tts(ai_acknowledgement, request_generation)

            conversation_history.append(AIMessage(content=ai_acknowledgement))

            guess_count = 0

            while playing_heads_up:
                queued_input = input_text_queue.get()
                input_text_queue.task_done()
                request_generation, input_text = unpack_generation_item(queued_input)
                if not generation_is_current(request_generation):
                    continue
                if not ai_on:
                    continue

                logging.debug(f"user input from voice: {input_text}")
                stt_queue.put(False)
                user_input = input_text

                if ("don't want" in user_input.lower() and "play" in user_input.lower()) or \
                   ("do not want" in user_input.lower() and "play" in user_input.lower()) or \
                   ("exit" in user_input.lower()) or \
                   ("quit" in user_input.lower()) or \
                   ("stop" in user_input.lower() and ("game" in user_input.lower() or "playing" in user_input.lower())) or \
                   "退出" in user_input or "不玩了" in user_input or \
                   ("停止" in user_input and "游戏" in user_input) or "结束游戏" in user_input:
                    playing_heads_up = False
                    queue_tts("Okay, exiting the heads up game. Thanks for playing!", request_generation)
                    logging.debug("User requested to exit heads up game")
                    continue

                conversation_history.append(HumanMessage(content=user_input))
                ai_answer = cancellable_chat_response(
                    multi_model, conversation_history, request_generation)
                if ai_answer is None:
                    continue
                logging.debug(f"ai answer: {ai_answer}")

                guess_count+=1

                conversation_history.append(AIMessage(content=ai_answer))

                if "that's it!" in ai_answer.lower() and heads_up_word.lower() in ai_answer.lower():
                    queue_tts(
                        f"Congratulations! You guessed the word: {heads_up_word}, in {guess_count} guesses!",
                        request_generation,
                    )
                    playing_heads_up = False
                else:
                    queue_tts(ai_answer, request_generation)
        else:
            logging.debug("text response start!")
            response = cancellable_ai_text_response(
                conversation, user_input, request_generation)
            logging.debug(f"text response end: {response}")
            if response is not None:
                send_response(response, request_generation)
        time.sleep(0.05)


def tts_task():
    """
    Synthesize and play generation-tagged audio with immediate VAD cancellation.
    """
    global tts_interrupt_flag, tts_active

    logging.debug("tts task start.")
    amixer_control = os.environ.get('AMIXER_CONTROL', 'PCM')
    os.system(f"amixer -c 0 sset '{amixer_control}' 100%")

    tts_client, voice, audio_config = google_api.init_text_to_speech()
    global voice0, cur_voice
    voice0 = voice
    cur_voice = voice
    output_device = choose_output_device(echo_cancellation_enabled)
    if output_device is not None:
        logging.info("TTS output device: %s", sd.query_devices(output_device).get("name"))
    logging.debug("init tts end.")
    while True:
        logging.debug("tts wait for Gemini response text... ...")
        queued_output = output_text_queue.get()
        output_text_queue.task_done()
        generation, out_text = unpack_generation_item(queued_output)
        if not generation_is_current(generation):
            logging.info("Discarding stale queued audio from generation %s", generation)
            continue
        out_text = remove_asterisk_text(remove_emojis(out_text))
        if not out_text or not ai_on:
            continue

        tts_interrupt_flag.clear()
        tts_active = True
        interrupted = False
        try:
            synthesis_input = texttospeech.SynthesisInput(text=out_text)
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=cur_voice,
                audio_config=audio_config,
            )

            # A barge-in may arrive while the cloud TTS call is outstanding.
            # Never let that completed-but-stale request start playback.
            if not generation_is_current(generation) or tts_interrupt_flag.is_set():
                interrupted = True
                continue

            try:
                audio_data, sample_rate = sf.read(
                    BytesIO(response.audio_content), dtype="int16", always_2d=False)
            except Exception:
                audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
                sample_rate = 24000

            sd.play(audio_data, sample_rate, device=output_device, blocking=False)
            while True:
                if not generation_is_current(generation) or tts_interrupt_flag.is_set():
                    interrupted = True
                    sd.stop()
                    break
                try:
                    if not sd.get_stream().active:
                        break
                except Exception:
                    break
                time.sleep(0.02)
        except Exception as exc:
            logging.error("TTS synthesis/playback error: %s", exc)
        finally:
            tts_active = False
            if interrupted:
                logging.info("TTS generation %s interrupted by VAD", generation)
            tts_interrupt_flag.clear()

        if (not interrupted and generation_is_current(generation)
                and out_text in ({GAME_TEXT} | set(GAME_TEXTS.values()))):
            text = "I am playing rock paper scissors. Tell me what is this? rock paper or scissors? Only in one word, no punctuation and all in lowercase."
            queue_llm(text, generation)


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
    global VAD_THRESH, MIN_SPEECH_MS, MIN_SILENCE_MS, echo_cancellation_enabled

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
    VAD_THRESH = float(os.environ.get("VAD_THRESH", "0.6"))
    MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "500"))
    MIN_SILENCE_MS = int(os.environ.get("MIN_SILENCE_MS", "1200"))
    echo_cancellation_enabled = configure_echo_cancellation()
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
    logging.info("AI APP 8 - VAD-BASED BARGE-IN")
    logging.info("  - VAD threshold: %.2f", VAD_THRESH)
    logging.info("  - Minimum speech: %sms", MIN_SPEECH_MS)
    logging.info("  - End-of-turn silence: %sms", MIN_SILENCE_MS)
    logging.info("  - Echo cancellation: %s", "enabled" if echo_cancellation_enabled else "fallback")
    logging.info("  - Speech-start cancels LLM/TTS and invalidates queued audio")
    logging.info("Enhanced with visual status indicators:")
    logging.info("  🟡 Yellow: Calibrating noise profile")
    logging.info("  🔴 Red: Actively listening to speech")
    logging.info("  🟢 Green: Speech processing completed")
    logging.info("  ⚪ White: Ready for next command")
    logging.info("Optimized for noisy environments:")
    logging.info("  - TV noise, dog barking, people talking")
    logging.info("  - Lawn mowers, traffic, background music")
    logging.info("Auto foreign language TTS:")
    logging.info("  - Detects language of incoming speech via Google STT")
    logging.info("  - Switches TTS voice to match detected foreign language")
    logging.info("Uses: continuous WebRTC VAD + streaming Chirp 3 + generation-based cancellation")
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
