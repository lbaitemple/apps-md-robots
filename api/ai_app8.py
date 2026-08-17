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
# Description: Full-duplex multilingual assistant for Raspberry Pi.
# PipeWire/PulseAudio's WebRTC AEC/NS removes this application's exact playback
# signal before local WebRTC VAD decides whether a human is speaking. A new
# utterance cancels stale TTS/LLM work after STT rejects residual playback echo.
#

import logging
import os
import time
import re
import atexit
import signal
import subprocess
from difflib import SequenceMatcher
import numpy as np
from scipy.signal import resample_poly
from PIL import Image
import pyaudio
import sounddevice as sd
import soundfile as sf
from io import BytesIO
import threading
import queue as queue_module
from collections import deque
from dataclasses import dataclass
from google.cloud import texttospeech
from google.cloud import translate_v3
from google.api_core.client_options import ClientOptions
import google.auth
from langchain_google_vertexai import ChatVertexAI
import random
from langchain.schema import HumanMessage, AIMessage
import webrtcvad


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_queue import input_text_queue, output_text_queue, gif_queue, image_queue, movement_queue, heads_up_queue
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
RPS_WIN_CONDITIONS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
RPS_GESTURES = ("rock", "paper", "scissors")
RPS_MOVE_ALIASES = {
    "rock": {"stone", "fist", "wrong", "roc", "rack"},
    "paper": {"cloth", "fabric", "sheet", "leaf", "pepper", "papers"},
    "scissors": {"scissor", "shears", "sizzors"},
}
RPS_INVALID_MOVE = {
    "Chinese": "请说石头、剪刀或布。",
    "Japanese": "グー、チョキ、パーのどれかを言ってください。",
    "Korean": "바위, 가위, 보 중 하나를 말해 주세요.",
    "Spanish": "Di piedra, papel o tijeras.",
    "French": "Dites pierre, feuille ou ciseaux.",
    "German": "Sag Stein, Papier oder Schere.",
}
RPS_UNCLEAR_GESTURE = {
    "Chinese": "我没有看清你的手势，请再试一次。",
    "Japanese": "手の形が見えませんでした。もう一度試してください。",
    "Korean": "손동작을 확인하지 못했어요. 다시 시도해 주세요.",
    "Spanish": "No pude ver tu gesto. Inténtalo de nuevo.",
    "French": "Je n'ai pas pu voir votre geste. Réessayez.",
    "German": "Ich konnte deine Geste nicht erkennen. Versuch es noch einmal.",
}

rps_game_lang = None  # language name of the current/last RPS game
rps_round_pending = False  # True once "Shoot!" has been requested, until a move is scored
rps_countdown_active = False
rps_gesture_display_active = False


def get_rps_voice_move(command_text, allow_stt_aliases=False):
    """Return the single spoken rock/paper/scissors move, or None.

    A bare move ("play rock") answers a pending round. A phrase naming more
    than one gesture ("rock paper scissors") is a request to start a game,
    not a move, so it is left for the game-start branch to handle.
    """
    matches = [word for word in RPS_GESTURES if word in command_text]
    if len(matches) == 1:
        return matches[0]
    if matches or not allow_stt_aliases:
        return None

    tokens = set(command_text.split())
    alias_matches = [
        gesture for gesture, aliases in RPS_MOVE_ALIASES.items()
        if tokens.intersection(aliases)
    ]
    if len(alias_matches) == 1:
        logging.info(
            "Mapped STT move %r to %s during an armed round",
            command_text, alias_matches[0],
        )
        return alias_matches[0]
    return None


def score_rps_voice_move(voice_move):
    """Score a spoken move against a random robot gesture and return the result text."""
    puppy_gesture = random.choice(RPS_GESTURES)
    if RPS_WIN_CONDITIONS.get(voice_move) == puppy_gesture:
        result = RPS_WIN.get(rps_game_lang, "You win!")
    elif voice_move == puppy_gesture:
        result = RPS_TIE.get(rps_game_lang, "It's a tie!")
    else:
        result = RPS_LOSE.get(rps_game_lang, "You lose!")
    logging.info(
        "Rock-paper-scissors scored: human=%s robot=%s result=%r",
        voice_move, puppy_gesture, result,
    )
    show_rps_gesture(puppy_gesture)
    return result
ai_on = True

# A generation changes only after STT confirms a non-empty human utterance.
# Local VAD is deliberately provisional: a knock or other noise must not make
# pending LLM/TTS work stale before any actual words have been recognized.
generation_lock = threading.Lock()
current_generation = 0
tts_interrupt_flag = threading.Event()
tts_active_event = threading.Event()
tts_playback_started_at = 0.0
tts_playback_text = ""
audio_routing = None

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
voice_IL = texttospeech.VoiceSelectionParams(language_code="he-IL", name="he-IL-Standard-A")
voice_KR = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Neural2-A")

lang_voices = {
    "English": voice0,
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
    "en":      "English",
    "en-US":   "English",
    "en-GB":   "English",
    "ja-JP":   "Japanese",
    "ja":      "Japanese",
    "cmn-CN":  "Chinese",
    "cmn-Hans-CN": "Chinese",
    "zh-CN":   "Chinese",
    "zh-TW":   "Chinese",
    "zh":      "Chinese",
    "it-IT":   "Italian",
    "it":      "Italian",
    "de-DE":   "German",
    "de":      "German",
    "fr-FR":   "French",
    "fr":      "French",
    "yue-HK":  "Cantonese",
    "yue":     "Cantonese",
    "es-US":   "Spanish",
    "es-ES":   "Spanish",
    "es":      "Spanish",
    "he-IL":   "Hebrew",
    "he":      "Hebrew",
    "ko-KR":   "Korean",
    "ko":      "Korean",
}

cur_voice = voice0

#heads up variables
playing_heads_up = False
heads_up_word = ""
heads_up_questions = 0
heads_up_game_language = None

# Track last response for translation
last_response = ""

# The STT, LLM, and TTS threads share one display state. Transitions are
# serialized so completion from older work cannot overwrite a newer state.
status_image_lock = threading.Lock()
current_status_image = None


def env_flag(name, default=False):
    """Read a conventional boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LanguageChoice:
    name: str
    code: str
    voice: object


@dataclass(frozen=True)
class InputItem:
    generation: int
    text: str
    language: LanguageChoice
    command_text: str = ""


@dataclass(frozen=True)
class OutputItem:
    generation: int
    text: str
    language: LanguageChoice
    must_finish: bool = False
    release_rps_gesture: bool = False
    opens_rps_round: bool = False


DEFAULT_LANGUAGE = LanguageChoice("English", "en-US", voice0)
generation_languages = {0: DEFAULT_LANGUAGE}
conversation_language_hint = None


def configure_default_language(code, voice_name):
    """Apply the configured default voice to generation-aware queue items."""
    global voice0, cur_voice, DEFAULT_LANGUAGE
    voice0 = texttospeech.VoiceSelectionParams(language_code=code, name=voice_name)
    cur_voice = voice0
    normalized_code = code.replace("_", "-")
    default_name = (
        lang_code_to_name.get(normalized_code) or
        lang_code_to_name.get(normalized_code.split("-", 1)[0]) or
        "English"
    )
    lang_voices[default_name] = voice0
    DEFAULT_LANGUAGE = LanguageChoice(default_name, code, voice0)
    with generation_lock:
        generation_languages[0] = DEFAULT_LANGUAGE


def generation_is_current(generation):
    with generation_lock:
        return generation == current_generation


def get_current_generation():
    with generation_lock:
        return current_generation


def stop_tts_playback():
    """Ask the active TTS callback to abort at its next audio block."""
    tts_interrupt_flag.set()


def transcript_matches_tts_echo(transcript, reference_text):
    """Return True when a recognized barge-in is probably the current TTS."""
    transcript_text = normalize_command_text(transcript)
    reference = normalize_command_text(reference_text)
    if not transcript_text or not reference:
        return False

    def compact_cjk_echo_text(text):
        compact = re.sub(r"[^\w]", "", text.casefold())
        # Chinese STT frequently swaps these common homophones in short echo
        # fragments (for example, 它的高低 -> 他的高低).
        return compact.translate(str.maketrans({
            "他": "它", "她": "它",
            "地": "的", "得": "的",
        }))

    transcript_cjk = compact_cjk_echo_text(transcript)
    reference_cjk = compact_cjk_echo_text(reference_text)
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in transcript_cjk)
    if has_cjk and len(transcript_cjk) >= 2:
        if transcript_cjk in reference_cjk:
            return True
        if len(transcript_cjk) >= 4:
            match = SequenceMatcher(None, transcript_cjk, reference_cjk)
            covered = match.find_longest_match().size / len(transcript_cjk)
            if covered >= 0.75:
                return True
    elif len(transcript_cjk) >= 4 and transcript_cjk in reference_cjk:
        # Short English echo fragments such as "okay" previously bypassed
        # both the eight-character containment rule and the three-word rule.
        return True

    # This also handles languages that are commonly written without spaces.
    if len(transcript_text) >= 8 and transcript_text in reference:
        return True

    # Streaming STT commonly changes one short word in playback echo (for
    # example, "It's a tie" -> "It's a time"). Exact containment and token
    # overlap both miss that case, so compare the normalized phrases fuzzily.
    if min(len(transcript_text), len(reference)) >= 6:
        similarity = SequenceMatcher(None, transcript_text, reference).ratio()
        if similarity >= 0.88:
            return True

    transcript_words = transcript_text.split()
    reference_words = set(reference.split())
    if len(transcript_words) < 3:
        return False
    overlap = sum(word in reference_words for word in transcript_words)
    return overlap / len(transcript_words) >= 0.8


def begin_human_turn():
    """Invalidate old work after a human utterance has been confirmed."""
    global current_generation
    with generation_lock:
        current_generation += 1
        generation = current_generation
        generation_languages[generation] = DEFAULT_LANGUAGE
        # Keep this dictionary bounded during a long-running session.
        for old_generation in list(generation_languages):
            if old_generation < generation - 8:
                generation_languages.pop(old_generation, None)
    stop_tts_playback()
    logging.info("Human speech confirmed; generation %d cancels older work", generation)
    return generation


def set_generation_language(generation, language):
    with generation_lock:
        generation_languages[generation] = language


def language_for_generation(generation):
    with generation_lock:
        return generation_languages.get(generation, DEFAULT_LANGUAGE)


def get_conversation_language_hint():
    with generation_lock:
        return conversation_language_hint


def set_conversation_language_hint(language):
    global conversation_language_hint
    with generation_lock:
        conversation_language_hint = language


def enqueue_input(text, generation=None, language=None, command_text=None):
    generation = get_current_generation() if generation is None else generation
    language = language or language_for_generation(generation)
    if command_text is None:
        command_text = normalize_command_text(text)
    input_text_queue.put(InputItem(generation, text, language, command_text))


def enqueue_output(
    text,
    generation=None,
    language=None,
    must_finish=False,
    release_rps_gesture=False,
    opens_rps_round=False,
):
    if not text:
        return
    generation = get_current_generation() if generation is None else generation
    language = language or language_for_generation(generation)
    output_text_queue.put(OutputItem(
        generation,
        str(text),
        language,
        must_finish,
        release_rps_gesture,
        opens_rps_round,
    ))


class WebRTCAudioRouting:
    """Create and verify a Pulse-compatible WebRTC AEC source/sink pair.

    The output sink is the AEC far-end reference.  The paired source is the
    microphone after WebRTC acoustic echo cancellation and noise suppression.
    PULSE_SOURCE/PULSE_SINK route PortAudio through that exact pair.
    """

    def __init__(self):
        self.enabled = env_flag("ENABLE_TTS_BARGE_IN", True)
        self.required = env_flag("REQUIRE_WEBRTC_AEC", True)
        self.source_name = os.environ.get("AEC_SOURCE_NAME", "ai_app8_echo_cancel")
        self.sink_name = os.environ.get("AEC_SINK_NAME", "ai_app8_echo_cancel_sink")
        self.module_id = None
        self.loaded_here = False

    @staticmethod
    def _pactl(*args):
        try:
            return subprocess.run(
                ["pactl", *args], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout.strip()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pactl is missing. Install pulseaudio-utils plus PipeWire's "
                "audio modules before enabling full-duplex barge-in."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"pactl {' '.join(args)} failed: {detail}") from exc

    @classmethod
    def _names(cls, object_type):
        rows = cls._pactl("list", "short", object_type).splitlines()
        return {row.split("\t", 2)[1] for row in rows if "\t" in row}

    def _matching_echo_module(self):
        for row in self._pactl("list", "short", "modules").splitlines():
            fields = row.split("\t", 3)
            if len(fields) < 3 or fields[1] != "module-echo-cancel":
                continue
            arguments = fields[2]
            if (f"source_name={self.source_name}" in arguments and
                    f"sink_name={self.sink_name}" in arguments and
                    "aec_method=webrtc" in arguments):
                return fields[0]
        return None

    @staticmethod
    def _validate_name(name, label):
        if not name or any(ch.isspace() for ch in name):
            raise RuntimeError(f"Invalid {label} name: {name!r}")

    def setup(self):
        if not self.enabled:
            logging.warning(
                "ENABLE_TTS_BARGE_IN=0: using safe half-duplex mode; the "
                "microphone will pause during TTS."
            )
            return

        self._validate_name(self.source_name, "AEC source")
        self._validate_name(self.sink_name, "AEC sink")
        try:
            self._pactl("info")
            sources = self._names("sources")
            sinks = self._names("sinks")
            source_exists = self.source_name in sources
            sink_exists = self.sink_name in sinks
            if source_exists != sink_exists:
                raise RuntimeError(
                    "Only one member of the configured AEC source/sink pair exists. "
                    "Unload the stale module or choose new AEC_SOURCE_NAME/AEC_SINK_NAME values."
                )

            if source_exists and not self._matching_echo_module():
                raise RuntimeError(
                    "The configured source/sink names exist, but pactl does not show "
                    "a matching module-echo-cancel with aec_method=webrtc."
                )

            if not source_exists:
                master_source = os.environ.get("AEC_MASTER_SOURCE") or self._pactl("get-default-source")
                master_sink = os.environ.get("AEC_MASTER_SINK") or self._pactl("get-default-sink")
                if master_source.endswith(".monitor"):
                    raise RuntimeError(
                        f"Default input {master_source!r} is a monitor, not a microphone; "
                        "set AEC_MASTER_SOURCE to the physical microphone source."
                    )
                if master_source == self.source_name or master_sink == self.sink_name:
                    raise RuntimeError("AEC master devices cannot point back to the virtual AEC pair.")
                if master_source not in sources:
                    raise RuntimeError(f"AEC master source {master_source!r} was not found.")
                if master_sink not in sinks:
                    raise RuntimeError(f"AEC master sink {master_sink!r} was not found.")

                module_args = [
                    "load-module", "module-echo-cancel", "aec_method=webrtc",
                    f"source_master={master_source}", f"sink_master={master_sink}",
                    f"source_name={self.source_name}", f"sink_name={self.sink_name}",
                    "rate=48000", "channels=1",
                ]
                aec_args = os.environ.get("WEBRTC_AEC_ARGS", "").strip()
                if aec_args:
                    module_args.append(f"aec_args={aec_args}")
                self.module_id = self._pactl(*module_args)
                self.loaded_here = True

            sources = self._names("sources")
            sinks = self._names("sinks")
            if self.source_name not in sources or self.sink_name not in sinks:
                raise RuntimeError("WebRTC AEC module loaded without creating the configured source/sink pair.")
            if not self._matching_echo_module():
                raise RuntimeError("Could not verify the loaded WebRTC module in pactl's module list.")

            os.environ["PULSE_SOURCE"] = self.source_name
            os.environ["PULSE_SINK"] = self.sink_name
            logging.info(
                "Verified WebRTC AEC/NS routing: capture=%s playback/reference=%s",
                self.source_name, self.sink_name,
            )
        except RuntimeError:
            if self.loaded_here:
                self.close()
            self.enabled = False
            if self.required:
                raise
            logging.exception(
                "AEC unavailable; REQUIRE_WEBRTC_AEC=0 permits safe half-duplex fallback."
            )

    def close(self):
        if self.loaded_here and self.module_id:
            try:
                self._pactl("unload-module", self.module_id)
            except RuntimeError as exc:
                logging.warning("Could not unload AEC module %s: %s", self.module_id, exc)
            self.loaded_here = False


def audio_setup_error_message(exc):
    return (
        f"{exc}\n\nFull-duplex barge-in needs a running PulseAudio-compatible server. "
        "On Ubuntu 24.04 install pipewire, pipewire-pulse, wireplumber, "
        "libspa-0.2-modules and pulseaudio-utils, then run the app as the "
        "logged-in audio user (not through sudo). A missing user session bus "
        "must be fixed before pactl or systemctl --user can work."
    )


def select_pyaudio_input(py_audio, require_pulse):
    """Select the Pulse/PipeWire PortAudio input routed by PULSE_SOURCE."""
    requested = os.environ.get("MIC_INPUT_DEVICE", "").strip()
    candidates = []
    for index in range(py_audio.get_device_count()):
        info = py_audio.get_device_info_by_index(index)
        if int(info.get("maxInputChannels", 0)) > 0:
            candidates.append((index, info))

    if requested:
        if requested.isdigit():
            index = int(requested)
            info = py_audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) < 1:
                raise RuntimeError(f"MIC_INPUT_DEVICE={requested} has no input channels.")
            if require_pulse and not any(
                token in str(info.get("name", "")).casefold()
                for token in ("pulse", "pipewire")
            ):
                raise RuntimeError("Full-duplex MIC_INPUT_DEVICE must be the Pulse/PipeWire device.")
            return index, info
        for index, info in candidates:
            if requested.casefold() in str(info.get("name", "")).casefold():
                if require_pulse and not any(
                    token in str(info.get("name", "")).casefold()
                    for token in ("pulse", "pipewire")
                ):
                    raise RuntimeError("Full-duplex MIC_INPUT_DEVICE must be the Pulse/PipeWire device.")
                return index, info
        raise RuntimeError(f"MIC_INPUT_DEVICE={requested!r} did not match a PortAudio input.")

    if require_pulse:
        for index, info in candidates:
            name = str(info.get("name", "")).casefold()
            if "pulse" in name or "pipewire" in name:
                return index, info
        raise RuntimeError(
            "WebRTC AEC was created, but PortAudio has no Pulse/PipeWire input. "
            "Install the ALSA Pulse plugin or set MIC_INPUT_DEVICE explicitly."
        )

    info = py_audio.get_default_input_device_info()
    return int(info["index"]), info


def select_sounddevice_output(require_pulse):
    """Select the Pulse/PipeWire output routed by PULSE_SINK."""
    requested = os.environ.get("TTS_OUTPUT_DEVICE", "").strip()
    devices = sd.query_devices()
    candidates = [
        (index, info) for index, info in enumerate(devices)
        if int(info.get("max_output_channels", 0)) > 0
    ]
    if requested:
        if requested.isdigit():
            index = int(requested)
            if index >= len(devices) or int(devices[index].get("max_output_channels", 0)) < 1:
                raise RuntimeError(f"TTS_OUTPUT_DEVICE={requested} is not an output device.")
            if require_pulse and not any(
                token in str(devices[index].get("name", "")).casefold()
                for token in ("pulse", "pipewire")
            ):
                raise RuntimeError("Full-duplex TTS_OUTPUT_DEVICE must be the Pulse/PipeWire device.")
            return index
        for index, info in candidates:
            if requested.casefold() in str(info.get("name", "")).casefold():
                if require_pulse and not any(
                    token in str(info.get("name", "")).casefold()
                    for token in ("pulse", "pipewire")
                ):
                    raise RuntimeError("Full-duplex TTS_OUTPUT_DEVICE must be the Pulse/PipeWire device.")
                return index
        raise RuntimeError(f"TTS_OUTPUT_DEVICE={requested!r} did not match a SoundDevice output.")

    if require_pulse:
        for index, info in candidates:
            name = str(info.get("name", "")).casefold()
            if "pulse" in name or "pipewire" in name:
                return index
        raise RuntimeError(
            "WebRTC AEC was created, but SoundDevice has no Pulse/PipeWire output. "
            "Install the ALSA Pulse plugin or set TTS_OUTPUT_DEVICE explicitly."
        )
    return None


def validate_portaudio_routing(require_pulse):
    """Fail during startup instead of leaving dead audio worker threads behind."""
    py_audio = pyaudio.PyAudio()
    try:
        input_index, input_info = select_pyaudio_input(py_audio, require_pulse)
        output_index = select_sounddevice_output(require_pulse)
        logging.info(
            "PortAudio preflight passed: input=%d (%s), output=%s",
            input_index, input_info.get("name"),
            output_index if output_index is not None else "system default",
        )
    finally:
        py_audio.terminate()


class ChirpStreamingSession:
    """One Google Speech-to-Text V2 Chirp stream, fed after local VAD fires."""

    _END = object()

    def __init__(self, sample_rate, language_codes=None):
        from google.cloud import speech_v2 as cloud_speech

        self.cloud_speech = cloud_speech
        credentials, detected_project = google.auth.default()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or detected_project
        if not project_id:
            raise RuntimeError("Google credentials did not provide a project ID.")
        location = os.environ.get("GOOGLE_STT_LOCATION", "us")
        recognizer_id = os.environ.get("GOOGLE_STT_RECOGNIZER", "_")
        self.recognizer = (
            f"projects/{project_id}/locations/{location}/recognizers/{recognizer_id}"
        )
        endpoint = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
        self.client = cloud_speech.SpeechClient(
            credentials=credentials,
            client_options=ClientOptions(api_endpoint=endpoint),
        )
        self.sample_rate = sample_rate
        self.language_codes = language_codes
        self.audio_queue = queue_module.Queue()
        self.transcript = ""
        self.live_transcript = ""
        self.transcript_lock = threading.Lock()
        self.language_code = None
        self.error = None
        self.thread = None

    def _requests(self):
        speech = self.cloud_speech
        language_codes = self.language_codes or [
            code.strip() for code in
            os.environ.get("GOOGLE_STT_LANGUAGE_CODES", "auto").split(",")
            if code.strip()
        ] or ["auto"]
        config = speech.RecognitionConfig(
            explicit_decoding_config=speech.ExplicitDecodingConfig(
                encoding=speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            language_codes=language_codes,
            model=os.environ.get("GOOGLE_STT_MODEL", "chirp_3"),
            features=speech.RecognitionFeatures(enable_automatic_punctuation=True),
        )
        yield speech.StreamingRecognizeRequest(
            recognizer=self.recognizer,
            streaming_config=speech.StreamingRecognitionConfig(
                config=config,
                streaming_features=speech.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            ),
        )
        while True:
            chunk = self.audio_queue.get()
            if chunk is self._END:
                return
            yield speech.StreamingRecognizeRequest(audio=chunk)

    def _run(self):
        try:
            final_parts = []
            interim = ""
            for response in self.client.streaming_recognize(requests=self._requests()):
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if getattr(result, "language_code", None):
                        self.language_code = result.language_code
                    if result.is_final:
                        if text:
                            final_parts.append(text)
                        interim = ""
                    else:
                        interim = text
                    live_text = " ".join(final_parts + ([interim] if interim else []))
                    with self.transcript_lock:
                        self.live_transcript = live_text.strip()
            self.transcript = " ".join(final_parts).strip() or interim
            with self.transcript_lock:
                self.live_transcript = self.transcript
        except Exception as exc:
            self.error = exc

    def start(self, pre_roll):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        for chunk in pre_roll:
            self.feed(chunk)

    def feed(self, chunk):
        if self.thread:
            self.audio_queue.put(chunk)

    def current_transcript(self):
        with self.transcript_lock:
            return self.live_transcript

    def finish(self):
        if not self.thread:
            return None, None
        self.audio_queue.put(self._END)
        timeout = float(os.environ.get("GOOGLE_STT_FINAL_TIMEOUT", "8"))
        self.thread.join(timeout=max(1.0, timeout))
        if self.thread.is_alive():
            try:
                self.client.transport.close()
            except Exception:
                pass
            raise TimeoutError(f"Streaming STT did not finish within {timeout:.1f}s")
        if self.error:
            raise self.error
        return self.transcript or None, self.language_code


def show_status_image(status, expected_statuses=None):
    """
    Display status images for different speech recognition stages.
    Only displays images when AI is active (ai_on=True).
    When AI is off, keeps the logo2.png image displayed.

    Args:
        status: 'calibrating', 'listening', 'completed', 'thinking', 'talking', or 'ready' (hello_ready.png)
    """
    global ai_on, current_status_image

    # Don't change image when AI is off (close_ai mode)
    if not ai_on:
        logging.debug(f"AI is off - keeping logo2.png, ignoring status '{status}'")
        return

    try:
        status_files = {
            "calibrating": "hello_y.png",
            "listening": "hello_r.png",
            "completed": "hello_g.png",
            "thinking": "hello_think.png",
            "talking": "hello_talk.png",
            "ready": "hello_ready.png",
        }
        filename = status_files.get(status)
        if not filename:
            logging.warning("Unknown display status %r", status)
            return False

        with status_image_lock:
            if rps_gesture_display_active:
                logging.debug(
                    "Keeping rock-paper-scissors gesture; ignoring status %r",
                    status,
                )
                return False
            if (expected_statuses is not None and
                    current_status_image not in expected_statuses):
                logging.debug(
                    "Ignoring stale display transition %s -> %s",
                    current_status_image, status,
                )
                return False
            if current_status_image == status:
                return True
            image = Image.open(f"{RES_DIR}/{filename}")
            image_queue.put(image)
            current_status_image = status
        logging.info("Display state=%s (%s)", status, filename)
        return True
    except Exception as e:
        logging.error(f"Failed to display status image for '{status}': {e}")
        return False


def show_rps_gesture(gesture):
    """Show and protect a robot gesture until its result speech completes."""
    global current_status_image, rps_gesture_display_active
    try:
        image = Image.open(f"{RES_DIR}/{gesture}.jpg")
        with status_image_lock:
            rps_gesture_display_active = True
            current_status_image = f"rps:{gesture}"
            image_queue.put(image)
        logging.info("Holding rock-paper-scissors gesture=%s", gesture)
    except Exception:
        logging.exception("Could not display rock-paper-scissors gesture=%s", gesture)
        with status_image_lock:
            rps_gesture_display_active = False


def release_rps_gesture_display():
    """Release the protected gesture and atomically restore the ready image."""
    global current_status_image, rps_gesture_display_active
    try:
        ready_image = Image.open(f"{RES_DIR}/hello_ready.png")
        with status_image_lock:
            if not rps_gesture_display_active:
                return
            rps_gesture_display_active = False
            current_status_image = "ready"
            image_queue.put(ready_image)
        logging.info("Rock-paper-scissors result finished; gesture released")
    except Exception:
        logging.exception("Could not release rock-paper-scissors gesture display")
        with status_image_lock:
            rps_gesture_display_active = False


def finish_rps_countdown(open_round):
    """Arm the move only when the spoken countdown completed successfully."""
    global rps_countdown_active, rps_round_pending
    rps_countdown_active = False
    rps_round_pending = bool(open_round)
    if open_round:
        logging.info("Rock-paper-scissors countdown completed; move window is open")
    else:
        logging.info("Rock-paper-scissors countdown cancelled; move window stays closed")


def rps_move_window_reached(playback_fraction):
    threshold = min(1.0, max(
        0.0,
        float(os.environ.get("RPS_MOVE_WINDOW_FRACTION", "0.80")),
    ))
    return playback_fraction >= threshold


def rps_input_mode():
    mode = os.environ.get("RPS_INPUT_MODE", "camera").strip().casefold()
    return mode if mode in {"camera", "voice"} else "camera"


def queue_rps_camera_capture(generation, language):
    """Queue one explicit camera-scoring turn after the countdown."""
    global rps_countdown_active, rps_round_pending
    rps_countdown_active = False
    rps_round_pending = False
    if not generation_is_current(generation):
        logging.info("Skipping stale rock-paper-scissors camera capture")
        return
    enqueue_input(
        "Capture and score the player's rock-paper-scissors hand gesture.",
        generation,
        language,
        command_text="capture rps gesture",
    )
    logging.info("Rock-paper-scissors countdown completed; camera capture queued")


class NoiseFloorTracker:
    """Adaptive estimate of the ambient background-noise RMS.

    Fed only non-speech frames captured while TTS is silent, so a noisy room
    raises the floor and a quiet one lowers it, instead of callers comparing
    against one fixed RMS constant that is wrong as soon as the room isn't.
    The rise is faster than the fall so a burst of noise (a door, a fan
    kicking on) is tracked quickly, while a brief lull right after loud
    noise doesn't undershoot the floor and make the next quiet moment look
    like speech.
    """

    def __init__(self, initial_rms, rise_alpha, fall_alpha):
        self.floor = float(initial_rms)
        self.rise_alpha = rise_alpha
        self.fall_alpha = fall_alpha

    def update(self, rms):
        alpha = self.rise_alpha if rms > self.floor else self.fall_alpha
        self.floor += alpha * (rms - self.floor)


class AudioInputStreamClosed(RuntimeError):
    """Signal that PortAudio input must be opened again."""


class NoiseRobustSTT:
    """
    VAD and utterance buffering over the WebRTC AEC/NS-cleaned source.

    AEC/NS deliberately lives in the audio server so it receives the exact TTS
    playback reference.  Local VAD is only a speech boundary detector; it is
    not acoustic echo cancellation.
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
        silence_ms = int(os.environ.get("VAD_END_SILENCE_MS", "450"))
        start_confirm_ms = int(os.environ.get("VAD_START_CONFIRM_MS", "100"))
        barge_in_confirm_ms = int(os.environ.get("BARGE_IN_CONFIRM_MS", "100"))
        self.barge_in_guard_ms = max(0, int(os.environ.get("BARGE_IN_GUARD_MS", "80")))
        self.num_silent_frames_threshold = max(1, silence_ms // self.vad_frame_duration_ms)
        self.num_speech_frames_threshold = max(1, start_confirm_ms // self.vad_frame_duration_ms)
        self.num_barge_in_frames_threshold = max(
            self.num_speech_frames_threshold,
            barge_in_confirm_ms // self.vad_frame_duration_ms,
        )
        barge_in_max_gap_ms = max(
            0, int(os.environ.get("BARGE_IN_MAX_GAP_MS", "60"))
        )
        self.barge_in_max_gap_frames = (
            barge_in_max_gap_ms // self.vad_frame_duration_ms
        )
        self.cancel_barge_in_on_vad = env_flag("BARGE_IN_CANCEL_ON_VAD", False)
        self.barge_in_min_rms = max(0, int(os.environ.get("BARGE_IN_MIN_RMS", "120")))
        self.silence_threshold = int(os.environ.get("VAD_RMS_FALLBACK_THRESHOLD", "500"))
        pre_roll_ms = int(os.environ.get("PRE_ROLL_MS", "400"))
        self.ring_buffer_size = max(10, pre_roll_ms // self.vad_frame_duration_ms)

        # Adaptive background-noise floor. barge_in_min_rms/silence_threshold
        # above remain hard minimums; the effective gate is whichever is
        # higher of that minimum and (tracked ambient noise + margin), so a
        # noisy room stops false-triggering barge-in and a quiet room still
        # catches soft speech near the old fixed floor.
        self.noise_floor = NoiseFloorTracker(
            initial_rms=float(os.environ.get("NOISE_FLOOR_INITIAL_RMS", "150")),
            rise_alpha=float(os.environ.get("NOISE_FLOOR_RISE_ALPHA", "0.1")),
            fall_alpha=float(os.environ.get("NOISE_FLOOR_FALL_ALPHA", "0.02")),
        )
        self.barge_in_margin_rms = max(0, int(os.environ.get("BARGE_IN_MARGIN_RMS", "150")))
        self.voice_max_crest_factor = max(
            1.0, float(os.environ.get("BARGE_IN_VOICE_MAX_CREST_FACTOR", "10.0"))
        )
        self.voice_max_spectral_flatness = min(
            1.0, max(0.0, float(os.environ.get("BARGE_IN_VOICE_MAX_FLATNESS", "0.65")))
        )
        self.voice_min_speech_band_ratio = min(
            1.0, max(0.0, float(os.environ.get("BARGE_IN_VOICE_MIN_BAND_RATIO", "0.55")))
        )
        self.voice_min_active_ratio = min(
            1.0, max(0.0, float(os.environ.get("BARGE_IN_VOICE_MIN_ACTIVE_RATIO", "0.03")))
        )

        logging.info(
            "STT front end initialized: WebRTC VAD=%d, pre-roll=%dms, "
            "speech confirmation=%dms, barge confirmation=%dms, "
            "barge guard=%dms, max barge gap=%dms, cancel-on-VAD=%s, "
            "barge min RMS=%d (+adaptive margin=%d over tracked noise floor, "
            "starting at %.0f), end silence=%dms",
            vad_aggressiveness, self.ring_buffer_size * 20,
            self.num_speech_frames_threshold * 20,
            self.num_barge_in_frames_threshold * 20,
            self.barge_in_guard_ms,
            self.barge_in_max_gap_frames * 20,
            self.cancel_barge_in_on_vad,
            self.barge_in_min_rms,
            self.barge_in_margin_rms,
            self.noise_floor.floor,
            self.num_silent_frames_threshold * 20,
        )

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
        """
        try:
            resampled = self._resample_to_vad_rate(audio_data)
            expected_size = int(self._VAD_RATE * self.vad_frame_duration_ms / 1000)
            if len(resampled) < expected_size:
                resampled = np.pad(resampled, (0, expected_size - len(resampled)), 'constant')
            else:
                resampled = resampled[:expected_size]
            return self.vad.is_speech(resampled.tobytes(), self._VAD_RATE)

        except Exception as e:
            # Amplitude fallback — gate against whichever is higher of the
            # configured floor and the adaptively tracked ambient noise, so a
            # noisy room doesn't get stuck permanently "hearing" speech.
            adaptive_threshold = max(self.silence_threshold, self.noise_floor.floor * 1.5)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            result = rms >= adaptive_threshold
            logging.warning(f"VAD exception ({e}), amplitude fallback: RMS={rms:.0f} threshold={adaptive_threshold:.0f} speech={result}")
            return result

    def is_voice_like_candidate(self, frames):
        """Reject impulsive and spectrally flat sounds before immediate barge-in."""
        if not frames:
            return False
        samples = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
        if samples.size < max(1, int(self.sample_rate * 0.08)):
            return False

        samples -= np.mean(samples)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms <= 1.0:
            return False
        peak = float(np.max(np.abs(samples)))
        crest_factor = peak / rms
        active_ratio = float(np.mean(np.abs(samples) >= peak * 0.25))

        windowed = samples * np.hanning(samples.size)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        analysis_mask = (frequencies >= 80.0) & (frequencies <= 8000.0)
        speech_mask = (frequencies >= 80.0) & (frequencies <= 4000.0)
        analysis_power = power[analysis_mask]
        if not analysis_power.size or float(np.sum(analysis_power)) <= 0.0:
            return False
        safe_power = np.maximum(analysis_power, 1e-12)
        spectral_flatness = float(
            np.exp(np.mean(np.log(safe_power))) / np.mean(safe_power)
        )
        speech_band_ratio = float(
            np.sum(power[speech_mask]) / np.sum(analysis_power)
        )

        accepted = (
            crest_factor <= self.voice_max_crest_factor and
            spectral_flatness <= self.voice_max_spectral_flatness and
            speech_band_ratio >= self.voice_min_speech_band_ratio and
            active_ratio >= self.voice_min_active_ratio
        )
        logging.info(
            "Barge-in voice gate accepted=%s: crest=%.2f, flatness=%.3f, "
            "speech-band=%.3f, active=%.3f",
            accepted, crest_factor, spectral_flatness,
            speech_band_ratio, active_ratio,
        )
        return accepted

    def transcribe_audio(self, audio_bytes):
        """
        Transcribe audio using Google Speech-to-Text API.
        Returns a tuple of (transcript, detected_language_code).
        """
        try:
            from google.cloud import speech

            # Log audio info
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            duration = len(audio_array) / self.sample_rate
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            logging.info(f"Transcribing audio: duration={duration:.2f}s, RMS={rms:.0f}, size={len(audio_bytes)} bytes")

            audio = speech.RecognitionAudio(content=audio_bytes)

            model_type = "latest_short" if duration < 3.0 else "latest_long"
            preferred_codes = preferred_stt_language_codes()
            primary_code = preferred_codes[0] if preferred_codes else self.language_code
            configured_fallbacks = [
                code.strip() for code in os.environ.get(
                    "GOOGLE_STT_FALLBACK_LANGUAGE_CODES",
                    "es-ES,cmn-CN,fr-FR",
                ).split(",")
                if code.strip()
            ]
            fallback_codes = []
            for code in (preferred_codes[1:] if preferred_codes else []) + configured_fallbacks:
                if code != primary_code and code not in fallback_codes:
                    fallback_codes.append(code)
            fallback_codes = fallback_codes[:3]

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=primary_code,
                enable_automatic_punctuation=True,
                model=model_type,
                audio_channel_count=self.channels,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
                alternative_language_codes=fallback_codes,
            )
            logging.debug(f"Using model: {model_type} for {duration:.2f}s audio")

            response = self.speech_client.recognize(config=config, audio=audio)

            if response.results:
                result = response.results[0]
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    try:
                        detected_lang = result.language_code or primary_code
                    except Exception:
                        detected_lang = primary_code
                    logging.info(f"Transcription: '{transcript}' (confidence: {confidence:.2%}, detected lang: {detected_lang})")
                    return transcript, detected_lang
                else:
                    logging.warning("No alternatives in transcription result")
            else:
                logging.warning("No results from Speech API - audio may be too quiet, too short, or not contain clear speech")

            return None, None

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None, None

    def listen_full_duplex(self, stream):
        """Capture one cleaned utterance and stream it to multilingual STT.

        Returns ``(transcript, language_code, generation, was_barge_in)``.  The
        current generation is snapshotted at VAD confirmation, but a new one
        is allocated only after cloud STT confirms actual human speech.  This
        prevents noise-only VAD events from invalidating pending responses.
        """
        frame_ms = self.vad_frame_duration_ms
        pre_roll = deque(maxlen=self.ring_buffer_size)
        captured = []
        streaming = None
        speech_frames = 0
        candidate_gap_frames = 0
        silent_frames = 0
        generation = None
        was_barge_in = False
        deferred_barge_in_validation = False
        barge_in_playback_stopped = False
        last_barge_in_partial = ""
        candidate_during_tts = False
        barge_in_reference = ""
        candidate_peak_rms = 0.0
        candidate_frames = []
        state = "IDLE"
        max_frames = int(float(os.environ.get("MAX_UTTERANCE_SECONDS", "20")) * 1000 / frame_ms)
        min_speech_frames = max(1, int(os.environ.get("MIN_SPEECH_MS", "300")) // frame_ms)
        voiced_frames = 0

        while True:
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except OSError as exc:
                stream_is_closed = (
                    getattr(exc, "errno", None) == -9988 or
                    "stream closed" in str(exc).casefold()
                )
                if stream_is_closed:
                    if streaming is not None:
                        try:
                            streaming.finish()
                        except Exception:
                            logging.debug(
                                "Could not finish STT stream after input closure",
                                exc_info=True,
                            )
                    logging.error("Audio input stream closed; requesting reopen")
                    raise AudioInputStreamClosed(str(exc)) from exc
                logging.warning("Audio input overflow; resetting VAD candidate: %s", exc)
                pre_roll.clear()
                speech_frames = 0
                candidate_gap_frames = 0
                candidate_during_tts = False
                candidate_peak_rms = 0.0
                candidate_frames.clear()
                state = "IDLE"
                time.sleep(frame_ms / 1000.0)
                continue

            # Without verified AEC, listening during TTS would let the robot
            # interrupt itself.  Keep draining PortAudio, but discard the audio.
            if not audio_routing.enabled and tts_active_event.is_set():
                pre_roll.clear()
                speech_frames = 0
                candidate_gap_frames = 0
                candidate_during_tts = False
                candidate_peak_rms = 0.0
                candidate_frames.clear()
                state = "IDLE"
                continue

            tts_is_active = tts_active_event.is_set()
            if (tts_is_active and tts_playback_started_at and
                    (time.monotonic() - tts_playback_started_at) * 1000 < self.barge_in_guard_ms):
                # Ignore the playback onset while the echo canceller adapts to
                # the new far-end signal. This transient otherwise looks like
                # local speech and aborts TTS before AEC has converged.
                pre_roll.clear()
                speech_frames = 0
                candidate_gap_frames = 0
                candidate_during_tts = False
                candidate_peak_rms = 0.0
                candidate_frames.clear()
                state = "IDLE"
                continue

            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
            is_speech = self.is_speech_vad(audio_data)

            # Track ambient noise only from confirmed-quiet frames while TTS
            # is silent, so playback residual and the utterance itself never
            # feed the estimate. barge_in_min_rms stays a hard floor beneath
            # the adaptive gate so a near-silent room can't drive it to ~0.
            if not tts_is_active and not is_speech:
                self.noise_floor.update(audio_rms)
            effective_barge_in_min_rms = max(
                self.barge_in_min_rms,
                self.noise_floor.floor + self.barge_in_margin_rms,
            )
            if tts_is_active and audio_rms < effective_barge_in_min_rms:
                # WebRTC VAD can label faint residual far-end speech as voice.
                # Require near-end energy above the adaptive noise floor as
                # well before allowing local TTS cancellation; normal
                # listening remains VAD-only.
                is_speech = False

            if state in {"IDLE", "BARGE_CANDIDATE"}:
                pre_roll.append(data)
                if is_speech:
                    if speech_frames == 0:
                        candidate_during_tts = tts_is_active
                        candidate_frames.clear()
                    elif tts_is_active:
                        candidate_during_tts = True
                    candidate_peak_rms = max(candidate_peak_rms, audio_rms)
                    candidate_frames.append(data)
                    speech_frames += 1
                    candidate_gap_frames = 0
                    state = "BARGE_CANDIDATE"
                else:
                    candidate_gap_frames += 1
                    allowed_gap_frames = (
                        self.barge_in_max_gap_frames
                        if candidate_during_tts else 0
                    )
                    if (state != "BARGE_CANDIDATE" or
                            candidate_gap_frames > allowed_gap_frames):
                        speech_frames = 0
                        candidate_gap_frames = 0
                        candidate_during_tts = False
                        candidate_peak_rms = 0.0
                        candidate_frames.clear()
                        state = "IDLE"

                required_frames = (
                    self.num_barge_in_frames_threshold
                    if candidate_during_tts else self.num_speech_frames_threshold
                )
                if speech_frames < required_frames:
                    continue

                if (candidate_during_tts and
                        not self.is_voice_like_candidate(candidate_frames)):
                    logging.info(
                        "Rejected non-voice barge-in candidate; playback continues"
                    )
                    speech_frames = 0
                    candidate_gap_frames = 0
                    candidate_during_tts = False
                    candidate_peak_rms = 0.0
                    candidate_frames.clear()
                    state = "IDLE"
                    continue

                was_barge_in = candidate_during_tts
                heads_up_barge_in = (
                    was_barge_in and
                    playing_heads_up and
                    heads_up_game_language is not None
                )
                camera_countdown_barge_in = (
                    was_barge_in and
                    rps_countdown_active and
                    rps_input_mode() == "camera"
                )
                transcript_validated_game_barge_in = (
                    heads_up_barge_in or
                    camera_countdown_barge_in
                )
                deferred_barge_in_validation = was_barge_in and (
                    not self.cancel_barge_in_on_vad or
                    transcript_validated_game_barge_in
                )
                # VAD is only a candidate signal.  Keep the current generation
                # until STT returns real text, otherwise a knock can invalidate
                # an LLM response that has not reached TTS yet.
                generation = get_current_generation()
                if deferred_barge_in_validation:
                    barge_in_reference = tts_playback_text
                    logging.info(
                        "%s barge-in candidate buffered; waiting for a "
                        "non-echo streaming transcript before stopping playback",
                        ("Heads Up" if heads_up_barge_in else
                         "Rock-paper-scissors countdown" if camera_countdown_barge_in else
                         "Deferred"),
                    )
                elif was_barge_in:
                    # Optional low-latency mode may stop active audio on VAD,
                    # but it still must not invalidate queued work until STT
                    # verifies that the event contained words.
                    stop_tts_playback()
                    barge_in_playback_stopped = True
                    barge_in_reference = tts_playback_text
                    logging.info(
                        "Immediate local barge-in playback stop requested "
                        "(peak RMS=%.0f, threshold=%d); generation remains "
                        "provisional until STT validation",
                        candidate_peak_rms,
                        self.barge_in_min_rms,
                    )
                else:
                    logging.info(
                        "Speech candidate buffered for generation %d; waiting "
                        "for STT validation before invalidating pending output",
                        generation,
                    )
                state = "LISTENING"
                # Show the listening indicator as soon as local VAD confirms
                # a candidate, even for a deferred barge-in that cloud STT
                # hasn't validated yet. Waiting for that round trip means the
                # indicator would lag or never show for genuine barge-ins
                # that go straight from talking -> completed -> thinking; the
                # RMS/gap gating above already keeps pure TTS echo from
                # reaching this point in the first place.
                show_status_image("listening")
                captured = list(pre_roll)
                voiced_frames = speech_frames
                logging.info(
                    "VAD state BARGE_CANDIDATE -> LISTENING (%dms confirmed, %dms pre-roll)",
                    speech_frames * frame_ms, len(pre_roll) * frame_ms,
                )
                try:
                    streaming_languages = preferred_stt_language_codes()
                    if streaming_languages:
                        logging.info(
                            "Conditioning streaming STT on conversation languages: %s",
                            ", ".join(streaming_languages),
                        )
                    streaming = ChirpStreamingSession(
                        self.sample_rate,
                        language_codes=streaming_languages,
                    )
                    streaming.start(captured)
                except Exception as exc:
                    streaming = None
                    logging.warning("Could not start Chirp streaming; buffered v1 STT fallback will be used: %s", exc)
                continue

            captured.append(data)
            if streaming:
                streaming.feed(data)

            if (deferred_barge_in_validation and
                    not barge_in_playback_stopped and
                    tts_active_event.is_set() and streaming):
                partial = streaming.current_transcript().strip()
                compact_partial = re.sub(r"[^\w]", "", partial)
                has_cjk_partial = any(
                    "\u3400" <= char <= "\u9fff" for char in compact_partial
                )
                partial_is_actionable = (
                    len(compact_partial) >= (2 if has_cjk_partial else 4)
                )
                if (partial_is_actionable and
                        partial != last_barge_in_partial):
                    last_barge_in_partial = partial
                    if transcript_matches_tts_echo(partial, barge_in_reference):
                        logging.info(
                            "Streaming barge-in partial %r matches current "
                            "TTS; playback continues",
                            partial,
                        )
                    else:
                        stop_tts_playback()
                        barge_in_playback_stopped = True
                        logging.info(
                            "Streaming barge-in partial %r is new human "
                            "speech; playback stop requested",
                            partial,
                        )

            if is_speech:
                voiced_frames += 1
                silent_frames = 0
            else:
                silent_frames += 1

            utterance_frames = len(captured)
            speech_finished = silent_frames >= self.num_silent_frames_threshold
            length_limit = utterance_frames >= max_frames
            if not speech_finished and not length_limit:
                continue

            # A deferred barge-in candidate leaves the display at "talking"
            # (or "ready", if TTS finished naturally mid-capture) instead of
            # "listening", so all three are valid predecessors here.
            show_status_image("completed", expected_statuses={"listening", "talking", "ready"})
            if voiced_frames < min_speech_frames:
                logging.info("Ignoring %dms VAD event as too short", voiced_frames * frame_ms)
                if streaming:
                    try:
                        streaming.finish()
                    except Exception:
                        pass
                show_status_image("ready", expected_statuses={"completed"})
                return None, None, generation, was_barge_in

            full_audio = b"".join(captured)
            transcript = None
            detected_language = None
            if streaming:
                try:
                    transcript, detected_language = streaming.finish()
                except Exception as exc:
                    logging.warning("Chirp streaming failed; retrying buffered audio with STT v1: %s", exc)
            if not transcript:
                transcript, detected_language = self.transcribe_audio(full_audio)

            if was_barge_in:
                if not transcript:
                    logging.info(
                        "Ignoring untranscribed barge-in candidate; generation "
                        "was not changed"
                    )
                    show_status_image("ready", expected_statuses={"completed"})
                    return None, None, generation, was_barge_in
                if transcript_matches_tts_echo(transcript, barge_in_reference):
                    logging.info(
                        "Ignoring probable TTS echo transcript %r; playback %s",
                        transcript,
                        ("had already stopped" if barge_in_playback_stopped
                         else "was allowed to continue"),
                    )
                    show_status_image("ready", expected_statuses={"completed"})
                    return None, None, generation, was_barge_in

            if transcript and rps_gesture_display_active:
                # Camera scoring and its short winner announcement are one
                # atomic game result. A transcript already in flight (the
                # countdown tail commonly becomes "好") must not advance the
                # generation between score calculation and result playback.
                logging.info(
                    "Ignoring transcript %r while rock-paper-scissors result "
                    "is protected",
                    transcript,
                )
                show_status_image("ready", expected_statuses={"completed"})
                return None, None, generation, was_barge_in

            if transcript:
                if not generation_is_current(generation):
                    logging.info(
                        "Discarding speech candidate for stale generation %d", generation
                    )
                    show_status_image("ready", expected_statuses={"completed"})
                    return None, None, generation, was_barge_in
                generation = begin_human_turn()
                logging.info(
                    "Validated human transcript before invalidating pending output"
                )

            if transcript:
                show_status_image("thinking", expected_statuses={"completed"})
            else:
                show_status_image("ready", expected_statuses={"completed"})
            logging.info(
                "STT finalized generation=%s language=%s transcript=%r",
                generation, detected_language, transcript,
            )
            return transcript, detected_language, generation, was_barge_in


# Track last response for translation
last_response = ""

def send_response(text, generation=None, language=None):
    """
    Send response to output queue and track it for potential translation.
    """
    global last_response
    last_response = text
    enqueue_output(text, generation, language)


#heads up ai prompt
def create_heads_up_prompt(secret_word, response_language="English"):
    prompt = f"""
You are an AI assistant playing a guessing game similar to "Heads-Up" or "20 Questions."
I have a secret word in mind, and I will tell it to you now.
Your role is to be the "Knower" or "Answerer." I will be the "Guesser."

**The Secret Word is: {secret_word}**
**The player's language is: {response_language}. Reply naturally in that language.**

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
15.  **Never output stage directions or placeholders** such as "waiting for the user to ask a question". If the input is not a real question, respond only with a short natural acknowledgement.
16.  **Readiness means clue:** If the player says "okay", "ready", "please", "go ahead", "continue", or a similar acknowledgement, immediately provide one concise new clue. Do not respond with another acknowledgement.

Let's begin. I have provided the secret word above. Await my first question after your acknowledgment. My next message will be a question.
"""
    return prompt


HEADS_UP_READY_TEXTS = {
    "Chinese": "好的，我已经记住这个词了。请问第一个问题吧。",
    "Cantonese": "好，我記住咗呢個詞喇。請問第一個問題啦。",
}

HEADS_UP_ERROR_TEXTS = {
    "Chinese": "抱歉，我刚才无法回答。请再问一次。",
    "Cantonese": "唔好意思，我頭先答唔到。請再問一次。",
}


def heads_up_ready_text(language_name):
    return HEADS_UP_READY_TEXTS.get(
        language_name,
        "Okay, I have the secret word. I'm ready for your first question.",
    )


def heads_up_error_text(language_name):
    return HEADS_UP_ERROR_TEXTS.get(
        language_name,
        "Sorry, I couldn't answer that. Please ask again.",
    )


def heads_up_success_text(language_name, secret_word, guess_count):
    if language_name == "Chinese":
        return f"恭喜你！你用了{guess_count}次猜中答案：{secret_word}！"
    if language_name == "Cantonese":
        return f"恭喜你！你用咗{guess_count}次就估中答案：{secret_word}！"
    return (
        f"Congratulations! You guessed the word: {secret_word}, "
        f"in {guess_count} guesses!"
    )


def heads_up_answer_is_correct(answer, secret_word):
    """Recognize affirmative correct-guess answers in English or Chinese."""
    normalized_answer = normalize_command_text(answer)
    compact_answer = re.sub(r"[^\w]", "", answer.casefold())
    compact_word = re.sub(r"[^\w]", "", secret_word.casefold())
    if not compact_word or compact_word not in compact_answer:
        return False
    success_phrases = (
        "that's it",
        "that is it",
        "you got it",
        "correct",
        "答对",
        "答對",
        "猜对",
        "猜對",
        "没错",
        "沒錯",
        "就是它",
        "就係佢",
        "估中",
    )
    return any(
        phrase in normalized_answer or phrase in compact_answer
        for phrase in success_phrases
    )


def heads_up_answer_is_placeholder(answer):
    compact = re.sub(r"[^\w]", "", str(answer or "").casefold())
    placeholders = (
        "等待用户提问",
        "等待提问",
        "waitingfortheusertoaskaquestion",
        "waitingforuserinput",
    )
    return any(phrase in compact for phrase in placeholders)


def heads_up_requests_clue(command_text):
    """Treat readiness/filler turns as a request for a clue, not an ack loop."""
    normalized = normalize_command_text(command_text)
    exact_requests = {
        "ok", "okay", "please", "okay please", "ok please",
        "ready", "i'm ready", "im ready", "go ahead", "continue", "next",
        "start", "let's start", "lets start", "what do you mean okay",
    }
    clue_phrases = (
        "give me a clue", "give me a hint", "tell me a clue",
        "tell me a hint", "next clue", "next hint",
    )
    return (
        normalized in exact_requests or
        any(phrase in normalized for phrase in clue_phrases)
    )


def model_response_text(response):
    """Return non-empty text from a LangChain response, including block content."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip() if content is not None else ""

def detect_lang_usage(prompt, lang):
    prompt = prompt.casefold()
    lang = lang.casefold()
    adjectives = ['food', 'culture', 'characters', 'novel', 'history']
    language_phrases = [f'in {lang}', f'to {lang}', f'say in {lang}', f'translate to {lang}']

    for phrase in language_phrases:
        if phrase in prompt:
            return "Language choice"

    for adj in adjectives:
        if f'{lang} {adj}' in prompt:
            return "Adjective"

    return "Unknown"

def detect_script_voice(transcript):
    """Return a voice when the written script identifies the language reliably."""
    def _any_in_range(text, lo, hi):
        return any(lo <= ord(c) <= hi for c in text)

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

    return None


def detect_voice_from_transcript(transcript, stt_lang_code=None):
    """
    Detect the appropriate TTS voice from Unicode script or the STT language.

    An unambiguous written script wins over STT metadata. This prevents an
    erroneous ``language=en`` result from assigning an English voice to text
    that is visibly Chinese, Japanese, Korean, or Hebrew.
    """
    script_voice = detect_script_voice(transcript)
    if script_voice:
        return script_voice

    if stt_lang_code:
        normalized_code = stt_lang_code.replace("_", "-")
        lang_name = lang_code_to_name.get(normalized_code)
        if not lang_name:
            lang_name = lang_code_to_name.get(normalized_code.split("-", 1)[0])
        if lang_name:
            voice = lang_voices.get(lang_name)
            if voice:
                logging.info(f"Detected language '{lang_name}' via STT code '{stt_lang_code}'")
                return voice

    return None  # Language not in lang_voices — caller uses default voice

def get_voice(prompt=None):
    """
    Determine the voice to be used based on the input prompt.
    """
    if not prompt:
        logging.debug(f"select key voice: None,default is voice0")
        return None, voice0
    folded_prompt = prompt.casefold()
    for key, value in lang_voices.items():
        if key.casefold() in folded_prompt:
            if detect_lang_usage(folded_prompt, key) == "Language choice":
                logging.info(f"select key: {key}")
                return key, value
    logging.info(f"no mapping, default is voice0")
    return None, voice0


LANGUAGE_TTS_CODES = {
    "English": "en-US",
    "Japanese": "ja-JP",
    "Chinese": "cmn-CN",
    "Italian": "it-IT",
    "German": "de-DE",
    "French": "fr-FR",
    "Cantonese": "yue-HK",
    "Spanish": "es-US",
    "Hebrew": "he-IL",
    "Korean": "ko-KR",
}

CLOUD_TRANSLATE_TARGET_CODES = {
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Italian": "it",
    "German": "de",
    "French": "fr",
    "Cantonese": "yue",
    "Spanish": "es",
    "Hebrew": "he",
    "Korean": "ko",
}


def preferred_stt_language_codes():
    """Narrow Chirp language ID using established conversation context."""
    hint = get_conversation_language_hint()
    if not hint:
        return None
    stt_code = {
        "Chinese": "cmn-Hans-CN",
        "Cantonese": "yue-Hant-HK",
    }.get(hint.name, hint.code)
    common_codes = ["en-US", "cmn-Hans-CN"]
    codes = [stt_code]
    for code in common_codes:
        if code not in codes:
            codes.append(code)
    return codes[:3]


def choose_language(transcript, stt_language_code=None):
    """Choose one dominant response/TTS language for this utterance.

    An explicit request wins, followed by an unambiguous Unicode script or an
    explicit STT language. Only short transcripts without usable STT language
    metadata retain the established conversation language.
    """
    requested_name, requested_voice = get_voice(transcript)
    if requested_name:
        language = LanguageChoice(
            requested_name,
            LANGUAGE_TTS_CODES[requested_name],
            requested_voice,
        )
        set_conversation_language_hint(language)
        return language

    normalized_stt_code = (stt_language_code or "").replace("_", "-")
    has_explicit_stt_language = normalized_stt_code.casefold() not in {
        "", "und", "auto",
    }
    voice = detect_voice_from_transcript(transcript, stt_language_code)
    if voice:
        language_name = next(
            (name for name, candidate in lang_voices.items() if candidate == voice),
            "English",
        )
        detected_language = LanguageChoice(
            language_name,
            LANGUAGE_TTS_CODES[language_name],
            voice,
        )
        script_voice = detect_script_voice(transcript)
        hint = get_conversation_language_hint()
        words = re.findall(r"\w+", transcript, flags=re.UNICODE)
        short_latin_utterance = (
            not script_voice and
            len(words) <= 3 and
            len(transcript.strip()) <= 20
        )
        if (hint and hint.name != detected_language.name and
                short_latin_utterance and not has_explicit_stt_language):
            logging.info(
                "Keeping conversation language %s for ambiguous short STT "
                "result %r without an explicit language",
                hint.name, transcript,
            )
            return hint
        if script_voice or has_explicit_stt_language or not short_latin_utterance:
            set_conversation_language_hint(detected_language)
        return detected_language
    hint = get_conversation_language_hint()
    words = re.findall(r"\w+", transcript, flags=re.UNICODE)
    if hint and len(words) <= 3 and len(transcript.strip()) <= 20:
        logging.info(
            "Keeping conversation language %s for short STT result %r "
            "without a usable language code",
            hint.name, transcript,
        )
        return hint
    return DEFAULT_LANGUAGE


def language_for_tts_text(text, fallback_language):
    """Choose TTS from response script, independent of imperfect input STT."""
    script_voice = detect_script_voice(text)
    if not script_voice:
        return fallback_language

    language_name = next(
        (name for name, candidate in lang_voices.items() if candidate == script_voice),
        fallback_language.name,
    )
    hint = get_conversation_language_hint()
    if language_name == "Chinese":
        if fallback_language.name == "Cantonese":
            return fallback_language
        if hint and hint.name == "Cantonese":
            return hint
    language = LanguageChoice(
        language_name,
        LANGUAGE_TTS_CODES[language_name],
        script_voice,
    )
    set_conversation_language_hint(language)
    return language


def response_has_wrong_script(text, expected_language):
    """Detect clear English/non-Latin response-language mismatches."""
    script_voice = detect_script_voice(text)
    script_name = next(
        (name for name, candidate in lang_voices.items()
         if candidate == script_voice),
        None,
    )
    if expected_language.name == "English":
        return script_name is not None
    if expected_language.name in {"Chinese", "Cantonese"}:
        return script_name != "Chinese"
    if expected_language.name in {"Japanese", "Korean", "Hebrew"}:
        return script_name != expected_language.name
    # Latin-script languages cannot be distinguished reliably from English
    # with Unicode alone; their explicit model instruction remains primary.
    return False


def with_language_instruction(text, language):
    # State the language on every turn. Relying on chat history for English
    # lets recent Chinese media or conversation pull an English question back
    # into Chinese even when STT identified the question correctly.
    return (
        f"{text}\n"
        f"Reply naturally and concisely in {language.name} only. "
        "Use the language of this instruction even if earlier turns used a "
        "different language."
    )


def normalize_command_text(text):
    """Normalize an English translation before deterministic intent matching."""
    without_punctuation = re.sub(r"[^\w\s']", " ", text.casefold())
    return re.sub(r"\s+", " ", without_punctuation).strip()


HEADS_UP_CHINESE_ALIASES = (
    "你比我猜",
    "你说我猜",
    "你說我猜",
    "猜词游戏",
    "猜詞遊戲",
    "猜单词游戏",
    "猜單詞遊戲",
    "额头猜词",
    "額頭猜詞",
    "抬头猜词",
    "抬頭猜詞",
    "疯狂猜词",
    "瘋狂猜詞",
    "猜猜我是谁",
    "猜猜我是誰",
)

HEADS_UP_CHINESE_CANCEL_PHRASES = (
    "不想",
    "不要",
    "别玩",
    "別玩",
    "停止",
    "退出",
    "结束",
    "結束",
)


def is_heads_up_start_command(command_text, original_text=""):
    """Recognize an English command or a commonly spoken Chinese game name."""
    if "heads up" in command_text and any(
        action in command_text for action in ("play", "start", "begin")
    ):
        return True

    # Check the original transcript because machine translation often renders
    # Chinese game names as generic phrases such as "guessing game", losing
    # the specific Heads Up intent needed by the deterministic router.
    compact_original = re.sub(r"[^\w]", "", original_text.casefold())
    if any(phrase in compact_original for phrase in HEADS_UP_CHINESE_CANCEL_PHRASES):
        return False
    return any(alias in compact_original for alias in HEADS_UP_CHINESE_ALIASES)


class CommandTranslator:
    """Cloud translation for command routing and response-language repair."""

    def __init__(self):
        credentials, detected_project = google.auth.default()
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or detected_project
        if not project_id:
            raise RuntimeError("Google credentials did not provide a project ID for translation.")
        self.client = translate_v3.TranslationServiceClient(credentials=credentials)
        self.parent = f"projects/{project_id}/locations/global"

    def translate(self, text, target_language_code):
        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "target_language_code": target_language_code,
            },
            timeout=float(os.environ.get("GOOGLE_TRANSLATE_TIMEOUT", "5")),
        )
        if not response.translations:
            raise RuntimeError("Cloud Translation returned no translation.")
        translated = response.translations[0].translated_text.strip()
        if not translated:
            raise RuntimeError("Cloud Translation returned an empty translation.")
        return translated

    def to_english(self, text):
        return self.translate(text, "en")

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

move_command_aliases = {
    "move forward": "move forwards",
    "walk forward": "move forwards",
    "go forward": "move forwards",
    "move backward": "move backwards",
    "walk backward": "move backwards",
    "go backward": "move backwards",
    "move to the left": "move left",
    "go left": "move left",
    "move to the right": "move right",
    "go right": "move right",
    "look to the left": "look left",
    "look to the right": "look right",
    "look upward": "look up",
    "look downward": "look down",
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
    for translated_phrase, command_key in move_command_aliases.items():
        if re.search(r'\b' + re.escape(translated_phrase) + r'\b', input_text):
            return command_key
    return None

def close_ai():
    global ai_on, current_status_image, rps_gesture_display_active
    ai_on = False
    stop_tts_playback()
    image = Image.open(f"{RES_DIR}/logo2.png")
    with status_image_lock:
        rps_gesture_display_active = False
        image_queue.put(image)
        current_status_image = "off"

def open_ai():
    global ai_on
    ai_on = True
    show_status_image("ready")
    enqueue_output("OK, my friend.")

def reboot():
    command = "sudo reboot"
    shell_api.execute_command(command)

def power_off():
    command = "sudo poweroff"
    shell_api.execute_command(command)

def restart_service():
    """Restart the robot UI without ever opening an interactive sudo prompt."""
    subprocess.run(
        ["sudo", "-n", "/usr/bin/systemctl", "restart", "robot.service"],
        check=True,
    )

sys_cmds_functions = {
        "shut up": close_ai,
        "be quiet": close_ai,
        "speak please": open_ai,
        "start speaking": open_ai,
        "reboot": reboot,
        "power off": power_off,
        "turn off": power_off,
        "shut down": power_off,
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


def stt_task_v8():
    """Continuously listen to the AEC/NS source, including while TTS plays."""
    global cur_voice, rps_game_lang, rps_round_pending, rps_countdown_active

    logging.info("Starting full-duplex WebRTC AEC/NS -> VAD -> streaming STT front end")
    py_audio = google_api.init_pyaudio()
    speech_client = google_api.init_speech_to_text()  # buffered fallback
    input_index, device_info = select_pyaudio_input(py_audio, audio_routing.enabled)
    native_rate = int(device_info["defaultSampleRate"])
    native_chunk = int(native_rate * 0.020)
    logging.info(
        "Capture device %d (%s), %d Hz through PULSE_SOURCE=%s",
        input_index, device_info.get("name"), native_rate,
        os.environ.get("PULSE_SOURCE", "system default"),
    )
    recognizer = NoiseRobustSTT(
        speech_client=speech_client,
        py_audio=py_audio,
        sample_rate=native_rate,
        chunk_size=native_chunk,
        vad_aggressiveness=int(os.environ.get("VAD_AGGRESSIVENESS", "1")),
        language_code=os.environ.get("LANGUAGE_CODE", "en-US"),
    )
    def open_input_stream():
        return py_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=native_rate,
            input=True,
            input_device_index=input_index,
            frames_per_buffer=native_chunk,
        )

    stream = open_input_stream()
    show_status_image("ready")

    try:
        command_translator = CommandTranslator()
        logging.info("Cloud Translation command normalization is ready")
    except Exception:
        command_translator = None
        logging.exception(
            "Command translation is unavailable; non-English text will remain "
            "conversational and will not trigger robot/system actions"
        )

    interrupt_phrases = {
        "stop", "enough", "that is enough", "that's enough", "be quiet",
        "shut up",
    }
    while True:
        try:
            user_input, detected_code, generation, was_barge_in = (
                recognizer.listen_full_duplex(stream)
            )
        except AudioInputStreamClosed:
            try:
                stream.close()
            except Exception:
                logging.debug("Closed PortAudio stream rejected close()", exc_info=True)
            time.sleep(0.25)
            try:
                stream = open_input_stream()
            except Exception:
                logging.exception("Could not reopen audio input; retrying in one second")
                time.sleep(1.0)
                continue
            logging.info("Audio input stream reopened after unexpected closure")
            continue
        if not user_input:
            continue
        if not generation_is_current(generation):
            logging.info("Discarding stale STT result for generation %d", generation)
            continue

        spoken_language = choose_language(user_input, detected_code)
        command_text = ""
        if command_translator:
            try:
                english_translation = command_translator.to_english(user_input)
                command_text = normalize_command_text(english_translation)
                logging.info("English command translation: %r", command_text)
            except Exception:
                logging.exception("Translation failed for generation %d", generation)
                if spoken_language.name == "English":
                    command_text = normalize_command_text(user_input)
                    logging.warning("Using the original English transcript for command matching")
                else:
                    logging.warning("Suppressing action matching for the untranslated transcript")
        elif spoken_language.name == "English":
            command_text = normalize_command_text(user_input)

        if was_barge_in and any(
            command_text == phrase or command_text.startswith(f"{phrase} ")
            for phrase in interrupt_phrases
        ):
            logging.info("Barge-in stop command transcribed; remaining silent")
            show_status_image("ready", expected_statuses={"thinking"})
            continue

        requested_name, requested_voice = get_voice(command_text)
        if requested_name:
            language = LanguageChoice(
                requested_name,
                LANGUAGE_TTS_CODES[requested_name],
                requested_voice,
            )
        else:
            language = spoken_language
        if playing_heads_up and heads_up_game_language is not None:
            # Keep the language chosen when the game started. Short answers
            # such as numbers often have STT language "und" and must not
            # silently switch a Chinese game back to the English default.
            language = heads_up_game_language
        set_generation_language(generation, language)
        cur_voice = language.voice  # compatibility for game code outside the queues
        logging.info(
            "Generation %d language=%s (%s), STT reported=%s",
            generation, language.name, language.code, detected_code,
        )

        move_key = get_move_cmd(command_text, move_cmd_functions)
        sys_cmd_key, sys_cmd_func = get_sys_cmd(command_text, sys_cmds_functions)
        heads_up_requested = is_heads_up_start_command(command_text, user_input)

        if playing_heads_up:
            enqueue_input(user_input, generation, language, command_text)
        elif sys_cmd_key:
            sys_cmd_func()
        elif move_key in {"sit", "action"}:
            movement_queue.put(move_key)
            enqueue_output("OK, my friend.", generation, language)
        elif "walk" in command_text or "come" in command_text:
            movement_queue.put("move forwards")
            enqueue_output("My friend, here I come.", generation, language)
        elif move_key:
            movement_queue.put(move_key)
            enqueue_output(f"OK, my friend, {move_key} immediately.", generation, language)
        elif not ai_on:
            continue
        elif heads_up_requested:
            # Canonicalize the routing intent while retaining the original
            # transcript and its detected language for the game interaction.
            enqueue_input(user_input, generation, language, "play heads up")
        elif (("don't want" in command_text and "play" in command_text) or
              ("do not want" in command_text and "play" in command_text) or
              "exit" in command_text or "quit" in command_text or
              ("stop" in command_text and
               ("game" in command_text or "playing" in command_text))):
            rps_round_pending = False
            rps_countdown_active = False
            enqueue_input(user_input, generation, language, command_text)
        elif rps_round_pending:
            voice_move = get_rps_voice_move(
                command_text,
                allow_stt_aliases=True,
            )
            if voice_move:
                rps_round_pending = False
                result = score_rps_voice_move(voice_move)
                enqueue_output(
                    result,
                    generation,
                    language,
                    must_finish=True,
                    release_rps_gesture=True,
                )
            else:
                logging.info(
                    "Unrecognized move %r while round remains armed",
                    command_text,
                )
                enqueue_output(
                    RPS_INVALID_MOVE.get(
                        rps_game_lang,
                        "Please say rock, paper, or scissors.",
                    ),
                    generation,
                    language,
                )
        elif rps_countdown_active:
            logging.info(
                "Ignoring speech while rock-paper-scissors countdown is active"
            )
        elif (all(word in command_text for word in RPS_GESTURES) or
              ("game" in command_text and "play" in command_text)):
            rps_round_pending = False
            rps_countdown_active = True
            rps_game_lang = language.name
            enqueue_output(
                GAME_TEXTS.get(rps_game_lang, GAME_TEXT),
                generation,
                language,
                opens_rps_round=True,
            )
        elif get_rps_voice_move(command_text):
            logging.info(
                "Ignoring bare rock-paper-scissors move because no round is armed"
            )
        elif requested_name and last_response and any(
            phrase in command_text for phrase in
            ("say that", "repeat that", "translate that", "say it", "repeat it")
        ):
            enqueue_input(
                f"Translate this to {language.name}: {last_response}",
                generation, language, command_text,
            )
        elif "test" == command_text:
            enqueue_output("test", generation, language)
        else:
            enqueue_input(
                with_language_instruction(user_input, language),
                generation, language, command_text,
            )


def gemini_task():
    """
    Task for handling Gemini AI interactions.
    """
    global last_response, playing_heads_up, heads_up_word
    global heads_up_game_language, rps_round_pending

    logging.debug("gemini task start.")
    history_file_path = "res/ece_history.json"
    conversation = google_api.create_conversation(history_file_path)

    init_input =  "From here on, always answer as if a human being is saying things off the top of his head which is always concise, relevant and contains a good conversational tone. so you will only and only answer in one breath responses. If the input contains a language other than English, for example, language A, please answer the question in language A."
    response = google_api.ai_text_response(conversation, init_input)
    logging.debug(f"init llm and first response: {response}")
    response_translator = None

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
    show_status_image(
        "ready",
        expected_statuses={None, "calibrating", "ready"},
    )

    while True:
        logging.debug("tts wait for gemini responese text... ...")
        queued_input = input_text_queue.get()
        input_text_queue.task_done()
        if isinstance(queued_input, InputItem):
            generation = queued_input.generation
            language = queued_input.language
            input_text = queued_input.text
            decision_text = queued_input.command_text
        else:
            generation = get_current_generation()
            language = language_for_generation(generation)
            input_text = str(queued_input)
            decision_text = normalize_command_text(input_text)
        if not generation_is_current(generation):
            logging.info("Discarding stale LLM input for generation %d", generation)
            continue
        if not ai_on:
            continue

        show_status_image(
            "thinking",
            expected_statuses={"ready", "completed", "thinking"},
        )

        def respond(text, must_finish=False, release_rps_gesture=False):
            if must_finish or generation_is_current(generation):
                enqueue_output(
                    text,
                    generation,
                    language,
                    must_finish=must_finish,
                    release_rps_gesture=release_rps_gesture,
                )
            else:
                if release_rps_gesture:
                    release_rps_gesture_display()
                logging.info("Discarding stale LLM output for generation %d", generation)

        logging.debug(f"user input from voice: {input_text}")
        user_input = input_text
        response = ""
        if not user_input:
            logging.debug(f"no input!")
        elif "clear history" in decision_text:
            conversation.memory.clear()
        elif "photo" in decision_text or "picture" in decision_text or "expression" in decision_text:
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
            respond(response)
        elif decision_text == "capture rps gesture":
            ms_start = int(time.time() * 1000)
            logging.info("Rock-paper-scissors camera scoring started")
            human_image = media_api.take_photo()
            logging.debug("Rock-paper-scissors photo capture finished")

            if human_image is None:
                logging.warning("Rock-paper-scissors camera returned no image")
                respond(RPS_UNCLEAR_GESTURE.get(
                    rps_game_lang,
                    "I couldn't see your gesture. Please try again.",
                ))
                continue

            puppy_gesture = random.choice(RPS_GESTURES)
            show_rps_gesture(puppy_gesture)

            vision_prompt = (
                "Look only at the player's hand gesture. Respond with exactly "
                "one lowercase word: rock, paper, scissors, or unknown. "
                "No punctuation or explanation."
            )
            try:
                raw_gesture = google_api.ai_image_response(
                    multi_model,
                    image=human_image,
                    text=vision_prompt,
                )
            except Exception:
                logging.exception(
                    "Rock-paper-scissors vision request failed; treating the "
                    "gesture as unknown"
                )
                raw_gesture = "unknown"
            normalized_gesture = normalize_command_text(str(raw_gesture or ""))
            human_gesture = get_rps_voice_move(
                normalized_gesture,
                allow_stt_aliases=True,
            )
            logging.info(
                "Rock-paper-scissors vision result: raw=%r normalized=%r",
                raw_gesture,
                human_gesture,
            )

            rps_round_pending = False
            if human_gesture is None:
                result = RPS_UNCLEAR_GESTURE.get(
                    rps_game_lang,
                    "I couldn't see your gesture. Please try again.",
                )
            elif RPS_WIN_CONDITIONS.get(human_gesture) == puppy_gesture:
                result = RPS_WIN.get(rps_game_lang, "You win!")
            elif human_gesture == puppy_gesture:
                result = RPS_TIE.get(rps_game_lang, "It's a tie!")
            else:
                result = RPS_LOSE.get(rps_game_lang, "You lose!")
            logging.info(
                "Rock-paper-scissors scored: human=%s robot=%s result=%r",
                human_gesture or "unknown",
                puppy_gesture,
                result,
            )
            logging.debug(
                "Rock-paper-scissors camera scoring finished in %dms",
                int(time.time() * 1000) - ms_start,
            )
            respond(
                result,
                must_finish=True,
                release_rps_gesture=True,
            )

        elif "what is this" in decision_text or ("what" in decision_text and "holding" in decision_text):
            ms_start = int(time.time() * 1000)
            logging.debug(f"identify take photo")
            input_image = media_api.take_photo()
            logging.debug(f"identify take photo finish")

            shown_object = google_api.ai_image_response(multi_model,image=input_image, text=user_input)
            logging.debug(f"shown_object is: {shown_object}")

            response = shown_object
            respond(response)

        elif "read this" in decision_text:
            ms_start = int(time.time() * 1000)
            logging.debug(f"read take photo")
            input_image = media_api.take_photo()
            logging.debug(f"read take photo finish")

            shown_text = google_api.ai_image_response(multi_model,image=input_image, text=user_input)
            logging.debug(f"shown_text is: {shown_text}")

            response = shown_text
            respond(response)

        elif is_heads_up_start_command(decision_text, user_input):
            conversation.memory.clear()

            ms_start = int(time.time() * 200)
            logging.debug(f"read take photo")
            input_image = media_api.take_photo()
            logging.debug(f"read take photo finish")

            shown_text = google_api.ai_image_response(multi_model,image=input_image, text="Tell me what the word on the paper is. Respond only with what is on the paper all in lowercase. Do not begin with a space. If there is no word on a card respond with 'no word' ")
            logging.debug(f"shown_text is: '{shown_text}'")

            heads_up_word = str(shown_text or "").strip()

            if not heads_up_word or "no word" in heads_up_word.lower():
                playing_heads_up = False
                heads_up_game_language = None
                logging.debug("no word on heads up card, ending heads up sequence")
                respond("No word was provided, ending heads up sequence")
                continue

            heads_up_game_language = language
            playing_heads_up = True
            conversation.memory.clear()

            heads_up_prompt = create_heads_up_prompt(heads_up_word, language.name)
            ai_acknowledgement = heads_up_ready_text(language.name)
            conversation_history = [
                HumanMessage(content=heads_up_prompt),
                AIMessage(content=ai_acknowledgement),
            ]
            respond(ai_acknowledgement)

            guess_count = 0

            while playing_heads_up:
                queued_input = input_text_queue.get()
                input_text_queue.task_done()
                if isinstance(queued_input, InputItem):
                    generation = queued_input.generation
                    language = heads_up_game_language or queued_input.language
                    input_text = queued_input.text
                    decision_text = queued_input.command_text
                else:
                    generation = get_current_generation()
                    language = (
                        heads_up_game_language or
                        language_for_generation(generation)
                    )
                    input_text = str(queued_input)
                    decision_text = normalize_command_text(input_text)
                if not generation_is_current(generation):
                    continue
                if not ai_on:
                    continue

                show_status_image(
                    "thinking",
                    expected_statuses={"ready", "completed", "thinking"},
                )

                logging.debug(f"user input from voice: {input_text}")
                user_input = input_text

                if (("don't want" in decision_text and "play" in decision_text) or
                    ("do not want" in decision_text and "play" in decision_text) or
                    "exit" in decision_text or "quit" in decision_text or
                    ("stop" in decision_text and
                     ("game" in decision_text or "playing" in decision_text))):
                    playing_heads_up = False
                    respond("Okay, exiting the heads up game. Thanks for playing!")
                    heads_up_game_language = None
                    logging.debug("User requested to exit heads up game")
                    continue

                user_question = user_input.strip()
                if not user_question:
                    logging.warning("Ignoring an empty Heads Up question")
                    show_status_image("ready", expected_statuses={"thinking"})
                    continue

                model_question = user_question
                if heads_up_requests_clue(decision_text):
                    model_question = (
                        "The player is ready. Give exactly one concise new clue "
                        "about the secret word now. Do not say okay, do not "
                        "acknowledge readiness, do not ask a question, and do "
                        f"not reveal the word. Reply in {language.name}."
                    )
                    logging.info(
                        "Mapped Heads Up acknowledgement %r to a clue request",
                        user_question,
                    )

                conversation_history.append(HumanMessage(content=model_question))
                try:
                    response = multi_model.invoke(conversation_history)
                    ai_answer = model_response_text(response)
                    if not ai_answer:
                        raise ValueError("Gemini returned empty content")
                except Exception:
                    # Remove the unanswered user turn so the next request has
                    # a valid alternating history and the worker stays alive.
                    conversation_history.pop()
                    logging.exception("Heads Up answer generation failed")
                    respond(heads_up_error_text(language.name))
                    continue
                logging.debug(f"ai answer: {ai_answer}")

                if heads_up_answer_is_placeholder(ai_answer):
                    # Do not synthesize model stage directions. They sound
                    # especially chaotic and can themselves leak back into
                    # STT as a new fake Heads Up turn.
                    conversation_history.pop()
                    logging.info(
                        "Suppressing Heads Up placeholder response %r",
                        ai_answer,
                    )
                    show_status_image("ready", expected_statuses={"thinking"})
                    continue

                guess_count+=1

                conversation_history.append(AIMessage(content=ai_answer))

                if heads_up_answer_is_correct(ai_answer, heads_up_word):
                    respond(
                        heads_up_success_text(
                            language.name,
                            heads_up_word,
                            guess_count,
                        )
                    )
                    playing_heads_up = False
                    heads_up_game_language = None
                else:
                    respond(ai_answer)
        else:
            logging.debug("text response start!")
            response = google_api.ai_text_response(conversation, user_input)
            if response_has_wrong_script(response, language):
                # The model can follow the language of recent turns despite
                # the per-turn instruction. Correct a clear script mismatch
                # before TTS; otherwise, for example, an English answer is
                # synthesized using the Chinese voice selected for the turn.
                logging.warning(
                    "Model response script does not match the %s turn; "
                    "translating before TTS",
                    language.name,
                )
                try:
                    if response_translator is None:
                        response_translator = CommandTranslator()
                    response = response_translator.translate(
                        response,
                        CLOUD_TRANSLATE_TARGET_CODES[language.name],
                    )
                except Exception:
                    logging.exception(
                        "Could not translate mismatched model response to %s",
                        language.name,
                    )
            logging.debug("text response end: {response}")
            if generation_is_current(generation):
                send_response(response, generation, language)
            else:
                logging.info("Discarding stale text response for generation %d", generation)
        time.sleep(0.05)


def decode_tts_audio(audio_content):
    """Decode Google's LINEAR16 WAV response without treating its header as audio."""
    samples, sample_rate = sf.read(BytesIO(audio_content), dtype="int16", always_2d=False)
    if samples.size == 0:
        raise RuntimeError("Google TTS returned an empty audio stream.")
    return samples, int(sample_rate)


def resample_tts_audio(samples, source_rate, target_rate):
    """Resample synthesized speech locally to the PipeWire graph rate."""
    if source_rate == target_rate:
        return samples
    from math import gcd
    divisor = gcd(source_rate, target_rate)
    converted = resample_poly(
        samples.astype(np.float32),
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    )
    return np.clip(np.rint(converted), -32768, 32767).astype(np.int16)


def play_buffered_audio(samples, sample_rate, device, latency, blocksize, stop_requested):
    """Play a complete in-memory buffer with low-overhead interruptible callbacks."""
    playback_samples = samples.reshape(-1, 1) if samples.ndim == 1 else samples
    channels = playback_samples.shape[1]
    finished = threading.Event()
    # The ALSA Pulse plugin can report the PortAudio callback complete while a
    # latency window is still queued in PipeWire. Queue silence after the final
    # sample so closing the stream cannot discard the last spoken syllable.
    drain_frames = max(1, int(latency * sample_rate) + max(0, blocksize))
    state = {
        "offset": 0,
        "drain_frames": drain_frames,
        "interrupted": False,
        "statuses": [],
    }

    def audio_callback(outdata, frames, _time_info, status):
        if status:
            state["statuses"].append(str(status))
        if stop_requested():
            state["interrupted"] = True
            outdata.fill(0)
            raise sd.CallbackAbort

        offset = state["offset"]
        count = min(frames, len(playback_samples) - offset)
        if count:
            outdata[:count] = playback_samples[offset:offset + count]
        if count < frames:
            outdata[count:].fill(0)
        state["offset"] = offset + count
        if state["offset"] >= len(playback_samples):
            state["drain_frames"] -= frames - count
            if state["drain_frames"] <= 0:
                raise sd.CallbackStop

    duration = len(playback_samples) / sample_rate
    drain_duration = drain_frames / sample_rate
    timeout = duration + drain_duration + max(5.0, latency * 4)
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device,
        blocksize=blocksize,
        latency=latency,
        callback=audio_callback,
        finished_callback=finished.set,
    ) as playback:
        if not finished.wait(timeout):
            playback.abort(ignore_errors=True)
            raise RuntimeError(f"TTS playback timed out after {timeout:.1f}s")

    return state["interrupted"], state["statuses"]


def tts_task_v8():
    """Synthesize and play complete responses with generation-safe interruption."""
    global tts_playback_started_at, tts_playback_text
    logging.info("Starting generation-aware TTS playback")
    amixer_control = os.environ.get("AMIXER_CONTROL", "").strip()
    amixer_volume = os.environ.get("AMIXER_VOLUME", "85%").strip()
    if amixer_control:
        try:
            subprocess.run(
                ["amixer", "-c", os.environ.get("AMIXER_CARD", "0"),
                 "sset", amixer_control, amixer_volume],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logging.warning("amixer is not installed; leaving hardware volume unchanged")

    tts_client = texttospeech.TextToSpeechClient()
    playback_sample_rate = int(os.environ.get("TTS_PLAYBACK_SAMPLE_RATE_HZ", "48000"))
    playback_latency = float(os.environ.get("TTS_LATENCY_SECONDS", "0.08"))
    playback_blocksize = int(os.environ.get("TTS_BLOCKSIZE_FRAMES", "1024"))
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )
    output_device = select_sounddevice_output(audio_routing.enabled)
    logging.info(
        "TTS output device=%s through PULSE_SINK=%s, playback-rate=%dHz, "
        "latency=%.0fms, blocksize=%d",
        output_device if output_device is not None else "system default",
        os.environ.get("PULSE_SINK", "system default"),
        playback_sample_rate,
        playback_latency * 1000,
        playback_blocksize,
    )

    while True:
        queued_output = output_text_queue.get()
        output_text_queue.task_done()
        if isinstance(queued_output, OutputItem):
            item = queued_output
        else:
            generation = get_current_generation()
            item = OutputItem(generation, str(queued_output), language_for_generation(generation))
        if ((not generation_is_current(item.generation) and not item.must_finish) or
                not ai_on):
            logging.info("Discarding stale/disabled TTS output for generation %d", item.generation)
            if item.opens_rps_round:
                finish_rps_countdown(False)
            if item.release_rps_gesture:
                release_rps_gesture_display()
            continue

        text = remove_asterisk_text(remove_emojis(item.text)).strip()
        if not text:
            if item.opens_rps_round:
                finish_rps_countdown(False)
            if item.release_rps_gesture:
                release_rps_gesture_display()
            show_status_image("ready", expected_statuses={"thinking"})
            continue
        tts_language = language_for_tts_text(text, item.language)
        if tts_language.name != item.language.name:
            logging.info(
                "TTS language corrected from %s to %s using response script",
                item.language.name, tts_language.name,
            )
        tts_interrupt_flag.clear()
        try:
            synthesis_started = time.monotonic()
            response = tts_client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=tts_language.voice,
                audio_config=audio_config,
            )
            synthesis_elapsed = time.monotonic() - synthesis_started
            samples, native_sample_rate = decode_tts_audio(response.audio_content)
            resample_started = time.monotonic()
            samples = resample_tts_audio(
                samples, native_sample_rate, playback_sample_rate,
            )
            resample_elapsed = time.monotonic() - resample_started
            sample_rate = playback_sample_rate
            logging.info(
                "TTS prepared %d characters: cloud=%.2fs, native=%dHz, "
                "local-resample=%.3fs, playback=%dHz",
                len(text), synthesis_elapsed, native_sample_rate,
                resample_elapsed, sample_rate,
            )
        except Exception:
            logging.exception("TTS synthesis failed for language %s", tts_language.name)
            if item.opens_rps_round:
                finish_rps_countdown(False)
            if item.release_rps_gesture:
                release_rps_gesture_display()
            show_status_image("ready", expected_statuses={"thinking"})
            continue

        if (not item.must_finish and
                (not generation_is_current(item.generation) or tts_interrupt_flag.is_set())):
            logging.info("Discarding synthesized audio invalidated during cloud TTS")
            if item.opens_rps_round:
                finish_rps_countdown(False)
            if item.release_rps_gesture:
                release_rps_gesture_display()
            continue

        interrupted = False
        playback_failed = False
        playback_started = time.monotonic()
        playback_duration = len(samples) / sample_rate
        tts_playback_text = text
        tts_playback_started_at = time.monotonic()
        tts_active_event.set()
        show_status_image(
            "talking",
            expected_statuses={"ready", "thinking", "talking"},
        )
        logging.info(
            "Speaking generation=%d language=%s duration=%.2fs",
            item.generation, tts_language.name, playback_duration,
        )
        try:
            interrupted, callback_statuses = play_buffered_audio(
                samples,
                sample_rate,
                output_device,
                playback_latency,
                playback_blocksize,
                (lambda: False) if item.must_finish else tts_interrupt_flag.is_set,
            )
            interrupted = interrupted or (
                not item.must_finish and not generation_is_current(item.generation)
            )
            if callback_statuses:
                logging.warning(
                    "PortAudio reported during TTS playback: %s",
                    ", ".join(dict.fromkeys(callback_statuses)),
                )
        except Exception:
            playback_failed = True
            logging.exception("TTS playback failed")
        finally:
            playback_fraction = min(
                1.0,
                (time.monotonic() - playback_started) / max(0.001, playback_duration),
            )
            tts_active_event.clear()
            tts_playback_started_at = 0.0
            tts_playback_text = ""
            if item.release_rps_gesture:
                release_rps_gesture_display()
            else:
                show_status_image("ready", expected_statuses={"talking"})

        if interrupted:
            if item.opens_rps_round:
                if rps_input_mode() == "voice":
                    accept_move = rps_move_window_reached(playback_fraction)
                    finish_rps_countdown(accept_move)
                    logging.info(
                        "Rock-paper-scissors countdown interrupted at %.0f%%; "
                        "voice move window %s",
                        playback_fraction * 100,
                        "opened" if accept_move else "closed",
                    )
                else:
                    finish_rps_countdown(False)
                    logging.info(
                        "Rock-paper-scissors camera countdown interrupted at "
                        "%.0f%%; capture cancelled",
                        playback_fraction * 100,
                    )
            logging.info("TTS generation %d interrupted by human speech", item.generation)
            continue
        if playback_failed:
            if item.opens_rps_round:
                finish_rps_countdown(False)
            continue
        logging.info("TTS generation %d playback completed", item.generation)

        if item.opens_rps_round:
            if rps_input_mode() == "camera":
                queue_rps_camera_capture(item.generation, item.language)
            else:
                finish_rps_countdown(True)


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
    global audio_routing
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
    audio_routing = WebRTCAudioRouting()
    try:
        audio_routing.setup()
    except RuntimeError as exc:
        raise SystemExit(audio_setup_error_message(exc)) from exc
    atexit.register(audio_routing.close)
    try:
        validate_portaudio_routing(audio_routing.enabled)
    except Exception as exc:
        audio_routing.close()
        raise SystemExit(audio_setup_error_message(exc)) from exc

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
    configure_default_language(lang_code, lang_name)

    logging.info("="*60)
    logging.info("AI APP 8 - FULL-DUPLEX MULTILINGUAL BARGE-IN")
    logging.info("Audio: %s", "verified WebRTC AEC/NS" if audio_routing.enabled else "safe half-duplex fallback")
    logging.info(
        "Barge-in: local WebRTC VAD with %sms confirmation after a %sms AEC guard; "
        "cancellation=%s",
        os.environ.get("BARGE_IN_CONFIRM_MS", "100"),
        os.environ.get("BARGE_IN_GUARD_MS", "80"),
        "immediate" if env_flag("BARGE_IN_CANCEL_ON_VAD", False) else "after STT validation",
    )
    logging.info("STT: Google Chirp 3 streaming with language identification")
    logging.info("Commands: Cloud Translation to English before deterministic action matching")
    logging.info("TTS: generation-aware 20ms playback chunks routed to the AEC reference sink")
    logging.info("="*60)

    stt_thread = threading.Thread(target=stt_task_v8, name="stt-v8")
    stt_thread.start()
    logging.debug("stt thread start.")

    gemini_thread = threading.Thread(target=gemini_task)
    gemini_thread.start()
    logging.debug("gemini thread start.")

    tts_thread = threading.Thread(target=tts_task_v8, name="tts-v8")
    tts_thread.start()

    gif_thread = threading.Thread(target=gif_task)
    gif_thread.start()

    image_thread = threading.Thread(target=image_task)
    image_thread.start()

    move_thread = threading.Thread(target=move_task)
    move_thread.start()

    heads_up_thread = threading.Thread(target=heads_up_task)
    heads_up_thread.start()

    try:
        stt_thread.join()
        gemini_thread.join()
        tts_thread.join()
        gif_thread.join()
        image_thread.join()
        move_thread.join()
        heads_up_thread.join()
    except KeyboardInterrupt:
        # The worker threads above are non-daemon and block on queues/audio
        # I/O, so they would otherwise keep the process alive after Ctrl-C.
        # A second Ctrl-C while the restart is still in flight would send
        # SIGINT to this whole foreground process group, killing the
        # sudo/systemctl child before it finishes and leaving robot.service
        # untouched. Ignoring SIGINT here is inherited across exec, so the
        # child is protected too; restart_service() logs its own completion.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info(
            "Ctrl-C received; restarting robot.service (this can take a few "
            "seconds while the robot control process shuts down cleanly, "
            "please don't press Ctrl-C again)"
        )
        try:
            audio_routing.close()
        except Exception:
            logging.exception("Failed to release audio routing before restart")
        try:
            restart_service()
        except (OSError, subprocess.CalledProcessError):
            logging.exception(
                "Could not restart robot.service without a password; "
                "check the sudoers NOPASSWD rule for systemctl"
            )
            os._exit(1)
        logging.info("robot.service restarted; exiting")
        os._exit(0)


if __name__ == '__main__':
    main()
