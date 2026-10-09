import queue
import threading
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from process.app_core.audio.voice_input import VoiceInput
from process.app_core.audio.voice_segments import Segment
from process.app_core.audio.wake_word import WakeWord
from process.app_core.kernel.turns import RuntimeStatus
from process.app_core.kernel.audio_config import audio_sections


def test_capture_only_queues_post_keyword_request_for_asr(tmp_path, monkeypatch):
    now = [9.9]
    monkeypatch.setattr('process.app_core.audio.wake_word.time.monotonic', lambda: now[0])
    config = SimpleNamespace(root=tmp_path, character_name='Riko', raw={'voice': {}}, **audio_sections({}))
    wake = WakeWord(config)
    keyword, activation, request, silence = (bytes([marker, 0]) * 512 for marker in (1, 2, 3, 0))
    monkeypatch.setattr(wake, 'feed', lambda frame, speaking: wake.activate(after_keyword=True) if frame == activation else None)
    class VAD:
        def __call__(self, samples, rate): return float(samples[0] != 0)
        def reset_states(self): pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda samples: samples))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(load_silero_vad=VAD))
    voice = VoiceInput.__new__(VoiceInput)
    voice.session = SimpleNamespace(config=config, wake=wake, state=SimpleNamespace(mic_enabled=True),
        set_user_speaking=lambda speaking: None, status=RuntimeStatus, voice_anchor=lambda: None)
    voice.closed, voice.jobs, voice._last_overflow = threading.Event(), queue.Queue(), 0
    class Frames(queue.Queue):
        def get(self, **kwargs):
            if self.empty():
                voice.closed.set()
                raise queue.Empty
            frame, timestamp = super().get(**kwargs)
            now[0] = timestamp
            return frame, timestamp
    voice.frames = Frames()
    packets = [(keyword, 9.9), (activation, 10.0), (keyword, 10.1)]
    packets += [(silence, 10.2 + index * .032) for index in range(10)]
    packets += [(request, 10.6)]
    packets += [(silence, 10.7 + index * .032) for index in range(32)]
    for packet in packets: voice.frames.put(packet)
    try:
        voice._run()
        segments = list(voice.jobs.queue)
        assert segments and sum(part.final for part in segments) == 1
        assert b''.join(part.pcm for part in segments) == request + silence * 10
    finally:
        wake.close()


def test_asr_keeps_legitimate_wake_name_mentions_in_request(monkeypatch):
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=object))
    voice = VoiceInput.__new__(VoiceInput)
    voice.closed, voice._parts = threading.Event(), {}
    text = 'Riko is the character in my story.'
    voice.session = SimpleNamespace(config=SimpleNamespace(raw={'voice': {}}, **audio_sections({})),
        wake=SimpleNamespace(calibrating=False, testing=False, mode='wake_word', phrase='Riko'), transcribe=lambda pcm, **options: text)
    submitted = []
    voice.responses = SimpleNamespace(submit=lambda *args: submitted.append(args))
    class Jobs(queue.Queue):
        def task_done(self):
            super().task_done()
            voice.closed.set()
    voice.jobs = Jobs()
    voice.jobs.put(Segment('request', bytes(1024), None, 10, 11, 1, True))
    voice._asr()
    assert len(submitted) == 1 and submitted[0][1] == text


class NoRaw(dict):
    def get(self, *args): raise AssertionError('voice settings must come from config.voice, checked once by load_config')
    __getitem__ = get


@pytest.mark.parametrize('threshold, heard', [(.5, True), (.75, False)])
def test_each_frame_compares_speech_probability_with_the_typed_vad_threshold(tmp_path, monkeypatch, threshold, heard):
    config = SimpleNamespace(root=tmp_path, character_name='Riko', raw=NoRaw(), **audio_sections({'voice': {'mode': 'continuous', 'vad_threshold': threshold}}))
    wake = WakeWord(config)
    class VAD:
        def __call__(self, samples, rate): return .6 if samples[0] else 0.0
        def reset_states(self): pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext, from_numpy=lambda samples: samples))
    monkeypatch.setitem(sys.modules, 'silero_vad', SimpleNamespace(load_silero_vad=VAD))
    speaking = []
    voice = VoiceInput.__new__(VoiceInput)
    voice.session = SimpleNamespace(config=config, wake=wake, state=SimpleNamespace(mic_enabled=True),
        set_user_speaking=speaking.append, status=RuntimeStatus, voice_anchor=lambda: None)
    voice.closed, voice.jobs, voice._last_overflow = threading.Event(), queue.Queue(), 0
    class Frames(queue.Queue):
        def get(self, **kwargs):
            if self.empty():
                voice.closed.set()
                raise queue.Empty
            return super().get(**kwargs)
    voice.frames = Frames()
    for index in range(3): voice.frames.put((bytes([1, 0]) * 512, 10 + index * .032))
    try:
        voice._run()
        assert speaking == [heard] * 3
    finally: wake.close()
