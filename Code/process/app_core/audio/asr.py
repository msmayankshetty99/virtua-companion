"""The one faster-whisper factory, and the session's one model built with it (AsrService). auto/default resolve to what
this CTranslate2 build can run here (it has no Metal backend, and its CPU path rejects float16), and a configured pair
it cannot run falls back once, with one warning, instead of failing every utterance."""
from functools import lru_cache
import logging
import threading

logger = logging.getLogger(__name__)
MODEL = 'distil-small.en'
# Fastest first, matching CTranslate2's own 'auto'. CPU 'int8' runs as int8_float32.
PREFERRED = {'cuda': ('int8_float16', 'int8_float32', 'float16', 'float32'), 'cpu': ('int8', 'float32')}
_warned = set()


@lru_cache(maxsize=1)
def supported_types():
    """{device: compute types} that this CTranslate2 build supports; 'cuda' only with a visible device."""
    try: import ctranslate2
    except ImportError: return {'cpu': frozenset({'float32'})}
    types = {'cpu': frozenset(ctranslate2.get_supported_compute_types('cpu'))}
    try:
        if ctranslate2.get_cuda_device_count() > 0: types['cuda'] = frozenset(ctranslate2.get_supported_compute_types('cuda'))
    except Exception as exc: logger.warning('CTranslate2 CUDA probe failed; speech recognition stays on the CPU: %s', exc)
    return types


def resolve(voice):
    """(device, compute_type, note) to construct with; note says why a configured pair was replaced, else ''."""
    wanted = str(voice.get('asr_device') or 'auto'), str(voice.get('asr_compute_type') or 'default')
    supported, (device, precision), notes = supported_types(), wanted, []
    if device == 'auto': device = 'cuda' if 'cuda' in supported else 'cpu'
    if device not in supported:
        notes.append('no CUDA device is visible to CTranslate2' if device == 'cuda' else f'unknown device {device}')
        device = 'cpu'
    best = next((name for name in PREFERRED[device] if name in supported[device]), 'float32')
    if precision in ('default', 'auto'): precision = best
    elif precision not in supported[device]: notes.append(f'{device} does not support {precision}'); precision = best
    return device, precision, '; '.join(notes)


def create_whisper(voice):
    from faster_whisper import WhisperModel
    device, precision, note = resolve(voice)
    if note and note not in _warned:
        _warned.add(note)
        logger.warning('Speech recognition %s/%s cannot run here (%s); using %s/%s. Choose auto and default in Settings.',
            voice.get('asr_device', 'auto'), voice.get('asr_compute_type', 'default'), note, device, precision)
    return WhisperModel(voice.get('asr_model', MODEL), device=device, compute_type=precision)


class AsrService:
    """The session's one Whisper model (SessionManager.asr), shared by the microphone (VoiceInput), startup warmup and
    Discord's /transcribe: built through create_whisper on first use, kept until close(), one decode at a time."""
    def __init__(self, voice, *, model=None):
        self.voice, self.model, self.lock, self.closed = voice, model, threading.Lock(), False

    def transcribe(self, samples, **options):
        """The text of float32 16 kHz mono samples; options go to WhisperModel.transcribe."""
        with self.lock:
            if self.closed: raise RuntimeError('Speech recognition is closed')
            model = self.model
            if model is None:  # a caller that arrives meanwhile waits for this one instead of building another
                model = create_whisper(self.voice)
                if not self.closed: self.model = model  # one finished after close() is not kept
            segments, _ = model.transcribe(samples, **options)
            return ' '.join(segment.text.strip() for segment in segments).strip()  # decoding runs while iterating

    def transcribe_pcm(self, pcm, **options):
        """The text of mono 16 kHz signed little-endian PCM16."""
        import numpy as np
        return self.transcribe(np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768, **options)

    def warm(self):
        import numpy as np
        self.transcribe(np.zeros(16000, dtype='float32'), beam_size=1, vad_filter=False)

    def close(self):  # never waits for a decode in progress
        self.closed, self.model = True, None
