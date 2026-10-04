"""Snapshots originate at mutation events, never periodic status requests."""
import json
import logging
import threading

from .outbox import Outbox

logger = logging.getLogger(__name__)


class ResourceEvents:
    def __init__(self, bus, getters):
        self.bus, self.getters = bus, getters
        self.lock = threading.RLock()
        self.previous = {}
        self.generation = {}
        self.outbox = Outbox(bus)
        self.unsubscribe = bus.subscribe(self.observe)

    def snapshot(self):
        result = {}
        for topic, getter in self.getters.items():
            try: result[topic] = getter()
            except Exception: # Runtime may be starting/stopping.
                logger.debug('Resource %s unavailable for snapshot', topic, exc_info=True)
                result[topic] = None
        return result

    def emit(self, topic):
        # Getters may take session locks, so they run unlocked. A newer emit of the same
        # topic supersedes an older one still computing, and the outbox publishes in queue
        # order after the lock is released, so subscribers never see a stale value last.
        with self.lock: generation = self.generation[topic] = self.generation.get(topic, 0) + 1
        try: value = self.getters[topic]()
        except Exception as exc:
            # Getters are HTTP handlers: an HTTP error (status_code) means the feature is off.
            level = logging.DEBUG if hasattr(exc, 'status_code') else logging.WARNING
            logger.log(level, 'Resource %s could not be read; panels keep the previous value', topic, exc_info=True)
            return
        key = json.dumps(value, sort_keys=True, default=str)
        with self.lock:
            if generation == self.generation[topic] and self.previous.get(topic) != key:
                self.previous[topic] = key
                self.outbox.put('resource.' + topic, **value)
        self.outbox.flush()

    def observe(self, event):
        kind = event.type
        if kind.startswith('resource.'): return
        if kind.startswith('tool.approval'): self.emit('approvals')
        elif kind.startswith(('initiative.', 'environment.')): self.emit('initiative')
        elif kind.startswith('animation.'): self.emit('animation')
        elif kind == 'avatar.models_changed': self.emit('avatar_models')
        elif kind == 'task.changed': self.emit('tasks')
        elif kind in {'voice.starting','voice.ready','voice.stopped','voice.error','voice.wake_status','voice.activated','voice.follow_up','voice.started','voice.resumed','voice.utterance_ended','voice.transcribing','voice.transcript','voice.waiting'}: self.emit('voice')
        elif kind == 'runtime.ready':
            for topic in self.getters: self.emit(topic)

    def close(self): self.unsubscribe()
