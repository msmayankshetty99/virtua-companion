"""Explicit wiring: the factory hands ChatService its collaborators as ChatDeps and the desktop tools a DesktopServices of
their own (so two services in one process never overwrite each other), close() owns what the factory built, the probe
asks EmotionWorker whether playback drives the expression, and each turn reaches ChatService as a TurnContext."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from process.app_core import factory
from process.app_core.configuration.config import AppConfig
from process.app_core.conversation.chat import ChatDeps, ChatService
from process.app_core.desktop.state import DesktopState, get_desktop_state
from process.app_core.desktop.tools import DesktopServices, EffectTool, iter_tools
from process.app_core.emotion.worker import EmotionWorker
from process.app_core.inference.provider import BaseProvider
from process.app_core.kernel.messages import ChatMessage, ModelResponse
from process.app_core.runtime.actions import ActionController
from process.app_core.runtime.session import SessionManager
from process.app_core.kernel.audio_config import audio_sections


class ProbeProvider(BaseProvider):
    """A provider that captures for the emotion probe: attach_probe keeps the hook."""
    hook = interval = None
    def __init__(self, close): self.close = close
    @property
    def probe_host(self): return self
    def attach_probe(self, hook, interval_tokens): self.hook, self.interval = hook, interval_tokens


def build(root, monkeypatch, closed, **emotion):
    """create_chat_service over stubs for the provider, tool registry and memory store, with emotion as given."""
    seen = {}
    provider = ProbeProvider(close=lambda: closed.append(('provider', root.name)))
    monkeypatch.setattr(factory, 'create_provider', lambda runtime: provider)
    def registry(config, activity=None, desktop=None):
        seen['desktop'] = desktop
        stub = SimpleNamespace(register_mcp=lambda client, **keys: None, choice_resolver=None)
        stub.close = lambda: (stub.choice_resolver.close(), closed.append(('registry', root.name)))
        return stub
    monkeypatch.setattr(factory.ToolRegistry, 'from_config', registry)
    monkeypatch.setattr(factory, 'MemoryStore', lambda *args, **kwargs: seen.setdefault('memory', kwargs) and SimpleNamespace(close=lambda: closed.append(('memory', root.name))))
    config = AppConfig(root=root)
    config.runtime.warmup = False
    config.memory.history_file = root / 'persistent_memories' / 'chat_history.json'  # never the cwd's (user data)
    for key, value in emotion.items(): setattr(config.emotion, key, value)
    return factory.create_chat_service(config), provider, seen


def test_the_factory_passes_collaborators_explicitly_and_close_owns_them(tmp_path, monkeypatch):
    closed = []
    (first_root := tmp_path / 'first').mkdir(); (second_root := tmp_path / 'second').mkdir()
    for root in (first_root, second_root): (root / 'character_files').mkdir(); (root / 'character_files' / 'face.png').write_bytes(b'png')
    first, provider, seen = build(first_root, monkeypatch, closed)
    second, _, _ = build(second_root, monkeypatch, closed)
    try:
        deps = first.deps
        assert seen['desktop'] is deps.desktop and deps.desktop.state is get_desktop_state() and deps.desktop.actions is deps.action_controller
        assert isinstance(deps.action_controller, ActionController) and isinstance(deps.cleanup, ExitStack)
        assert deps.initiative_provider is provider and deps.task_mcp.store is deps.task_store  # provider.lanes: one lane without slots
        assert seen['memory']['reflection_provider'] is provider and seen['memory']['parallelism'] == 1 and seen['memory']['token_counter'] == provider.count_text_tokens
        assert deps.context_limit == 8192 and deps.emotion_worker is None and first.emotion_worker is None  # min(n_ctx, memory + reply)
        # A second service in the same process keeps its own media root, effects and gestures instead of replacing the first's.
        assert first.deps.desktop is not second.deps.desktop
        assert first.deps.desktop.media_resolver('character_files/face.png') == (first_root / 'character_files' / 'face.png').resolve()
        assert second.deps.desktop.media_resolver('character_files/face.png') == (second_root / 'character_files' / 'face.png').resolve()
        assert first.deps.desktop.effect_library.directory == first_root / 'effects' / 'greenscreens'
    finally:
        first.close(); first.close()  # the second close finds the stack already empty
        second.close()
    assert closed == [('memory', 'first'), ('registry', 'first'), ('provider', 'first'), ('memory', 'second'), ('registry', 'second'), ('provider', 'second')]


def test_the_probe_holds_its_reading_back_while_playback_drives_the_expression(tmp_path, monkeypatch):
    engine = SimpleNamespace(start_turn=lambda *args: None, observe_input=lambda *args, **kwargs: None, observe_output=lambda *args, **kwargs: None,
        choose_tool_input=None, close=lambda: None)
    monkeypatch.setattr(factory, 'JuliaEmotionEngine', lambda *args, **kwargs: engine)
    updates, probes = [], []
    monkeypatch.setattr(factory, 'ExpressionActionBridge', lambda on_emotion: SimpleNamespace(update=updates.append))
    monkeypatch.setattr('process.app_core.emotion.probe.EmotionProbe', lambda *args, **kwargs: probes.append(kwargs) or 'probe')
    service, provider, _ = build(tmp_path, monkeypatch, [], enabled=True, probe={'enabled': True})
    try:
        worker = service.emotion_worker
        assert isinstance(worker, EmotionWorker) and service.deps.emotion_worker is worker and worker.engine is engine
        assert provider.interval == 32 and provider.hook.probe is None  # attached before the model loads
        assert provider.hook.start({'gguf_sha256': 'x'}, lambda: True) == 'probe' and provider.hook.probe == 'probe'
        predicted = probes[0]['on_prediction']
        predicted('calm')
        worker.submit('speech', 'Hello there.', final=True)  # a sentence started playing: it drives the expression now
        predicted('excited')
        probes[0]['on_fallback']('excited')
        worker.submit('start', 'next-turn')
        predicted('curious')
        assert updates == ['calm', 'curious']
    finally: service.close()
    assert worker.closed


class Speech:
    def __init__(self, *args): pass
    def submit(self, *args): return False
    def cancel(self): pass
    def close(self): pass


def test_a_turn_reaches_chat_service_as_a_turn_context(monkeypatch):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    requests, captures, submitted = [], [], []
    provider = BaseProvider()
    provider.generate = lambda messages, **kwargs: requests.append(messages) or ModelResponse(ChatMessage('assistant', 'Hi there.'))
    memory = SimpleNamespace(remember=lambda text, context: captures.append(context), retrieve=lambda text: '', set_foreground=lambda value: None, close=lambda: None)
    worker = SimpleNamespace(submit=lambda kind, *args, **kwargs: submitted.append(kind), generation=lambda *args, **kwargs: None,
        invalidate=lambda: None, transcript=lambda *args: None, close=lambda: None)
    chat = ChatService(provider, system_prompt='Riko', memory_store=memory, deps=ChatDeps(emotion_worker=worker))
    config = SimpleNamespace(raw={'animation': {'enabled': False}}, root=Path('.'), character_name='Riko',
        tools=SimpleNamespace(max_iterations=8), runtime=SimpleNamespace(pause_background_on_live=True), **audio_sections({}))
    session = SessionManager(config, chat, DesktopState())
    origin = {'source': 'discord', 'conversation_id': 'discord:client:dm:1', 'user_id': '1', 'channel_id': '1', 'message_id': 'd1'}
    try:
        assert session.respond('hello', 'Ana', origin=origin, speak=False).message.content == 'Hi there.'
        user = chat.history[0]
        assert (user.content, user.source, user.conversation_id) == ('Ana: hello', 'discord', 'discord:client:dm:1')
        observation = [m.content for m in requests[0] if m.role == 'system' and m.content.startswith('Current runtime observation')]
        assert len(observation) == 1 and '"source": "discord"' in observation[0]  # the session's runtime_snapshot
        assert {'runtime', 'desktop', 'perception'} <= set(captures[0])  # the memory capture records the runtime context
        assert 'output' not in submitted and submitted[:2] == ['start', 'input']  # sentences feed Julia as they play, not the stream
    finally: session.close()


def test_close_owns_the_factory_stack_or_what_the_chat_was_built_with(monkeypatch):
    stack, calls = ExitStack(), []
    stack.callback(calls.append, 'factory resources')
    owned = ChatService(SimpleNamespace(close=lambda: calls.append('provider')), system_prompt='', deps=ChatDeps(cleanup=stack))
    owned.close(); owned.close()
    assert calls == ['factory resources']  # only the stack: it owns the provider too
    built = ChatService(SimpleNamespace(close=lambda: calls.append('provider')), system_prompt='',
        tool_registry=SimpleNamespace(close=lambda: calls.append('registry')), memory_store=SimpleNamespace(close=lambda: calls.append('memory')))
    built.close()
    assert calls[1:] == ['memory', 'registry', 'provider']
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={'animation': {'enabled': False}}, root=Path('.'), character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    injected = ActionController()
    session = SessionManager(config, ChatService(BaseProvider(), system_prompt=''), DesktopState(), injected)
    own = SessionManager(config, ChatService(BaseProvider(), system_prompt=''), DesktopState())
    session.close(); own.close()
    injected.start('gesture')  # still open: the factory's ActionController is closed with the factory's stack
    with pytest.raises(RuntimeError, match='closed'): own.actions.start('gesture')  # the session's own is closed with it
    injected.close()


def test_desktop_tools_act_through_the_services_they_are_given(tmp_path):
    services = DesktopServices(DesktopState(), media_resolver=lambda path: tmp_path / path)
    tools = iter_tools(services)
    assert all(tool.services is services and tool.state is services.state for tool in tools)
    bare = EffectTool()  # no services: the shared state, and nothing to resolve media with
    assert bare.state is get_desktop_state() and bare.services.media_resolver is None
    with pytest.raises(RuntimeError, match='Media resolver unavailable'): bare.execute(action='play', name='rain.mp4')
