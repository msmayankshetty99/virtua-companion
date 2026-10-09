from __future__ import annotations

from .configuration.config import AppConfig
from .inference.providers import create_provider
from .tools.registry import ToolRegistry
from .conversation.chat import ChatDeps, ChatService
from .emotion import JuliaEmotionEngine, ExpressionActionBridge
from .emotion.worker import EmotionWorker
from .desktop.state import get_desktop_state
from .desktop.tools import DesktopServices
from .persistence.memory import MemoryStore
from .runtime.actions import ActionController
from .persistence.tasks import TaskStore, TaskMCP, TASK_RULES
from .desktop.media import resolve_media
from .desktop.effects import EffectLibrary
from contextlib import ExitStack
from .kernel.lifecycle import close_bounded


def create_chat_service(config: AppConfig) -> ChatService:
    """Build the complete application core without importing audio or UI code. The service's close() owns everything built
    here; a failure part way closes what was built so far."""
    cleanup = ExitStack()
    try: return _build_chat_service(config, cleanup)
    except BaseException:
        cleanup.close()
        raise


def _build_chat_service(config, cleanup):
    def own(resource):
        cleanup.callback(close_bounded, resource)
        return resource
    emotion_engine = emotion_worker = None
    actions = own(ActionController())
    effects_directory = config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens')
    desktop = DesktopServices(get_desktop_state(), actions=actions, effects_directory=effects_directory,
        media_resolver=lambda path: resolve_media(config.root, path, effects_directory),
        effect_library=EffectLibrary(config.root / effects_directory))
    if config.emotion.enabled:
        state = get_desktop_state()
        def on_emotion(emotion):
            state.set_emotion(emotion)
            actions.set_emotion(emotion)
        bridge = ExpressionActionBridge(on_emotion=on_emotion)
        emotion_engine = own(JuliaEmotionEngine(
            config.emotion.model_path,
            model_id=config.emotion.model_id,
            cache_dir=config.emotion.cache_dir,
            device=config.emotion.device,
            strict_encoding=config.emotion.strict_encoding,
            max_length=config.emotion.max_length,
            context_tokens=config.emotion.context_tokens,
            update_interval_tokens=config.emotion.update_interval_tokens,
            temperature=config.emotion.temperature,
            fallback=config.emotion.fallback,
            on_event=lambda event: publish_teacher(event),
        ))
        emotion_worker = own(EmotionWorker(emotion_engine))
    provider = own(create_provider(config.runtime))
    def publish_teacher(event):
        probe = getattr(provider, 'probe', None)
        if probe: probe.publish_teacher(event, bridge.update)
        else: bridge.update(event.state)
    if config.emotion.probe.get('enabled'):
        from .emotion.probe import EmotionProbe, ProbeConfig
        from .emotion.probe_storage import probe_directory, training_directory
        if not hasattr(provider, 'probe_factory'):
            raise ValueError('Selected provider does not expose hidden-state probe capture')
        probe_config = ProbeConfig.from_raw(config.emotion.probe)
        provider.set_probe_interval(probe_config.interval_tokens)
        playback_active = emotion_worker.playback_active  # the probe needs emotion.enabled (load_config), so the worker exists
        def expression(state):  # while sentences play, they drive the expression: the probe's reading waits
            if not playback_active(): bridge.update(state)
        provider.probe_factory = lambda identity, idle: EmotionProbe(
            probe_directory(config.root, config.runtime), identity, emotion_engine,
            probe_config, idle=idle, legacy_directory=config.root / 'persistent_memories' / 'emotion_probes',
            training_directory=training_directory(config.root, config.runtime),
            on_prediction=expression, on_fallback=expression)
        # Julia remains the teacher and low-confidence fallback. Student
        # events use the same expression/action bridge, never tool execution.
    if config.runtime.warmup and hasattr(provider, 'warmup'): provider.warmup()
    task_path = config.raw.get('tasks', {}).get('store_file', 'persistent_memories/tasks.sqlite3')
    task_store = TaskStore(config.root / task_path)
    registry = own(ToolRegistry.from_config(config, activity=get_desktop_state(), desktop=desktop))
    from .tools.choices import ChoiceResolver
    registry.choice_resolver = ChoiceResolver(emotion_engine.choose_tool_input if emotion_engine else None,
        enabled=config.tools.best_fit_inputs, timeout=config.tools.best_fit_timeout_seconds,
        confidence=config.tools.best_fit_min_confidence)
    task_mcp = TaskMCP(task_store)
    registry.register_mcp(task_mcp)
    memory = own(MemoryStore(config.memory, reflection_provider=getattr(provider, 'reflection', provider), start_worker=not config.runtime.warmup))
    if hasattr(provider, 'count_text_tokens'): memory.token_counter = provider.count_text_tokens
    if config.runtime.warmup:
        from .runtime.warmup import warm_core
        warm_core(memory, emotion_engine, config.runtime.startup_timeout_seconds)
        memory.start()
    return ChatService(
        provider,
        system_prompt=config.system_prompt + '\n' + TASK_RULES + '\n' + str(config.raw.get('tasks', {}).get('rules', '')),
        character_name=config.character_name,
        tool_registry=registry,
        history_file=config.memory.history_file,
        emotion_engine=emotion_engine,
        memory_store=memory,
        deps=ChatDeps(action_controller=actions, task_store=task_store, task_mcp=task_mcp,
            initiative_provider=getattr(provider, 'initiative', provider),
            context_limit=min(config.runtime.n_ctx, config.memory.context_window_tokens + config.runtime.max_output_tokens),
            emotion_worker=emotion_worker, desktop=desktop, cleanup=cleanup),
    )
