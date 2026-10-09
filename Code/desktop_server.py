"""Local HTTP/WebSocket bridge for the Electron desktop client."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictInt
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from process.app_core.integrations.discord.launcher import DiscordLauncher

from process.app_core.configuration.config import load_config
from process.app_core.factory import create_chat_service
from process.app_core.desktop.state import get_desktop_state
from process.app_core.events.bus import event_bus
from process.app_core.runtime.session import SessionManager
from process.app_core.kernel.cancellation import TurnBusy, TurnCancelled
from process.app_core.runtime.initiative import Initiative
from process.app_core.persistence.tasks import TaskConflict
from process.app_core.events.stream import stream_events
from process.app_core.persistence.conversation_store import ConversationStore
from process.app_core.persistence.atomic import atomic_write
from process.app_core.kernel.lifecycle import close_bounded, run_bounded
from process.app_core.desktop.media import resolve_media
from fastapi.responses import FileResponse

config = load_config(recover='setup')  # a broken section leaves Settings up to repair it (lifespan)
chat = None
state = get_desktop_state()
session = None
conversation_store = None
_desktop_settings_loaded = False
logger = logging.getLogger(__name__)
startup_error = ''
discord_launcher = DiscordLauncher(config.root, state)
resource_events = None
task_file_events = None
from process.app_core.resources.gpu_memory import GPUMonitor
gpu_monitor = GPUMonitor()
from process.app_core.desktop.whiteboard_image import WhiteboardImages, board_revision
board_images = WhiteboardImages()

def settings_store():
    import os
    from process.app_core.configuration.settings_store import SettingsStore
    # The config this backend started with, so Settings can say which saved settings still wait for a restart.
    return SettingsStore(os.environ.get('RIKO_CONFIG', str(config.root / 'character_config.yaml')), running=config)

def _state_event(event_type, value):
    # Publish a complete state after every mutation so clients never need to
    # reconstruct nested whiteboard/tool objects from partial events.
    current = snapshot()
    event_bus.publish("state.snapshot", **current)
    if event_type.startswith('whiteboard') or event_type == 'surface_result': board_images.observe(current, event_bus)


def _action_event(event):
    if not event.type.startswith("action."): return
    action = event.payload.get("action")
    if not action: return
    with state._lock:
        state.actions = [action, *[item for item in state.actions if item.get("id") != action.get("id")][:19]]
    state._emit("actions", state.actions)


@asynccontextmanager
async def lifespan(_app):
    global chat, session, conversation_store, _desktop_settings_loaded, startup_error, resource_events, task_file_events
    # Importing the ASGI module must not start model/memory/audio workers.
    startup_error = ''
    session = None
    try:
        # A section load_config could not read (recover='setup') would otherwise run with its defaults: repair it first.
        if config.load_errors: raise ValueError('; '.join(config.load_errors))
        chat = await run_in_threadpool(create_chat_service, config)
        # load_config has checked the voice, speech and GPT-SoVITS values; whatever else fails building the session
        # (after the model has loaded) takes the same fallback, so Settings can repair it instead of every launch failing.
        try: session = SessionManager(config, chat, state, chat.deps.action_controller)
        except BaseException:
            await run_in_threadpool(close_bounded, chat, 6)
            raise
    except Exception as exc:
        if not config.raw.get('desktop', {}).get('setup_on_startup_error', False): raise
        logger.exception('Backend unavailable; keeping settings available')
        startup_error = str(exc)
        chat = None
        session = None
        conversation_store = None
        unsubscribe = state.subscribe(_state_event)
        try: yield
        finally: unsubscribe()
        return
    conversation_store = None
    unsubscribers = []
    _desktop_settings_loaded = False
    saved = config.root / 'persistent_memories' / 'desktop_settings.json'
    try:
        if config.runtime.warmup and hasattr(session, 'speech'):
            from process.app_core.runtime.warmup import warm_session
            await run_in_threadpool(warm_session, session)
        conversation_store = ConversationStore(config.root / 'persistent_memories' / 'conversations.sqlite3',
            provider=config.runtime.provider, legacy=chat.conversation.snapshot())
        unsubscribe_history = event_bus.subscribe(conversation_store.observe)
        unsubscribers.append(unsubscribe_history)
        state.configure_board_store(config.root / 'persistent_memories' / 'whiteboard.json')
        unsubscribers.append(state.subscribe(_state_event))
        unsubscribers.append(event_bus.subscribe(_action_event))
        if saved.exists():
            try:
                state.update_geometry('avatar', **json.loads(saved.read_text(encoding='utf-8'))['avatar_geometry'])
                _desktop_settings_loaded = True
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning('Ignoring invalid desktop settings; original file preserved')
        session.initiative = Initiative(session)
        from process.app_core.events.resources import ResourceEvents
        resource_events = ResourceEvents(event_bus, resource_getters())
        from process.app_core.persistence.task_file_events import TaskFileEvents
        if chat.deps.task_store: task_file_events = TaskFileEvents(chat.deps.task_store, event_bus)
        event_bus.publish("runtime.ready", provider=config.runtime.provider)
        yield
    finally:
        # Electron allows the whole shutdown 15 s: run_server spends at most 1 s stopping the turn and 2 s draining
        # requests, this block at most about 8.5 s, and llama_native 2 s destroying the native context at exit.
        discord = asyncio.create_task(run_in_threadpool(discord_launcher.stop))  # a separate process; stop it meanwhile
        if task_file_events: task_file_events.close(); task_file_events = None
        if resource_events: resource_events.close(); resource_events = None
        if session: await run_in_threadpool(close_bounded, session, 8)
        elif chat: await run_in_threadpool(close_bounded, chat, 6)
        await discord
        for unsubscribe in reversed(unsubscribers): unsubscribe()
        if conversation_store: conversation_store.close()
        event_bus.publish("runtime.stopped")

def stop_turn():
    """run_server calls this before uvicorn drains requests, so an in-flight chat returns now instead of generating
    through the drain. Bounded: a stuck cancel must not hold up the rest of shutdown."""
    if session is not None and session.is_open: run_bounded(session.cancel, 1, 'SessionManager.cancel')

_dev = os.environ.get('RIKO_DEV') == '1'  # The API map is published only while developing.
app = FastAPI(title="Riko Local Desktop Runtime", lifespan=lifespan,
    docs_url='/docs' if _dev else None, redoc_url='/redoc' if _dev else None, openapi_url='/openapi.json' if _dev else None)

_secrets = {}
def api_secrets():
    """(token, confirmation key) for this start, minted once (see app_core/desktop/api_guard.py).
    run_server mints them right after binding the port; tests mint them on first use."""
    from process.app_core.desktop.api_guard import issue_secrets
    if config.root not in _secrets: _secrets[config.root] = issue_secrets(config.root)
    return _secrets[config.root]

def api_token(): return api_secrets()[0]
discord_launcher.token = api_token  # The Discord worker it starts receives this start's token.

from process.app_core.desktop.api_guard import Confirmations
confirmations = Confirmations()

def require_confirmation(action, changes, provided):
    """Security-sensitive changes need a single-use signature that Electron main makes with the
    confirmation key, which never leaves the machine's files, after the user approves a dialog."""
    if not changes or confirmations.consume(api_secrets()[1], action, changes, provided): return
    raise HTTPException(428, {'detail': 'Confirm this change in the Riko window', 'confirm': confirmations.challenge(action, changes)})

# Settings that load code, start programs, choose models or network endpoints, move user data, widen
# file access, or decide what runs unattended. Classified by name, so settings added later with these
# names are covered; any other value that is a URL or an absolute path is treated the same way.
SENSITIVE_SETTING_NAMES = {'native_library', 'mcp_config', 'executable', 'arguments', 'auto_start', 'api_key', 'url',
    'provider', 'model_id', 'embedding_model', 'asr_model', 'require_approval', 'system1_enabled'}
LOCATION = re.compile(r'\s*([a-z][a-z0-9+.-]*://|[/\\~]|[a-z]:[/\\])', re.IGNORECASE)
def security_sensitive(key, value=None):
    # An empty mapping in the YAML is one editable value, so a whole section can arrive at once: check each leaf.
    if isinstance(value, dict): return security_sensitive(key) or any(security_sensitive(f'{key}.{name}', item) for name, item in value.items())
    if isinstance(value, (list, tuple)): return security_sensitive(key) or any(security_sensitive(key, item) for item in value)
    section, _, name = key.rpartition('.')
    if (name in SENSITIVE_SETTING_NAMES or name.startswith('hf_') or key == 'emotion.enabled'
            or name.endswith(('_path', '_dir', '_directory', '_folder', '_root', '_file', '_url', '_model_id', '_library', '_config', '_executable'))):
        return True
    return isinstance(value, str) and key != 'avatar.model' and bool(LOCATION.match(value))  # the avatar endpoint serves only .vrm files
from process.app_core.integrations.discord.api import create_router as discord_router, client_event_filter
app.include_router(discord_router(lambda: session, lambda: discord_launcher))

@app.get('/api/discord/process')
def discord_process(): return discord_launcher.status()

@app.post('/api/discord/start')
def start_discord():
    if session is None or not session.is_open: raise HTTPException(503, 'Start the companion runtime before Discord')
    try: return discord_launcher.start()
    except (ValueError, RuntimeError) as exc: raise HTTPException(400, str(exc)) from exc

@app.post('/api/discord/stop-client')
def stop_discord():
    discord_launcher.stop()
    return discord_launcher.status()

@app.get('/api/discord/settings')
def discord_settings():
    try: return discord_launcher.access.read()
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc

class DiscordSettingsRequest(BaseModel):
    values: dict
    revision: str

@app.put('/api/discord/settings')
def save_discord_settings(request: DiscordSettingsRequest, x_riko_confirmation: str | None = Header(default=None)):
    # Allow-lists decide who can drive the companion from Discord: changes need confirmation.
    try: current = discord_launcher.access.read()['values']
    except ValueError: current = {}
    require_confirmation('discord_access', {key: value for key, value in request.values.items() if current.get(key) != value}, x_riko_confirmation)
    try: return discord_launcher.configure(request.values, request.revision)
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(409, str(exc)) from exc

@app.get('/api/discord/inbox')
def discord_inbox(): return discord_launcher.inbox_snapshot()

def probe_host():
    """The provider's ProbeHost, or None (no chat service yet, or a provider without hidden-state capture)."""
    return chat.provider.probe_host if chat is not None else None

def current_probe():
    host=probe_host()
    probe=host.probe if host else None
    if probe is None: raise HTTPException(409,host.probe_error.rstrip('.')+'; fix the cause and restart Python' if host and host.probe_error
        else 'Enable the expression probe and a compatible in-process native library, then restart Python')
    return probe

@app.get('/api/neural/status')
def neural_status():
    host=probe_host()
    probe=host.probe if host else None
    error=host.probe_error if host else ''  # the probe failed to start; chat runs without it
    return {'available':probe is not None, **(probe.status() if probe else {'mode':'unavailable','samples':0}), 'probe_error':error,
        'note':error.rstrip('.')+'. Chat works without it; fix the cause and restart Python.' if error else 'Requires hidden-state capture from the compatible in-process riko-native library.'}

@app.post('/api/neural/train')
def neural_train():
    try: return current_probe().request_training()
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@app.get('/api/neural/data')
def neural_data(offset:int=0,limit:int=40,group:str|None=None):
    if offset<0 or not 1<=limit<=100: raise HTTPException(400,'Invalid page')
    return current_probe().data_page(offset,limit,group)

@app.get('/api/neural/groups')
def neural_groups(offset:int=0,limit:int=20):
    if offset<0 or not 1<=limit<=100: raise HTTPException(400,'Invalid page')
    return current_probe().data_groups(offset,limit)

class NeuralEdit(BaseModel):
    values:dict
    revision:str

@app.patch('/api/neural/data/{sample_id}')
def neural_edit(sample_id:str,request:NeuralEdit):
    try: return current_probe().edit_sample(sample_id,request.values,request.revision)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc

@app.get('/api/neural/corpora')
def neural_corpora():
    from process.app_core.emotion.probe_storage import corpus_files
    results=[]
    for file in corpus_files(config.root):
        try:
            values=json.loads(file.read_text(encoding='utf-8'))
            results.append({'key':file.parent.name,'examples':len(values.get('examples',[])),
                'model':file.parents[1].name if file.parents[2].name=='expression' else file.parents[2].name if file.parent.parent.name=='expression probe' else 'Legacy dataset'})
        except (OSError,ValueError): continue
    return {'corpora':results}

@app.post('/api/neural/replay/{key}')
def neural_replay(key:str):
    if len(key)!=64 or any(c not in '0123456789abcdef' for c in key): raise HTTPException(400,'Invalid model key')
    probe=current_probe()
    from process.app_core.emotion.probe_storage import corpus_files
    path=next((file for file in corpus_files(config.root) if file.parent.name==key),None)
    if path is None: raise HTTPException(404,'Retained examples not found')
    try: examples=json.loads(path.read_text(encoding='utf-8'))['examples']
    except (OSError,ValueError,KeyError,TypeError): raise HTTPException(404,'Retained examples not found')
    if not isinstance(examples,list) or not examples: raise HTTPException(400,'No retained text is available')
    try: probe.replay(examples,probe_host().replay_lane)  # at background priority on the capture slot: a live turn preempts it
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc
    return {'queued':True,'examples':len(examples),'note':'Old text is replayed; new activations and labels are collected. Old weights are not reused.'}

@app.websocket('/ws/discord/client')
async def discord_client(websocket: WebSocket, instance: str):
    # attach/detach publish state that reaches session locks: run them off the event loop.
    try: await run_in_threadpool(discord_launcher.attach, instance)
    except (ValueError, TypeError):
        await websocket.accept(); await websocket.close(code=1008, reason='A Discord client is already connected or the instance ID is invalid'); return
    async def report(value): await run_in_threadpool(discord_launcher.report, instance, value)
    def resources():
        values = resource_events.snapshot() if resource_events else {'discord': discord_launcher.status()}
        return {key: values[key] for key in ('discord', 'approvals') if key in values}
    try:
        # Only what bot.py acts on (client_event_filter): local transcripts, chats and desktop state stay local.
        await stream_events(websocket, event_bus, lambda: {}, initial=resources, event_filter=client_event_filter(), on_message=report)
    finally:
        # Shielded: detach must still free the client slot if this handler is being cancelled.
        await asyncio.shield(run_in_threadpool(discord_launcher.detach, instance))

@app.get('/api/resources/gpu')
def gpu_status():
    return gpu_monitor.sample(getattr(chat, 'provider', None))

def approval_registry():
    registry = getattr(chat, 'tool_registry', None)
    if not registry or not getattr(registry, 'approvals', None): raise HTTPException(503, 'Tool approvals unavailable')
    return registry

@app.get('/api/tools/approvals')
def tool_approvals():
    registry = approval_registry()
    # Rules belong to (source, name) (tools/approval.py); policy and tools name them as the model calls them.
    return {**registry.approvals.snapshot(registry.tools), 'tools': [{'name': t.name, 'description': t.description, 'source': t.source} for t in list(registry.tools.values())]}

@app.put('/api/tools/approvals')
def approval_policy(request: dict, x_riko_confirmation: str | None = Header(default=None)):
    registry = approval_registry()
    # Switching approval off lets the model run that tool unattended: confirm it.
    policy, current = request.get('policy'), registry.approvals.snapshot(registry.tools)
    if isinstance(policy, dict):
        require_confirmation('tool_approvals', {name: False for name, value in policy.items()
            if value is False and current['policy'].get(name, current['default_required'])}, x_riko_confirmation)
    try: return registry.approvals.configure(policy, registry.tools)
    except ValueError as exc: raise HTTPException(400, str(exc))
    except OSError as exc: raise HTTPException(409, f'Could not save tool permissions: {exc}')

@app.post('/api/tools/approvals/{request_id}')
def resolve_approval(request_id: str, request: dict):
    try: approval_registry().approvals.resolve(request_id, request.get('approved'))
    except ValueError as exc: raise HTTPException(409, str(exc))
    return {'ok': True}

class ElectronProcesses(BaseModel):
    processes: list[dict]

@app.post('/api/resources/electron')
def electron_processes(request: ElectronProcesses):
    try: gpu_monitor.register_electron(request.processes)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class ResourceEstimate(BaseModel):
    changes: dict = Field(default_factory=dict)

@app.post('/api/resources/estimate')
def resource_estimate(request: ResourceEstimate):
    import os
    import tempfile
    from pathlib import Path
    from process.app_core.resources.vram_estimate import estimate
    from process.app_core.configuration.settings_store import LOCK
    store = settings_store()
    with LOCK:
        _, output, errors = store.prepare(request.changes)
        # A prompt+output mismatch does not prevent estimating the selected KV
        # allocation. Show feedback without silently changing either parameter;
        # normal settings save still rejects this draft.
        budget_only = bool(errors) and set(errors) == {'runtime.n_ctx'} and errors['runtime.n_ctx'].startswith('Context must fit')
        if errors and not budget_only: raise HTTPException(400, 'Invalid estimate settings: ' + '; '.join(f'{key}: {value}' for key, value in errors.items()))
        descriptor, temporary = tempfile.mkstemp(prefix='.resource-estimate-', suffix='.yaml', dir=store.path.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as file: file.write(output)
            candidate = load_config(temporary)
        finally: Path(temporary).unlink(missing_ok=True)
    initiative = getattr(session, 'initiative', None)
    if 'initiative.context_window_tokens' in request.changes:
        candidate.runtime.initiative_n_ctx = request.changes['initiative.context_window_tokens']
    projected = estimate(candidate, gpu_monitor.sample(getattr(chat, 'provider', None)))
    if errors: projected.setdefault('warnings', []).extend('Unsavable draft: ' + value for value in errors.values())
    return {'estimate': projected, 'validation_errors':errors, 'draft': bool(request.changes)}

class ChatRequest(BaseModel):
    text: str
    user_name: str = "User"

def snapshot():
    current = state.snapshot()
    current['whiteboard_revision'] = board_revision(current)
    current["avatar"] = config.avatar
    current["runtime"] = session.runtime_snapshot()['runtime'] if session else {'generating': False, 'ready': False}
    animation = getattr(session, 'animation', None)
    current['animation'] = animation.status() if animation else {'enabled': False, 'error': getattr(session, 'animation_error', '')}
    current['startup_error'] = startup_error
    current['character_name'] = config.character_name
    current['session_id'] = conversation_store.session_id if conversation_store else None
    return current

@app.get("/api/status")
def status(): return snapshot()

@app.middleware('http')
async def require_runtime(request, call_next):
    if startup_error and request.url.path.startswith('/api/') and not request.url.path.startswith(('/api/settings', '/api/resources', '/api/status', '/api/displays', '/api/media', '/api/avatar/model')):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': 'Runtime unavailable. Fix Settings and restart Python.', 'error': startup_error}, status_code=503)
    return await call_next(request)

@app.get('/api/settings')
def get_settings(): return settings_store().snapshot()

class SettingsPatch(BaseModel):
    changes: dict
    revision: str = ''

@app.post('/api/settings/validate')
def validate_settings(request: SettingsPatch):
    try: return settings_store().validate(request.changes)
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

@app.put('/api/settings')
def save_settings(request: SettingsPatch, x_riko_confirmation: str | None = Header(default=None)):
    from process.app_core.configuration.settings_store import SettingsConflict
    try: current = settings_store().snapshot()['values']
    except (OSError, ValueError, TypeError): current = {}
    require_confirmation('settings', {key: value for key, value in request.changes.items()
        if security_sensitive(key, value) and current.get(key) != value}, x_riko_confirmation)
    try:
        result = settings_store().save(request.changes, request.revision)
        if result.get('saved'):
            from process.app_core.configuration.schema import live_hooks
            for name, paths in live_hooks(request.changes).items(): LIVE[name](result['values'], paths)
        return result
    except SettingsConflict as exc: raise HTTPException(409, str(exc))
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))
    except OSError as exc: raise HTTPException(500, 'Could not write YAML. Check file permissions or close another editor holding the file.') from exc

# Saved settings that apply without a restart, by the hook their schema entry names (Setting.live, restart scope 'none'
# unless noted): each takes the saved values and the changed paths.
def live_probe_interval(values, paths):
    interval = values['emotion.probe.interval_tokens']
    config.emotion.probe['interval_tokens'] = interval
    host = probe_host()
    if host: host.set_probe_interval(interval)

def live_background_pause(values, paths):
    enabled = values['runtime.pause_background_on_live']
    config.runtime.pause_background_on_live = enabled
    if chat is not None: chat.provider.set_pause_background(enabled)
    memory = getattr(chat, 'memory_store', None)
    if memory: memory.set_foreground(bool(enabled and session is not None and session.status().generating))

def live_avatar(values, paths):
    from copy import deepcopy
    from process.app_core.configuration.settings_store import LOCK
    with LOCK:
        config.avatar = deepcopy(load_config(settings_store().path).avatar)
        config.raw['avatar'] = deepcopy(config.avatar)
    event_bus.publish('state.snapshot', **snapshot())

def live_initiative_budgets(values, paths):  # the running initiative takes them; resizing the KV pool still needs a restart
    initiative = getattr(session, 'initiative', None)
    if initiative: initiative.update({key.split('.')[-1]: values[key] for key in ('initiative.context_window_tokens', 'initiative.max_output_tokens')})

LIVE = {'probe_interval': live_probe_interval, 'background_pause': live_background_pause, 'avatar': live_avatar,
    'initiative_budgets': live_initiative_budgets}

class PathCheck(BaseModel):
    value: str

@app.post('/api/settings/path')
def check_settings_path(request: PathCheck):
    try: return settings_store().check_path(request.value)
    except (ValueError, OSError) as exc: raise HTTPException(400, str(exc))

@app.get('/api/settings/huggingface/search')
def search_huggingface(query: str = '', gguf: bool = True):
    if len(query) > 200: raise HTTPException(400, 'Search is too long')
    import httpx
    params = {'search': query, 'limit': 12, 'sort': 'downloads', 'direction': -1}
    if gguf: params['filter'] = 'gguf'
    try:
        response = httpx.get('https://huggingface.co/api/models', params=params, timeout=8)
        response.raise_for_status()
        return {'models': [{'id': m['id'], 'downloads': m.get('downloads', 0)} for m in response.json()]}
    except (httpx.HTTPError, ValueError) as exc: raise HTTPException(502, 'Hugging Face search unavailable; enter an ID manually') from exc

@app.get('/api/settings/huggingface/files')
def huggingface_files(repo: str, revision: str = 'main'):
    import re
    import httpx
    from urllib.parse import quote
    if len(repo)>200 or '..' in repo or not re.fullmatch(r'[\w-]+/[\w.-]+', repo) or not revision or len(revision) > 200: raise HTTPException(400, 'Use owner/repository and a valid revision')
    try:
        response = httpx.get(f'https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe="")}', timeout=8)
        response.raise_for_status()
        data = response.json()
        return {'revision': data.get('sha'), 'files': [x['rfilename'] for x in data.get('siblings', []) if x['rfilename'].lower().endswith('.gguf') and 'mmproj' not in x['rfilename'].lower()],
            'context_length': data.get('gguf', {}).get('context_length')}
    except (httpx.HTTPError, ValueError) as exc: raise HTTPException(502, 'Cannot list that repository/revision; manual filenames remain available') from exc

@app.get('/api/chat/history')
def chat_history(before: int | None = None, limit: int = 40):
    if before is not None and before < 1 or not 1 <= limit <= 100: raise HTTPException(400, 'Invalid history page')
    if conversation_store is None: raise HTTPException(503, 'History store is not ready')
    return conversation_store.page(before, limit)

@app.get('/api/media')
def media(path: str):
    try: asset = resolve_media(config.root, path, config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens'))
    except ValueError as exc: raise HTTPException(400, str(exc))
    return FileResponse(asset)

@app.get('/api/whiteboard/image')
def whiteboard_image():
    from fastapi.responses import Response
    return Response(board_images.image(state.snapshot(), config.root), media_type='image/png', headers={'Cache-Control': 'no-store'})

@app.post('/api/whiteboard/image')
async def whiteboard_capture(request: Request, revision: str):
    if request.headers.get('content-type', '').split(';')[0] != 'image/png': raise HTTPException(415, 'Send a PNG capture of the whiteboard only')
    png = bytearray()
    async for chunk in request.stream():
        if len(png) + len(chunk) > 8 * 1024 * 1024: raise HTTPException(413, 'Board image exceeds 8 MiB')
        png.extend(chunk)
    # PIL verification, hashing and the publish run off the event loop.
    try: accepted = await run_in_threadpool(lambda: board_images.capture(revision, bytes(png), state.snapshot(), event_bus))
    except (ValueError, OSError) as exc: raise HTTPException(400, 'Invalid board image') from exc
    return {'accepted': accepted}

@app.get('/api/avatar/model')
def avatar_model():
    from pathlib import Path
    asset = Path(config.avatar.get('model', 'character_files/Mita.vrm')).expanduser()
    asset = (asset if asset.is_absolute() else config.root / asset).resolve()
    # Only the explicitly configured VRM, not arbitrary renderer-supplied paths.
    if asset.suffix.lower() != '.vrm' or not asset.is_file(): raise HTTPException(404, 'Configured VRM model not found')
    return FileResponse(asset, media_type='model/gltf-binary', headers={'Cache-Control': 'no-store'})

@app.get('/api/avatar/models')
def avatar_models():
    from process.app_core.desktop.avatar_models import AvatarModels
    return AvatarModels(config.root).listing()

@app.post('/api/avatar/models/import')
def import_avatar_model(request: PathCheck):
    from process.app_core.desktop.avatar_models import AvatarModels
    try: result = AvatarModels(config.root).import_model(request.value)
    except (ValueError, OSError) as exc: raise HTTPException(400, str(exc)) from exc
    event_bus.publish('avatar.models_changed')
    return result

@app.get('/api/avatar/animation')
def avatar_animation(path: str):
    try:
        asset = resolve_media(config.root, path, config.raw.get('desktop', {}).get('effects_directory', 'effects/greenscreens'), extensions={'.vrma'})
    except ValueError as exc: raise HTTPException(400, str(exc))
    if asset.stat().st_size > 32 * 1024 * 1024: raise HTTPException(400, 'Animation asset exceeds 32 MiB')
    return FileResponse(asset, media_type='model/gltf-binary')

class AnimationResult(BaseModel):
    action_id: str
    status: str
    error: str = ''

@app.post('/api/avatar/animation/result')
def animation_result(result: AnimationResult):
    if session is None: raise HTTPException(503, 'Runtime is not ready')
    action = next((item for item in session.actions.active() if item['id'] == result.action_id and (item['kind'] == 'wake_animation' or item['kind'].startswith('motion.'))), None)
    if action is None: raise HTTPException(404, 'Animation action expired or replaced')
    if result.status not in {'started', 'completed', 'cancelled', 'error'}: raise HTTPException(400, 'Invalid animation status')
    if result.status == 'completed': session.actions.complete(result.action_id)
    elif result.status in {'error', 'cancelled'}:
        session.actions.cancel(result.action_id)
        if result.status == 'error' and getattr(session, 'animation', None): session.animation.renderer_error(action, result.error)
    event_bus.publish('wake.feedback.error' if result.status == 'error' else 'wake.feedback.animation',
        action_id=result.action_id, asset='animation', status=result.status, error=result.error[:1000])
    return {'ok': True}

def animation_service():
    service = getattr(session, 'animation', None)
    if service is None: raise HTTPException(503, 'Animation service is disabled or unavailable')
    return service

@app.get('/api/animation')
def animation_status():
    service = animation_service()
    return {**service.status(), 'entries': service.library.list()}

class AnimationImport(BaseModel):
    path: str = Field(max_length=2048)
    metadata: dict = Field(default_factory=dict)

@app.post('/api/animation/import')
def animation_import(request: AnimationImport):
    service = animation_service()
    try: entry = service.library.import_file(request.path, request.metadata)
    except (ValueError, OSError, TypeError) as exc: raise HTTPException(400, str(exc))
    service.invalidate_library()
    return entry

@app.patch('/api/animation/assets/{identifier}')
def animation_metadata(identifier: str, request: dict):
    service = animation_service()
    try: entry = service.library.update(identifier, request)
    except (ValueError, TypeError, OSError) as exc: raise HTTPException(400, str(exc))
    service.invalidate_library()
    return entry

@app.get('/api/animation/assets/{identifier}/file')
def animation_asset(identifier: str):
    try: path = animation_service().library.path(identifier)
    except ValueError as exc: raise HTTPException(404, str(exc))
    return FileResponse(path)

@app.post('/api/animation/assets/{identifier}/preview')
def animation_preview(identifier: str):
    try: action = animation_service().preview(identifier)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'action_id': action.id}

class AnimationInteraction(BaseModel):
    kind: str
    pointer: dict | None = None
    target: dict | None = None

@app.post('/api/animation/interaction')
def animation_interaction(request: AnimationInteraction):
    try: animation_service().interact(request.kind, request.pointer, request.target)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class AnimationCapabilities(BaseModel):
    bones: list[str]
    expressions: list[str]

@app.post('/api/animation/capabilities')
def animation_capabilities(request: AnimationCapabilities):
    try: animation_service().report_capabilities(request.bones, request.expressions)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'ok': True}

class WalkDestination(BaseModel):
    x: StrictInt
    y: StrictInt

@app.post('/api/animation/walk')
def animation_walk(request: WalkDestination):
    try: action = animation_service().walk_to(request.x, request.y)
    except ValueError as exc: raise HTTPException(400, str(exc))
    return {'action_id': action.id}

@app.post('/api/animation/stop')
def animation_stop():
    service = animation_service()
    service.stop_movement()
    for action in session.actions.active():
        if action['kind'] == 'motion.preview': session.actions.cancel(action['id'])
    return {'ok': True}

class SurfaceResult(BaseModel):
    surface: str
    command_id: str
    status: str
    error: str = ''
    bounds: dict[str, float] | None = None

@app.post('/api/surfaces/result')
def surface_result(request: SurfaceResult):
    if len(request.error) > 2000: raise HTTPException(400, 'Error exceeds 2000 characters')
    try: return {'accepted': state.surface_result(**request.model_dump())}
    except ValueError as exc: raise HTTPException(400, str(exc))

class BoardSurface(BaseModel):
    visible: bool | None = None
    geometry: dict[str, StrictInt] | None = None

@app.patch('/api/surfaces/whiteboard')
def board_surface(request: BoardSurface):
    try: state.set_whiteboard_surface(**request.model_dump())
    except ValueError as exc: raise HTTPException(400, str(exc))
    return snapshot()

@app.get('/api/tasks')
def tasks(include_closed: bool = True, query: str = ''):
    return {'tasks': chat.deps.task_store.list(include_closed=include_closed, query=query, limit=100)}

class DisplayBounds(BaseModel):
    x: StrictInt
    y: StrictInt
    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)

class Display(BaseModel):
    index: StrictInt = Field(ge=0)
    id: StrictInt
    label: str = Field(max_length=512)
    primary: bool
    bounds: DisplayBounds
    scaleFactor: float = Field(gt=0, allow_inf_nan=False)

@app.post('/api/displays')
def displays(request: list[Display]):
    indices = [item.index for item in request]
    if not request or len(request) > 32 or indices != list(range(len(request))) or sum(item.primary for item in request) != 1:
        raise HTTPException(400, 'Provide ordered display indices and exactly one primary display')
    values = [item.model_dump() for item in request]
    with state._lock:
        if (not state.displays and not _desktop_settings_loaded) or state.avatar_geometry['screen'] not in indices:
            state.avatar_geometry['screen'] = next(item.index for item in request if item.primary)
        state.displays = values
    state._emit('displays', values)
    return {'accepted': True}


desktop_settings_write = threading.Lock()  # avatar geometry PATCHes still write in order

@app.patch('/api/surfaces/avatar')
def avatar_surface(request: dict[str, StrictInt]):
    try: state.update_geometry('avatar', **request)
    except ValueError as exc: raise HTTPException(400, str(exc))
    # Write outside DesktopState's lock (bus listeners and turns take it): a durable write can take a while.
    with desktop_settings_write:
        with state._lock: payload = json.dumps({'avatar_geometry': state.avatar_geometry})
        atomic_write(config.root / 'persistent_memories' / 'desktop_settings.json', payload)
    return snapshot()

@app.get('/api/tasks/{task_id}')
def get_task(task_id: str):
    try: return chat.deps.task_store.get(task_id)
    except KeyError: raise HTTPException(404, 'Task not found')

class TaskCreateRequest(BaseModel):
    title: str
    description: str = ''
    next_step: str = ''

@app.post('/api/tasks')
def create_task(request: TaskCreateRequest):
    try: return chat.deps.task_store.create(**request.model_dump(), actor='user_api', reason='Explicit user creation')
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

class TaskUpdateRequest(BaseModel):
    expected_revision: int
    changes: dict
    reason: str = 'Explicit user correction'

@app.patch('/api/tasks/{task_id}')
def update_task(task_id: str, request: TaskUpdateRequest):
    try: return chat.deps.task_store.update(task_id, **request.model_dump(), actor='user_api')
    except TaskConflict as exc: raise HTTPException(409, str(exc))
    except KeyError: raise HTTPException(404, 'Task not found')
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

@app.get('/api/initiative')
def initiative_status(): return session.initiative.snapshot()

@app.put('/api/initiative')
def initiative_settings(request: dict):
    try: return session.initiative.update(request)
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

class CustomEventRequest(BaseModel):
    event: str = 'user.custom'
    context: str = ''

@app.post('/api/initiative/event')
def custom_event(request: CustomEventRequest):
    if request.event not in session.initiative.triggers or not request.event.startswith('user.'):
        raise HTTPException(400, 'Only registered user.* events can be published here')
    if len(request.context) > 2000: raise HTTPException(400, 'Context exceeds 2000 characters')
    event = event_bus.publish(request.event, context=request.context)
    return {'event_id': event.id}

@app.get("/api/memories")
def memories():
    return {"records": chat.memory_store.list_records(), "pipeline": chat.memory_store.status()}

class MemoryUpdate(BaseModel):
    text: str | None = None
    memory_type: str | None = None
    importance: float | None = None
    tags: list[str] | None = None
    active: bool | None = None

@app.patch("/api/memories/{record_id}")
def update_memory(record_id: str, request: MemoryUpdate):
    try:
        return chat.memory_store.update(record_id, **request.model_dump(exclude_unset=True))
    except KeyError: raise HTTPException(404, "Memory not found")
    except (ValueError, TypeError) as exc: raise HTTPException(400, str(exc))

@app.delete("/api/memories/{record_id}")
def delete_memory(record_id: str):
    try: chat.memory_store.delete(record_id)
    except KeyError: raise HTTPException(404, "Memory not found")
    return {"deleted": record_id}

BUSY = 'Riko is still handling another reply; send again when it finishes'

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    # Plain def: FastAPI runs it in the thread pool, so the state/snapshot calls below
    # (which take session locks) never block the event loop.
    try:
        response = session.respond(request.text, request.user_name)
    except TurnCancelled:
        return {"cancelled": True}
    except TurnBusy as exc:
        # Another turn (voice, Discord, a reply still stopping) holds the session: a conflict, not a server failure.
        raise HTTPException(409, BUSY) from exc
    state.set_speech(response.message.content)
    return {"text": response.message.content, "emotion": snapshot()["emotion"]}

@app.post("/api/chat/stop")
def stop_chat():
    session.cancel()
    return {"stopping": True}

class VoiceActivityRequest(BaseModel):
    speech_seconds: float

class InterjectionRequest(BaseModel):
    text: str
    started_at: float
    ended_at: float

@app.post("/api/voice/activity")
def voice_activity(request: VoiceActivityRequest):
    session.voice_activity(request.speech_seconds)
    return {"received": True}

@app.post("/api/voice/interjection")
def voice_interjection(request: InterjectionRequest):
    # 'fresh' words belong to no reply in flight: the client sends them as a chat turn instead. 'reply' words cut the
    # reply in flight and are already in history, so the reply is redone here, as VoiceInput does.
    anchor = session.voice_anchor()
    disposition = session.voice_transcript(request.text, request.started_at, request.ended_at, anchor) if anchor else 'fresh'
    if disposition == 'reply':
        def redo():
            try: session.respond(request.text, record_user=False, reply_to=anchor[0], wait=lambda: False, origin={'source': 'microphone'})
            except TurnCancelled: pass
            except Exception: logger.exception('Could not redo the reply an interjection cut')
        threading.Thread(target=redo, daemon=True, name='interjection-redo').start()
    return {"accepted": disposition in {'preserved', 'reply'}}

@app.post("/api/mic/toggle")
def toggle_mic():
    enabled = state.toggle_mic()
    if enabled: session.start_listening()
    else: session.stop_listening()
    return {"enabled": enabled}

@app.post("/api/audio/toggle")
def toggle_audio(): return {"enabled": state.toggle_audio()}

class AudioVolumeRequest(BaseModel):
    volume: float = Field(ge=0, le=1, strict=True, allow_inf_nan=False)

@app.patch('/api/audio/volume')
def set_audio_volume(request: AudioVolumeRequest):
    try: return {'volume': state.set_audio_volume(request.volume), 'enabled': state.audio_enabled}
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc

@app.post("/api/sleep/toggle")
def toggle_sleep():
    enabled = not state.sleep_mode
    state.set_sleep(enabled)
    if enabled: session.cancel()
    return {"sleep": enabled}

@app.websocket("/ws/events")
async def events(websocket: WebSocket):
    await stream_events(websocket, event_bus, snapshot,
        initial=lambda: resource_events.snapshot() if resource_events else {})

def resource_getters():
    return {'approvals': tool_approvals, 'initiative': initiative_status,
        'animation': animation_status, 'voice': voice_status,
        'avatar_models': avatar_models, 'discord': discord_launcher.status,
        'tasks': lambda: {'tasks': chat.deps.task_store.list(include_closed=True)}}

@app.websocket('/ws/resources/gpu')
async def gpu_events(websocket: WebSocket):
    unsubscribe = await run_in_threadpool(gpu_monitor.subscribe, lambda: getattr(chat, 'provider', None))
    try:
        await stream_events(websocket, event_bus, lambda: {},
            initial=lambda: {'gpu': gpu_monitor.sample(getattr(chat, 'provider', None))},
            event_filter=lambda event: event.type == 'resource.gpu')
    finally: await run_in_threadpool(unsubscribe)

@app.post("/api/voice/start")
def start_voice():
    if not state.mic_enabled: state.toggle_mic()
    session.start_listening()
    return {"starting": True}

@app.post("/api/voice/stop")
def stop_voice():
    if state.mic_enabled: state.toggle_mic()
    session.stop_listening()
    return {"stopped": True}

@app.get("/api/voice/status")
def voice_status():
    runtime = session.runtime_snapshot()['runtime']
    return {'listening': runtime['listening'], 'capture_running': session.voice is not None and not session.voice.closed.is_set(),
            'microphone_status': runtime['microphone_status'], 'phase': runtime.get('voice_phase','stopped'),
            'latest_transcript': runtime['latest_transcript'], 'wake': runtime['wake']}

@app.post("/api/voice/activate")
def activate_voice():
    if not state.mic_enabled: state.toggle_mic()
    session.start_listening()
    session.wake.activate()
    return {"active": True}

class CalibrationRequest(BaseModel):
    action: str
    threshold: float | None = None

@app.post("/api/voice/calibration")
def voice_calibration(request: CalibrationRequest):
    from fastapi import HTTPException
    operations = {"begin": session.wake.begin_calibration, "record": session.wake.record,
                  "save": session.wake.finish_calibration, "cancel": session.wake.cancel_calibration,
                  "test": lambda: session.wake.set_testing(True),
                  "stop_test": lambda: session.wake.set_testing(False),
                  "threshold": lambda: session.wake.set_threshold(request.threshold)}
    try:
        if request.action not in operations: raise ValueError("Unknown calibration operation")
        # Also between sentences of a reply: Riko's own voice must never become a wake-word sample.
        if request.action in {"begin", "test"} and session.status().replying():
            raise ValueError("Stop the current reply before calibration or testing")
        operations[request.action]()
        return session.wake.status()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/voice/devices")
def voice_devices():
    import sounddevice as sd
    return {"devices": [{"index": index, "name": device["name"]}
                        for index, device in enumerate(sd.query_devices()) if device["max_input_channels"] > 0]}

@app.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            request = await websocket.receive_json()
            text = str(request.get("text", "")).strip()
            if not text: continue
            await websocket.send_json({"type": "chat.started"})
            # Tool-enabled turns use the reliable completion path; this endpoint
            # is reserved for providers that support token streaming.
            try: response = await run_in_threadpool(session.respond, text, request.get("user_name", "User"))
            except TurnBusy:  # the socket's 409: say so and keep the connection for the next message
                await websocket.send_json({"type": "chat.busy", "payload": {"status": 409, "detail": BUSY}}); continue
            except TurnCancelled:
                await websocket.send_json({"type": "chat.cancelled", "payload": {}}); continue
            await websocket.send_json({"type": "chat.completed", "payload": {"text": response.message.content}})
    except (WebSocketDisconnect, RuntimeError):
        return

# Every route, HTTP and WebSocket, needs the install's API token and a loopback Host.
from process.app_core.desktop.api_guard import LocalAPIGuard
app.add_middleware(LocalAPIGuard, token=api_token)
# Outermost: even setup-mode 503 responses need readable CORS headers.
app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
