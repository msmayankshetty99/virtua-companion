from types import SimpleNamespace
import importlib
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from process.app_core.desktop.state import DesktopState


def client_for(backend, **options):
    """A client that passes the local API guard: loopback Host plus this test data root's token."""
    return TestClient(backend.app, base_url='http://127.0.0.1:8765', headers={'Authorization': 'Bearer ' + backend.api_token()}, **options)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    # Load settings with the real loader before the server-import startup stub.
    # Otherwise isolated API tests capture the no-argument stub in settings_store.
    importlib.import_module('process.app_core.configuration.settings_store')
    config = SimpleNamespace(root=tmp_path, raw={}, avatar={}, character_name='Test character', runtime=SimpleNamespace(provider='fake', warmup=False))
    monkeypatch.setattr('process.app_core.configuration.config.load_config', lambda: config)
    server = importlib.import_module('desktop_server')
    monkeypatch.setattr(server, 'config', config)
    monkeypatch.setattr(server, 'state', DesktopState())
    monkeypatch.setattr(server, 'chat', None)
    monkeypatch.setattr(server, 'session', SimpleNamespace(runtime_snapshot=lambda: {'runtime': {}}))
    monkeypatch.setattr(server, '_desktop_settings_loaded', False)
    monkeypatch.setattr(server, 'conversation_store', None)
    monkeypatch.setattr(server, 'startup_error', '')
    return server


def test_server_lifespan_owns_workers_and_removes_subscriptions(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: calls.append('create') or SimpleNamespace())
    class Session:
        def __init__(self, *args): pass
        def runtime_snapshot(self): return {'runtime': {}}
        def close(self): calls.append('close')
    monkeypatch.setattr(backend, 'SessionManager', Session)
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    before = len(backend.event_bus._listeners)
    assert calls == []
    with client_for(backend) as client:
        assert calls == ['create']
        assert client.get('/api/status').status_code == 200
        assert len(backend.state._listeners) == 1
    assert calls == ['create', 'close']
    assert backend.state._listeners == []
    assert len(backend.event_bus._listeners) == before


def test_animation_assets_and_results_are_scoped(backend, monkeypatch):
    from process.app_core.runtime.actions import ActionController
    directory = backend.config.root / 'character_files'
    directory.mkdir()
    path = directory / 'wake.vrma'
    path.write_bytes(b'test animation')
    outside = backend.config.root / 'outside.vrma'
    outside.touch()
    actions = ActionController()
    monkeypatch.setattr(backend.session, 'actions', actions, raising=False)
    client = client_for(backend)
    try:
        assert client.get('/api/avatar/animation', params={'path': 'character_files/wake.vrma'}).content == b'test animation'
        assert client.get('/api/avatar/animation', params={'path': 'outside.vrma'}).status_code == 400
        assert client.get('/api/media', params={'path': 'character_files/wake.vrma'}).status_code == 400
        action = actions.start('wake_animation', {'path': str(path)}, 10)
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'started'}).status_code == 200
        assert action.status == 'running'
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'completed'}).status_code == 200
        assert action.status == 'complete'
        assert client.post('/api/avatar/animation/result', json={'action_id': action.id, 'status': 'started'}).status_code == 404
    finally: actions.close()


def test_audio_volume_is_live_bounded_and_independent_of_mute(backend):
    client = client_for(backend)
    backend.state.toggle_audio()
    result = client.patch('/api/audio/volume', json={'volume': .35})
    assert result.status_code == 200
    assert result.json() == {'volume': .35, 'enabled': False}
    assert backend.state.snapshot()['audio_volume'] == .35
    assert not backend.state.audio_enabled
    for value in (-.1, 1.1, True, '0.5', None):
        assert client.patch('/api/audio/volume', json={'volume': value}).status_code == 422
    assert backend.state.audio_volume == .35
    assert client.post('/api/audio/toggle').json()['enabled']
    assert backend.state.audio_volume == .35


def test_discord_start_is_explicit_and_runtime_scoped(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, 'discord_launcher', SimpleNamespace(status=lambda: {'running': False, 'error': ''}, start=lambda: calls.append('start') or {'running': True, 'error': ''}))
    client = client_for(backend)
    assert client.get('/api/discord/process').json()['running'] is False
    assert calls == []
    assert client.post('/api/discord/start').json()['running'] is True
    assert calls == ['start']
    monkeypatch.setattr(backend, 'session', None)
    assert client.post('/api/discord/start').status_code == 503
    assert calls == ['start']


def test_avatar_library_import_and_settings_apply_live(backend, monkeypatch):
    from test_avatar_models import vrm_bytes
    from process.app_core.configuration.settings_store import SettingsStore, load_config
    path = backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    monkeypatch.setattr(backend, 'settings_store', lambda: store)
    monkeypatch.setattr(backend, 'load_config', load_config)
    source = backend.config.root/'source.vrm'; source.write_bytes(vrm_bytes())
    client = client_for(backend)
    imported = client.post('/api/avatar/models/import',json={'value':str(source)})
    assert imported.status_code == 200
    model = imported.json()['path']
    assert client.get('/api/avatar/models').json()['entries'][0]['path'] == model
    result = client.put('/api/settings',json={'revision':store.snapshot()['revision'], 'changes':{'avatar.model':model,'avatar.format':'auto'}})
    assert result.status_code == 200 and result.json()['saved']
    assert client.get('/api/status').json()['avatar']['model'] == model
    assert client.get('/api/avatar/model').content == source.read_bytes()
    other = backend.config.root/'second.vrm'; other.write_bytes(vrm_bytes('vrm0'))
    imported0 = client.post('/api/avatar/models/import',json={'value':str(other)}).json()['path']
    revision = store.snapshot()['revision']
    invalid = client.put('/api/settings',json={'revision':revision, 'changes':{'avatar.model':imported0,'avatar.format':'vrm1'}})
    assert not invalid.json()['saved']
    assert client.get('/api/status').json()['avatar']['model'] == model
    switched = client.put('/api/settings',json={'revision':revision, 'changes':{'avatar.model':imported0,'avatar.format':'vrm0'}})
    assert switched.json()['saved'] and not switched.json()['restart_required']
    assert client.get('/api/status').json()['avatar'] == {'model':imported0,'format':'vrm0'}
    # A query value cannot request an arbitrary local path or the old model.
    response = client.get('/api/avatar/model',params={'selection':str(source)})
    assert response.content == other.read_bytes() and response.headers['cache-control'] == 'no-store'
    assert client.post('/api/avatar/models/import',json={'value':str(path)}).status_code == 400


def test_background_pause_setting_applies_live_without_restart(backend,monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore
    path=backend.config.root/'character_config.yaml';path.write_text('runtime:\n  provider: lm_studio\n')
    store=SettingsStore(path)
    monkeypatch.setattr(backend,'settings_store',lambda:store)
    calls=[]
    provider=SimpleNamespace(set_pause_background=lambda value:calls.append(('pause',value)))
    memory=SimpleNamespace(set_foreground=lambda value:calls.append(('memory',value)))
    monkeypatch.setattr(backend,'chat',SimpleNamespace(provider=provider,memory_store=memory))
    monkeypatch.setattr(backend.session,'_generation_active',True,raising=False)
    result=client_for(backend).put('/api/settings',json={'revision':store.snapshot()['revision'],'changes':{'runtime.pause_background_on_live':False}})
    assert result.status_code==200 and result.json()['saved'] and not result.json()['restart_required']
    assert calls==[('pause',False),('memory',False)]


def test_gpu_draft_feedback_can_estimate_budget_mismatch_without_fixing_values(backend,monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore,load_config
    path=backend.config.root/'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  n_ctx: 8192\n  max_output_tokens: 1024\nmemory:\n  context_window_tokens: 7168\n')
    store=SettingsStore(path)
    before=path.read_bytes()
    monkeypatch.setattr(backend,'settings_store',lambda:store)
    monkeypatch.setattr(backend,'load_config',load_config)
    monkeypatch.setattr(backend.gpu_monitor,'sample',lambda *args:{})
    monkeypatch.setattr('process.app_core.resources.vram_estimate.estimate',lambda config,telemetry:{'n_ctx':config.runtime.n_ctx,'warnings':[]})
    client=client_for(backend)
    result=client.post('/api/resources/estimate',json={'changes':{'runtime.n_ctx':2048}})
    assert result.status_code==200 and result.json()['estimate']['n_ctx']==2048
    assert result.json()['validation_errors'] and result.json()['estimate']['warnings']
    assert not store.validate({'runtime.n_ctx':2048})['valid']
    assert path.read_bytes()==before


def test_animation_library_api_import_assignment_preview_and_interaction(backend, monkeypatch):
    import json
    import threading
    from process.app_core.runtime.actions import ActionController
    from process.app_core.animation.runtime import AnimationRuntime
    session = SimpleNamespace(config=backend.config, chat=SimpleNamespace(), state=backend.state,
        actions=ActionController(), _voice_lock=threading.RLock(), _voice_status='ready',
        _user_speaking=False, _playing=None, _generation_active=False, _speech_pending=0)
    service = AnimationRuntime(session, start=False)
    session.animation = service
    monkeypatch.setattr(backend, 'session', session)
    source = backend.config.root / 'source.pose.json'
    source.write_text(json.dumps({'version': 1, 'bones': {'head': [0, .1, 0]}}))
    client = client_for(backend)
    try:
        assert client.post('/api/animation/capabilities', json={'bones': ['head'], 'expressions': []}).status_code == 200
        response = client.post('/api/animation/import', json={'path': str(source), 'metadata': {'states': ['idle']}})
        assert response.status_code == 200
        entry = response.json()
        assert client.get(f"/api/animation/assets/{entry['id']}/file").content == source.read_bytes()
        assert client.get('/api/animation/assets/not-an-asset/file').status_code == 404
        assert client.patch(f"/api/animation/assets/{entry['id']}", json={'mask': ['tail']}).status_code == 400
        assert client.patch(f"/api/animation/assets/{entry['id']}", json={'speed': .5}).status_code == 200
        preview = client.post(f"/api/animation/assets/{entry['id']}/preview")
        assert preview.status_code == 200
        assert client.post('/api/avatar/animation/result', json={'action_id': preview.json()['action_id'], 'status': 'cancelled'}).status_code == 200
        assert client.post('/api/animation/interaction', json={'kind': 'hold'}).status_code == 200
        service.step()
        assert client.get('/api/animation').json()['state']['mode'] == 'held'
        assert client.post('/api/animation/interaction', json={'kind': 'pointer', 'pointer': {'x': 3, 'y': 0, 'near': True}}).status_code == 400
        assert client.post('/api/animation/stop').status_code == 200
    finally: service.close(); session.actions.close()


def test_resource_estimate_previews_drafts_without_saving(backend, monkeypatch):
    from process.app_core.configuration.settings_store import load_config
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: openai\nvoice:\n  asr_device: cpu\n')
    before = path.read_bytes()
    monkeypatch.setattr(backend, 'load_config', load_config)
    monkeypatch.setattr(backend.gpu_monitor, 'sample', lambda *args: {'available': False, 'gpus': []})
    client = client_for(backend)
    response = client.post('/api/resources/estimate', json={'changes': {'memory.reflection_context_window_tokens': 8192}})
    assert response.status_code == 200
    assert response.json()['estimate']['kv']['reflection_context_tokens'] == 8192
    assert response.json()['draft']
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.resource-estimate-*'))
    assert client.post('/api/resources/estimate', json={'changes': {'runtime.n_ctx': -1}}).status_code == 400


def test_tool_approval_api_policy_and_decision(backend, monkeypatch, tmp_path):
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    registry = ToolRegistry()
    registry.tools['example'] = RegisteredTool('example', 'Test tool', {}, lambda args: 'ok')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json')
    monkeypatch.setattr(backend, 'chat', SimpleNamespace(tool_registry=registry))
    client = client_for(backend)
    try:
        assert client.get('/api/tools/approvals').json()['tools'][0]['name'] == 'example'
        assert client.put('/api/tools/approvals', json={'policy': {'example': True}}).status_code == 200
        assert client.put('/api/tools/approvals', json={'policy': {'missing': True}}).status_code == 400
        assert client.post('/api/tools/approvals/expired', json={'approved': True}).status_code == 409
    finally: registry.close()


def test_resource_websocket_bootstrap_and_source_triggered_updates(backend, monkeypatch):
    from process.app_core.events.bus import EventBus
    from process.app_core.events.resources import ResourceEvents
    bus = EventBus()
    pending = []
    bridge = ResourceEvents(bus, {'approvals': lambda: {'pending': list(pending)}})
    monkeypatch.setattr(backend, 'event_bus', bus)
    monkeypatch.setattr(backend, 'resource_events', bridge)
    client = client_for(backend)
    try:
        with client.websocket_connect('ws://127.0.0.1:8765/ws/events') as socket:
            assert socket.receive_json()['type'] == 'state.snapshot'
            assert socket.receive_json()['payload']['approvals']['pending'] == []
            pending.append({'id':'request'})
            bus.publish('tool.approval_requested')
            event = socket.receive_json()
            assert event['type'] == 'resource.approvals'
            assert event['payload']['pending'] == [{'id':'request'}]
    finally: bridge.close()


def test_corrupt_saved_desktop_settings_do_not_abort_startup(backend, monkeypatch):
    path = backend.config.root / 'persistent_memories' / 'desktop_settings.json'
    path.parent.mkdir()
    path.write_text('{"avatar_geometry":{"width":0}}', encoding='utf-8')
    original = path.read_bytes()
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: SimpleNamespace())
    monkeypatch.setattr(backend, 'SessionManager', lambda *args: SimpleNamespace(
        runtime_snapshot=lambda: {'runtime': {}}, close=lambda: None))
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    with client_for(backend):
        assert backend.state.avatar_geometry['width'] == 480
        assert not backend._desktop_settings_loaded
    assert path.read_bytes() == original


def test_surface_api_rejects_empty_bounds_and_noninteger_geometry(backend):
    client = client_for(backend)
    command = backend.state.add_whiteboard('text', {'text': 'hello'})
    response = client.post('/api/surfaces/result', json={
        'surface': 'whiteboard', 'command_id': command, 'status': 'rendered', 'bounds': {}})
    assert response.status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'width': 1}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'x': True}).status_code == 422
    assert client.patch('/api/surfaces/whiteboard', json={'geometry': {'screen': -1}}).status_code == 400


def test_failed_session_start_closes_constructed_chat(backend,monkeypatch):
    closed=[]
    monkeypatch.setattr(backend,'create_chat_service',lambda config:SimpleNamespace(close=lambda:closed.append('chat')))
    def fail(*args): raise RuntimeError('session startup failed')
    monkeypatch.setattr(backend,'SessionManager',fail)
    with pytest.raises(RuntimeError,match='session startup failed'):
        with client_for(backend): pass
    assert closed==['chat']


def test_a_multi_word_companion_name_starts_the_real_session(backend, monkeypatch):
    """Regression: WakeWord refused a two-word name such as the fixture's 'Test character' in every mode, which the
    packaged setup accepts, and the lifespan aborted after the model had loaded."""
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: SimpleNamespace(provider=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    with client_for(backend) as client:
        wake = client.get('/api/voice/status').json()['wake']
        assert (wake['mode'], wake['wake_word'], wake['error']) == ('wake_word', 'Test', '')
        assert client.get('/api/status').json()['startup_error'] == ''
    assert backend.session._closed


def test_setup_mode_survives_a_session_that_cannot_start(backend, monkeypatch):
    """Voice settings are checked only after the model has loaded; their errors keep Settings up like a model error."""
    backend.config.raw.update(desktop={'setup_on_startup_error': True}, voice={'mode': 'push_to_talk'})
    (backend.config.root / 'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    closed = []
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: SimpleNamespace(close=lambda: closed.append('chat'), provider=None))
    listeners = len(backend.event_bus._listeners)
    with client_for(backend) as client:
        assert 'voice.mode must be' in client.get('/api/status').json()['startup_error']
        assert client.get('/api/settings').status_code == 200
        assert client.get('/api/voice/status').status_code == 503
        assert closed == ['chat'] and backend.chat is None and backend.session is None  # the loaded model is released
    assert len(backend.event_bus._listeners) == listeners


def test_setup_mode_keeps_settings_accessible_with_cors(backend,monkeypatch):
    backend.config.raw['desktop']={'setup_on_startup_error':True}
    (backend.config.root/'character_config.yaml').write_text('runtime:\n  provider: lm_studio\n')
    def fail(config): raise RuntimeError('llama-server missing')
    monkeypatch.setattr(backend,'create_chat_service',fail)
    with client_for(backend) as client:
        assert client.get('/api/settings').status_code==200
        assert client.get('/api/status').json()['startup_error']=='llama-server missing'
        response=client.get('/api/voice/status',headers={'Origin':'http://127.0.0.1:5173'})
        assert response.status_code==503
        assert response.headers['access-control-allow-origin']=='http://127.0.0.1:5173'


def test_history_api_is_paginated_without_replaying_events(backend,monkeypatch):
    from process.app_core.persistence.conversation_store import ConversationStore
    from process.app_core.conversation.messages import ChatMessage
    store=ConversationStore(backend.config.root/'history.sqlite3',legacy=[ChatMessage('user',str(i)) for i in range(10)])
    monkeypatch.setattr(backend,'conversation_store',store)
    client=client_for(backend)
    try:
        page=client.get('/api/chat/history?limit=3').json()
        assert [m['text'] for m in page['messages']]==['7','8','9']
        older=client.get('/api/chat/history',params={'limit':3,'before':page['before']}).json()
        assert [m['text'] for m in older['messages']]==['4','5','6']
        assert client.get('/api/chat/history?limit=1000').status_code==400
    finally: store.close()


def test_display_api_validates_payload_and_defaults_to_primary(backend):
    client = client_for(backend)
    assert client.post('/api/displays', json=[{}]).status_code == 422
    assert client.post('/api/displays', json=[]).status_code == 400
    displays = [{'index': index, 'id': index + 10, 'label': f'Screen {index}', 'primary': index == 1,
                 'bounds': {'x': index * 1920, 'y': 0, 'width': 1920, 'height': 1080}, 'scaleFactor': 1}
                for index in range(2)]
    assert client.post('/api/displays', json=displays).status_code == 200
    assert backend.state.avatar_geometry['screen'] == 1
    assert client.patch('/api/surfaces/avatar', json={'screen': 9}).status_code == 400
    assert client.patch('/api/surfaces/avatar', json={'screen': 0}).status_code == 200
    # Removing the selected display safely falls back to the surviving primary.
    assert client.post('/api/displays', json=[{**displays[1], 'index': 0}]).status_code == 200
    assert backend.state.avatar_geometry['screen'] == 0


def test_every_route_needs_the_install_token_and_a_loopback_host(backend):
    token = backend.api_token()
    assert client_for(backend).get('/api/discord/inbox').status_code == 200
    anonymous = TestClient(backend.app, base_url='http://127.0.0.1:8765')
    assert anonymous.get('/api/discord/inbox').status_code == 401  # e.g. a web page in a browser tab
    assert anonymous.get('/api/discord/inbox', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    rebound = TestClient(backend.app, base_url='http://attacker.example:8765', headers={'Authorization': 'Bearer ' + token})
    assert rebound.get('/api/discord/inbox').status_code == 403  # DNS rebinding keeps the attacker's Host
    folder = backend.config.root / 'persistent_memories'
    key = backend.api_secrets()[1]
    assert (folder / 'api_token').read_text() == token and (folder / 'confirm_key').read_text() == key and len(token) >= 32 and key != token
    if os.name != 'nt': assert all((folder / name).stat().st_mode & 0o777 == 0o600 for name in ('api_token', 'confirm_key'))


def test_websockets_need_the_token_and_an_app_origin(backend):
    from starlette.websockets import WebSocketDisconnect
    url = 'ws://127.0.0.1:8765/ws/events'
    with client_for(backend).websocket_connect(url, headers={'Origin': 'file://'}) as socket:
        assert socket.receive_json()['type'] == 'state.snapshot'
    with pytest.raises(WebSocketDisconnect):
        with TestClient(backend.app).websocket_connect(url): pass  # no token
    with pytest.raises(WebSocketDisconnect):
        with client_for(backend).websocket_connect(url, headers={'Origin': 'https://attacker.example'}): pass


def test_api_docs_are_off_outside_development(backend):
    client = client_for(backend)
    assert client.get('/docs').status_code == 404 and client.get('/openapi.json').status_code == 404


def test_security_sensitive_settings_need_a_signature_only_electron_main_can_make(backend, monkeypatch):
    from process.app_core.configuration.settings_store import SettingsStore
    from process.app_core.desktop.api_guard import signature
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\n')
    store = SettingsStore(path)
    monkeypatch.setattr(backend, 'settings_store', lambda: store)
    client = client_for(backend)
    body = {'revision': store.snapshot()['revision'], 'changes': {'sovits_ping_config.executable': '/tmp/not-sovits'}}
    asked = client.put('/api/settings', json=body)
    assert asked.status_code == 428 and path.read_text() == 'runtime:\n  provider: lm_studio\n'
    challenge = asked.json()['detail']['confirm']
    assert client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': 'forged'}).status_code == 428
    # The API token is on the wire with every request, so it must not be able to sign confirmations.
    assert client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': signature(backend.api_token(), challenge)}).status_code == 428
    signed = client.put('/api/settings', json=body, headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert signed.status_code == 200 and signed.json()['saved'] and 'not-sovits' in path.read_text()
    path.write_text('runtime:\n  provider: lm_studio\n')  # changed back; the old approval must not apply again
    replayed = client.put('/api/settings', json={**body, 'revision': store.snapshot()['revision']}, headers={'X-Riko-Confirmation': signature(backend.api_secrets()[1], challenge)})
    assert replayed.status_code == 428 and 'not-sovits' not in path.read_text()


def test_turning_tool_approval_off_needs_confirmation(backend, monkeypatch, tmp_path):
    from process.app_core.desktop.api_guard import signature
    from process.app_core.tools.registry import ToolRegistry, RegisteredTool
    from process.app_core.tools.approval import ToolApprovals
    registry = ToolRegistry()
    registry.tools['example'] = RegisteredTool('example', 'Test tool', {}, lambda args: 'ok')
    registry.approvals = ToolApprovals(tmp_path / 'approvals.json', True)
    monkeypatch.setattr(backend, 'chat', SimpleNamespace(tool_registry=registry))
    client = client_for(backend)
    try:
        assert client.put('/api/tools/approvals', json={'policy': {'example': True}}).status_code == 200  # stricter: no prompt
        asked = client.put('/api/tools/approvals', json={'policy': {'example': False}})
        assert asked.status_code == 428 and registry.approvals.snapshot()['policy']['example'] is True
        proof = signature(backend.api_secrets()[1], asked.json()['detail']['confirm'])
        assert client.put('/api/tools/approvals', json={'policy': {'example': False}}, headers={'X-Riko-Confirmation': proof}).status_code == 200
        assert registry.approvals.snapshot()['policy']['example'] is False
    finally: registry.close()


def test_discord_client_receives_only_its_own_turns_not_local_activity(backend, monkeypatch):
    import uuid
    from process.app_core.events.bus import EventBus
    bus = EventBus()
    monkeypatch.setattr(backend, 'event_bus', bus)
    monkeypatch.setattr(backend, 'resource_events', None)
    instance = str(uuid.uuid4())
    with client_for(backend).websocket_connect(f'ws://127.0.0.1:8765/ws/discord/client?instance={instance}') as socket:
        assert socket.receive_json() == {'type': 'state.snapshot', 'payload': {}}  # no local desktop state
        assert set(socket.receive_json()['payload']) <= {'discord', 'approvals'}
        bus.publish('voice.transcript', text='private words spoken at the desk')
        bus.publish('chat.delta', turn_id='local', text='a local reply', source='message')
        bus.publish('model.started', turn_id='remote', source='discord')
        bus.publish('chat.delta', turn_id='remote', text='hello Discord', source='discord')
        bus.publish('initiative.presented', message='marker')
        received = [socket.receive_json() for _ in range(2)]
    assert [(event['type'], event['payload'].get('text') or event['payload'].get('message')) for event in received] == [
        ('chat.delta', 'hello Discord'), ('initiative.presented', 'marker')]


def test_confirmations_are_single_use_bound_to_the_change_and_expire():
    from process.app_core.desktop.api_guard import Confirmations, signature
    confirmations, key = Confirmations(), 'k' * 43
    challenge = confirmations.challenge('settings', {'runtime.native_library': 'a.dll'})
    proof = signature(key, challenge)
    assert not confirmations.consume(key, 'settings', {'runtime.native_library': 'b.dll'}, proof)  # a different change
    assert not confirmations.consume(key, 'discord_access', {'runtime.native_library': 'a.dll'}, proof)
    assert not confirmations.consume('other' * 9, 'settings', {'runtime.native_library': 'a.dll'}, proof)
    assert confirmations.consume(key, 'settings', {'runtime.native_library': 'a.dll'}, proof)
    assert not confirmations.consume(key, 'settings', {'runtime.native_library': 'a.dll'}, proof)  # replay
    expired = Confirmations(ttl=-1)
    challenge = expired.challenge('settings', {'tools.mcp_config': 'x'})
    assert not expired.consume(key, 'settings', {'tools.mcp_config': 'x'}, signature(key, challenge))


@pytest.mark.parametrize('key', ['runtime.native_library', 'runtime.provider', 'runtime.base_url', 'runtime.model_path',
    'runtime.hf_repo_id', 'tools.mcp_config', 'emotion.enabled', 'emotion.model_id', 'memory.system1_enabled',
    'memory.system1_model_id', 'memory.store_file', 'sovits_ping_config.url', 'sovits_ping_config.ref_audio_path',
    'sovits_ping_config.executable', 'sovits_ping_config.arguments', 'voice.asr_model'])
def test_settings_that_run_code_or_send_data_elsewhere_are_security_sensitive(backend, key):
    assert backend.security_sensitive(key)


@pytest.mark.parametrize('key', ['runtime.pause_background_on_live', 'runtime.temperature', 'avatar.scale', 'speech.max_words', 'voice.vad_threshold'])
def test_everyday_settings_save_without_confirmation(backend, key):
    assert not backend.security_sensitive(key)


def test_sections_directories_and_locations_cannot_slip_past_confirmation(backend, monkeypatch):
    sensitive = backend.security_sensitive
    assert sensitive('tools', {'mcp_config': '/tmp/evil/mcp.json', 'require_approval': False})  # an empty YAML mapping is one editable value
    assert sensitive('sovits_ping_config', {'nested': {'auto_start': True}}) and not sensitive('animation', {'walk_speed': 2})
    assert sensitive('desktop.effects_directory', 'effects') and sensitive('initiative.rules', [{'id': 'x', 'webhook': 'https://example.com'}])
    assert sensitive('presets.default.name', 'C:\\Windows') and sensitive('voice.wake_word', '~/words')
    assert not sensitive('avatar.model', '/Users/me/Avatar.vrm') and not sensitive('presets.default.name', 'Riko')
    from process.app_core.configuration.settings_store import SettingsStore
    path = backend.config.root / 'character_config.yaml'
    path.write_text('runtime:\n  provider: lm_studio\ntools: {}\n')
    store = SettingsStore(path)
    monkeypatch.setattr(backend, 'settings_store', lambda: store)
    asked = client_for(backend).put('/api/settings', json={'revision': store.snapshot()['revision'], 'changes': {'tools': {'mcp_config': '/tmp/evil/mcp.json'}}})
    assert asked.status_code == 428 and 'evil' not in path.read_text()


def test_secrets_are_fresh_each_start_and_never_inherited_by_children(tmp_path, monkeypatch):
    from process.app_core.desktop.api_guard import issue_secrets, client_token
    first, second = issue_secrets(tmp_path), issue_secrets(tmp_path)
    assert first != second and (tmp_path / 'persistent_memories' / 'api_token').read_text() == second[0]
    monkeypatch.setenv('RIKO_API_TOKEN', 'a' * 43)
    monkeypatch.setenv('RIKO_CONFIRM_KEY', 'b' * 43)
    assert issue_secrets(tmp_path / 'packaged') == ('a' * 43, 'b' * 43)  # Electron's values, no files
    assert 'RIKO_API_TOKEN' not in os.environ and 'RIKO_CONFIRM_KEY' not in os.environ
    assert not (tmp_path / 'packaged' / 'persistent_memories').exists()
    monkeypatch.setenv('RIKO_API_TOKEN', 'short')
    monkeypatch.setenv('RIKO_CONFIRM_KEY', 'b' * 43)
    token, _ = issue_secrets(tmp_path)
    assert token != 'short' and 'RIKO_CONFIRM_KEY' not in os.environ
    # A client such as the Discord worker finds the token beside the backend's config.
    monkeypatch.setenv('RIKO_DATA_DIR', str(tmp_path))
    monkeypatch.delenv('RIKO_CONFIG', raising=False)
    assert client_token() == token
    monkeypatch.setenv('RIKO_API_TOKEN', 'c' * 43)
    assert client_token() == 'c' * 43


def test_stop_turn_cancels_the_live_session_before_the_drain(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, 'session', SimpleNamespace(_closed=False, cancel=lambda: calls.append('cancel')))
    backend.stop_turn()
    assert calls == ['cancel']
    monkeypatch.setattr(backend, 'session', SimpleNamespace(_closed=True, cancel=lambda: calls.append('closed')))
    backend.stop_turn()
    monkeypatch.setattr(backend, 'session', None)  # setup mode: nothing to stop
    backend.stop_turn()
    stuck = threading.Event()
    monkeypatch.setattr(backend, 'session', SimpleNamespace(_closed=False, cancel=lambda: stuck.wait(5)))
    started = time.monotonic()
    try: backend.stop_turn()
    finally: stuck.set()
    assert calls == ['cancel'] and time.monotonic() - started < 3  # a stuck cancel is abandoned after 1 s


def test_chat_while_another_turn_runs_is_a_conflict_not_a_server_error(backend, monkeypatch):
    from pathlib import Path
    from process.app_core.runtime.session import SessionManager
    class Speech:
        def __init__(self, *args): pass
        def submit(self, *args): return False
        def cancel(self): pass
        def close(self): pass
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={}, root=Path('.'), character_name='Riko', tools=SimpleNamespace(max_iterations=8))
    session = SessionManager(config, SimpleNamespace(history=[], _save_history=lambda: None, close=lambda: None), backend.state)
    monkeypatch.setattr(backend, 'session', session)
    client = client_for(backend, raise_server_exceptions=False)
    session._turn_lock.acquire()  # a voice or Discord turn, or a reply still stopping
    try:
        response = client.post('/api/chat', json={'text': 'hello'})
        assert response.status_code == 409 and 'still handling another reply' in response.json()['detail']
    finally: session._turn_lock.release(); session.close()
    def fail(*args): raise RuntimeError('model exploded')
    monkeypatch.setattr(backend, 'session', SimpleNamespace(respond=fail))
    assert client.post('/api/chat', json={'text': 'hello'}).status_code == 500  # any other failure is still a server error


def test_lifespan_stops_discord_while_the_session_closes(backend, monkeypatch):
    closing, calls = threading.Event(), []
    monkeypatch.setattr(backend, 'create_chat_service', lambda config: SimpleNamespace())
    class Session:
        def __init__(self, *args): pass
        def runtime_snapshot(self): return {'runtime': {}}
        def close(self): closing.set(); calls.append('close')
    monkeypatch.setattr(backend, 'SessionManager', Session)
    monkeypatch.setattr(backend, 'Initiative', lambda session: None)
    monkeypatch.setattr(backend, 'discord_launcher', SimpleNamespace(status=lambda: {'running': False, 'error': ''},
        stop=lambda: calls.append(('discord', closing.wait(5)))))
    with client_for(backend): pass
    assert ('discord', True) in calls and 'close' in calls  # Discord's stop overlapped the session close


def test_an_interjection_that_cuts_the_reply_in_flight_redoes_it(backend, monkeypatch):
    redone, calls = threading.Event(), []
    def respond(text, **kwargs):
        calls.append((text, kwargs['record_user'], kwargs['reply_to']))
        redone.set()
    monkeypatch.setattr(backend, 'session', SimpleNamespace(voice_anchor=lambda: ('turn-1', 0, False),
        voice_transcript=lambda text, started_at, ended_at, anchor: 'reply', respond=respond))
    client = client_for(backend)
    body = {'text': 'and tomorrow', 'started_at': 1, 'ended_at': 2}
    assert client.post('/api/voice/interjection', json=body).json() == {'accepted': True}
    assert redone.wait(5) and calls == [('and tomorrow', False, 'turn-1')]  # the words are already in history
    def unexpected(*args): raise AssertionError('no reply is in flight')
    monkeypatch.setattr(backend, 'session', SimpleNamespace(voice_anchor=lambda: None, voice_transcript=unexpected))
    assert client.post('/api/voice/interjection', json=body).json() == {'accepted': False}  # the client sends a chat turn
