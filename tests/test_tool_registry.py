"""Every tool enters through ToolRegistry.register: valid names, no silent shadowing, isolation declared by the tool,
read-only tools chosen by their metadata, and tool-specific input rules kept in the tools themselves."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from process.app_core.conversation.chat import ChatService
from process.app_core.desktop.state import DesktopState
from process.app_core.desktop.tools import AvatarGestureTool, DesktopServices, EffectTool
from process.app_core.persistence.tasks import TaskMCP, TaskStore
from process.app_core.runtime.session import SessionManager
from process.app_core.tools.choices import ChoiceResolver
from process.app_core.tools.mcp import StdioMCPClient
from process.app_core.tools.registry import ToolRegistry
from process.app_core.tools.tool import RIKO, RegisteredTool, ToolActivity
from process.app_core.kernel.audio_config import audio_sections


class Server:
    """An in-process MCP client listing the given tool names, recording each call."""
    def __init__(self, *names, annotations=None): self.names, self.annotations, self.calls = names, annotations, []
    def list_tools(self):
        return [{'name': name, 'description': f'remote {name}', 'inputSchema': {'type': 'object'}, **({'annotations': self.annotations} if self.annotations else {})}
                for name in self.names]
    def call(self, name, arguments): self.calls.append((name, arguments)); return {'content': [{'type': 'text', 'text': f'remote {name}'}]}
    def close(self): pass


class Notes(ToolActivity):
    def __init__(self): self.notes = []
    def notify(self, source, text, level='info'): self.notes.append((source, text, level))


try: from process.app_core.inference.provider import BaseProvider as Provider  # the provider contract's do-nothing defaults
except ImportError:  # before that contract (Wave 5 segment 3, providers), SessionManager needs only close()
    class Provider:
        def close(self): pass


class Speech:
    def __init__(self, *args): pass
    def submit(self, *args): return True
    def cancel(self): pass
    def close(self): pass


def local(name, result='local', **metadata): return RegisteredTool(name, f'local {name}', {'type': 'object', 'properties': {}}, lambda args: result, **metadata)


def test_riko_tools_need_valid_names_and_a_clash_raises_instead_of_replacing():
    registry = ToolRegistry()
    try:
        assert registry.register(local('lookup'), source=RIKO) == 'lookup'
        with pytest.raises(ValueError, match='already registered'): registry.register(local('lookup', 'second'), source=RIKO)
        assert registry.execute('lookup', {}).content == 'local'
        registry.register(local('lookup', 'replaced'), source=RIKO, replace=True)
        assert registry.execute('lookup', {}).content == 'replaced'
        for name in ('has space', 'dotted.name', '', 'x' * 65, 'café'):  # OpenAI's ^[a-zA-Z0-9_-]{1,64}$
            with pytest.raises(ValueError, match='Invalid tool name'): registry.register(local(name), source=RIKO)
    finally: registry.close()


def test_a_server_tool_never_shadows_riko_s_own_and_is_called_by_its_server_name():
    server = Server('scientific_calculator', 'read_file')
    registry = ToolRegistry()
    try:
        registry.register(local('scientific_calculator'), source=RIKO)
        assert registry.register_mcp(server, source='mcp:my files') == ['my_files__scientific_calculator', 'read_file']
        assert registry.execute('scientific_calculator', {}).content == 'local' and server.calls == []
        assert registry.execute('my_files__scientific_calculator', {'x': 1}).content == 'remote scientific_calculator'
        assert server.calls == [('scientific_calculator', {'x': 1})]
        assert [tool['function']['name'] for tool in registry.definitions()] == ['scientific_calculator', 'my_files__scientific_calculator', 'read_file']
        assert registry.tools['read_file'].approval_key == ('mcp:my files', 'read_file')
        assert registry.tools['my_files__scientific_calculator'].approval_key == ('mcp:my files', 'scientific_calculator')
    finally: registry.close()


def test_riko_s_tool_registered_after_a_server_takes_its_name_back():
    # SessionManager registers runtime_status after mcp.json's servers have loaded.
    server = Server('runtime_status')
    registry = ToolRegistry()
    try:
        registry.register_mcp(server, source='mcp:spy')
        assert registry.execute('runtime_status', {}).content == 'remote runtime_status'
        registry.register(local('runtime_status', 'session'), source=RIKO)
        assert registry.execute('runtime_status', {}).content == 'session'
        assert registry.execute('spy__runtime_status', {}).content == 'remote runtime_status'
        assert server.calls == [('runtime_status', {}), ('runtime_status', {})]
    finally: registry.close()


def test_server_tools_with_invalid_or_taken_names_are_renamed_or_left_out_with_a_notice():
    activity = Notes()
    registry = ToolRegistry(activity=activity)
    try:
        assert registry.register_mcp(Server('files.read', 'search'), source='mcp:fs') == ['fs__files_read', 'search']
        assert registry.register_mcp(Server('search'), source='mcp:web') == ['web__search']
        assert registry.register_mcp(Server('search'), source='mcp:' + 'w' * 60) == [None]  # no valid name is left
        assert set(registry.tools) == {'fs__files_read', 'search', 'web__search'} and activity.notes[-1][2] == 'error'
        assert registry.execute('fs__files_read', {}).content == 'remote files.read'
    finally: registry.close()


def test_built_ins_declare_isolation_themselves_whatever_module_they_are_imported_from(tmp_path, monkeypatch):
    # Isolation used to follow the module path (process.app_core.tools.builtin.*): under any other name a built-in ran in-process.
    (tmp_path / 'relocated_builtin.py').write_text('import os\nfrom process.app_core.tools.builtin.base import BaseTool\n'
        'class Tool(BaseTool):\n    TOOL_NAME = "where"\n    TOOL_DESCRIPTION = "Name the process"\n'
        '    def _call(self, label: str = "") -> str: return label + str(os.getpid())\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    from relocated_builtin import Tool
    class InProcess:  # a desktop-style tool: its execute signature is its schema, and it runs here
        TOOL_NAME, TOOL_DESCRIPTION = 'here', 'Name the process'
        def execute(self, label: str = '') -> str: return label + str(os.getpid())
    registry = ToolRegistry(timeout_seconds=60)
    try:
        registry.register_local(Tool({}, {}), owner='builtin')
        registry.register_local(InProcess(), owner='desktop')
        assert registry.tools['where'].isolated == {'module': 'relocated_builtin', 'class': 'Tool', 'config': {}, 'context': {}}
        assert registry.tools['here'].isolated is None
        assert registry.tools['where'].schema == registry.tools['here'].schema
        isolated, here = registry.execute('where', {'label': 'pid '}), registry.execute('here', {'label': 'pid '})
        assert not isolated.is_error and isolated.content.startswith('pid ') and isolated.content != here.content == f'pid {os.getpid()}'
    finally: registry.close()


def test_only_riko_s_declared_read_only_tools_are_offered_as_read_only(tmp_path):
    registry = ToolRegistry()
    try:
        registry.register(local('peek', read_only=True), source=RIKO)
        registry.register(local('poke'), source=RIKO)
        registry.register_mcp(Server('remote_peek', annotations={'readOnlyHint': True}), source='mcp:remote')  # untrusted hint
        registry.register_mcp(TaskMCP(TaskStore(tmp_path / 'tasks.sqlite3')), source=RIKO, owner='tasks')
        assert [tool['function']['name'] for tool in registry.definitions(read_only=True)] == ['peek', 'task_list', 'task_get']
        assert not registry.tools['remote_peek'].read_only and registry.tools['task_get'].owner == 'tasks'
    finally: registry.close()


def test_generic_choice_repair_knows_no_tool_and_each_tool_narrows_its_own_choices():
    resolver = ChoiceResolver()
    try:  # the exemption for empty name/asset_id fields now lives in the tools that have them
        with pytest.raises(ValueError, match='Invalid name'): resolver.normalize('any_tool', {'name': ''}, {'name': ['nod', 'wave']})
    finally: resolver.close()
    library = SimpleNamespace(assets={'video': [Path('effects/stars.mp4')]}, rules=[], directory=Path('effects'))
    effect = EffectTool(DesktopServices(DesktopState(), effect_library=library))
    actions = {'action': ['list', 'play', 'stop']}
    assert effect.input_choices() == {**actions, 'name': ['stars', 'stars.mp4']}
    for arguments in ({'action': 'list', 'name': 'junk'}, {'action': 'stop'}, {'action': 'play', 'asset': 'x.mp4', 'name': 'junk'}, {'action': 'play', 'name': ''}):
        assert effect.input_choices(arguments) == actions
    animation = SimpleNamespace(library=SimpleNamespace(list=lambda: [{'id': 'wave-1'}]))
    tools = {tool.name: tool for tool in SessionManager.runtime_tools(SimpleNamespace(animation=animation, avatar_animation=None))}
    assert tools['avatar_animation'].choices() == {'action': ['list', 'preview', 'stop'], 'asset_id': ['wave-1']}
    assert tools['avatar_animation'].choices({'action': 'list', 'asset_id': ''}) == {'action': ['list', 'preview', 'stop']}
    cancelled = []
    gesture = AvatarGestureTool(DesktopServices(DesktopState(), actions=SimpleNamespace(cancel=lambda action: cancelled.append(action) or True)))
    assert gesture.input_choices() == {'name': ['nod', 'shake', 'wave']} and gesture.input_choices({'cancel_id': 'a1', 'name': ''}) == {}
    registry = ToolRegistry()
    registry.choice_resolver, approved = ChoiceResolver(), []
    registry.approvals = SimpleNamespace(authorize=lambda name, arguments, *rest, **keys: approved.append(arguments) or True, close=lambda: None)
    try:
        registry.register_local(gesture, owner='desktop')
        assert registry.execute('avatar_gesture', {'cancel_id': 'a1', 'name': ''}).content == 'Cancelled' and cancelled == ['a1']
        approved.clear()
        registry.approvals = SimpleNamespace(authorize=lambda name, arguments, *rest, **keys: approved.append(arguments) or False, close=lambda: None)
        registry.register_local(effect, owner='desktop')
        assert 'approval denied' in registry.execute('visual_effect', {'action': 'list', 'name': 'junk'}).content  # not repaired, not refused
        assert approved == [{'action': 'list', 'name': 'junk'}]
    finally: registry.close()


def test_the_session_registers_its_tools_like_every_other_tool(tmp_path, monkeypatch):
    monkeypatch.setattr('process.app_core.runtime.session.SpeechQueue', Speech)
    config = SimpleNamespace(raw={'voice': {}, 'animation': False}, root=tmp_path, character_name='Riko', tools=SimpleNamespace(max_iterations=8), **audio_sections({}))
    registry = ToolRegistry()
    registry.register_mcp(Server('interrupt_user'), source='mcp:spy', owner='spy')  # loaded before the session, as mcp.json is
    session = SessionManager(config, ChatService(Provider(), system_prompt='Riko', tool_registry=registry), DesktopState())
    try:
        assert {name: (tool.source, tool.owner) for name, tool in registry.tools.items()} == {
            'spy__interrupt_user': ('mcp:spy', 'spy'), 'runtime_status': (RIKO, 'session'), 'interrupt_user': (RIKO, 'session')}
    finally: session.close()


def test_a_stdio_server_that_exits_is_restarted_in_place_by_the_next_call(tmp_path):
    server = tmp_path / 'server.py'
    server.write_text('''import json, os, sys
for line in sys.stdin:
    message = json.loads(line)
    reply = lambda result: print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)
    if message.get('method') == 'initialize': reply({})
    elif message.get('method') == 'tools/call': reply({'content': [{'type': 'text', 'text': str(os.getpid())}]}); sys.exit()
''', encoding='utf-8')
    client = StdioMCPClient(sys.executable, [str(server)])
    try:
        first = client.call('pid', {})
        client.process.wait(timeout=30)
        second = client.call('pid', {})
        assert first != second and client.process.poll() is None
    finally: client.close()


def test_old_import_paths_still_resolve_for_one_release():
    from process.app_core.tools import mcp, registry, schema, tool
    assert registry.StdioMCPClient is mcp.StdioMCPClient and registry.HTTPMCPClient is mcp.HTTPMCPClient
    assert registry.resolve_command is mcp.resolve_command and registry.local_definition is schema.local_definition
    assert registry.RegisteredTool is tool.RegisteredTool and registry.ToolActivity is tool.ToolActivity and registry.ToolCancelled is tool.ToolCancelled
