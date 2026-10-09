"""run_server.main and configure_logging, without the real desktop_server app (whose lifespan builds the services) and
without binding 127.0.0.1:8765, where a developer's backend may be running."""
import logging
from pathlib import Path
import socket as real_socket
import sys
from types import SimpleNamespace

import pytest

import run_server
from process.app_core.configuration.debug_logging import SafeFormatter, configure_logging
from process.app_core.configuration.paths import DataPaths

LOGGERS = ('', 'process.app_core', 'desktop_server', 'process.app_core.kernel.metrics')  # '' is the root logger


@pytest.fixture(autouse=True)
def restore_logging():
    """configure_logging sets process-wide logger levels and adds a root handler: put both back."""
    root = logging.getLogger()
    levels, handlers = {name: logging.getLogger(name).level for name in LOGGERS}, list(root.handlers)
    yield
    for handler in [handler for handler in root.handlers if handler not in handlers]:
        root.removeHandler(handler)
        handler.close()
    for name, level in levels.items(): logging.getLogger(name).setLevel(level)


def config(tmp_path, *, debug=False, **logging_settings):
    return SimpleNamespace(root=tmp_path, paths=DataPaths.at(tmp_path), runtime=SimpleNamespace(api_key=None),
        raw={'desktop': {'debug': debug}, 'logging': {'file_enabled': False, **logging_settings}})


@pytest.mark.parametrize('debug', [False, True])
def test_desktop_debug_sets_application_levels_only(tmp_path, debug):
    configure_logging(config(tmp_path, debug=debug))
    expected = logging.DEBUG if debug else logging.INFO
    assert logging.getLogger('process.app_core').level == logging.getLogger('desktop_server').level == expected
    assert not (tmp_path / 'logs').exists()  # file_enabled: false


def test_explicit_level_wins_and_unknown_levels_fall_back_to_info(tmp_path):
    configure_logging(config(tmp_path, debug=True, level='warning', inference_timings=False))
    assert logging.getLogger('process.app_core').level == logging.WARNING
    assert logging.getLogger('process.app_core.kernel.metrics').level == logging.WARNING
    configure_logging(config(tmp_path, level='verbose'))
    assert logging.getLogger('desktop_server').level == logging.INFO


def test_debug_log_file_redacts_secrets(tmp_path, monkeypatch):
    secret = 'example-secret-value-0123456789'
    monkeypatch.setenv('RIKO_TEST_API_TOKEN', secret)
    settings = config(tmp_path, file_enabled=True)
    settings.runtime.api_key = 'sk-local-model-key-987654'
    configure_logging(settings)
    handler = next(handler for handler in logging.getLogger().handlers if isinstance(handler.formatter, SafeFormatter))
    assert handler.baseFilename == str(tmp_path / 'logs' / 'debug.log') and handler.level == logging.INFO
    logging.getLogger('process.app_core.test').warning('token=%s key %s header Bearer abc.def', secret, settings.runtime.api_key)
    handler.flush()
    text = (tmp_path / 'logs' / 'debug.log').read_text(encoding='utf-8')
    assert 'Diagnostic logging enabled level=INFO' in text
    assert secret not in text and 'sk-local-model-key-987654' not in text and 'abc.def' not in text
    assert text.count('[REDACTED]') >= 3


class Recorder:
    """Stands in for desktop_server, the listening socket, uvicorn.Config and run_server.Server, recording the order."""
    def __init__(self, bind_failures=0, interrupt=False):
        self.events, self.bind_failures, self.interrupt, self.configs, self.given = [], bind_failures, interrupt, [], []
        self.stop_turn = lambda: None
        self.app = SimpleNamespace(state=SimpleNamespace(backend=SimpleNamespace(stop_turn=self.stop_turn, api_secrets=lambda: self.events.append('api_secrets'))))
        self.desktop_server = SimpleNamespace(create_app=lambda config: self.events.append('create_app') or self.given.append(config) or self.app)
        recorder = self

        class Listener:
            def setsockopt(self, level, option, value): recorder.events.append(('setsockopt', option, value))
            def bind(self, address):
                if recorder.bind_failures:
                    recorder.bind_failures -= 1
                    recorder.events.append('bind_failed')
                    raise OSError(48, 'Address already in use')
                recorder.events.append(('bind', address))
            def listen(self, backlog): recorder.events.append(('listen', backlog))

        self.listener = Listener()
        self.socket = SimpleNamespace(AF_INET=real_socket.AF_INET, SOCK_STREAM=real_socket.SOCK_STREAM, SOL_SOCKET=real_socket.SOL_SOCKET,
            SO_REUSEADDR=real_socket.SO_REUSEADDR, SO_EXCLUSIVEADDRUSE=getattr(real_socket, 'SO_EXCLUSIVEADDRUSE', -1),
            socket=lambda family, kind: self.events.append(('socket', family, kind)) or self.listener)

        class Server:
            def __init__(self, config, stop_turn): recorder.server = (config, stop_turn)
            def run(self, sockets):
                recorder.events.append(('run', sockets))
                if recorder.interrupt: raise KeyboardInterrupt  # uvicorn re-raises Ctrl+C after its graceful shutdown

        self.Server = Server

    def config(self, app, **kwargs):
        self.configs.append((app, kwargs))
        return 'uvicorn-config'


def start(monkeypatch, tmp_path, recorder, yaml='', clock=None, data=None):
    """main() on a YAML in tmp_path, with the recorder's desktop_server, socket and uvicorn. data: RIKO_DATA_DIR relative to
    tmp_path's parent, the cwd main() starts in (default: tmp_path, absolute)."""
    (tmp_path / 'character_config.yaml').write_text(yaml, encoding='utf-8')
    monkeypatch.chdir(tmp_path if data is None else tmp_path.parent)  # main() changes the cwd to RIKO_DATA_DIR; this puts it back afterwards
    monkeypatch.setenv('RIKO_DATA_DIR', str(tmp_path) if data is None else data)
    monkeypatch.setenv('RIKO_CONFIG', str(tmp_path / 'character_config.yaml'))
    monkeypatch.delenv('RIKO_MANAGED', raising=False)  # the managed path reads stdin and arms a faulthandler exit
    monkeypatch.setattr(sys, 'argv', ['run_server.py'])
    run_server.load_config(), run_server.uvicorn  # imports what they import lazily now, so no module is first imported with a fake below
    loaded, load_config = [], run_server.load_config
    monkeypatch.setattr(run_server, 'load_config', lambda **options: loaded.append(options) or load_config(**options))
    monkeypatch.setitem(sys.modules, 'desktop_server', recorder.desktop_server)
    monkeypatch.setitem(sys.modules, 'socket', recorder.socket)  # main() imports socket and time when it binds
    if clock: monkeypatch.setitem(sys.modules, 'time', clock)
    monkeypatch.setattr(run_server.uvicorn, 'Config', recorder.config)
    monkeypatch.setattr(run_server, 'Server', recorder.Server)
    run_server.main()
    assert loaded == [{'recover': 'setup'}]  # one load, whose result the app serves (a test that stops startup earlier stops here)


@pytest.mark.parametrize('debug', [False, True])
def test_application_debug_does_not_enable_websocket_packet_dump(monkeypatch, tmp_path, debug):
    recorder = Recorder()
    start(monkeypatch, tmp_path, recorder, f'desktop:\n  debug: {str(debug).lower()}\nlogging:\n  file_enabled: false\n')
    # Uvicorn DEBUG logs every WebSocket frame (voice.level, chat.delta); desktop.debug raises only Riko's own loggers.
    assert recorder.configs == [(recorder.app, {'host': '127.0.0.1', 'port': 8765, 'log_level': 'info', 'timeout_graceful_shutdown': 2})]
    assert recorder.server == ('uvicorn-config', recorder.stop_turn)
    assert [config.root for config in recorder.given] == [tmp_path] and recorder.given[0].raw['desktop']['debug'] is debug  # main()'s own config
    assert logging.getLogger('process.app_core').level == logging.getLogger('desktop_server').level == (logging.DEBUG if debug else logging.INFO)
    assert not (tmp_path / 'logs').exists()


def test_port_is_held_before_the_token_is_minted_and_ctrl_c_exits_cleanly(monkeypatch, tmp_path):
    recorder = Recorder(bind_failures=2, interrupt=True)
    clock = SimpleNamespace(now=0.0, sleeps=[])
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda seconds: (clock.sleeps.append(seconds), setattr(clock, 'now', clock.now + seconds))
    start(monkeypatch, tmp_path, recorder, 'logging:\n  file_enabled: false\n', clock)  # returns: KeyboardInterrupt is swallowed
    socket = recorder.socket
    option = socket.SO_EXCLUSIVEADDRUSE if run_server.os.name == 'nt' else socket.SO_REUSEADDR
    # The app is built (starting nothing, minting nothing) before the port is held; its secrets only once it is.
    assert recorder.events == ['create_app', ('socket', socket.AF_INET, socket.SOCK_STREAM), ('setsockopt', option, 1),
        'bind_failed', 'bind_failed', ('bind', ('127.0.0.1', 8765)), ('listen', 2048), 'api_secrets', ('run', [recorder.listener])]
    assert clock.sleeps == [.5, .5]  # a previous backend may still be shutting down


def test_port_held_by_another_process_stops_startup_with_a_clear_message(monkeypatch, tmp_path):
    recorder = Recorder(bind_failures=1000)
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda seconds: setattr(clock, 'now', clock.now + seconds)
    with pytest.raises(SystemExit, match='Port 8765 is already in use'):
        start(monkeypatch, tmp_path, recorder, 'logging:\n  file_enabled: false\n', clock)
    assert 'api_secrets' not in recorder.events and not any(event[0] == 'run' for event in recorder.events if isinstance(event, tuple))
    assert recorder.events.count('bind_failed') == 41  # every half second for 20 s, then once more at the deadline


def test_main_works_in_the_data_folder_and_hands_children_its_absolute_path(monkeypatch, tmp_path):
    start(monkeypatch, tmp_path, Recorder(), 'logging:\n  file_enabled: false\n', data=tmp_path.name)
    exported = Path(run_server.os.environ['RIKO_DATA_DIR'])  # tool workers, MCP servers and load_config read it after the chdir
    assert exported.is_absolute() and exported.resolve() == Path.cwd().resolve() == tmp_path.resolve()
