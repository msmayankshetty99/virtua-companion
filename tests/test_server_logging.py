"""run_server.main and configure_logging, without the real desktop_server (whose import loads the cwd's private YAML and
builds services) and without binding 127.0.0.1:8765, where a developer's backend may be running."""
import logging
import socket as real_socket
import sys
from types import SimpleNamespace

import pytest

import run_server
from process.app_core.configuration.debug_logging import SafeFormatter, configure_logging

LOGGERS = ('', 'process.app_core', 'desktop_server', 'process.app_core.inference.metrics')  # '' is the root logger


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
    return SimpleNamespace(root=tmp_path, runtime=SimpleNamespace(api_key=None),
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
    assert logging.getLogger('process.app_core.inference.metrics').level == logging.WARNING
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
        self.events, self.bind_failures, self.interrupt, self.configs = [], bind_failures, interrupt, []
        self.app, self.stop_turn = object(), lambda: None
        self.desktop_server = SimpleNamespace(app=self.app, stop_turn=self.stop_turn, api_secrets=lambda: self.events.append('api_secrets'))
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


def start(monkeypatch, tmp_path, recorder, yaml='', clock=None):
    """main() on a YAML in tmp_path, with the recorder's desktop_server, socket and uvicorn."""
    (tmp_path / 'character_config.yaml').write_text(yaml, encoding='utf-8')
    monkeypatch.chdir(tmp_path)  # main() changes the cwd to RIKO_DATA_DIR; this puts it back afterwards
    monkeypatch.setenv('RIKO_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('RIKO_CONFIG', str(tmp_path / 'character_config.yaml'))
    monkeypatch.delenv('RIKO_MANAGED', raising=False)  # the managed path reads stdin and arms a faulthandler exit
    monkeypatch.setattr(sys, 'argv', ['run_server.py'])
    run_server.load_config()  # imports what load_config imports lazily now, so no module is first imported with a fake below
    monkeypatch.setitem(sys.modules, 'desktop_server', recorder.desktop_server)
    monkeypatch.setitem(sys.modules, 'socket', recorder.socket)  # main() imports socket and time when it binds
    if clock: monkeypatch.setitem(sys.modules, 'time', clock)
    monkeypatch.setattr(run_server.uvicorn, 'Config', recorder.config)
    monkeypatch.setattr(run_server, 'Server', recorder.Server)
    run_server.main()


@pytest.mark.parametrize('debug', [False, True])
def test_application_debug_does_not_enable_websocket_packet_dump(monkeypatch, tmp_path, debug):
    recorder = Recorder()
    start(monkeypatch, tmp_path, recorder, f'desktop:\n  debug: {str(debug).lower()}\nlogging:\n  file_enabled: false\n')
    # Uvicorn DEBUG logs every WebSocket frame (voice.level, chat.delta); desktop.debug raises only Riko's own loggers.
    assert recorder.configs == [(recorder.app, {'host': '127.0.0.1', 'port': 8765, 'log_level': 'info', 'timeout_graceful_shutdown': 2})]
    assert recorder.server == ('uvicorn-config', recorder.stop_turn)
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
    assert recorder.events == [('socket', socket.AF_INET, socket.SOCK_STREAM), ('setsockopt', option, 1),
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
