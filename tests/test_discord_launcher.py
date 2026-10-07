from types import SimpleNamespace
import pytest
from process.app_core.integrations.discord import launcher


def test_launcher_is_explicit_idempotent_and_uses_only_the_transport(tmp_path, monkeypatch):
    (tmp_path / 'Code').mkdir()
    (tmp_path / 'Code' / 'discord_bot.py').write_text('# test stub')
    monkeypatch.setattr(launcher.os, 'environ', {'Discord_bot_token': 'fake-token', 'Discord_admins': '1'})
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {})
    monkeypatch.setattr(launcher.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None))
    calls = []
    class Process:
        code = None
        def poll(self): return self.code
        def terminate(self): self.code = 0
        def wait(self, timeout=None): return self.code
    process = Process()
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda command, **kwargs: calls.append((command, kwargs)) or process)
    service = launcher.DiscordLauncher(tmp_path)
    service.token = lambda: 'x' * 43  # desktop_server hands the launcher this backend start's token
    assert not service.status()['running']
    assert calls == []
    assert service.start()['running']
    assert service.start()['running']
    assert len(calls) == 1
    assert calls[0][0] == [launcher.sys.executable, str(tmp_path / 'Code' / 'discord_bot.py')]
    assert 'fake-token' not in str(calls[0][0])  # credentials never on the command line
    # The child inherits the environment as before, plus only the backend API token, data root and exit rule.
    env = calls[0][1]['env']
    assert set(env) - set(launcher.os.environ) == {'RIKO_API_TOKEN', 'RIKO_DATA_DIR', 'RIKO_EXIT_WITH_BACKEND'}
    assert calls[0][1]['stdin'] == launcher.subprocess.PIPE  # closes when this backend exits
    assert env['RIKO_API_TOKEN'] == 'x' * 43 and env['RIKO_DATA_DIR'] == str(tmp_path)
    assert not calls[0][1].get('shell')
    service.stop()
    assert process.code == 0
    assert not service.status()['running']


def test_launcher_configuration_failures_never_spawn_or_expose_values(tmp_path, monkeypatch):
    monkeypatch.setattr('dotenv.dotenv_values', lambda path: {})
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Unexpected process launch'))
    monkeypatch.setattr(launcher.os, 'environ', {})
    service = launcher.DiscordLauncher(tmp_path)
    with pytest.raises(ValueError, match='Discord_bot_token'): service.start()
    monkeypatch.setattr(launcher.os, 'environ', {'Discord_bot_token': 'private-test-token', 'Discord_admins': 'invalid-private-id'})
    with pytest.raises(ValueError, match='configuration is invalid'): service.start()
    assert 'private' not in str(service.status())


def test_launcher_reports_exit_without_exposing_child_output(tmp_path):
    service = launcher.DiscordLauncher(tmp_path)
    process = SimpleNamespace(wait=lambda: 1)
    service.process = process
    service._watch(process)
    assert not service.status()['running']
    assert 'exited' in service.status()['error']


def test_worker_started_by_the_backend_exits_when_the_backend_goes_away():
    import subprocess, sys
    code = "import sys, types; sys.path.insert(0, 'Code'); import discord_bot; discord_bot.exit_with_backend(types.SimpleNamespace())"
    child = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE)
    child.stdin.close()  # what the OS does when the backend process ends
    assert child.wait(timeout=20) == 0
