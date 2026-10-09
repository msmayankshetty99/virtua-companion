"""MCP over stdio: one persistent server process per configured server, restarted by the next call after it exits."""
from __future__ import annotations

from concurrent.futures import TimeoutError
import json
import os
import queue
import shutil
import subprocess
import threading
import time

from ..tool import ToolCancelled


def resolve_command(command, env):
    """Find an MCP server command on the PATH its process will get. Windows CreateProcess applies neither PATHEXT nor
    the child's PATH, so npx/uvx (.cmd shims) fail by bare name; a resolved .cmd/.bat path runs through cmd.exe."""
    if os.path.dirname(command): return command  # an explicit path runs as written (CreateProcess still adds .exe)
    search = os.pathsep.join(os.get_exec_path(env))
    found = shutil.which(command, path=search)
    if not found: raise FileNotFoundError(f'MCP server command {command!r} was not found on PATH: {search}')
    return found


class StdioMCPClient:
    """Minimal MCP JSON-RPC stdio client with a persistent server process."""
    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.command, self.args, self.env = command, args, env
        self._restart_lock = threading.Lock()
        self._start()

    def _start(self):
        # MCP stdio is UTF-8. Python servers on Windows write the ANSI code page unless told otherwise, and one byte that
        # does not decode would stop the stderr drain; PYTHONIOENCODING changes only their stdio, not their file encodings.
        environment = {'PYTHONIOENCODING': 'utf-8', **os.environ, **(self.env or {})}
        process = subprocess.Popen([resolve_command(self.command, environment), *(self.args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1, env=environment)
        responses = queue.Queue(maxsize=256)
        # A restart replaces the lock too: a request still holding the old one finishes against the old, exited process.
        self.process, self._responses, self._counter, self._lock = process, responses, 0, threading.Lock()
        def read_stdout():
            try:
                for line in process.stdout:
                    value = json.loads(line)
                    while True:
                        try: responses.put(value, timeout=.1); break
                        except queue.Full:
                            if process.poll() is not None: return
            except Exception as exc:
                try: responses.put_nowait(exc)
                except queue.Full: pass
            finally:
                try: responses.put_nowait(RuntimeError('MCP server closed stdout'))
                except queue.Full: pass
        threading.Thread(target=read_stdout, daemon=True, name='mcp-stdout').start()
        def drain_stderr():
            try:
                for _ in process.stderr: pass
            except (OSError, ValueError): pass
        threading.Thread(target=drain_stderr, daemon=True, name='mcp-stderr').start()
        try: self._initialize()
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=30):
        try: response = self._responses.get(timeout=timeout)
        except queue.Empty: raise TimeoutError('MCP response deadline exceeded')
        if isinstance(response, Exception): raise response
        return response

    def _request(self, method, params=None, cancelled=lambda: False):
        with self._lock:
            if cancelled(): raise ToolCancelled('MCP request cancelled')  # stopped while another request held the server
            self._counter += 1
            request = {"jsonrpc": "2.0", "id": self._counter, "method": method, "params": params or {}}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + 30
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError('MCP request deadline exceeded')
                if cancelled():
                    # MCP cancellation: the server should stop the request. A late reply carries this id, which later requests skip.
                    try:
                        self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/cancelled',
                            'params': {'requestId': request['id'], 'reason': 'The user stopped the reply'}}) + '\n')
                        self.process.stdin.flush()
                    except (OSError, ValueError): pass  # the server already exited; the next call restarts it
                    raise ToolCancelled('MCP request cancelled')
                try: response = self._read(min(remaining, .05))
                except TimeoutError: continue
                if response.get("id") == request["id"]:
                    if "error" in response: raise RuntimeError(response["error"])
                    return response.get("result", {})

    def _initialize(self):
        self._request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "riko", "version": "0.1"}})
        with self._lock:
            self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
            self.process.stdin.flush()

    def list_tools(self): return self._request("tools/list").get("tools", [])
    def call(self, name, arguments, cancelled=lambda: False):
        if self.process.poll() is not None:
            with self._restart_lock:
                if self.process.poll() is not None: self._start()
        return self._request("tools/call", {"name": name, "arguments": arguments}, cancelled)
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream: stream.close()
