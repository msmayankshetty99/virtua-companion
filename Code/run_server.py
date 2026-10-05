"""Backend entry point: standalone in development, Electron-owned in releases."""
from pathlib import Path
import asyncio
import os
import logging

import uvicorn

from process.app_core.configuration.config import load_config


class Server(uvicorn.Server):
    """uvicorn, except that the active turn ends before requests drain: a reply still generating would hold its
    request for the whole drain, and Electron allows the entire shutdown 15 s (electron/release.cjs)."""
    def __init__(self, config, stop_turn):
        super().__init__(config)
        self.stop_turn = stop_turn

    async def shutdown(self, sockets=None):
        await asyncio.to_thread(self.stop_turn)
        await super().shutdown(sockets=sockets)


def main():
    import sys
    if '--release-check' in sys.argv:
        import ctypes
        library = Path(sys.argv[sys.argv.index('--release-check') + 1]).resolve()
        directory = os.add_dll_directory(str(library.parent)) if os.name == 'nt' else None
        try:
            dll = ctypes.CDLL(str(library))
            for symbol in ('riko_create', 'riko_request', 'riko_set_interval', 'riko_stop', 'riko_destroy'): getattr(dll, symbol)
            import torch, faster_whisper, sounddevice, silero_vad, onnxruntime
        finally:
            if directory: directory.close()
        return
    if '--tool-worker' in sys.argv:
        import runpy
        runpy.run_module('process.app_core.tools.worker', run_name='__main__')
        return
    if '--discord-worker' in sys.argv:
        import runpy
        runpy.run_module('discord_bot', run_name='__main__')
        return
    os.chdir(Path(os.environ.get('RIKO_DATA_DIR', Path(__file__).resolve().parents[1])))
    config = load_config()
    from process.app_core.configuration.debug_logging import configure_logging
    configure_logging(config)
    # Debug our application, not WebSocket frames. Uvicorn DEBUG dumps every
    # voice.level/chat.delta packet and can dominate the capture/UI event loop.
    print("Riko AI server: http://127.0.0.1:8765 — Ctrl+C to stop", flush=True)
    import desktop_server
    # Hold the port before the model loads, so no other process (or account) can listen on it and
    # collect the API token Electron sends. In development this start's token is minted only now, so a
    # squatter could only have seen the previous one; packaged Electron sends its token only after the
    # line below. Requests queue until startup finishes.
    import socket
    import time
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE if os.name == 'nt' else socket.SO_REUSEADDR, 1)
    deadline = time.monotonic() + 20  # a previous backend may still be shutting down after a quick relaunch
    while True:
        try: listener.bind(('127.0.0.1', 8765)); break
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(f'Port 8765 is already in use ({exc}); stop the other process and try again.') from None
            time.sleep(.5)
    listener.listen(2048)
    desktop_server.api_secrets()
    if os.environ.get('RIKO_MANAGED') == '1': print('RIKO_BACKEND_LISTENING', flush=True)  # see electron/release.cjs
    server = Server(uvicorn.Config(desktop_server.app, host="127.0.0.1", port=8765,
                log_level="info", timeout_graceful_shutdown=2), desktop_server.stop_turn)
    if os.environ.get('RIKO_MANAGED') == '1':
        import faulthandler
        import threading
        def managed_shutdown():
            for line in sys.stdin:
                if line.strip() == 'shutdown': break
            # Electron kills us 15 s after asking (electron/release.cjs). If teardown hangs, write every thread's
            # stack to the launch log and exit at 14 s instead of outliving the app with :8765 and the model.
            try: faulthandler.dump_traceback_later(14, exit=True)
            except Exception: pass  # no usable stderr
            server.should_exit = True
        threading.Thread(target=managed_shutdown, name='release-shutdown', daemon=True).start()
    try: server.run(sockets=[listener])
    except KeyboardInterrupt: pass  # uvicorn re-raises Ctrl+C after its graceful shutdown; uvicorn.run() swallows it too


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
