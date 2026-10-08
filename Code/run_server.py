"""Backend entry point: standalone in development, Electron-owned in releases."""
from pathlib import Path
import os
import logging
import sys

# Imported lazily or dynamically, so a module or data file the frozen build misses would otherwise fail only on a
# user's machine (tools/release/build.py collects them).
RELEASE_MODULES = ('numpy', 'torch', 'transformers', 'sentence_transformers', 'faster_whisper', 'ctranslate2', 'silero_vad',
    'onnxruntime', 'sounddevice', 'soundfile', 'scipy.signal', 'ruamel.yaml', 'pypdf', 'openai', 'huggingface_hub', 'discord',
    'discord_bot', 'uvicorn')
# kernel/, which code imports partly inside functions (chat's metrics, the background budgets): the release check imports
# each by name, as it loads every name of the lazy process.app_core facade. tests/test_release_build.py keeps it equal to kernel/.
KERNEL_MODULES = tuple(f'process.app_core.kernel.{name}' for name in ('background_budget', 'cancellation', 'lifecycle', 'messages',
    'metrics', 'output_filter', 'streaming', 'torch_device', 'workers'))
# Installed only with an NVIDIA display driver: a bundle that loads it at load time fails everywhere else.
DRIVER_LIBRARIES = {'nvcuda.dll', 'libcuda.so', 'libcuda.so.1', 'libcuda.dylib'}


def is_driver(name):
    """The NVIDIA driver library under any name it loads by; Linux maps list libcuda.so.<driver version>."""
    import re
    return name.lower() == 'nvcuda.dll' or bool(re.fullmatch(r'libcuda\.(dylib|so(\.\d+)*)', name))


def server_class():
    import asyncio
    import uvicorn

    class Server(uvicorn.Server):
        """uvicorn, except that the active turn ends before requests drain: a reply still generating would hold its
        request for the whole drain, and Electron allows the entire shutdown 15 s (electron/release.cjs)."""
        def __init__(self, config, stop_turn):
            super().__init__(config)
            self.stop_turn = stop_turn

        async def shutdown(self, sockets=None):
            await asyncio.to_thread(self.stop_turn)
            await super().shutdown(sockets=sockets)
    return Server


def __getattr__(name):
    """uvicorn, Server and load_config load on first use and stay: frozen, every built-in tool call and the Discord client
    re-execute this binary (--tool-worker, --discord-worker), and main() dispatches those before any of them loads."""
    if name == 'uvicorn': import uvicorn as value
    elif name == 'Server': value = server_class()
    elif name == 'load_config': from process.app_core.configuration.config import load_config as value
    else: raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value


def loaded_libraries(names):
    """{name: path} for the shared libraries among `names` that this process has loaded."""
    import ctypes
    if os.name == 'nt':
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetModuleHandleW.argtypes, kernel32.GetModuleHandleW.restype = [wintypes.LPCWSTR], wintypes.HMODULE
        kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel32.GetModuleFileNameW.restype = wintypes.DWORD
        found = {}
        for name in names:
            buffer = ctypes.create_unicode_buffer(32768)
            module = kernel32.GetModuleHandleW(name)
            if module and kernel32.GetModuleFileNameW(module, buffer, len(buffer)): found[name] = Path(buffer.value)
        return found
    if sys.platform == 'darwin':
        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.argtypes, dyld._dyld_get_image_name.restype = [ctypes.c_uint32], ctypes.c_char_p
        paths = [os.fsdecode(name) for name in map(dyld._dyld_get_image_name, range(dyld._dyld_image_count())) if name]
    else:
        with open('/proc/self/maps', encoding='utf-8', errors='surrogateescape') as maps:
            paths = [fields[5].rstrip('\n') for fields in (line.split(None, 5) for line in maps) if len(fields) == 6 and fields[5].startswith('/')]
    return {Path(path).name: Path(path) for path in paths if Path(path).name in names or is_driver(Path(path).name)}


def native_bundle_problems(library, loaded):
    """Why a loaded native bundle cannot ship: one of its libraries resolved from elsewhere (a toolkit, SDK or
    LD_LIBRARY_PATH that users lack was shadowing it), or the NVIDIA driver library loaded with it."""
    bundle = library.resolve().parent
    return [f'{name} was loaded from {path}' for name, path in sorted(loaded.items())
            if is_driver(name) or path.resolve().parent != bundle]


def release_check(library=None):
    """`--release-check [library]`: can this build do what an installed app needs? tools/release/build.py runs it on the
    frozen backend once per native backend, without the build's toolkit paths; tests/test_release_build.py runs it on this
    tree without a library. It needs no GPU, audio device, credentials, network or model, and touches no user data."""
    import ctypes
    import importlib
    import subprocess
    import tempfile
    from process.app_core.configuration.native_backends import backends_for, library_name
    shipped, name = backends_for(sys.platform), library_name(sys.platform)  # the frozen copy bundled:<backend> resolves with
    if library and (Path(library).name != name or Path(library).absolute().parent.name not in shipped):
        raise SystemExit(f'{library} is not where bundled:<backend> looks on {sys.platform}: native/<{"|".join(shipped)}>/{name}')
    if library:
        library = Path(library).resolve()
        directory = os.add_dll_directory(str(library.parent)) if os.name == 'nt' else None
        try:
            dll = ctypes.CDLL(str(library))
            for symbol in ('riko_create', 'riko_request', 'riko_set_interval', 'riko_stop', 'riko_destroy'): getattr(dll, symbol)
        finally:
            if directory: directory.close()
        problems = native_bundle_problems(library, loaded_libraries({file.name for file in library.parent.iterdir()} | DRIVER_LIBRARIES))
        if problems: raise SystemExit('The native bundle is not self-contained:\n  ' + '\n  '.join(problems))
    for module in RELEASE_MODULES + KERNEL_MODULES: importlib.import_module(module)
    core = importlib.import_module('process.app_core')
    for name in core.__all__: getattr(core, name)
    from eff_word_net.audio_processing import Resnet50_Arc_loss
    from silero_vad import load_silero_vad
    Resnet50_Arc_loss(); load_silero_vad()  # wake words and voice activity: both models are package data files
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data:
        # desktop_server loads the config on import: the defaults, in an empty data root, never a developer's files.
        os.environ.update(RIKO_DATA_DIR=data, RIKO_CONFIG=str(Path(data) / 'character_config.yaml'))
        os.chdir(data)
        try:
            import desktop_server  # the FastAPI app and the whole app_core import graph
            from process.app_core.tools.builtin.scientific_calculator import Tool
            from process.app_core.tools.registry import ToolRegistry
            registry = ToolRegistry(timeout_seconds=300)  # through the real worker: --tool-worker when frozen
            try:
                registry.register_local(Tool({}, {}))
                result = registry.execute(Tool.TOOL_NAME, {'expression': '2**10'})
            finally: registry.close()
            if result.is_error or result.content != '1024': raise SystemExit(f'A built-in tool failed in its worker: {result.content}')
            # As DiscordLauncher starts it, but --dry-run stops before the credentials check and the login.
            script = Path(__file__).with_name('discord_bot.py')
            command = [sys.executable, '--discord-worker'] if getattr(sys, 'frozen', False) else [sys.executable, str(script)]
            worker = subprocess.run([*command, '--dry-run'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            if worker.returncode or 'RIKO_DISCORD_DRY_RUN_OK' not in worker.stdout:
                raise SystemExit(f'The Discord worker failed its dry run (exit {worker.returncode}):\n{worker.stdout}{worker.stderr}')
        finally: os.chdir(previous)
    print('RIKO_RELEASE_CHECK_OK', flush=True)


def main():
    if '--release-check' in sys.argv:
        arguments = sys.argv[sys.argv.index('--release-check') + 1:]
        release_check(arguments[0] if arguments else None)
        return
    if '--tool-worker' in sys.argv:
        import runpy
        runpy.run_module('process.app_core.tools.worker', run_name='__main__')
        return
    if '--discord-worker' in sys.argv:
        import runpy
        runpy.run_module('discord_bot', run_name='__main__')
        return
    this = sys.modules[__name__]  # uvicorn, Server and load_config resolve through __getattr__ (and tests replace them there)
    os.chdir(Path(os.environ.get('RIKO_DATA_DIR', Path(__file__).resolve().parents[1])))
    config = this.load_config()
    from process.app_core.configuration.debug_logging import configure_logging
    configure_logging(config)
    # Debug our application, not WebSocket frames. Uvicorn DEBUG dumps every
    # voice.level/chat.delta packet and can dominate the capture/UI event loop.
    # Redirected stdout uses the ANSI code page on Windows (cp932 has no em dash); never let output stop startup.
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(errors='backslashreplace')
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
    server = this.Server(this.uvicorn.Config(desktop_server.app, host="127.0.0.1", port=8765,
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
