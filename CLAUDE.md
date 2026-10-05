# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Riko is a local desktop AI companion with a VRM avatar.
- **Python backend** (`Code/`, FastAPI/uvicorn on `127.0.0.1:8765`) owns conversation, in-process llama.cpp inference, microphone capture/ASR, TTS playback, memory, tools and persistence. Core package: `Code/process/app_core/` (written `app_core/` below). Entry points: `Code/run_server.py` (backend), `Code/desktop_server.py` (FastAPI app), `Code/discord_bot.py`, `Code/task_mcp_server.py`.
- **Electron + React + three.js** (`electron/`) owns every window, the avatar and the UI, and talks to the backend only over HTTP/WebSocket.

This layout is on `main` (formerly the `rewrite-2` branch). `origin/llama.cpp` is an older pre-rewrite tree, kept for reference, whose tip fails to parse (`Code/process/llm_scripts/module.py:238`). `old llm_scripts/` here is dead code (the space in its name makes it unimportable).

## Commands

No linter or formatter is configured. Besides tests, the README's only checks are `node --check main.cjs` and `node --check preload.cjs` in `electron/`.

```bash
# Python >= 3.11 (CI uses 3.11). Tests need the full runtime deps; the [test] extra pulls the [runtime] extra, which tests/test_dependency_manifests.py keeps equal to requirements-runtime.txt.
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps EfficientWord-Net
# Linux CI pre-installs CPU torch: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# install_reqs.sh (bash, set -euo pipefail) picks torch wheels per machine: PyPI on macOS, cu130/cu126 by NVIDIA driver, rocm7.2 with rocminfo, else cpu; RIKO_TORCH_INDEX overrides.

python -u Code/run_server.py   # backend; run from the repo root; Ctrl+C stops it

# Python tests, from the repo root (pyproject sets pythonpath=Code, testpaths=tests)
HF_HUB_OFFLINE=1 python -m pytest -q
python -m pytest tests/test_tasks.py -q
python -m pytest tests/test_tasks.py::test_real_stdio_mcp_handshake_and_task_round_trip -q
RIKO_RELEASE_BUILD=1 python -m pytest tests/test_release_build.py tests/test_release_config.py tests/test_llama_native.py tests/test_responses_wire.py tests/test_settings_store.py -q   # everything CI runs

# Electron, from electron/ (CI uses Node 22)
npm ci && npm run build && npm run start   # loads dist/; start the backend first
npm run dev                                # Vite on :5173, then in another shell: RIKO_DEV=1 npm run start
npm test                                   # node --test src/*.test.mjs
node --test src/release.test.mjs                                          # one file
node --test --test-name-pattern='refuses overwrite' src/release.test.mjs  # one test

python -u Code/task_mcp_server.py --store persistent_memories/tasks.sqlite3   # optional stdio MCP server for external clients
```

Test caveats:
- Use `HF_HUB_OFFLINE=1`: otherwise `tests/test_emotion.py` downloads Julia-1 (unpinned) and imports code from it.
- `tests/test_server_logging.py` is stale (2 failures). `tests/test_settings_store.py::test_current_character_configuration_has_valid_settings` needs the private, gitignored `character_config.yaml` and only skips under `RIKO_RELEASE_BUILD=1`.
- Some tests use cwd-relative paths (`Path('Code')`, `Path('.')`), and `electron/src/formatted_text.test.mjs` / `polished_ui.test.mjs` start Vite SSR servers, so keep the working directories above.
- Every GPU and audio-device path is mocked; a green suite says nothing about CUDA, Metal or ROCm behaviour.

### Native llama.cpp bridge

`tools/llama_cpp/README.md` (PowerShell-oriented) is the reference. Check out llama.cpp at `b92761a515ea31e852e7fbc1fad5f874b46f3718`, run `git apply --check` and then `git apply` with `tools/llama_cpp/emotion-probe.patch`, then:

```bash
cmake -S .native/llama.cpp -B .native/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DRIKO_NATIVE_BRIDGE_SOURCE="$PWD/tools/llama_cpp/riko-native.cpp"
cmake --build .native/llama.cpp/build --config Release --target riko-native -j 2
```

`-DGGML_CUDA=OFF` gives a CPU-only build on Windows/Linux. On macOS the same configure builds Metal by default (embedded shader library, no full Xcode needed) and produces `libriko-native.dylib`. Point `runtime.native_library` at the built library and keep its llama/ggml libraries beside it.

Release (CI only, `.github/workflows/release.yml`): `python tools/release/build.py` clones llama.cpp into `.native/llama.cpp-release` (it refuses if that exists), builds the `cuda` and `vulkan` backends, freezes `riko-backend` with PyInstaller into `release-stage/` and runs `--release-check`; electron-builder 26.0.12 then packages from `electron/`.

## Architecture

### Process model
- **Dev:** backend and Electron start separately; Electron never starts or stops Python. **Packaged:** Electron spawns and stops the frozen backend.
- Python child processes: a fresh interpreter per built-in tool call (`app_core/tools/worker.py`, `--tool-worker` when frozen), stdio MCP servers from `mcp.json`, and the Discord client (`--discord-worker` when frozen). GPT-SoVITS is always a separate HTTP server.
- Every Electron window loads the same Vite bundle, routed by URL hash in `electron/src/main.jsx`: `#/overlay` (transparent click-through avatar layer), `#/control` (chat/settings; "closing" collapses it to a dock), `#/whiteboard`, `#/effects`, `#/setup` (packaged first run), `#/neural-data`.
- One `electron/preload.cjs` exposes every `contextBridge` API to every window. The `ipcMain` handlers in `electron/main.cjs` check that `event.sender` owns the request; new handlers must do the same.

### Startup and service assembly
- `run_server.main` chdirs to `$RIKO_DATA_DIR` (default: repo root), loads config, configures logging, then imports `desktop_server`. That import itself calls `load_config()` and builds `DiscordLauncher`, `GPUMonitor` and `WhiteboardImages`, so tests stub `load_config` before the first import (see `tests/test_desktop_api.py`).
- The FastAPI lifespan calls `create_chat_service` (`app_core/factory.py`), which builds everything inside an `ExitStack` that rolls back on failure: `ActionController` → optional Julia emotion engine → `create_provider` (+ optional emotion probe) → `provider.warmup()` (loads the GGUF, no timeout) → `TaskStore` / `ToolRegistry` → `MemoryStore` → `ChatService`. Then come `SessionManager`, `warm_session` (ASR/VAD/wake/TTS, bounded by `runtime.startup_timeout_seconds`), `ConversationStore`, `Initiative` and the resource/task-file watchers, and finally `runtime.ready` is published.
- `run_server.py` binds 127.0.0.1:8765 before the model loads (retrying for 20 s while a previous backend exits) and mints the API secrets right after; connections queue until lifespan startup finishes, so requests hang rather than fail while the model loads.
- With `desktop.setup_on_startup_error: true`, a failing `create_chat_service` leaves only the settings/status/resource/media routes up (the rest return 503) so Settings can repair the YAML.
- Shutdown: `run_server.Server.shutdown` first cancels the active turn (`desktop_server.stop_turn`, bounded 1 s), then uvicorn drains for 2 s; the lifespan stops Discord in parallel with `close_bounded(session, 8)` (`app_core/runtime/lifecycle.py`; a hung close is abandoned to a daemon thread) → unwind the ExitStack. An atexit guard in `llama_native.py` destroys live native contexts, or `os._exit`s when one cannot be destroyed, because ggml-metal's static destructor aborts on leftover Metal buffers. The whole sequence stays under Electron's 15 s.

### Threading and events
- `event_bus` (`app_core/events/bus.py`) is synchronous: listeners run on the publisher's thread, their exceptions are swallowed, and every event carries a monotonic `sequence`.
- `SessionManager` (`app_core/runtime/session.py`) serialises foreground turns with `_turn_lock`. A second turn raises `RuntimeError('Riko is already handling another turn')`, which `app_core/integrations/discord/api.py` string-matches into a 409. The `_voice_lock` RLock guards voice/playback state, and bus listeners can take it. Lock order is `_capture_lock` (request handlers only) → `_voice_lock` → component locks. **Never publish or run callbacks while holding a component lock**: queue them on an `Outbox` (`app_core/events/outbox.py`) and call `flush()` after releasing it. `tests/test_lock_discipline.py` enforces this with a static scan and deadlock regressions.
- Hidden coupling: voice input, animation, initiative, Discord and the server read private `SessionManager` fields (`_voice_lock`, `_active_turn`, `_generation_active`). Attributes are monkey-patched onto `ChatService` and providers and read back through `getattr` defaults. Renames break silently.
- Background work uses `DaemonExecutor` (`app_core/runtime/workers.py`; bounded queue, daemon threads). Each turn is cancelled through its own `threading.Event` and `TurnCancelled`.

### Inference
- `create_provider` (`app_core/inference/providers.py`) picks the provider:
  - `llama_cpp` → `InProcessLlamaProvider` (`llama_native.py`). It calls the C ABI in `tools/llama_cpp/riko-native.cpp` via ctypes: five `riko_*` exports, serving only `/v1/responses`, `/apply-template`, `/tokenize`, `/props` and `/slots`, with argv from `native_arguments` in `llama_context.py`. There is no HTTP listener, llama-server or llama-cpp-python, and no fallback: `runtime.native_library` is required.
  - `llama_server` → `LlamaServerProvider` (`llama_server.py`): the same `LlamaContextProvider` protocol (slot 0 for live, exact `/apply-template` + `/tokenize` counts, `/v1/responses` streaming and tools) against a llama-server the user runs at `runtime.base_url` (default `http://127.0.0.1:8080`; validated by `server_address` in `llama_runtime.py`). It uses `http.client`, one connection per request, and cancels by shutting the socket down (closing alone does not wake a read blocked during prefill). Startup waits on `/health` (503 while loading), then requires exactly `parallel_slots` slots each holding the largest of the live, initiative and reflection budgets, and prints the matching `llama-server` command otherwise; a connection failure resets it so the next request checks again. The server owns the model and GPU settings, and there is no emotion probe.
  - `openai`, `lm_studio`, `openai_compatible`, `ollama`, `local_http` → `OpenAIProvider`: Responses API with a Chat Completions fallback, byte-based token estimates, no slots, no provider-level cancel.
- `SlotScheduler` (`llama_context.py`) reserves slot 0 for the `live` lane. `initiative` and `reflection` share slots 1..N-1 (`runtime.parallel_slots`, 2–4), and live turns preempt them when `runtime.pause_background_on_live` is set. The emotion probe captures only on slot 0.
- `--ctx-size` comes from `kv_budget.pool_capacity`. `runtime.flash_attn` is `auto`/`on`/`off` (YAML booleans mean on/off); `n_gpu_layers` -1 lets llama.cpp `--fit` keep 1 GiB free, -2 requires every layer, and only -1 passes `--fit on`. `split_mode: row` loads but the native provider refuses it. Budgets above the GGUF's training context fail before loading, and `riko_create` returns llama.cpp's own WARN/ERROR lines on failure and its offload/flash-attention notes on success. Per-role budgets are enforced only by Python-side packing (`context_budget.pack_context`, with exact native token counts) and by `max_output_tokens`.
- Turn flow: `SessionManager.respond` → `ChatService.respond` (`app_core/conversation/chat.py`):
  1. memory capture and recall;
  2. the prompt: a stable system + history prefix (so the KV cache is reused), then `context_kind='optional'` system messages;
  3. `provider.generate`, which streams deltas;
  4. the `ToolRegistry` loop, up to `tools.max_iterations`;
  5. save history.
- The native context is allocated once, so changing `n_ctx`, `parallel_slots`, the KV pool or the initiative/reflection budgets needs a Python restart. A native crash kills the whole backend.
- The llama.cpp pin and the patch's five-file footprint are enforced by `tests/test_release_build.py`.

### Voice pipeline
- The mic opens only on `/api/voice/start|activate` or `/api/mic/toggle`. `VoiceInput` (`app_core/audio/voice_input.py`) runs these threads:
  1. `microphone-capture`: sounddevice, 16 kHz mono int16, 512-sample frames.
  2. `voice-vad`: Silero VAD, `WakeWord` (EfficientWord-Net ONNX on CPU) and `VoiceSegments`.
  3. `voice-asr`: faster-whisper.
  4. `voice-turn`: calls `SessionManager.respond`.
- Output: streamed deltas → `SpeechChunks` (`speech.max_words`) → `SpeechQueue` (`app_core/audio/speech.py`) → GPT-SoVITS over HTTP (`sovits_ping_config.url`) → raw PCM16 played by sounddevice. `sovits_ping_config.sample_rate` must match the server.
- Barge-in uses the character offsets of spoken chunks. `Interjections` cancels after `voice.interruption_seconds`, and history is rewritten at the estimated cut point. There is no echo cancellation.

### Persistence
The data root is the directory holding `character_config.yaml`.
- `persistent_memories/` is user data: never delete it during cleanup. Runtime files:
  - `chat_history.json`: the model's context. Both `ChatService` and `SessionManager` rewrite the whole file (fsynced). An unreadable file is kept as `chat_history.json.unreadable-<timestamp>` before anything replaces it. Only a turn still generating or playing is cut by Stop/Sleep/shutdown; failed and cancelled turns keep the user's message.
  - `memory_store.json` (`app_core/persistence/memory.py`): written via temp file, fsync and replace; fails closed. Its semantic index is an in-memory float32 numpy matrix, re-embedded from record text on every start and never written (FAISS was dropped: its libomp crashed beside torch's on macOS); `memory.index_file` is ignored.
  - `conversations.sqlite3`: the UI archive, built from bus events. It is a second history and can diverge from `chat_history.json`.
  - `tasks.sqlite3` (`app_core/persistence/tasks.py`): revision-checked, and shared with `Code/task_mcp_server.py`.
  - The whiteboard, desktop, initiative, tool-approval and Discord JSON files, plus `wake_words/`.
- Also under the data root: `logs/debug.log` (rotating, secrets redacted), and emotion-probe datasets and weights under `models/` (`app_core/emotion/probe_storage.py`).

### Configuration and settings
- `load_config` (`app_core/configuration/config.py`) reads `$RIKO_CONFIG`, or `./character_config.yaml`, with PyYAML; a missing file silently yields defaults (provider `openai`). Relative paths resolve against the YAML's directory (`AppConfig.root`). Only `runtime`, `tools`, `memory` and `emotion` are typed dataclasses; other sections are read from `config.raw`.
- `character_config.yaml` is gitignored and no template is committed. `electron/main.cjs` also parses it at launch and aborts if it is missing.
- The Settings UI goes through `SettingsStore` (`app_core/configuration/settings_store.py`, module-level `LOCK`):
  - GET returns values, fields and a sha256 `revision`.
  - Validate writes a temporary `.settings-validation-*.yaml` beside the config and runs `load_config` on it.
  - PUT re-checks the revision (409 on conflict), keeps `character_config.yaml.previous`, and writes via mkstemp + fsync + `os.replace`. ruamel preserves comments and unknown keys.
- Only keys already in the YAML or defaulted in `SettingsStore._values` are editable, and each type is inferred from the current value. To add a setting:
  1. Give it a default in `_values`, plus entries in `ENUMS`, `RANGES`, `HELP` and `LABELS`.
  2. For it to apply without a restart, also update the restart flags in `field()`, `restart_required` in `save()`, and `save_settings` in `Code/desktop_server.py`.
- Applied live: `avatar.*`, `runtime.pause_background_on_live`, and `emotion.probe.interval_tokens` (if the probe was enabled at load). Everything else needs a Python restart. Electron reads `desktop.shortcuts`, `desktop.debug` and `presets.default.name` only at launch. `persistent_memories/initiative_settings.json` overrides the YAML `initiative` section.
- The backend's PyYAML parses YAML 1.1; ruamel and Electron's `yaml` parse 1.2. They disagree on `yes`/`no`/`on`/`off`.

### Electron ↔ backend contract
- `http://127.0.0.1:8765` is hard-coded in `electron/src/api.mjs`, `electron/main.cjs` and several components. CORS allows `null` (the packaged `file://` renderer) and the Vite origin.
- Every HTTP and WebSocket request needs `Authorization: Bearer <token>` and a loopback Host; WebSockets also need an app Origin (`LocalAPIGuard` in `app_core/desktop/api_guard.py`). The token and a separate confirmation key are fresh on every backend start, minted only after `run_server.py` has bound 127.0.0.1:8765 (before the model loads), and removed from the backend's environment so children never inherit them:
  - Packaged: Electron generates both and passes `RIKO_API_TOKEN`/`RIKO_CONFIRM_KEY` to the backend it spawns; nothing is written to disk. Main sends the token only after the backend prints `RIKO_BACKEND_LISTENING` on stdout (`watchListening` in `electron/release.cjs`), and stops when it exits.
  - Dev: the backend writes `persistent_memories/api_token` and `confirm_key` (mode 0600) beside the config; Electron main re-reads them when they change.
  - Electron main injects the header for every renderer request to `127.0.0.1:8765` (`injectApiToken` in `electron/main.cjs`), so renderer code never handles the token; main's own requests use `backendFetch`. The Discord worker gets `RIKO_API_TOKEN` from its launcher and exits when its stdin pipe closes, i.e. with that backend (`RIKO_EXIT_WITH_BACKEND`); started by hand, `client_token()` reads the file on every reconnect. Use `127.0.0.1`, not `localhost`, which may resolve to `::1`.
  - Tests use `client_for(backend)` in `tests/test_desktop_api.py`. API docs are served only with `RIKO_DEV=1`.
- Security-sensitive changes (`security_sensitive()` in `Code/desktop_server.py`: anything that loads code, models or libraries, starts programs, widens file access or sends data elsewhere, judged by key name, by every leaf of a section saved at once, and by values that are URLs or absolute paths; turning tool approval off; Discord access) return 428 with a single-use `confirm` challenge. `request()` in `electron/src/api.mjs` asks Electron main (`confirm-security-change`), which shows a native dialog and signs the challenge with the confirmation key, which never goes over the wire. A new setting that can do any of those things must be covered by `security_sensitive()`.
- `/ws/events` (`app_core/events/stream.py`) sends `state.snapshot`, then `resource.snapshot`, then every bus event. A client with 100 queued events is closed with code 1013.
- All windows share one socket (`electron/src/event_connection.mjs`, 1 s reconnect). Subscribe via `connectEvents`, `useRuntime` or `useResource`, and never poll: `electron/src/subscription_guard.test.mjs` forbids `setInterval`.
- Every `DesktopState` change republishes a full `state.snapshot`. Panels read `resource.<topic>` events from `ResourceEvents` (`app_core/events/resources.py`); a new topic needs an `observe()` mapping and a `resource_getters()` entry.
- `POST /api/chat` returns only after the full reply; deltas arrive over the socket. Renderer acknowledgements are authoritative (`/api/surfaces/result`, `/api/avatar/animation/result`).

### Discord
`Code/discord_bot.py` is a separate process and only a backend client; it loads no model. It holds one WebSocket at `/ws/discord/client?instance=<uuid>` and calls `/api/discord/*` (`app_core/integrations/discord/api.py`). Start it with `POST /api/discord/start` or the tray menu, or by hand once the backend runs. Credentials go in a local `.env` (template: `.env.discord.example`).

### Dev vs packaged
- **Dev:** the repo root is both the cwd and the data root.
- **Packaged, first launch:** Electron shows `#/setup`, which writes `<data>/character_config.yaml` (never overwriting) and `data-location.json` in Electron's userData (`electron/release.cjs`).
- **Packaged, later launches:**
  - Electron spawns `backend/riko-backend` with `RIKO_MANAGED=1`, `RIKO_DATA_DIR`, `RIKO_CONFIG` and `RIKO_BUNDLE_ROOT`.
  - Hugging Face and torch caches go under `<data>/models`.
  - A `shutdown` line or EOF on stdin stops the backend (it dumps stacks and exits itself after 14 s); Electron sends SIGTERM after 15 s and SIGKILL 3 s later. On macOS the backend's PATH gets the Homebrew prefixes so MCP servers and ffmpeg resolve (`finderPath`).
  - `runtime.native_library: bundled:<cuda|vulkan>` resolves to `$RIKO_BUNDLE_ROOT/native/<backend>/`.

## Conventions and gotchas
- Writes go to a temp file and then `os.replace`; the memory store, settings and chat history also fsync. Stores fail closed rather than resetting user data: keep an unreadable file with `persistence/preserve.py` (`<name>.unreadable-<timestamp>`), or refuse to save over it.
- Optimistic concurrency uses revisions throughout: the settings `revision`, task `expected_revision` (`TaskConflict` → 409), the Discord access revision, memory-record revisions, and whiteboard `board_revision`.
- File paths supplied by the renderer or the model must go through `resolve_media` (`app_core/desktop/media.py`: approved roots plus an extension allowlist).
- Heavy dependencies (torch, faster_whisper, sounddevice, silero_vad, sentence_transformers, openai, huggingface_hub) are imported lazily, and tests swap them through `sys.modules`. Services accept `start=False` / `start_worker=False`. Tests must close services and unsubscribe from the global `event_bus` in `finally`.
- The `app_core/` root may contain only `__init__.py` and `factory.py`, and every relative import must resolve (`tests/test_app_core_structure.py`). `app_core/__init__.py` eagerly imports `SessionManager`, audio and emotion.
- Built-in tools (`app_core/tools/builtin/`) run in an isolated subprocess. `mcp.json` (copied from `mcp.json.example`) loads only if `tools.mcp_config` points at it. Registering a tool name twice silently overwrites the first.
- Electron:
  - Pure logic lives in `.mjs`/`.cjs` files, each with a sibling `*.test.mjs`.
  - Several tests regex-match the source text of `electron/main.cjs`, the CSS and some components, so refactors must update those tests too.
  - `electron/vite.config.mjs` passes `react` without calling it, so JSX uses the classic transform and every `.jsx` must `import React`.
- Style: the Python and JS are dense, often with several statements per line; match the surrounding code.
- The README links to `docs/*.md`, but `/docs` is gitignored and absent.
- `tools/migrate_legacy_memories.py` and `tools/migrate_legacy_chat_history.py` only dry-run unless given `--write`; run them with the app stopped.

## Platform status

Releases ship CUDA and Vulkan bundles for Windows and Linux x64 only; nothing builds, packages, detects or tests Metal or ROCm. The Python inference layer is backend-agnostic: the backend is whatever llama.cpp build `runtime.native_library` loads, or whatever llama-server `runtime.provider: llama_server` talks to (any backend, without the emotion probe; checked against stock llama-server b11408 on Apple Silicon). The patched bridge, including the emotion probe, has been built and run on Apple Silicon Metal at the current pin.

Backend and device assumptions live here:
- **Native build and packaging:**
  - `tools/release/build.py`: only cuda/vulkan; collects only `.dll`/`.so` files; uses an `$ORIGIN` rpath, which macOS dyld ignores (macOS needs `@loader_path`).
  - `.github/workflows/release.yml`: windows-2022 and ubuntu-22.04, CUDA 12.4.1.
  - `electron/electron-builder.yml`: NSIS, AppImage and deb.
- **Bundled-backend allowlists and library names:** `app_core/configuration/config.py`, `electron/release.cjs` (detects GPUs via nvidia-smi and vulkaninfo) and `electron/src/first_setup.jsx`. They are asserted by `tests/test_release_config.py` and `electron/src/release.test.mjs`.
- **Library loading:** `os.add_dll_directory` is called only on Windows (`app_core/inference/llama_native.py`, `Code/run_server.py`).
- **NVIDIA-only telemetry:** `app_core/resources/gpu_memory.py` (nvidia-smi, DXGI vendor `0x10de`, PowerShell counters) and the CUDA rows in `app_core/resources/vram_estimate.py`.
- **ASR:** every faster-whisper model is built by `app_core/audio/asr.py:create_whisper`. Defaults are `auto`/`default`, resolved from what CTranslate2 reports (CPU int8 on macOS, which has no CTranslate2 GPU backend); an unsupported configured pair falls back once with a warning, and Settings refuses one when it is edited. GPU ASR on Metal or ROCm needs a different engine (whisper.cpp, mlx-whisper, or a HIP-built CTranslate2).
- **Torch devices:** `app_core/runtime/torch_device.py` accepts `auto`/`cpu`/`cuda[:n]`/`mps` (ROCm torch reports `cuda`). Julia maps `mps` to CPU until upstream moves inputs for MPS, retries a failed GPU load on CPU, and restores torch's thread count after loading. `memory.device` covers the memory classifier and embedder; the settings `ENUMS` and `tests/test_llama_native.py` pin the lists.
- **Windows-only behaviour:** `WindowsActivity` in `app_core/runtime/initiative.py`, the Win32 pointer hooks in `electron/main.cjs`, and the acrylic material in `electron/window_material.cjs`.
