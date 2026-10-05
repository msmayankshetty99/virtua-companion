import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

import run_server


def release_build():
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


def test_cmake_source_path_uses_forward_slashes_on_windows():
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    argument = build.cmake_file_definition('RIKO_NATIVE_BRIDGE_SOURCE',
        PureWindowsPath(r'D:\a\riko_project\riko_project\tools\llama_cpp\riko-native.cpp'))
    assert argument == '-DRIKO_NATIVE_BRIDGE_SOURCE:FILEPATH=D:/a/riko_project/riko_project/tools/llama_cpp/riko-native.cpp'
    assert '\\' not in argument


def test_release_patch_normalizes_windows_line_endings(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'tools/release/build.py'
    spec = importlib.util.spec_from_file_location('release_build', source)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    patch = tmp_path / 'change.patch'
    patch.write_bytes(b'patch line\r\nnext line\r\n')
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    build.apply_patch(tmp_path, patch)
    assert calls[0][0][0] == ['git', 'apply', '--check', '-']
    assert calls[1][0][0] == ['git', 'apply', '-']
    assert all(call[1]['input'] == b'patch line\nnext line\n' for call in calls)
    assert all(call[1]['check'] for call in calls)


def test_upstream_patch_is_limited_to_probe_and_build_glue():
    root = Path(__file__).resolve().parents[1]
    patch = (root / 'tools/llama_cpp/emotion-probe.patch').read_text()
    files = {line.split(' b/', 1)[1] for line in patch.splitlines() if line.startswith('diff --git ')}
    assert files == {'tools/server/CMakeLists.txt', 'tools/server/server-context.cpp',
        'tools/server/server-context.h', 'tools/server/server-task.cpp', 'tools/server/server-task.h'}
    assert not (root / 'tools/llama_cpp/cuda-infinity.patch').exists()


def test_frozen_backend_runs_in_utf8_mode():
    # PyInstaller apps ignore PYTHONUTF8, so the backend and its frozen --tool-worker/--discord-worker children get
    # UTF-8 stdio only from this build option (Windows otherwise uses the ANSI code page).
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--python-option', 'X utf8'" in source


def test_cuda_bundle_does_not_link_the_nvidia_driver():
    # ggml-cuda otherwise links nvcuda.dll/libcuda.so.1 directly, and the release check could not even load it on a
    # runner without an NVIDIA driver (run_server.DRIVER_LIBRARIES fails the check if it is loaded anyway).
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'-DGGML_CUDA_NO_VMM=ON'" in source
    assert 'env=environment' in source and 'release_check_environment(os.environ)' in source


def test_release_check_environment_drops_the_build_toolchain():
    toolkit = os.pathsep.join(['/usr/local/cuda-12.4/bin', '/tmp/vulkan-sdk/x86_64/bin', '/opt/NVIDIA GPU Computing Toolkit/v12.4/bin', '/usr/bin', ''])
    environment = release_build().release_check_environment({'PATH': toolkit, 'HOME': '/home/runner', 'LD_LIBRARY_PATH': '/usr/local/cuda/lib64',
        'CUDA_PATH': '/usr/local/cuda', 'CUDA_PATH_V12_4': 'x', 'VULKAN_SDK': '/tmp/vulkan-sdk', 'PYTHONPATH': 'Code',
        'RIKO_CONFIG': 'character_config.yaml', 'RIKO_DATA_DIR': '.'})
    assert environment == {'PATH': '/usr/bin', 'HOME': '/home/runner', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    assert release_build().release_check_environment({'Path': '/opt/Cuda/bin' + os.pathsep + '/windows'})['Path'] == '/windows'


def test_native_bundle_problems_name_shadowed_and_driver_libraries(tmp_path):
    bundle = tmp_path / 'native' / 'cuda'
    bundle.mkdir(parents=True)
    library = bundle / 'libriko-native.so'
    loaded = {'libriko-native.so': library, 'libcublas.so.12': bundle / 'libcublas.so.12'}
    assert run_server.native_bundle_problems(library, loaded) == []
    toolkit = tmp_path / 'cuda' / 'lib64' / 'libcublasLt.so.12'
    problems = run_server.native_bundle_problems(library, {**loaded, 'libcublasLt.so.12': toolkit, 'libcuda.so.1': bundle / 'libcuda.so.1'})
    assert problems == [f'libcublasLt.so.12 was loaded from {toolkit}', f'libcuda.so.1 was loaded from {bundle / "libcuda.so.1"}']


def test_loaded_libraries_finds_this_process_system_library():
    name = 'kernel32.dll' if os.name == 'nt' else 'libSystem.B.dylib' if sys.platform == 'darwin' else 'libc.so.6'
    found = run_server.loaded_libraries({name, 'not-a-loaded-library.so'})
    assert list(found) == [name] and found[name].is_absolute()


def test_release_check_passes_on_this_tree_without_a_native_library():
    # The frozen backend runs this same check (tools/release/build.py), so a dynamic import or data file the build
    # would miss, or a broken --tool-worker or --discord-worker path, fails here before a release does.
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'Code/run_server.py'), '--release-check'], cwd=root,
        env=release_build().release_check_environment(os.environ), capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RIKO_RELEASE_CHECK_OK' in result.stdout


def test_frozen_worker_entry_points_route_through_run_server(tmp_path):
    # Frozen, the backend is its own tool and Discord worker (ToolRegistry, DiscordLauncher): the same argv here.
    root = Path(__file__).resolve().parents[1]
    environment = {**release_build().release_check_environment(os.environ), 'RIKO_DATA_DIR': str(tmp_path)}
    server = [sys.executable, str(root / 'Code/run_server.py')]
    request = {'module': 'process.app_core.tools.builtin.scientific_calculator', 'class': 'Tool', 'arguments': {'expression': '6*7'}}
    tool = subprocess.run([*server, '--tool-worker'], input=json.dumps(request), capture_output=True, text=True, encoding='utf-8',
        env=environment, cwd=tmp_path, timeout=300)
    assert json.loads(tool.stdout.splitlines()[-1]) == {'result': '42'}, tool.stdout + tool.stderr
    discord = subprocess.run([*server, '--discord-worker', '--dry-run'], capture_output=True, text=True, encoding='utf-8',
        env=environment, cwd=tmp_path, timeout=300)
    assert discord.returncode == 0 and 'RIKO_DISCORD_DRY_RUN_OK' in discord.stdout, discord.stdout + discord.stderr
    assert not list(tmp_path.iterdir())  # a dry run writes nothing into the data root


def test_frozen_numpy_and_locked_release_dependencies():
    # numpy >= 2.3 imports numpy._core._exceptions only from C, so PyInstaller cannot see it.
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--hidden-import', 'numpy._core._exceptions'" in source
    workflow = (root / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert 'pip install --require-hashes -r tools/release/requirements-lock.txt' in workflow
    assert 'pip install --require-hashes --no-deps -r tools/release/requirements-lock-no-deps.txt' in workflow
    assert '-r requirements-runtime.txt' not in workflow and 'pyinstaller==' not in workflow
    # electron-builder comes from electron/package-lock.json through npm ci, never an ad-hoc install.
    assert 'npm install' not in workflow and 'npx --no-install electron-builder' in workflow


def test_driver_library_is_recognised_under_its_versioned_linux_name_and_ci_steps_stop_on_failure():
    import run_server
    assert all(run_server.is_driver(name) for name in ('nvcuda.dll', 'NVCUDA.DLL', 'libcuda.so', 'libcuda.so.1', 'libcuda.so.550.54.15', 'libcuda.dylib'))
    assert not any(run_server.is_driver(name) for name in ('libcudart.so.12', 'libcuda_helper.so', 'cudart64_12.dll'))
    library = Path('/bundle/libriko-native.so')
    assert run_server.native_bundle_problems(library, {'libcuda.so.550.54.15': Path('/usr/lib/x86_64-linux-gnu/libcuda.so.550.54.15')})
    workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert 'defaults:\n      run:\n' in workflow and 'shell: bash' in workflow
