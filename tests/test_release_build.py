import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

import pytest
import yaml

import run_server
from process.app_core.configuration.native_backends import backends_for, library_name


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
    assert '-DGGML_CUDA_NO_VMM=ON' in release_build().native.cmake_flags('cuda')
    assert '*native.cmake_flags(backend)]' in source
    assert 'env=environment' in source and 'release_check_environment(os.environ)' in source


def test_build_names_bundles_from_the_manifest_and_ships_it_to_the_frozen_backend():
    # electron/native_backends.json is the only place that names libraries; the frozen backend resolves bundled:<backend>
    # from the copy PyInstaller puts where native_backends.manifest_path() looks under sys._MEIPASS.
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert not any(name in source for name in ('riko-native.dll', 'libriko-native', 'GGML_CUDA=', 'GGML_VULKAN='))
    assert source.count('native.library_name(sys.platform)') == 2 and "'--add-data', manifest_data()" in source
    assert release_build().manifest_data() == f'{root / "electron" / "native_backends.json"}{os.pathsep}electron'


def test_release_check_refuses_a_library_bundled_resolution_would_not_find(tmp_path):
    for library in (tmp_path / 'native' / backends_for()[0] / 'riko-native.bogus', tmp_path / 'native' / 'other' / library_name()):
        with pytest.raises(SystemExit, match='bundled:<backend>'): run_server.release_check(library)


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


def test_each_os_builds_its_own_bundles_with_an_rpath_its_loader_expands():
    # dyld ignores $ORIGIN, so a macOS bundle with it would resolve its own libraries only from the build directory.
    build = release_build()
    metal = build.native_flags('metal', 'darwin')
    assert {'-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_MACOSX_RPATH=ON', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=@loader_path',
        '-DCMAKE_BUILD_RPATH=@loader_path', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0'} <= set(metal)
    assert not any('$ORIGIN' in flag or 'CUDA_FLAGS' in flag for flag in metal)
    for backend in ('cuda', 'vulkan'):  # unchanged for Linux: Release, then the install rpath for build-tree binaries too
        linux = build.native_flags(backend, 'linux')
        assert linux[-3:] == ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=$ORIGIN']
        assert not any('MACOSX' in flag or '@loader_path' in flag or 'DEPLOYMENT_TARGET' in flag for flag in linux)
        windows = build.native_flags(backend, 'win32')  # multi-config Visual Studio: --config Release, and no rpath
        assert f'-DGGML_{backend.upper()}=ON' in windows and not any('RPATH' in flag or 'BUILD_TYPE' in flag for flag in windows)
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert source.count('native.backends_for(sys.platform)') == 3  # what to build, what to release-check, what a smoke build may build
    assert "('cuda', 'vulkan')" not in source and 'RPATH=$ORIGIN' not in source


def symlink_or_skip(target, link):
    try: os.symlink(target, link)
    except OSError: pytest.skip('creating symlinks needs a privilege this host lacks')


def test_bundles_keep_versioned_library_links_as_links(tmp_path):
    # libriko-native links @rpath/libggml.0.dylib (libggml.so.0 on Linux): following the links copied every library three times.
    build, release = tmp_path / 'build', release_build()
    destination, outside = tmp_path / 'native' / 'metal', tmp_path / 'elsewhere'
    for folder in (build / 'bin', build / 'CMakeFiles', destination, outside): folder.mkdir(parents=True)
    (build / 'bin' / 'libggml.0.25.3.dylib').write_bytes(b'ggml')
    symlink_or_skip('libggml.0.25.3.dylib', build / 'bin' / 'libggml.0.dylib')
    symlink_or_skip('libggml.0.dylib', build / 'bin' / 'libggml.dylib')
    (outside / 'libextra.dylib').write_bytes(b'extra')
    symlink_or_skip(outside / 'libextra.dylib', build / 'bin' / 'libextra.dylib')
    for name in ('bin/libriko-native.dylib', 'bin/ggml.dll', 'bin/libggml.so.0', 'bin/libggml.so', 'CMakeFiles/notes.dylib.txt', 'bin/version.sorted'):
        (build / name).write_bytes(b'x')
    found = release.shared_libraries(build, 'darwin')
    assert [file.name for file in found] == ['libextra.dylib', 'libggml.0.25.3.dylib', 'libggml.0.dylib', 'libggml.dylib', 'libriko-native.dylib']
    for _ in range(2):  # a rerun replaces what an earlier run copied
        for file in found: release.copy_shared_library(file, destination)
    assert os.readlink(destination / 'libggml.dylib') == 'libggml.0.dylib' and os.readlink(destination / 'libggml.0.dylib') == 'libggml.0.25.3.dylib'
    assert not (destination / 'libggml.0.25.3.dylib').is_symlink() and (destination / 'libggml.dylib').read_bytes() == b'ggml'
    assert not (destination / 'libextra.dylib').is_symlink() and (destination / 'libextra.dylib').read_bytes() == b'extra'
    assert release.dangling_links(destination) == []
    (destination / 'libggml.0.25.3.dylib').unlink()
    assert release.dangling_links(destination) == ['libggml.0.dylib', 'libggml.dylib']
    assert [file.name for file in release.shared_libraries(build, 'linux')] == ['libggml.so', 'libggml.so.0']  # not version.sorted
    assert [file.name for file in release.shared_libraries(build, 'win32')] == ['ggml.dll']


def test_cuda_redistributables_go_only_into_the_cuda_bundle_with_their_links(tmp_path):
    release = release_build()
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert source.count('copy_cuda_redistributables(destination)') == 1 and "if backend == 'cuda': copy_cuda_redistributables(destination)" in source
    toolkit, destination = tmp_path / 'cuda' / 'targets' / 'x86_64-linux' / 'lib', tmp_path / 'native' / 'cuda'
    for folder in (toolkit, destination): folder.mkdir(parents=True)
    for library, version in (('libcudart', '12.4.127'), ('libcublas', '12.4.5.8'), ('libcublasLt', '12.4.5.8')):
        (toolkit / f'{library}.so.{version}').write_bytes(library.encode())
        symlink_or_skip(f'{library}.so.{version}', toolkit / f'{library}.so.12')
        symlink_or_skip(f'{library}.so.12', toolkit / f'{library}.so')
    (toolkit / 'libcuda.so').write_bytes(b'driver stub')
    release.copy_cuda_redistributables(destination, 'linux', tmp_path / 'cuda')
    assert sorted(file.name for file in destination.iterdir() if not file.is_symlink()) == ['libcublas.so.12.4.5.8', 'libcublasLt.so.12.4.5.8', 'libcudart.so.12.4.127']
    assert os.readlink(destination / 'libcublasLt.so.12') == 'libcublasLt.so.12.4.5.8' and release.dangling_links(destination) == []
    assert not (destination / 'libcuda.so').exists()
    with pytest.raises(RuntimeError, match='CUDA redistributable missing'): release.copy_cuda_redistributables(destination, 'linux', tmp_path / 'none')


def test_mac_icon_is_rendered_from_the_logo_where_electron_builder_packages_it(tmp_path, monkeypatch):
    root, release, calls = Path(__file__).resolve().parents[1], release_build(), []
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: calls.append([str(arg) for arg in args]))
    icns = release.mac_icon(root / 'assets/logo.svg', tmp_path / 'mac' / 'icon.icns')
    assert calls[-1] == ['iconutil', '-c', 'icns', str(tmp_path / 'mac' / 'icon.iconset'), '-o', str(icns)]
    renders = [call for call in calls if call[1:3] == ['-s', 'format']]
    assert len(renders) == 10 and all(call[7:9] == [str(root / 'assets/logo.svg'), '--out'] for call in renders)
    assert renders[-1][4:7] == ['-z', '880', '880'] and calls[-2][1:4] == ['-p', '1024', '1024']  # 512@2x: 880 px of art, padded
    assert sorted({int(call[2]) for call in calls if call[1] == '-p'}) == [16, 32, 64, 128, 256, 512, 1024]
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "if sys.platform == 'darwin': mac_icon(ROOT / 'assets/logo.svg', STAGE / 'mac' / 'icon.icns')" in source
    builder = yaml.safe_load((root / 'electron/electron-builder.yml').read_text(encoding='utf-8'))
    assert (root / 'electron' / builder['mac']['icon']).resolve() == (release.STAGE / 'mac' / 'icon.icns').resolve()


@pytest.mark.skipif(sys.platform != 'darwin', reason='sips and iconutil are part of macOS')
def test_mac_icon_renders_with_the_system_tools(tmp_path):
    icns = release_build().mac_icon(Path(__file__).resolve().parents[1] / 'assets/logo.svg', tmp_path / 'icon.icns')
    assert icns.read_bytes()[:4] == b'icns' and icns.stat().st_size > 10_000


def test_frozen_backend_copy_keeps_pyinstaller_symlinks():
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "shutil.copytree(STAGE / 'riko-backend', STAGE / 'backend', symlinks=True, dirs_exist_ok=True)" in source


def test_release_workflow_packages_a_macos_dmg_without_the_cuda_and_vulkan_steps():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8'))
    job = workflow['jobs']['package']
    assert job['strategy']['matrix']['os'] == ['windows-2022', 'ubuntu-22.04', 'macos-15']  # macos-15: Apple silicon
    steps = {step.get('name'): step for step in job['steps']}
    assert steps['CUDA build toolkit']['if'] == "runner.os != 'macOS'"
    assert steps['Linux build prerequisites']['if'] == steps['Linux Vulkan SDK shader compiler']['if'] == "runner.os == 'Linux'"
    assert steps['Windows Vulkan SDK']['if'] == "runner.os == 'Windows'"
    assert steps['Build the native backends and freeze Python']['run'] == 'python tools/release/build.py'
    package = steps['Package installers']
    assert package['env']['CSC_IDENTITY_AUTO_DISCOVERY'] == 'false'
    secrets = {key: value for key, value in package['env'].items() if key.startswith('MAC_')}
    assert len(secrets) == 5 and all(value.startswith("${{ runner.os == 'macOS' && secrets.") for value in secrets.values())
    # An empty CSC_LINK makes electron-builder import an empty certificate: export it only when the secret is set.
    script = package['run']
    assert 'CSC_LINK' not in package['env'] and script.index('if [ -n "$MAC_CSC_LINK" ]; then') < script.index('export CSC_LINK=')
    assert script.index('export CSC_LINK=') < script.index('export APPLE_API_KEY=') < script.index('npx --no-install electron-builder')
    upload = job['steps'][-1]['with']
    assert 'electron/release/*.dmg' in upload['path'] and upload['name'] == "Riko-${{ runner.os }}-${{ runner.os == 'macOS' && 'arm64' || 'x64' }}"


def test_pyinstaller_sees_the_app_package_when_its_spec_collects_submodules():
    # The spec's collect_submodules('process') runs before Analysis applies --paths: without Code/ on PYTHONPATH it returned
    # nothing, and the frozen --tool-worker failed the release check with "No module named process.app_core.tools.worker".
    root, release = Path(__file__).resolve().parents[1], release_build()
    source = (root / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "'--collect-submodules', 'process'" in source and "ROOT / 'Code/run_server.py', env=freeze_environment(os.environ))" in source
    assert release.freeze_environment({'PATH': '/usr/bin'}) == {'PATH': '/usr/bin', 'PYTHONPATH': str(root / 'Code')}
    assert release.freeze_environment({'PYTHONPATH': 'extra'})['PYTHONPATH'] == str(root / 'Code') + os.pathsep + 'extra'
    pytest.importorskip('PyInstaller', reason='release CI installs PyInstaller from the hash lock')  # what the spec then sees:
    found = subprocess.run([sys.executable, '-c', "from PyInstaller.utils.hooks import collect_submodules; print('\\n'.join(collect_submodules('process')))"],
        cwd=root, env=release.freeze_environment({key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}),
        capture_output=True, text=True, timeout=300).stdout.split()
    assert {'process.app_core.tools.worker', 'process.app_core.tools.builtin.scientific_calculator', 'process.app_core.tools.builtin.todo_list'} <= set(found)


def test_release_check_runs_per_shipped_bundle_on_the_stage_and_inside_the_signed_mac_app(tmp_path, monkeypatch):
    release, calls = release_build(), []
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: calls.append(([str(arg) for arg in args], kwargs['env'])))
    app = tmp_path / 'Riko.app'
    release.check_app(app)
    resources = app / 'Contents' / 'Resources'
    assert [call[0] for call in calls] == [[str(resources / 'backend' / 'riko-backend'), '--release-check', str(resources / 'native' / backend / library_name())]
        for backend in backends_for()]
    assert all(call[1] == release.release_check_environment(os.environ) for call in calls)
    source = (Path(__file__).resolve().parents[1] / 'tools/release/build.py').read_text(encoding='utf-8')
    assert "    release_check(executable, STAGE / 'native')\n" in source
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text(encoding='utf-8'))
    steps = {step.get('name'): step for step in workflow['jobs']['package']['steps']}
    check = steps['Release check inside the signed macOS app']
    assert check['if'] == "runner.os == 'macOS'" and check['run'] == 'python tools/release/build.py --check-app electron/release/mac-arm64/Riko.app'
    names = [step.get('name') for step in workflow['jobs']['package']['steps']]
    assert names.index('Package installers') < names.index('Release check inside the signed macOS app') < len(names) - 1  # before the upload


def test_probe_feature_version_matches_the_native_patch():
    # The bridge reports this string in /props and tags every sample with it; the provider refuses a mismatch at startup.
    from process.app_core.emotion.probe import FEATURE_VERSION
    patch = (Path(__file__).resolve().parents[1] / 'tools/llama_cpp/emotion-probe.patch').read_text(encoding='utf-8')
    assert f'"{FEATURE_VERSION}"' in patch
