from fnmatch import fnmatch
import json
from pathlib import Path
import sys

import pytest

from process.app_core.configuration import native_backends
from process.app_core.configuration.config import load_config
from process.app_core.configuration.native_backends import backends_for, bundled_library, cmake_flags, library_name, platform_build

# Written out, not derived: each OS is checked on every host (electron/src/release.test.mjs holds the same table).
LIBRARIES = {'win32': 'riko-native.dll', 'linux': 'libriko-native.so', 'darwin': 'libriko-native.dylib'}
SHIPPED = {'win32': ['cuda', 'vulkan'], 'linux': ['cuda', 'vulkan'], 'darwin': ['metal']}


def test_bundled_backend_resolves_current_install_location(tmp_path, monkeypatch):
    backend = backends_for()[0]  # cuda on Windows and Linux, metal on macOS
    config = tmp_path / 'data' / 'character_config.yaml'
    config.parent.mkdir()
    config.write_text(f'runtime:\n  provider: llama_cpp\n  native_library: bundled:{backend}\n  hf_repo_id: owner/model\n  hf_filename: model.gguf\n')
    resources = tmp_path / 'application'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    loaded = load_config(config)
    assert loaded.root == config.parent
    assert loaded.runtime.native_library == resources / 'native' / backend / LIBRARIES[sys.platform]
    assert loaded.memory.store_file.is_relative_to(config.parent)
    moved = tmp_path / 'upgraded-app'
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(moved))
    assert load_config(config).runtime.native_library.is_relative_to(moved)


def test_bundled_backend_rejects_untrusted_names(tmp_path, monkeypatch):
    config = tmp_path / 'character_config.yaml'
    config.write_text('runtime:\n  native_library: bundled:../../other\n')
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='Bundled native'): load_config(config)


@pytest.mark.parametrize('platform', sorted(LIBRARIES))
def test_bundled_library_names_and_backends_per_os(platform, tmp_path):
    assert library_name(platform) == LIBRARIES[platform]
    assert backends_for(platform) == SHIPPED[platform]
    for backend in SHIPPED[platform]:  # existing bundled:cuda and bundled:vulkan configs keep resolving
        assert bundled_library(backend, tmp_path, platform) == tmp_path / 'native' / backend / LIBRARIES[platform]
    for backend in {'cuda', 'vulkan', 'metal'} - set(SHIPPED[platform]):
        with pytest.raises(ValueError, match=f'not shipped for {platform}; use bundled:{SHIPPED[platform][0]}'):
            bundled_library(backend, tmp_path, platform)
    for backend in ('../../other', '', 'CUDA', 'cuda/../metal'):
        with pytest.raises(ValueError, match='Bundled native'): bundled_library(backend, tmp_path, platform)
    with pytest.raises(ValueError, match='packaged app'): bundled_library(SHIPPED[platform][0], None, platform)


def test_unsupported_platform_has_no_bundled_backend(tmp_path):
    with pytest.raises(ValueError, match='no bundled backend ships for freebsd'): bundled_library('cuda', tmp_path, 'freebsd')
    with pytest.raises(ValueError, match='freebsd'): library_name('freebsd')


def test_manifest_carries_what_the_release_build_needs():
    assert native_backends.validate(native_backends.manifest()) is native_backends.manifest()
    assert cmake_flags('cuda') == ['-DGGML_CUDA=ON', '-DGGML_VULKAN=OFF', '-DGGML_METAL=OFF',
        '-DCMAKE_CUDA_FLAGS=--diag-suppress=221', '-DGGML_CUDA_NO_VMM=ON']
    assert cmake_flags('vulkan') == ['-DGGML_CUDA=OFF', '-DGGML_VULKAN=ON', '-DGGML_METAL=OFF']
    assert cmake_flags('metal') == ['-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON',
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0']
    with pytest.raises(ValueError, match='Unknown native backend'): cmake_flags('hip')
    # macOS dyld ignores $ORIGIN; Windows finds a DLL's neighbours without an rpath.
    assert {platform: platform_build(platform).get('rpath') for platform in LIBRARIES} == {'win32': None, 'linux': '$ORIGIN', 'darwin': '@loader_path'}
    bundles = {'win32': ['riko-native.dll', 'ggml-cuda.dll', 'cublas64_12.dll'], 'linux': ['libriko-native.so', 'libggml-base.so', 'libcublas.so.12'],
        'darwin': ['libriko-native.dylib', 'libggml-metal.dylib', 'libllama.dylib']}
    for platform, files in bundles.items():
        patterns = platform_build(platform)['shared_libraries']
        assert all(any(fnmatch(file, pattern) for pattern in patterns) for file in files), platform
        assert not any(fnmatch(file, pattern) for pattern in patterns for other in set(bundles) - {platform} for file in bundles[other][:1])


def test_manifest_leaves_room_for_hip_and_cpu_bundles():
    data = native_backends.validate(json.loads(native_backends.manifest_path().read_text(encoding='utf-8')))
    data['backends'].insert(2, {'id': 'hip', 'label': 'ROCm', 'description': 'AMD', 'platforms': ['linux'], 'cmake_switch': 'GGML_HIP', 'cmake_flags': []})
    data['backends'].append({'id': 'cpu', 'label': 'CPU', 'description': 'no GPU', 'platforms': ['win32', 'linux', 'darwin'], 'cmake_switch': None})
    native_backends.validate(data)
    assert backends_for('linux', data) == ['cuda', 'vulkan', 'hip', 'cpu'] and backends_for('darwin', data) == ['metal', 'cpu']
    assert cmake_flags('cpu', data) == ['-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_HIP=OFF', '-DGGML_METAL=OFF']
    assert cmake_flags('hip', data)[2] == '-DGGML_HIP=ON'
    assert bundled_library('hip', '/app', 'linux', data) == Path('/app/native/hip/libriko-native.so')


@pytest.mark.parametrize('change, message', [
    (lambda data: data['backends'].append({**data['backends'][0], 'id': '../x'}), 'backend id'),
    (lambda data: data['backends'].append(dict(data['backends'][0])), 'listed twice'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'hip', 'platforms': ['linux-arm64']}), 'platforms'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'hip'}), 'cmake_switch'),
    (lambda data: data['backends'].append({**data['backends'][0], 'id': 'hip', 'cmake_switch': 'HIP'}), 'cmake_switch'),
    (lambda data: data['backends'][0].update(cmake_flags=['--diag-suppress=221']), 'cmake_flags'),
    (lambda data: data['backends'][0]['detect'].update(required='yes'), 'detect'),
    (lambda data: data['platforms']['win32'].update(library='native/riko-native.dll'), 'bare file name'),
    (lambda data: data['platforms']['win32'].update(library='..\\riko-native.dll'), 'bare file name'),
    (lambda data: data['platforms'].update(freebsd=data['platforms']['linux']), 'unknown platform'),
    (lambda data: data['platforms']['darwin'].pop('shared_libraries'), 'shared_libraries'),
])
def test_manifest_validation_rejects_entries_that_would_misresolve(change, message):
    data = json.loads(native_backends.manifest_path().read_text(encoding='utf-8'))
    change(data)
    with pytest.raises(ValueError, match=message): native_backends.validate(data)


def test_frozen_backend_reads_its_bundled_copy_of_the_manifest(tmp_path, monkeypatch):
    assert native_backends.manifest_path() == Path(__file__).resolve().parents[1] / 'electron' / 'native_backends.json'
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    assert native_backends.manifest_path() == tmp_path / native_backends.FROZEN_DIRECTORY / 'native_backends.json'


def test_macos_bundle_and_app_declare_the_same_oldest_macos():
    # The Metal library targets the macOS that Info.plist (LSMinimumSystemVersion) admits, not the build runner's.
    import yaml
    builder = yaml.safe_load((Path(__file__).resolve().parents[1] / 'electron/electron-builder.yml').read_text(encoding='utf-8'))
    assert backends_for('darwin') == ['metal'] and library_name('darwin') == 'libriko-native.dylib'
    assert f"-DCMAKE_OSX_DEPLOYMENT_TARGET={builder['mac']['minimumSystemVersion']}" in cmake_flags('metal')
    assert builder['mac']['target'] == [{'target': 'dmg', 'arch': ['arm64']}]
