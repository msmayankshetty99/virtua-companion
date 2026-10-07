from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

from process.app_core.configuration.config import load_config
from process.app_core.configuration.native_backends import backends_for, bundled_library
from process.app_core.configuration.settings_store import SettingsStore, SettingsConflict


@pytest.fixture
def store(tmp_path):
    path = tmp_path/'character_config.yaml'
    path.write_text('''# keep this explanation
your_name: User
runtime:
  provider: lm_studio # keep this comment
  n_ctx: 8192
  n_batch: 512
  n_ubatch: 512
  model_path: null
  hf_repo_id: null
  hf_filename: null
  n_threads: null
  type_v: f16
  flash_attn: false
voice:
  mode: wake_word
  wake_word: Riko
  wake_threshold: 0.9
presets:
  default:
    model_params:
      context_window_token_limit: 7168
      max_output_tokens: 1024
    memories: []
custom_extension:
  untouched: true
''', encoding='utf-8')
    return SettingsStore(path)


def test_round_trip_save_preserves_comments_and_unknown_settings(store):
    before = store.path.read_text()
    snapshot = store.snapshot()
    result = store.save({'your_name':'Senpai', 'voice.wake_threshold':.85}, snapshot['revision'])
    assert result['saved'] and result['restart_required']
    assert '# keep this explanation' in store.path.read_text()
    assert '# keep this comment' in store.path.read_text()
    assert result['values']['custom_extension.untouched'] is True
    assert store.path.with_suffix('.yaml.previous').read_text() == before


def test_validation_does_not_write_config_or_start_models(store):
    before = store.path.read_bytes()
    result = store.validate({'runtime.provider':'llama_cpp'})
    assert not result['valid']
    assert store.path.read_bytes() == before
    assert not list(store.path.parent.glob('.settings-validation-*'))


def test_replaced_preset_fields_are_not_editable_and_independent_values_are_preserved(store):
    snapshot=store.snapshot()
    paths={item['path'] for item in snapshot['fields']}
    assert not any(path.startswith('presets.default.model_params.') for path in paths)
    result=store.save({'runtime.temperature':.23},snapshot['revision'])
    assert result['saved']
    assert result['values']['runtime.temperature']==.23
    assert result['values']['presets.default.model_params.max_output_tokens']==1024
    assert result['values']['memory.context_window_tokens']==snapshot['values']['memory.context_window_tokens']
    result=store.save({'runtime.pause_background_on_live':False},result['revision'])
    assert result['saved'] and not result['restart_required']


@pytest.mark.parametrize('changes', [
    {'runtime.flash_attn':'false'}, {'runtime.n_ctx':-1}, {'runtime.n_batch':12.5},
    {'voice.wake_word':'two names'}, {'voice.wake_threshold':2}, {'unknown.setting':True},
    {'runtime.type_v':'not_a_type'}, {'presets.default.memories':[True]},
])
def test_invalid_drafts_never_replace_config(store, changes):
    before = store.path.read_bytes()
    result = store.save(changes, store.snapshot()['revision'])
    assert not result['saved'] and result['errors']
    assert store.path.read_bytes() == before


def test_conflicting_external_edit_is_preserved(store):
    snapshot = store.snapshot()
    store.path.write_text(store.path.read_text()+'\n# user edited outside UI\n')
    with pytest.raises(SettingsConflict): store.save({'your_name':'New name'}, snapshot['revision'])
    assert '# user edited outside UI' in store.path.read_text()


def test_nullable_threads_can_be_reset_after_saving(store):
    result = store.save({'runtime.n_threads':4}, store.snapshot()['revision'])
    spec = next(field for field in result['fields'] if field['path']=='runtime.n_threads')
    assert spec['nullable']
    assert store.save({'runtime.n_threads':None}, result['revision'])['saved']


def test_repository_model_and_cache_validation(store):
    valid = {'runtime.provider':'llama_cpp', 'runtime.hf_repo_id':'owner/model', 'runtime.hf_filename':'model.gguf'}
    assert store.validate(valid)['valid']
    result = store.validate({**valid,'runtime.type_v':'q8_0'})
    assert not result['valid']


def test_fixture_character_configuration_has_valid_settings(store):
    assert store.validate({}) == {'valid':True,'errors':{}}
    assert 'runtime.provider' in store.snapshot()['values']


def test_current_character_configuration_has_valid_settings():
    if os.environ.get('RIKO_RELEASE_BUILD') == '1':
        pytest.skip('Release builds exclude the private local configuration')
    path=Path(__file__).resolve().parents[1]/'character_config.yaml'
    if not path.is_file(): pytest.skip('No local character_config.yaml: it is private and gitignored, so a fresh clone has none')
    store=SettingsStore(path)
    assert store.validate({}) == {'valid':True,'errors':{}}
    assert 'runtime.provider' in store.snapshot()['values']


def test_background_budgets_are_exposed_and_validated(store):
    values = store.snapshot()['values']
    assert values['initiative.max_output_tokens'] == 1024
    assert values['memory.reflection_context_window_tokens'] == 4096
    assert store.validate({'initiative.max_output_tokens': 4096})['valid'] is False
    assert store.validate({'memory.reflection_max_output_tokens': 4096})['valid'] is False
    assert store.validate({'memory.reflection_max_output_tokens': 1536, 'memory.reflection_context_window_tokens': 8192})['valid']


def test_speech_recognition_pair_defaults_to_auto_and_is_checked_only_when_edited(store, monkeypatch):
    from process.app_core.audio import asr
    monkeypatch.setattr(asr, 'supported_types', lambda: {'cpu': frozenset({'float32', 'int8', 'int8_float32'})})
    values = store.snapshot()['values']
    assert (values['voice.asr_device'], values['voice.asr_compute_type']) == ('auto', 'default')
    assert store.validate({'voice.asr_device': 'cpu', 'voice.asr_compute_type': 'int8'})['valid']
    assert 'voice.asr_device' in store.validate({'voice.asr_device': 'cuda'})['errors']
    assert 'voice.asr_compute_type' in store.validate({'voice.asr_compute_type': 'int8_float16'})['errors']
    # A pair already in the file falls back at startup; it must not block saving anything else.
    store.path.write_text(store.path.read_text().replace('  wake_threshold: 0.9\n', '  wake_threshold: 0.9\n  asr_device: cuda\n  asr_compute_type: int8_float16\n'))
    assert store.save({'voice.wake_threshold': .8}, store.snapshot()['revision'])['saved']


def test_resource_fields_are_grouped_with_models(store):
    fields = {field['path']:field for field in store.snapshot()['fields']}
    for path in ('voice.asr_device','memory.embedding_model','emotion.max_length','memory.reflection_max_output_tokens','initiative.context_window_tokens','memory.system1_model_id'):
        assert fields[path]['group'] == 'models'
    assert fields['voice.wake_threshold']['group'] == 'voice'


def test_automatic_pool_is_recalculated_and_saved_with_other_settings(store):
    before = store.snapshot()
    result = store.save({'memory.reflection_context_window_tokens':8192,'runtime.kv_pool_auto':True}, before['revision'])
    assert result['saved']
    assert result['values']['runtime.kv_pool_tokens'] == 16384
    assert 'kv_pool_tokens: 16384' in store.path.read_text()
    assert 'kv_pool_auto: true' in store.path.read_text()


def test_saved_pool_and_initiative_budget_preserve_live_preferences(store):
    import json
    preferences = store.path.parent / 'persistent_memories' / 'initiative_settings.json'
    preferences.parent.mkdir()
    preferences.write_text(json.dumps({'enabled':True,'context_window_tokens':8192,'max_output_tokens':1536}))
    before = store.snapshot()
    assert before['values']['initiative.context_window_tokens'] == 8192
    result = store.save({'initiative.context_window_tokens':6144}, before['revision'])
    assert result['saved']
    assert result['values']['runtime.kv_pool_tokens'] == 14336
    assert result['values']['initiative.context_window_tokens'] == 6144
    assert json.loads(preferences.read_text())['enabled'] is True
    assert json.loads(preferences.read_text())['max_output_tokens'] == 1536


def test_missing_default_settings_can_be_saved_into_new_yaml_sections(store):
    snapshot=store.snapshot()
    assert snapshot['values']['memory.embeddings_enabled'] is True
    result=store.save({'memory.embeddings_enabled':False},snapshot['revision'])
    assert result['saved'] and result['values']['memory.embeddings_enabled'] is False


def test_explicit_path_check_never_reads_file_contents(store):
    result=store.check_path(str(store.path))
    assert result['exists'] and not result['directory']
    assert 'contents' not in result


def test_concurrent_same_name_tool_results_update_the_correct_activity():
    from process.app_core.desktop.state import DesktopState
    state=DesktopState()
    first=state.tool_started('lookup', {'id':'first'})
    second=state.tool_started('lookup', {'id':'second'})
    state.tool_finished('lookup','first result',activity_id=first)
    activities={item['id']:item for item in state.tool_activity}
    assert activities[first]['status']=='complete'
    assert activities[second]['status']=='running'
    assert activities[first]['duration_ms']>=0


def test_legacy_faiss_index_file_is_not_offered(store):
    assert 'memory.index_file' not in {field['path'] for field in store.snapshot()['fields']}


def test_memory_device_is_a_background_model_setting(store):
    snapshot = store.snapshot()
    spec = next(item for item in snapshot['fields'] if item['path'] == 'memory.device')
    assert snapshot['values']['memory.device'] == 'cpu' and spec['section'] == 'Background models' and spec['restart']
    assert store.validate({'memory.device': 'mps', 'emotion.device': 'auto'})['valid']
    assert not store.validate({'memory.device': 'vulkan'})['valid']
    assert not store.validate({'emotion.device': 'mps'})['valid']


def test_flash_attention_is_a_choice_that_keeps_old_boolean_files(store):
    snapshot = store.snapshot()
    spec = next(item for item in snapshot['fields'] if item['path'] == 'runtime.flash_attn')
    assert snapshot['values']['runtime.flash_attn'] == 'off' and spec['kind'] == 'text' and spec['options'] == ['auto', 'on', 'off']
    result = store.save({'runtime.flash_attn': 'on'}, snapshot['revision'])
    assert result['saved'] and result['values']['runtime.flash_attn'] == 'on'
    assert 'flash_attn: on' in store.path.read_text()  # ruamel writes it bare, so PyYAML reads true
    from process.app_core.configuration.config import load_config
    native = store.path.with_name('native.yaml')
    native.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  flash_attn: on\n')
    assert load_config(native).runtime.flash_attn == 'on'
    assert not store.save({'runtime.flash_attn': True}, store.snapshot()['revision'])['saved']


def test_layer_offload_and_split_choices_match_llama_cpp(store):
    assert store.validate({'runtime.n_gpu_layers': -2})['valid']
    assert not store.validate({'runtime.n_gpu_layers': -3})['valid']
    assert not store.validate({'runtime.split_mode': 'row'})['valid']
    store.path.write_text(store.path.read_text().replace('provider: lm_studio', 'provider: llama_cpp')
        .replace('  model_path: null', '  model_path: model.gguf').replace('  flash_attn: false', '  flash_attn: false\n  split_mode: row'))
    assert store.snapshot()['values']['runtime.split_mode'] == 'row'  # an old file still opens
    assert 'runtime.split_mode' in store.validate({})['errors']


def test_speech_recognition_error_names_the_value_that_cannot_run(store, monkeypatch):
    from process.app_core.audio import asr
    monkeypatch.setattr(asr, 'supported_types', lambda: {'cpu': frozenset({'float32', 'int8'}), 'cuda': frozenset({'float32', 'int8', 'int8_float16'})})
    store.path.write_text(store.path.read_text().replace('  wake_threshold: 0.9\n', '  wake_threshold: 0.9\n  asr_device: cuda\n  asr_compute_type: int8_float16\n'))
    errors = store.validate({'voice.asr_device': 'cpu'})['errors']  # CPU is fine; the stored precision is what cannot run there
    assert 'voice.asr_device' not in errors and 'cpu cannot run int8_float16' in errors['voice.asr_compute_type']
    assert store.validate({'voice.asr_device': 'cpu', 'voice.asr_compute_type': 'default'})['valid']


def test_live_budget_is_checked_only_when_edited_so_older_setups_can_save(tmp_path, monkeypatch):
    # The setup wizard used to write memory.context_window_tokens equal to n_ctx, which never leaves room for the reply.
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(tmp_path / 'application'))
    path = tmp_path / 'character_config.yaml'
    path.write_text(f'runtime:\n  provider: llama_cpp\n  native_library: bundled:{backends_for()[0]}\n  hf_repo_id: owner/model\n  hf_filename: model.gguf\n'
        '  n_ctx: 8192\n  max_output_tokens: 1024\nmemory:\n  context_window_tokens: 8192\n')
    store = SettingsStore(path)
    assert store.validate({'speech.max_words': 20}) == {'valid': True, 'errors': {}}
    assert 'Context must fit' in store.validate({'runtime.max_output_tokens': 512})['errors']['runtime.n_ctx']
    assert store.validate({'memory.context_window_tokens': 7168})['valid']
    path.write_text(path.read_text().replace('memory:\n  context_window_tokens: 8192\n', ''))
    assert store.snapshot()['values']['memory.context_window_tokens'] == 7168  # the prompt gets what the reply leaves
    assert store.validate({'runtime.n_ctx': 4096})['valid']
    assert 'runtime.n_ctx' in store.validate({'runtime.max_output_tokens': 8192})['errors']


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which('node') or not (ROOT / 'electron' / 'node_modules' / 'yaml').is_dir(), reason='needs node and the Electron dependencies')
def test_packaged_setup_writes_a_config_settings_can_save(tmp_path, monkeypatch):
    # Electron's setup refuses a backend whose library is not where it looks, so this also proves that both languages
    # resolve bundled:<backend> to the same file for this OS.
    resources, backend = tmp_path / 'application', backends_for()[0]
    library = bundled_library(backend, resources)
    library.parent.mkdir(parents=True); library.write_text('test')
    form = {'backend': backend, 'repo': 'owner/model', 'filename': 'model.gguf', 'context': 8192, 'output': 1024, 'threads': 4}
    script = "require('./release.cjs').saveSetup(process.argv[1], JSON.parse(process.argv[2]), process.argv[3])"
    subprocess.run(['node', '-e', script, str(tmp_path / 'data'), json.dumps(form), str(resources)], cwd=ROOT / 'electron', check=True, timeout=60)
    monkeypatch.setenv('RIKO_BUNDLE_ROOT', str(resources))
    store = SettingsStore(tmp_path / 'data' / 'character_config.yaml')
    assert load_config(store.path).runtime.native_library == library
    assert store.validate({'runtime.n_ctx': 8192}) == {'valid': True, 'errors': {}}  # the wizard's budget fits
    assert store.validate({'speech.max_words': 20}) == {'valid': True, 'errors': {}}


def test_legacy_preset_context_leaves_room_for_the_reply(tmp_path):
    from process.app_core.configuration.config import load_config
    path = tmp_path / 'character_config.yaml'
    path.write_text('runtime:\n  provider: llama_cpp\n  model_path: model.gguf\n  max_output_tokens: 1024\npresets:\n  default:\n    model_params:\n      context_window_token_limit: 8192\n')
    config = load_config(path)
    assert config.runtime.n_ctx == 8192 and config.memory.context_window_tokens == 7168
    assert SettingsStore(path).validate({'runtime.max_output_tokens': 512})['valid']
