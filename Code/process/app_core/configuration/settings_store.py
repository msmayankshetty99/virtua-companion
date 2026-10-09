"""Round-trip YAML settings with typed forms, validation and conflict-safe saves. What each setting is (its kind, range,
help, group, restart scope and live hook) comes from the settings schema (configuration/schema.py over kernel/schema.py),
where each package registers its own sections."""
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading

from . import schema
from .config import AppConfig, load_config
from .schema import UNKNOWN, field, flatten  # noqa: F401  (callers and tests import field and UNKNOWN from here)
from ..persistence.atomic import atomic_write

LOCK = threading.RLock()


class SettingsConflict(ValueError): pass


def unknown_warnings(raw):
    """{path: UNKNOWN} for keys the strict sections do not declare (configuration/schema.py unknown_settings): reported,
    never an error, so a typo or an obsolete key never blocks saving."""
    return {path: UNKNOWN for path in schema.unknown_settings(raw)}


def revision(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SettingsStore:
    def __init__(self, path, running=None):
        """running: the AppConfig the backend started with, so snapshot() can say which saved settings await a restart."""
        self.path, self.running = Path(path), running

    def _read(self):
        from ruamel.yaml import YAML
        text = self.path.read_text(encoding='utf-8')
        # Keep quotes: ruamel writes YAML 1.2, and the backend's PyYAML reads an unquoted yes or 42 as a boolean or number.
        yaml = YAML(); yaml.preserve_quotes = True
        raw = yaml.load(text)
        if not isinstance(raw, dict): raise ValueError('Configuration must be a YAML mapping')
        return text, raw

    def snapshot(self):
        text, raw = self._read()
        candidate = load_config(self.path, recover=True)  # open even while a section is broken, so Settings can repair it
        values = schema.values(candidate, raw)
        fields = [field(k, v) for k, v in values.items() if schema.offered(k, values)]
        warnings = unknown_warnings(raw)
        for item in fields:
            if any(item['path'] == path or item['path'].startswith(path + '.') for path in warnings): item.update(unknown=True, help=UNKNOWN)
        pending = self.pending(candidate)
        # restart_pending ({path: restart scope}) and rules (checks across settings, kernel/schema.py Rule) are additions;
        # restart_required used to be always true.
        result = {'revision': revision(text), 'values': values, 'fields': fields, 'path': str(self.path), 'restart_required': bool(pending),
            'restart_pending': pending, 'rules': schema.rules()}
        return {**result, 'warnings': warnings} if warnings else result

    def _values(self, raw): return schema.values(load_config(self.path, recover=True), raw)  # effective defaults, not just what the YAML holds

    def pending(self, candidate):
        """{path: restart scope} for saved settings that differ from what the running backend loaded, both read as it
        reads them (PyYAML); empty without a running backend."""
        if not isinstance(self.running, AppConfig): return {}
        before, after = schema.values(self.running, self.running.raw), schema.values(candidate, candidate.raw)
        changed = (path for path in sorted({*before, *after}) if before.get(path) != after.get(path))
        return {path: scope for path in changed if (scope := schema.restart_scope(path)) != 'none'}

    def prepare(self, changes):
        text, raw = self._read()
        values = self._values(raw); errors = {}
        if not isinstance(changes, dict) or len(changes) > 300: raise ValueError('Invalid settings patch')
        for path, value in changes.items():
            if path not in values: errors[path] = 'Unknown setting'; continue
            spec = field(path, values[path])
            if spec['readonly']: errors[path] = 'Legacy setting is not used by the current runtime'; continue
            if value is None and spec['nullable']: pass
            elif spec['kind'] == 'boolean' and type(value) is not bool: errors[path] = 'Choose on or off'
            elif spec['kind'] == 'number':
                if type(value) not in (int, float) or not math.isfinite(value): errors[path] = 'Enter a finite number'
                elif spec['integer'] and type(value) is not int: errors[path] = 'Enter a whole number'
                elif 'min' in spec and value < spec['min'] or 'max' in spec and value > spec['max']: errors[path] = f"Allowed range: {spec.get('min', '−∞')} to {spec.get('max', '∞')}"
            elif spec['kind'] == 'text' and not isinstance(value, str): errors[path] = 'Enter text'
            elif spec['kind'] == 'json' and not isinstance(value, type(values[path])) and values[path] is not None: errors[path] = 'Keep the same JSON structure type'
            if spec.get('options') and value not in spec['options']: errors[path] = 'Choose a supported option'
            if isinstance(value, str) and len(value) > 100000: errors[path] = 'Value is too long'
            if path not in errors:
                node = raw
                keys = path.split('.')
                for key in keys[:-1]:
                    if node.get(key) is None: node[key] = {}  # absent, or a section whose keys are all commented out
                    node = node[key]
                if schema.quoted(path) and isinstance(value, str):
                    from ruamel.yaml.scalarstring import DoubleQuotedScalarString
                    value = DoubleQuotedScalarString(value)
                node[keys[-1]] = deepcopy(value)
        if errors: return text, None, errors
        from ruamel.yaml import YAML
        yaml = YAML(); yaml.preserve_quotes = True
        stream = io.StringIO(); yaml.dump(raw, stream)
        output = stream.getvalue()
        fd, temporary = tempfile.mkstemp(suffix='.yaml', prefix='.settings-validation-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as file: file.write(output)
            candidate = load_config(temporary)  # every registered section's check, as the backend will run it
            # Each section's checks beyond load_config (configuration/schema.py review): the wake word, an edited speech
            # recognition pair or avatar, default memories, the llama.cpp budget and split, the initiative budget.
            errors.update(schema.review(candidate, {**values, **changes}, changes))
            for path, value in dict(flatten(raw)).items():
                if path.endswith(('_url', '.url')) and isinstance(value, str) and not value.startswith(('http://', 'https://')):
                    errors[path] = 'Use an http:// or https:// URL'
            writes = schema.derive(candidate)  # e.g. the automatic KV pool, saved with every change
            if writes:
                for path, value in writes.items():
                    node, (*parents, key) = raw, path.split('.')
                    for name in parents:
                        if node.get(name) is None: node[name] = {}
                        node = node[name]
                    node[key] = value
                stream = io.StringIO(); yaml.dump(raw, stream); output = stream.getvalue()
        except (ValueError, TypeError, KeyError) as exc: errors.setdefault('__all__', str(exc))
        finally: Path(temporary).unlink(missing_ok=True)
        return text, output, errors

    def validate(self, changes):
        with LOCK:
            _, _, errors = self.prepare(changes)
            warnings = unknown_warnings(self._read()[1])
            return {'valid': not errors, 'errors': errors, **({'warnings': warnings} if warnings else {})}

    def save(self, changes, expected_revision):
        with LOCK:
            text, output, errors = self.prepare(changes)
            if revision(text) != expected_revision or revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                raise SettingsConflict('Configuration changed outside this editor. Reload before saving.')
            if errors: return {'saved': False, 'valid': False, 'errors': errors}
            budget_changes = {key.split('.')[-1]: value for key, value in changes.items()
                if key in {'initiative.context_window_tokens', 'initiative.max_output_tokens'}}
            preferences = self.path.parent / 'persistent_memories' / 'initiative_settings.json'
            saved_preferences = None
            if budget_changes and preferences.exists():
                saved_preferences = json.loads(preferences.read_text(encoding='utf-8'))
                if not isinstance(saved_preferences, dict): raise ValueError('Live initiative preferences must be an object')
                saved_preferences.update(budget_changes)
            # Keep a previous config, including its comments and unknown settings.
            backup = self.path.with_suffix(self.path.suffix + '.previous')
            backup.write_text(text, encoding='utf-8')
            fd, temporary = tempfile.mkstemp(prefix='.settings-save-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    stream.write(output); stream.flush(); os.fsync(stream.fileno())
                if revision(self.path.read_text(encoding='utf-8')) != expected_revision:
                    raise SettingsConflict('Configuration changed while saving. Reload before saving.')
                os.replace(temporary, self.path)
                # Live initiative preferences share these budgets. Preserve all
                # other preferences while saving model/runtime budgets together.
                if saved_preferences is not None: atomic_write(preferences, json.dumps(saved_preferences, indent=2))
            finally: Path(temporary).unlink(missing_ok=True)
            # Whether this save needs a restart; snapshot()'s restart_pending also names earlier saves still waiting for one.
            return {**self.snapshot(), 'saved': True, 'restart_required': any(schema.restart_scope(key) != 'none' for key in changes)}

    def check_path(self, value):
        if not isinstance(value, str) or len(value) > 4096: raise ValueError('Invalid path')
        path = Path(value).expanduser()
        if not path.is_absolute(): path = self.path.parent / path
        executable = shutil.which(value)
        return {'resolved': executable or str(path.resolve()), 'exists': path.exists(), 'directory': path.is_dir(), 'executable': executable}
