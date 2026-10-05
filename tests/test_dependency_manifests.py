from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def requirement(text):
    name, spec = re.match(r'\s*([A-Za-z0-9._-]+)\s*(.*)', text).groups()
    return re.sub(r'[-_.]+', '-', name).lower(), spec.replace(' ', '')


def test_runtime_extra_mirrors_requirements_runtime_and_covers_every_extra():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    extras = project['optional-dependencies']
    lines = [line.split('#')[0].strip() for line in (ROOT / 'requirements-runtime.txt').read_text(encoding='utf-8').splitlines()]
    runtime = {requirement(line) for line in lines if line}
    assert {requirement(item) for item in extras['runtime']} == runtime
    for name in ('llama', 'desktop', 'discord'): assert {requirement(item) for item in extras[name]} <= runtime, name
    assert {requirement(item) for item in project['dependencies']} <= runtime
    assert 'riko-companion[runtime]' in extras['test']
