"""The boundary between the infrastructure and the environments: hosted_env/infra/ knows no concrete environment.

infra may import hosted_env.envs (the registry: make_env, receipt_spec, SetupError) and nothing below it, and its
code (not its comments or docstrings) names none of the SWE environment's concepts. A new environment is added
under hosted_env/envs/ without touching infra; this test is what keeps that true."""
import ast
import io
import tokenize
from pathlib import Path

import hosted_env

ROOT = Path(hosted_env.__file__).resolve().parent
INFRA = ROOT / 'infra'
# identifiers and string values that belong to the SWE environment (envs/swe/)
SWE_WORDS = ('model_patch', 'test_patch', 'fail_to_pass', 'pass_to_pass', 'report_kind', 'expected_tests',
             'start_tree', 'grader', 'second_check', 'reverted', 'ENV_PROBE', 'mimo_swe', 'SweEnv')


def _modules():
    return sorted(p for p in INFRA.rglob('*.py') if '__pycache__' not in p.parts)


def _imported(path: Path) -> set[str]:
    """Absolute names of every module path imports, at any depth (lazy imports inside functions too)."""
    pkg = '.'.join(('hosted_env', *path.relative_to(ROOT).with_suffix('').parts))
    parts = pkg.split('.')[:-1] if path.name != '__init__.py' else pkg.split('.')[:-1]
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = parts[:len(parts) - node.level + 1] if node.level else []
            mod = '.'.join([*base, node.module] if node.module else base)
            out.add(mod)
            out.update(f'{mod}.{a.name}' for a in node.names)
    return out


def test_infra_imports_no_concrete_environment():
    bad = {}
    for p in _modules():
        envs = {m for m in _imported(p) if m == 'hosted_env.envs' or m.startswith('hosted_env.envs.')}
        allowed = {'hosted_env.envs', 'hosted_env.envs.make_env', 'hosted_env.envs.receipt_spec',
                   'hosted_env.envs.SetupError', 'hosted_env.envs.KINDS'}
        if envs - allowed:
            bad[str(p.relative_to(ROOT))] = sorted(envs - allowed)
    assert not bad, f'infra imports concrete environments: {bad}'


def _code_tokens(text: str):
    """NAME and STRING tokens of the code, without comments and without docstrings (a string statement)."""
    docstrings = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add(body[0].value.lineno)
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NAME or (tok.type == tokenize.STRING and tok.start[0] not in docstrings):
            yield tok


def test_infra_code_names_no_swe_concept():
    hits = []
    for p in _modules():
        for tok in _code_tokens(p.read_text()):
            if any(w in tok.string for w in SWE_WORDS):
                hits.append(f'{p.relative_to(ROOT)}:{tok.start[0]}: {tok.string[:60]}')
    assert not hits, 'SWE concepts in infra code:\n' + '\n'.join(hits)


def test_importing_the_registry_stays_light():
    """A validator imports hosted_env.envs (receipt_spec) to verify attachments; the environments' CLI helpers
    registered there (pin-images talks to registries) must not pull an HTTP client in at import time."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, '-c', 'import sys, hosted_env.envs; print("httpx" in sys.modules)'],
                       capture_output=True, text=True, check=True)
    assert r.stdout.strip() == 'False'


def test_the_checks_catch_a_violation(tmp_path, monkeypatch):
    """The two checks above would fail on the shapes they guard against (a lazy import of envs.swe, a field name
    in a string), so passing them means something."""
    bad = tmp_path / 'infra'
    (bad / 'x').mkdir(parents=True)
    (bad / 'x' / 'm.py').write_text('"""docstring may say model_patch"""\n'
                                    'def f():\n    from ...envs.swe import SweEnv\n    return {"model_patch": 1}\n')
    monkeypatch.setattr(__import__(__name__), 'ROOT', tmp_path)
    monkeypatch.setattr(__import__(__name__), 'INFRA', bad)
    import pytest
    with pytest.raises(AssertionError, match='envs.swe'):
        test_infra_imports_no_concrete_environment()
    with pytest.raises(AssertionError, match='model_patch'):
        test_infra_code_names_no_swe_concept()
