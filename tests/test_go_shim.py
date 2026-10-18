"""GO_REPORT_SHIM (envs/swe/reports.py): the `go` put first on PATH for a report_kind='gotest' task. Driven with a fake
real go, so it runs without a Go toolchain."""
import json
import os
import shlex
import shutil
import stat
import subprocess

from hosted_env.envs.swe import GO_REPORT_SHIM, report_verdict

FAKE_GO = '''#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_LOG"
if [ "$1" = test ] && [ "$2" = -json ]; then
  printf '%s\\n' '{"Action":"run","Package":"ex/p","Test":"TestA"}'
  printf '%s\\n' '{"Action":"pass","Package":"ex/p","Test":"TestA"}'
  printf '%s\\n' '{"Action":"fail","Package":"ex/p","Test":"TestB"}'
fi
exit "${FAKE_RC:-0}"
'''


def _setup(tmp_path):
    real = tmp_path / 'real' / 'go'
    real.parent.mkdir()
    real.write_text(FAKE_GO)
    real.chmod(real.stat().st_mode | stat.S_IEXEC)
    bindir = tmp_path / 'shim'
    bindir.mkdir()
    shim = bindir / 'go'
    shim.write_text(GO_REPORT_SHIM.replace('__REAL_GO__', shlex.quote(str(real))))
    shim.chmod(0o755)
    report = tmp_path / 'rep' / 'report.ndjson'
    report.parent.mkdir()
    return bindir, report, tmp_path / 'calls.log'


def _run(bindir, report, log, cmd, rc=0, with_report=True):
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'FAKE_LOG': str(log), 'FAKE_RC': str(rc)}
    if with_report:
        env['MIMO_GO_REPORT'] = str(report)
    return subprocess.run(['sh', '-c', cmd], env=env, capture_output=True, text=True)


def test_go_test_gets_json_and_the_report(tmp_path):
    bindir, report, log = _setup(tmp_path)
    p = _run(bindir, report, log, "go test -v -count=1 -run 'TestA|TestB' ./p", rc=1)
    assert p.returncode == 1                                    # go test's own exit status, not cat's
    assert log.read_text().splitlines() == ["test -json -v -count=1 -run TestA|TestB ./p"]
    assert report.read_text() == p.stdout                       # same events on stdout and in the report
    assert [json.loads(x)['Action'] for x in report.read_text().splitlines()] == ['run', 'pass', 'fail']
    assert not list(report.parent.glob('*.out'))                # the buffer file is removed
    ok, why = report_verdict('gotest', report.read_bytes(), ['TestB'], ['TestA'])
    assert not ok and 'TestB=fail' in why
    assert report_verdict('gotest', report.read_bytes(), [], ['ex/p::TestA'])[0]


def test_report_accumulates_over_several_go_test_runs(tmp_path):
    bindir, report, log = _setup(tmp_path)
    assert _run(bindir, report, log, 'go test ./a && go test ./b').returncode == 0
    assert len(report.read_text().splitlines()) == 6


def test_passthrough_cases(tmp_path):
    """Other go commands, compile-only / list modes, an explicit -json, and no report requested: untouched."""
    bindir, report, log = _setup(tmp_path)
    for cmd in ('go env GOPATH', 'go build ./...', 'go test -c -o x.test ./p', 'go test -list . ./p',
                'go test -json ./p', 'go test -json=true ./p'):
        log.unlink(missing_ok=True)
        assert _run(bindir, report, log, cmd, rc=3).returncode == 3
        assert log.read_text().strip() == cmd[len('go '):]
    log.unlink()
    _run(bindir, report, log, 'go test ./p', with_report=False)
    assert log.read_text().strip() == 'test ./p'
    assert not report.exists()


def test_flags_after_args_belong_to_the_test_binary(tmp_path):
    """`-c` after -args is the test binary's flag, not go test's compile-only mode."""
    bindir, report, log = _setup(tmp_path)
    _run(bindir, report, log, 'go test ./p -args -c')
    assert log.read_text().strip() == 'test -json ./p -args -c'
    assert report.exists()


class _Sb:
    """Just enough of a sandbox for _install_go_shim: commands run locally under a fixed PATH."""

    def __init__(self, path):
        self.path = path

    def execute(self, cmd, cwd=None, timeout=60, login=False):
        p = subprocess.run(['/bin/sh', '-c', cmd], cwd=cwd, env={'PATH': self.path, 'HOME': '/tmp'},
                           capture_output=True, text=True)
        return {'returncode': p.returncode, 'output': p.stdout + p.stderr}

    def copy_text_to(self, text, remote_path):
        with open(remote_path, 'w') as f:
            f.write(text)


def _tools(tmp_path):
    """Only the utilities the install and the shim call, symlinked into one dir: the host's /usr/bin may hold a
    real go (GitHub's ubuntu runners do), which would make a 'no go on PATH' case find one."""
    tools = tmp_path / 'tools'
    if not tools.exists():
        tools.mkdir()
        for name in ('cat', 'rm', 'mkdir', 'chmod'):
            (tools / name).symlink_to(shutil.which(name))
    return str(tools)


def _install(tmp_path, path_dirs):
    from hosted_env.envs.swe import SweEnv
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    rdir = tmp_path / 'mimo-report-abc'
    rdir.mkdir(exist_ok=True)
    rpath, res = str(rdir / 'report.ndjson'), {}
    sb = _Sb(':'.join([*path_dirs, _tools(tmp_path)]))
    pre = SweEnv._install_go_shim(object(), sb, str(repo), str(rdir), rpath, res)
    return sb, repo, rpath, res, pre


def test_install_go_shim_wraps_the_image_go(tmp_path):
    real = tmp_path / 'goroot' / 'bin' / 'go'
    real.parent.mkdir(parents=True)
    real.write_text(FAKE_GO)
    real.chmod(0o755)
    sb, repo, rpath, res, pre = _install(tmp_path, [str(real.parent)])
    assert res['go_report_shim'] == str(real) and pre.startswith('PATH=') and 'MIMO_GO_REPORT=' in pre
    shim = tmp_path / 'mimo-report-abc-bin' / 'go'
    assert stat.S_IMODE(shim.stat().st_mode) == 0o755
    r = sb.execute(f'FAKE_LOG={tmp_path}/c.log {pre}go test ./p', cwd=str(repo))
    assert r['returncode'] == 0 and (tmp_path / 'c.log').read_text().strip() == 'test -json ./p'
    assert len(open(rpath).read().splitlines()) == 3


def test_install_go_shim_skips_without_a_usable_go(tmp_path):
    sb, repo, rpath, res, pre = _install(tmp_path, [])                  # no go on PATH at all
    assert pre == '' and res['go_report_shim'].startswith('skipped')
    inrepo = tmp_path / 'repo' / 'bin'                                   # only a go inside the repo: never wrapped
    inrepo.mkdir(parents=True)
    (inrepo / 'go').write_text('#!/bin/sh\nexit 0\n')
    (inrepo / 'go').chmod(0o755)
    sb, repo, rpath, res, pre = _install(tmp_path, [str(inrepo)])
    assert pre == '' and res['go_report_shim'].startswith('skipped')
    assert not (tmp_path / 'mimo-report-abc-bin' / 'go').exists()
