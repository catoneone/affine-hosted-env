"""Execution sandboxes (design §6, §7).

- DockerSandbox: one container per rollout, by default 1 core 2 GB (cgroup limits, not reservations),
  `--network none`, limited pids, no privileges. Commands run in the container via `bash -lc`, wrapped inside the
  container by coreutils timeout as a backstop.
- ImageCache: each image is pulled only once per machine (per-image lock + global concurrency cap); beyond the
  cap, idle images are evicted by least recent use.
- LocalSandbox: for tests and local demos only; commands run directly in a temp directory on the host with no
  isolation at all.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

DOCKER = os.environ.get('HOSTED_ENV_DOCKER', 'docker')
OUTPUT_CAP_BYTES = 4_000_000   # collect at most 4MB of output per command and drop the rest (the env truncates by characters later)
CONTAINER_LABEL = 'hosted-env=1'

# The Xiaomi images on Docker Hub are single-layer tarballs and lost every ENV in the image config (EXP-014,
# measured 2026-10-13). Probe directories following the conventions of the official golang/rust/ruby/openjdk
# images and restore the environment variables, only when a variable is not already set.
ENV_PROBE = r'''
out() { printf '%s=%s\n' "$1" "$2"; }
P="$PATH"
if [ -z "${GOPATH:-}" ] && [ -d /go ]; then out GOPATH /go; P="/go/bin:$P"; fi
[ -d /usr/local/go/bin ] && case ":$P:" in *:/usr/local/go/bin:*) ;; *) P="/usr/local/go/bin:$P";; esac
if [ -d /usr/local/cargo/bin ]; then
  [ -z "${CARGO_HOME:-}" ] && out CARGO_HOME /usr/local/cargo
  [ -z "${RUSTUP_HOME:-}" ] && [ -d /usr/local/rustup ] && out RUSTUP_HOME /usr/local/rustup
  P="/usr/local/cargo/bin:$P"
fi
if [ -z "${GEM_HOME:-}" ] && [ -d /usr/local/bundle ]; then
  out GEM_HOME /usr/local/bundle; out BUNDLE_APP_CONFIG /usr/local/bundle; out BUNDLE_SILENCE_ROOT_WARNING 1
  P="/usr/local/bundle/bin:$P"
fi
if [ -z "${JAVA_HOME:-}" ]; then
  for j in /opt/java/openjdk /usr/local/openjdk-* /usr/lib/jvm/*; do
    if [ -x "$j/bin/java" ]; then out JAVA_HOME "$j"; P="$j/bin:$P"; break; fi
  done
fi
[ -d /root/.local/bin ] && P="$P:/root/.local/bin"
[ -z "${LANG:-}" ] && out LANG C.UTF-8
[ "$P" != "$PATH" ] && out PATH "$P"
true
'''


class SandboxError(RuntimeError):
    """The sandbox itself is broken (fails to start, docker errors): recorded as an infra error, not the
    model's fault."""


def _pump(stream, buf: bytearray):
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if len(buf) < OUTPUT_CAP_BYTES:
            buf.extend(chunk[:OUTPUT_CAP_BYTES - len(buf)])


class DockerSandbox:
    def __init__(self, image: str, cwd: str, *, network: str = 'none', cpus: str = '1', memory: str = '2g',
                 pids: int = 4096, repair_env: bool = True):
        self.image, self.cwd = image, cwd
        self.run_args = ['--network', network, '--cpus', str(cpus), '--memory', str(memory),
                         '--pids-limit', str(pids), '--security-opt', 'no-new-privileges']
        self.cid: str | None = None
        self.has_timeout = False
        self.repair_env = repair_env
        self.env: dict[str, str] = {}

    def start(self):
        name = f'hosted-env-{uuid.uuid4().hex[:12]}'
        r = subprocess.run([DOCKER, 'run', '-d', '--name', name, '--label', CONTAINER_LABEL, '-w', self.cwd,
                            *self.run_args, self.image, 'sleep', 'infinity'],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise SandboxError(f'docker run failed: {r.stderr.strip()[-500:]}')
        self.cid = r.stdout.strip()
        self.has_timeout = self.execute('command -v timeout', timeout=30, raw=True)['returncode'] == 0
        if self.repair_env:
            out = self.execute(ENV_PROBE, timeout=60, raw=True)['output']
            self.env = dict(ln.split('=', 1) for ln in out.splitlines()
                            if '=' in ln and ln.split('=', 1)[0].isupper())

    def execute(self, command: str, cwd: str | None = None, timeout: int = 60, raw: bool = False) -> dict:
        """Returns {output, returncode, reason}; reason is ok / client_timeout."""
        if not self.cid:
            raise SandboxError('sandbox not started')
        inner = ['bash', '-lc', command]
        if self.has_timeout and not raw:
            inner = ['timeout', '-k', '5', str(int(timeout)), *inner]
        envs = [x for k, v in self.env.items() for x in ('-e', f'{k}={v}')]
        p = subprocess.Popen([DOCKER, 'exec', *envs, '-w', cwd or self.cwd, self.cid, *inner],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        buf = bytearray()
        th = threading.Thread(target=_pump, args=(p.stdout, buf), daemon=True)
        th.start()
        timed_out = False
        try:
            rc = p.wait(timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            p.kill()
            rc, timed_out = -1, True
        th.join(10)
        if self.has_timeout and not raw and rc == 124:
            timed_out = True
        if rc in (125, 126, 127) and not buf and not timed_out:
            # docker exec itself failed (container gone): indistinguishable from the command returning 127, so
            # check whether the container is still running
            alive = subprocess.run([DOCKER, 'inspect', '-f', '{{.State.Running}}', self.cid],
                                   capture_output=True, text=True, timeout=30)
            if alive.stdout.strip() != 'true':
                raise SandboxError('container is gone')
        return {'output': buf.decode('utf-8', 'replace'), 'returncode': rc,
                'reason': 'client_timeout' if timed_out else 'ok'}

    def copy_text_to(self, text: str, remote_path: str):
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', delete=False) as f:
            f.write(text)
            local = f.name
        try:
            r = subprocess.run([DOCKER, 'cp', local, f'{self.cid}:{remote_path}'], capture_output=True, text=True,
                               timeout=300)
            if r.returncode != 0:
                raise SandboxError(f'docker cp failed: {r.stderr.strip()[-300:]}')
        finally:
            os.unlink(local)

    def cleanup(self):
        if self.cid:
            subprocess.run([DOCKER, 'rm', '-f', self.cid], capture_output=True, timeout=120)
            self.cid = None


class LocalSandbox:
    """For tests / local demos: copy image (a template directory on the host) into a fresh temp directory root
    and run commands directly on the host in root; the task's cwd prefix is mapped to root. No isolation at all;
    never use in production."""

    def __init__(self, image: str, cwd: str, *, base_dir: str | None = None):
        self.image, self.cwd = image, cwd
        self.root = tempfile.mkdtemp(prefix='hosted-env-local-', dir=base_dir)
        home = os.path.join(self.root, '.home')
        self.env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': home, 'GIT_CONFIG_NOSYSTEM': '1',
                    'GIT_CEILING_DIRECTORIES': os.path.dirname(self.root), 'PYTHONDONTWRITEBYTECODE': '1',
                    'LANG': 'C.UTF-8'}

    def _map(self, s: str) -> str:
        # One pass, so the replacement text (which may itself live under /tmp, e.g. on Linux CI) is never
        # re-mapped; /tmp/ only matches at the start of a path, not inside one like /var/tmp/ or /x/tmp/.
        home = os.path.join(self.root, '.home') + '/'
        pat = re.compile(f'{re.escape(self.cwd)}|(?<![\\w./-])/tmp/')
        return pat.sub(lambda m: self.root if m.group(0) == self.cwd else home, s)

    def start(self):
        if self.image and os.path.isdir(self.image):
            shutil.copytree(self.image, self.root, symlinks=True, dirs_exist_ok=True)
        os.makedirs(self.env['HOME'], exist_ok=True)

    def execute(self, command: str, cwd: str | None = None, timeout: int = 60, raw: bool = False) -> dict:
        try:
            r = subprocess.run(['bash', '-c', self._map(command)], cwd=self._map(cwd) if cwd else self.root,
                               env=self.env, capture_output=True, timeout=timeout)
            out, rc, reason = r.stdout + r.stderr, r.returncode, 'ok'
        except subprocess.TimeoutExpired as e:
            out, rc, reason = (e.stdout or b'') + (e.stderr or b''), -1, 'client_timeout'
        return {'output': out[:OUTPUT_CAP_BYTES].decode('utf-8', 'replace'), 'returncode': rc, 'reason': reason}

    def copy_text_to(self, text: str, remote_path: str):
        with open(self._map(remote_path), 'w', encoding='utf-8', newline='') as f:
            f.write(text)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class ImageCache:
    """Image cache on the worker: per-image locked pulls, a global concurrency cap, and least-recently-used
    eviction of idle images once there are more than max_images."""

    def __init__(self, max_images: int = 10, max_parallel_pulls: int = 3, pull_timeout: int = 3600):
        self.max_images, self.pull_timeout = max_images, pull_timeout
        self._sem = threading.Semaphore(max_parallel_pulls)
        self._locks: dict[str, threading.Lock] = {}
        self._mu = threading.Lock()
        self._in_use: dict[str, int] = {}
        self._last_used: dict[str, float] = {}

    def _lock(self, image: str) -> threading.Lock:
        with self._mu:
            return self._locks.setdefault(image, threading.Lock())

    @staticmethod
    def present(image: str) -> bool:
        return subprocess.run([DOCKER, 'image', 'inspect', image], capture_output=True, timeout=60).returncode == 0

    def acquire(self, image: str) -> None:
        """Ensure the image is present locally and record one use; call release when done."""
        with self._lock(image):
            if not self.present(image):
                with self._sem:
                    err = ''
                    for attempt in range(3):
                        r = subprocess.run([DOCKER, 'pull', '-q', image], capture_output=True, text=True,
                                           timeout=self.pull_timeout)
                        if r.returncode == 0:
                            break
                        err = r.stderr.strip()[-300:]
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise SandboxError(f'docker pull failed: {err}')
            with self._mu:
                self._in_use[image] = self._in_use.get(image, 0) + 1
                self._last_used[image] = time.time()
        self._evict()

    def release(self, image: str) -> None:
        with self._mu:
            self._in_use[image] = max(0, self._in_use.get(image, 0) - 1)
            self._last_used[image] = time.time()

    def cached(self) -> list[str]:
        """Images present locally that this worker has used (reported to the gateway in heartbeats so task
        assignment prefers workers with the image cached)."""
        with self._mu:
            return sorted(self._last_used)

    def _evict(self) -> None:
        with self._mu:
            idle = sorted((t, img) for img, t in self._last_used.items() if not self._in_use.get(img))
            extra = len(self._last_used) - self.max_images
            victims = [img for _, img in idle[:max(0, extra)]]
            for img in victims:
                self._last_used.pop(img, None)
        for img in victims:
            subprocess.run([DOCKER, 'rmi', img], capture_output=True, timeout=600)
