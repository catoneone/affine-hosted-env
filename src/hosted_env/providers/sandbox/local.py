"""LocalSandbox: for tests and local demos only; commands run directly in a temp directory on the host with no
isolation at all."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from .base import OUTPUT_CAP_BYTES


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
