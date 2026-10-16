"""gateway state and rules (design §4, §5): quota, task assignment, global budget, sessions, worker registry.

Quota (configurable per env, set by the user on 2026-10-15):
- Each hotkey may have at most max_tasks_inflight (9) tasks running at once; at most max_tasks_per_epoch (20)
  may be started per epoch, counted as "ever started": finishing or abandoning does not refund (tasks voided by
  an infra error are refunded).
- Each (hotkey, task) may have at most max_concurrent (10) sessions at once; each (hotkey, task, nonce) can be
  opened only once.
- A session expires after idle_ttl (15 minutes) of inactivity.

A task is "running" from the moment the gateway assigns it (/tasks/open) until any of the following holds: the
miner calls finish_task; every nonce has been opened and there is no active session; there is no active session
and the last activity is older than idle_ttl; any rollout reports an infra error (void); its worker is lost
(void). A finished task cannot open new nonces for the rest of the epoch.

Task assignment: miners cannot choose. A task is assigned to only one miner at a time (pool table); once the group
is qualified (every nonce has a result, at least one completed and correct, one completed and wrong) it is not
assigned again this epoch; if not qualified or voided it goes back to the pool (the same hotkey never gets the
same task again). Tasks whose image is already cached on a worker with free slots are preferred, and all sessions
of a task are placed on the same worker.

Global budget: assigning a task reserves group_size container slots on its worker until the task ends. Global
in-flight <= max_containers (0 = limited only by worker capacity). When capacity runs out the request is refused
and the hotkey is recorded as "waiting"; while someone else is waiting, a hotkey that already holds its fair share
(assignable tasks / active hotkeys, at least 1) gets no new task, and freed slots are left for those waiting.

The prototype uses SQLite (serialized with BEGIN IMMEDIATE); a production switch to Postgres keeps the same
transaction boundaries.
"""
from __future__ import annotations

import json
import random
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

from nacl.signing import SigningKey

from ..auth import REQID_TTL, issue_token
from ..receipt import sign_group_seal

_HEX = frozenset('0123456789abcdef')

SESSION_OUTCOMES = ('completed', 'unresolved', 'infra_error', 'abandoned')


@dataclass(frozen=True)
class Limits:
    max_tasks_inflight: int = 9
    max_tasks_per_epoch: int = 20
    max_concurrent: int = 10
    idle_ttl: int = 15 * 60
    token_ttl: int = 4 * 3600
    max_containers: int = 0          # global in-flight container budget B; 0 = limited only by worker capacity
    worker_ttl: int = 90             # a worker with no heartbeat for this long counts as lost
    node_ttl: int = 120              # a node missing from (or not Ready in) node syncs for this long counts as lost
    demand_window: int = 60          # how long a "waiting" record is kept
    infra_error_policy: str = 'void'  # void: void the task, refund the start quota slot, return it to the pool; keep: record as usual


class QuotaError(Exception):
    status = 409

    def __init__(self, code: str, detail: str = '', status: int | None = None):
        super().__init__(f'{code}: {detail}' if detail else code)
        self.code, self.detail = code, detail
        if status is not None:
            self.status = status


class CapacityError(QuotaError):
    status = 429

    def __init__(self, code: str, detail: str = '', retry_after: int = 10):
        super().__init__(code, detail)
        self.retry_after = retry_after


SCHEMA = '''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS reqids (reqid TEXT PRIMARY KEY, seen_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY, url TEXT NOT NULL, capacity INTEGER NOT NULL, images TEXT NOT NULL,
    envs TEXT NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, epoch TEXT NOT NULL, hotkey TEXT NOT NULL, env_id TEXT NOT NULL, task TEXT NOT NULL,
    worker_id TEXT NOT NULL, slots INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'done', 'void')),
    started_at REAL NOT NULL, last_seen REAL NOT NULL, ended_at REAL, end_reason TEXT, node TEXT, seal TEXT,
    UNIQUE (epoch, hotkey, env_id, task));
CREATE INDEX IF NOT EXISTS tasks_hotkey ON tasks (epoch, hotkey, state);
CREATE INDEX IF NOT EXISTS tasks_state ON tasks (state, worker_id);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, task_id TEXT NOT NULL, nonce INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'closed', 'expired')),
    outcome TEXT, reward REAL, opened_at REAL NOT NULL, last_seen REAL NOT NULL, receipt_sha256 TEXT,
    UNIQUE (task_id, nonce));
CREATE INDEX IF NOT EXISTS sessions_task ON sessions (task_id, state);
CREATE INDEX IF NOT EXISTS sessions_active ON sessions (state, last_seen);
CREATE TABLE IF NOT EXISTS pool (
    epoch TEXT NOT NULL, env_id TEXT NOT NULL, task TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('assigned', 'solved')), task_id TEXT NOT NULL,
    PRIMARY KEY (epoch, env_id, task));
CREATE TABLE IF NOT EXISTS demand (epoch TEXT NOT NULL, hotkey TEXT NOT NULL, at REAL NOT NULL,
    PRIMARY KEY (epoch, hotkey));
CREATE TABLE IF NOT EXISTS nodes (
    name TEXT PRIMARY KEY, slots INTEGER NOT NULL, images TEXT NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS machines (
    provider TEXT NOT NULL, machine_id TEXT NOT NULL, name TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('planner', 'static', 'unmanaged')),
    state TEXT NOT NULL CHECK (state IN ('creating', 'running', 'orphaned', 'deleted', 'gone')),
    provider_state TEXT, kind TEXT NOT NULL DEFAULT '', price_per_hour REAL, slots INTEGER NOT NULL DEFAULT 0,
    public_ipv4 TEXT, public_ipv6 TEXT, created_at REAL NOT NULL, last_seen REAL, ended_at REAL,
    note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (provider, machine_id));
CREATE INDEX IF NOT EXISTS machines_state ON machines (state);
'''

# Machine inventory states. Live: creating (the planner asked for it, the boot is not done or was interrupted),
# running (it exists), orphaned (the planner's create failed and the provider could not delete it; the planner
# retries the delete). Ended: deleted (by the planner, or cleaned up after a failed create), gone (a machine the
# planner does not own stopped being listed by its provider).
LIVE_MACHINE_STATES = ('creating', 'running', 'orphaned')
_MACHINE_COLS = ('provider', 'machine_id', 'name', 'origin', 'state', 'provider_state', 'kind', 'price_per_hour',
                 'slots', 'public_ipv4', 'public_ipv6', 'created_at', 'last_seen', 'ended_at', 'note')


class Store:
    def __init__(self, db_path: str, envs: dict, signing_key: SigningKey, limits: Limits = Limits(),
                 rng: random.Random | None = None, placement: str = 'workers'):
        """placement: 'workers' -- capacity and image caches come from worker heartbeats (one host per worker);
        'nodes' -- they come from cluster nodes (`sync_nodes`), each task is pinned to one node, and workers only
        route (architecture v2)."""
        if placement not in ('workers', 'nodes'):
            raise ValueError(f'unknown placement {placement!r}')
        self.envs, self.signing_key, self.limits, self.placement = envs, signing_key, limits, placement
        self.rng = rng or random.SystemRandom()
        self.db = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, timeout=30)
        self.db.executescript(SCHEMA)
        for table, col in (('tasks', 'node'), ('tasks', 'seal'), ('sessions', 'receipt_sha256')):
            if col not in {r[1] for r in self.db.execute(f'PRAGMA table_info({table})')}:
                self.db.execute(f'ALTER TABLE {table} ADD COLUMN {col} TEXT')     # databases from before it
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='planner_machines'").fetchone():
            # databases from before the inventory: the planner's own machines move into `machines`
            self.db.executescript(
                'BEGIN; INSERT OR IGNORE INTO machines (provider, machine_id, name, origin, state, price_per_hour, '
                "slots, created_at) SELECT provider, machine_id, name, 'planner', 'running', price_per_hour, slots, "
                'created_at FROM planner_machines; DROP TABLE planner_machines; COMMIT;')
        self._mu = threading.RLock()       # one connection shared by many request threads

    # ------------------------------------------------------------ transactions
    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self):
                store._mu.acquire()
                self.cur = store.db.cursor()
                self.cur.execute('BEGIN IMMEDIATE')
                return self.cur

            def __exit__(self, et, ev, tb):
                try:
                    self.cur.execute('COMMIT' if et is None else 'ROLLBACK')
                finally:
                    store._mu.release()
                return False
        return _Tx()

    def _env(self, env_id: str):
        env = self.envs.get(env_id)
        if env is None:
            raise QuotaError('unknown_env', env_id, status=404)
        return env

    # ------------------------------------------------------------ replay protection
    def remember_reqid(self, reqid: str, now: float) -> bool:
        with self._tx() as cur:
            cur.execute('DELETE FROM reqids WHERE seen_at < ?', (now - REQID_TTL,))
            try:
                cur.execute('INSERT INTO reqids VALUES (?, ?)', (reqid, now))
            except sqlite3.IntegrityError:
                return False
        return True

    # ------------------------------------------------------------ worker registry
    def heartbeat(self, worker_id: str, url: str, capacity: int, images: list[str], env_ids: list[str],
                  now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._tx() as cur:
            cur.execute('INSERT INTO workers VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (worker_id) DO UPDATE SET '
                        'url=excluded.url, capacity=excluded.capacity, images=excluded.images, envs=excluded.envs, '
                        'last_seen=excluded.last_seen',
                        (worker_id, url, int(capacity), json.dumps(sorted(images)), json.dumps(sorted(env_ids)), now))

    # ------------------------------------------------------------ cluster nodes (placement='nodes')
    def sync_nodes(self, nodes, now: float | None = None) -> None:
        """Record the usable nodes from one cluster read (coordinator.nodes.k8s_nodes). A node that stops
        appearing is kept until node_ttl, then its running tasks are voided as node_lost."""
        now = time.time() if now is None else now
        with self._tx() as cur:
            for n in nodes:
                cur.execute('INSERT INTO nodes VALUES (?, ?, ?, ?) ON CONFLICT (name) DO UPDATE SET '
                            'slots=excluded.slots, images=excluded.images, last_seen=excluded.last_seen',
                            (n.name, int(n.slots), json.dumps(sorted(n.images)), now))
            cur.execute('DELETE FROM nodes WHERE last_seen < ? AND name NOT IN '
                        "(SELECT node FROM tasks WHERE state='running' AND node IS NOT NULL)",
                        (now - 10 * self.limits.node_ttl,))

    # ------------------------------------------------------------ ending / reaping
    def _end_task(self, cur, task_id: str, state: str, reason: str, now: float) -> None:
        row = cur.execute("SELECT epoch, env_id, task FROM tasks WHERE task_id=? AND state='running'",
                          (task_id,)).fetchone()
        if not row:
            return
        epoch, env_id, task = row
        cur.execute('UPDATE tasks SET state=?, ended_at=?, end_reason=? WHERE task_id=?', (state, now, reason, task_id))
        cur.execute("UPDATE sessions SET state='closed', last_seen=? WHERE task_id=? AND state='active'",
                    (now, task_id))
        qualified = False
        if state == 'done':
            env = self.envs.get(env_id)
            res = cur.execute("SELECT outcome, reward FROM sessions WHERE task_id=? AND outcome IN "
                              "('completed', 'unresolved')", (task_id,)).fetchall()
            if env is not None and len(res) == env.group_size:
                done = [r for o, r in res if o == 'completed']
                qualified = any(r >= env.success_reward for r in done) and any(r < env.success_reward for r in done)
        if qualified:
            cur.execute("UPDATE pool SET state='solved' WHERE epoch=? AND env_id=? AND task=? AND task_id=?",
                        (epoch, env_id, task, task_id))
        else:
            cur.execute('DELETE FROM pool WHERE epoch=? AND env_id=? AND task=? AND task_id=?',
                        (epoch, env_id, task, task_id))

    def _maybe_done(self, cur, task_id: str, now: float) -> None:
        row = cur.execute("SELECT t.env_id, t.last_seen, COUNT(s.id), COALESCE(SUM(s.state='active'), 0) "
                          "FROM tasks t LEFT JOIN sessions s ON s.task_id = t.task_id "
                          "WHERE t.task_id=? AND t.state='running' GROUP BY t.task_id", (task_id,)).fetchone()
        if not row:
            return
        env_id, last_seen, n, active = row
        env = self.envs.get(env_id)
        if active == 0 and ((env is not None and n >= env.group_size) or last_seen < now - self.limits.idle_ttl):
            self._end_task(cur, task_id, 'done', 'all_nonces_done' if env and n >= env.group_size else 'idle', now)

    def _reap(self, cur, now: float) -> None:
        lim = self.limits
        cur.execute("UPDATE sessions SET state='expired' WHERE state='active' AND last_seen < ?", (now - lim.idle_ttl,))
        lost = [r[0] for r in cur.execute(
            "SELECT t.task_id FROM tasks t LEFT JOIN workers w ON w.worker_id = t.worker_id "
            "WHERE t.state='running' AND (w.worker_id IS NULL OR w.last_seen < ?)", (now - lim.worker_ttl,))]
        for tid in lost:
            self._end_task(cur, tid, 'void', 'worker_lost', now)
        if self.placement == 'nodes':
            gone = [r[0] for r in cur.execute(
                "SELECT t.task_id FROM tasks t LEFT JOIN nodes n ON n.name = t.node "
                "WHERE t.state='running' AND (n.name IS NULL OR n.last_seen < ?)", (now - lim.node_ttl,))]
            for tid in gone:
                self._end_task(cur, tid, 'void', 'node_lost', now)
        for (tid,) in cur.execute("SELECT task_id FROM tasks WHERE state='running'").fetchall():
            self._maybe_done(cur, tid, now)
        cur.execute('DELETE FROM demand WHERE at < ?', (now - lim.demand_window,))

    def reap(self, now: float | None = None) -> None:
        with self._tx() as cur:
            self._reap(cur, time.time() if now is None else now)

    # ------------------------------------------------------------ miner: task assignment
    def open_task(self, epoch: str, hotkey: str, env_id: str, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        env = self._env(env_id)
        with self._tx() as cur:
            res = self._open_task(cur, epoch, hotkey, env_id, env, now)
        if isinstance(res, CapacityError):      # the "waiting" record must be committed, so raise after the transaction
            raise res
        return res

    def _open_task(self, cur, epoch, hotkey, env_id, env, now):
        lim = self.limits
        self._reap(cur, now)
        started, running = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(state='running'), 0) FROM tasks "
            "WHERE epoch=? AND hotkey=? AND state != 'void'", (epoch, hotkey)).fetchone()
        if started >= lim.max_tasks_per_epoch:
            raise QuotaError('task_quota_epoch', f'{started}/{lim.max_tasks_per_epoch} tasks started this epoch')
        if running >= lim.max_tasks_inflight:
            raise QuotaError('task_quota_inflight', f'{running}/{lim.max_tasks_inflight} tasks running')

        slots = env.group_size
        used = dict(cur.execute("SELECT worker_id, SUM(slots) FROM tasks WHERE state='running' "
                                'GROUP BY worker_id').fetchall())
        workers = []
        for wid, url, cap, images, envs in cur.execute(
                'SELECT worker_id, url, capacity, images, envs FROM workers WHERE last_seen >= ?',
                (now - lim.worker_ttl,)):
            if env_id in json.loads(envs):
                workers.append({'id': wid, 'url': url, 'cap': cap, 'free': cap - used.get(wid, 0),
                                'images': set(json.loads(images))})
        budget = sum(w['cap'] for w in workers)
        fits = [w for w in workers if w['free'] >= slots]
        in_flight = sum(used.values())
        nodes, node_fits = [], []
        if self.placement == 'nodes':
            # capacity and image caches come from cluster nodes; the worker (executor) only routes
            nused = dict(cur.execute("SELECT node, SUM(slots) FROM tasks WHERE state='running' AND node IS NOT NULL "
                                     'GROUP BY node').fetchall())
            for name, nslots, images in cur.execute('SELECT name, slots, images FROM nodes WHERE last_seen >= ?',
                                                    (now - lim.node_ttl,)):
                nodes.append({'name': name, 'cap': nslots, 'free': nslots - nused.get(name, 0),
                              'images': set(json.loads(images))})
            budget = min(budget, sum(n['cap'] for n in nodes))
            node_fits = [n for n in nodes if n['free'] >= slots]
            if not node_fits:
                fits = []
        if lim.max_containers:
            budget = min(budget, lim.max_containers)
        if not fits or in_flight + slots > (lim.max_containers or float('inf')):
            self._note_demand(cur, epoch, hotkey, now)
            return CapacityError('capacity_exhausted', 'no free environment capacity, retry later')

        # fair share: while someone else is waiting, a hotkey that holds its share gets no new task
        waiting = {r[0] for r in cur.execute('SELECT hotkey FROM demand WHERE epoch=? AND at >= ?',
                                             (epoch, now - lim.demand_window))}
        active = {r[0] for r in cur.execute("SELECT DISTINCT hotkey FROM tasks WHERE epoch=? AND state='running'",
                                            (epoch,))}
        share = max(1, (budget // slots) // max(1, len(active | waiting | {hotkey})))
        if waiting - {hotkey} and running >= share:
            self._note_demand(cur, epoch, hotkey, now)
            return CapacityError('fair_share', f'{running} tasks running, fair share is {share}')

        taken = {r[0] for r in cur.execute('SELECT task FROM pool WHERE epoch=? AND env_id=?', (epoch, env_id))}
        taken |= {r[0] for r in cur.execute('SELECT task FROM tasks WHERE epoch=? AND hotkey=? AND env_id=?',
                                            (epoch, hotkey, env_id))}
        free = [t for t in env.task_ids() if t not in taken]
        if not free:
            raise QuotaError('pool_exhausted', 'no task left for this hotkey in this epoch')
        places = node_fits if self.placement == 'nodes' else fits
        cached = set().union(*(p['images'] for p in places))
        hot = [t for t in free if env.image_of(t) in cached]
        task = self.rng.choice(hot or free)
        image = env.image_of(task)
        worker = max(fits, key=lambda w: (image in w['images'], w['free'], w['id']))
        node = None
        if self.placement == 'nodes':
            node = max(node_fits, key=lambda n: (image in n['images'], n['free'], n['name']))['name']

        task_id = secrets.token_hex(12)
        cur.execute("INSERT INTO pool VALUES (?, ?, ?, 'assigned', ?)", (epoch, env_id, task, task_id))
        cur.execute('INSERT INTO tasks (task_id, epoch, hotkey, env_id, task, worker_id, slots, state, started_at, '
                    "last_seen, node) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                    (task_id, epoch, hotkey, env_id, task, worker['id'], slots, now, now, node))
        cur.execute('DELETE FROM demand WHERE epoch=? AND hotkey=?', (epoch, hotkey))
        return {'task_id': task_id, 'env_id': env_id, 'task': task, 'worker_url': worker['url'],
                'group_size': slots}

    def _note_demand(self, cur, epoch, hotkey, now):
        cur.execute('INSERT INTO demand VALUES (?, ?, ?) ON CONFLICT (epoch, hotkey) DO UPDATE SET at=excluded.at',
                    (epoch, hotkey, now))

    # ------------------------------------------------------------ miner: open sessions / finish
    def open_sessions(self, epoch: str, hotkey: str, task_id: str, nonces: list[int],
                      now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        lim = self.limits
        if not isinstance(nonces, list) or not nonces or len(set(nonces)) != len(nonces):
            raise QuotaError('bad_nonce', 'nonces must be a non-empty list without duplicates', status=400)
        out = []
        with self._tx() as cur:
            self._reap(cur, now)
            row = cur.execute('SELECT epoch, hotkey, env_id, task, worker_id, state FROM tasks WHERE task_id=?',
                              (task_id,)).fetchone()
            if not row or row[0] != epoch or row[1] != hotkey:
                raise QuotaError('unknown_task', status=404)
            _, _, env_id, task, worker_id, state = row
            if state != 'running':
                raise QuotaError('task_finished', 'this task is finished for this epoch')
            env = self._env(env_id)
            for n in nonces:
                if type(n) is not int or not 0 <= n < env.group_size:
                    raise QuotaError('bad_nonce', f'nonce must be 0..{env.group_size - 1}', status=400)
            active = cur.execute("SELECT COUNT(*) FROM sessions WHERE task_id=? AND state='active'",
                                 (task_id,)).fetchone()[0]
            if active + len(nonces) > lim.max_concurrent:
                raise QuotaError('concurrency_quota', f'{active}/{lim.max_concurrent} active on this task')
            for n in nonces:
                sid = secrets.token_hex(16)
                try:
                    cur.execute('INSERT INTO sessions (id, task_id, nonce, state, opened_at, last_seen) '
                                "VALUES (?, ?, ?, 'active', ?, ?)", (sid, task_id, n, now, now))
                except sqlite3.IntegrityError:
                    raise QuotaError('nonce_used', f'nonce {n} already opened for this task') from None
                claims = {'sid': sid, 'task_id': task_id, 'epoch': epoch, 'hotkey': hotkey, 'env_id': env_id,
                          'task': task, 'nonce': n, 'worker_id': worker_id, 'exp': int(now) + lim.token_ttl}
                out.append({'nonce': n, 'session_id': sid, 'token': issue_token(self.signing_key, claims)})
            cur.execute('UPDATE tasks SET last_seen=? WHERE task_id=?', (now, task_id))
        return out

    def finish_task(self, epoch: str, hotkey: str, task_id: str, now: float | None = None) -> dict | None:
        """Miner finishes or abandons the task: close all its active sessions, end the task, free the "running"
        quota slot immediately. Returns the gateway-signed group seal (receipt.sign_group_seal) of a task that
        ended done, the same one on every call; None for a voided task."""
        now = time.time() if now is None else now
        with self._tx() as cur:
            row = cur.execute('SELECT epoch, hotkey FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            if not row or row[0] != epoch or row[1] != hotkey:
                raise QuotaError('unknown_task', status=404)
            self._end_task(cur, task_id, 'done', 'finished', now)
            return self._seal(cur, task_id)

    def _seal(self, cur, task_id: str) -> dict | None:
        epoch, hotkey, env_id, task, state, started, ended, seal = cur.execute(
            'SELECT epoch, hotkey, env_id, task, state, started_at, ended_at, seal FROM tasks WHERE task_id=?',
            (task_id,)).fetchone()
        if state != 'done':
            return None
        if seal:
            return json.loads(seal)
        size = self._env(env_id).group_size
        got = dict(cur.execute('SELECT nonce, receipt_sha256 FROM sessions WHERE task_id=?', (task_id,)).fetchall())
        out = sign_group_seal(self.signing_key, epoch=epoch, hotkey=hotkey, env_id=env_id, task=task,
                              task_id=task_id, group_size=size, opened_at_ms=int(started * 1000),
                              finished_at_ms=int(ended * 1000), receipts=[got.get(n) for n in range(size)])
        cur.execute('UPDATE tasks SET seal=? WHERE task_id=?', (json.dumps(out), task_id))
        return out

    # ------------------------------------------------------------ worker internal API
    def _worker_task(self, cur, worker_id: str, task_id: str):
        row = cur.execute("SELECT env_id, task FROM tasks WHERE task_id=? AND worker_id=? AND state='running'",
                          (task_id, worker_id)).fetchone()
        if not row:
            raise QuotaError('unknown_task', status=404)
        return row

    def worker_task_spec(self, worker_id: str, task_id: str) -> dict:
        with self._tx() as cur:
            env_id, task = self._worker_task(cur, worker_id, task_id)
            node = cur.execute('SELECT node FROM tasks WHERE task_id=?', (task_id,)).fetchone()[0]
        env = self._env(env_id)
        return {'env_id': env_id, 'task': task, 'group_size': env.group_size, 'spec': env.public_spec(task),
                'node': node}

    def worker_grade_inputs(self, worker_id: str, session_id: str) -> dict:
        with self._tx() as cur:
            row = cur.execute("SELECT task_id FROM sessions WHERE id=? AND state='active'", (session_id,)).fetchone()
            if not row:
                raise QuotaError('unknown_session', status=404)
            env_id, task = self._worker_task(cur, worker_id, row[0])
        return self._env(env_id).grade_inputs(task)

    def worker_touch(self, worker_id: str, session_id: str, now: float | None = None) -> bool:
        """The worker reports once per step (and periodically during long operations) to refresh the idle timer; returns False if the session is closed or expired, or the task is void."""
        now = time.time() if now is None else now
        with self._tx() as cur:
            n = cur.execute("UPDATE sessions SET last_seen=? WHERE id=? AND state='active' AND last_seen >= ? AND "
                            "task_id IN (SELECT task_id FROM tasks WHERE worker_id=? AND state='running')",
                            (now, session_id, now - self.limits.idle_ttl, worker_id)).rowcount
            if n:
                cur.execute('UPDATE tasks SET last_seen=? WHERE task_id=(SELECT task_id FROM sessions WHERE id=?)',
                            (now, session_id))
        return n == 1

    def worker_close(self, worker_id: str, session_id: str, outcome: str, reward: float | None,
                     now: float | None = None, receipt_sha256: str | None = None) -> None:
        """receipt_sha256: receipt.receipt_digest of the receipt the worker issued; the group seal lists it."""
        now = time.time() if now is None else now
        if outcome not in SESSION_OUTCOMES:
            raise QuotaError('bad_outcome', status=400)
        if receipt_sha256 is not None and (len(receipt_sha256) != 64 or not set(receipt_sha256) <= _HEX):
            raise QuotaError('bad_receipt_sha256', status=400)
        with self._tx() as cur:
            row = cur.execute('SELECT s.task_id FROM sessions s JOIN tasks t ON t.task_id = s.task_id '
                              "WHERE s.id=? AND t.worker_id=?", (session_id, worker_id)).fetchone()
            if not row:
                raise QuotaError('unknown_session', status=404)
            task_id = row[0]
            cur.execute("UPDATE sessions SET state='closed', outcome=?, reward=?, last_seen=?, receipt_sha256=? "
                        "WHERE id=? AND state IN ('active', 'expired') AND outcome IS NULL",
                        (outcome, reward, now, receipt_sha256, session_id))
            cur.execute('UPDATE tasks SET last_seen=? WHERE task_id=?', (now, task_id))
            if outcome == 'infra_error' and self.limits.infra_error_policy == 'void':
                self._end_task(cur, task_id, 'void', 'infra_error', now)
            else:
                self._maybe_done(cur, task_id, now)

    # ------------------------------------------------------------ machine inventory
    def machines(self, *, live: bool = True, provider: str | None = None, origin: str | None = None) -> list[dict]:
        """Every machine we know of (survives restarts). live=True: only creating / running / orphaned ones;
        live=False also returns the history (deleted, gone)."""
        q, args = f'SELECT {", ".join(_MACHINE_COLS)} FROM machines WHERE 1=1', []
        if live:
            q += f' AND state IN ({", ".join("?" * len(LIVE_MACHINE_STATES))})'
            args += LIVE_MACHINE_STATES
        for col, v in (('provider', provider), ('origin', origin)):
            if v is not None:
                q += f' AND {col} = ?'
                args.append(v)
        with self._tx() as cur:
            rows = cur.execute(q + ' ORDER BY created_at, provider, machine_id', args).fetchall()
        return [dict(zip(_MACHINE_COLS, r, strict=True)) for r in rows]

    def machine_created(self, provider: str, machine_id: str, name: str, kind: str, price_per_hour: float,
                        slots: int, now: float) -> None:
        """The planner asked `provider` for a machine and got `machine_id` (recorded before the slow boot)."""
        with self._tx() as cur:
            cur.execute('INSERT OR REPLACE INTO machines (provider, machine_id, name, origin, state, kind, '
                        "price_per_hour, slots, created_at, last_seen) VALUES (?, ?, ?, 'planner', 'creating', ?, ?, "
                        '?, ?, ?)', (provider, machine_id, name, kind, price_per_hour, slots, now, now))

    def machine_set_state(self, provider: str, machine_id: str, state: str, now: float, note: str | None = None):
        ended = state not in LIVE_MACHINE_STATES
        with self._tx() as cur:
            cur.execute('UPDATE machines SET state = ?, ended_at = ?, note = COALESCE(?, note) '
                        'WHERE provider = ? AND machine_id = ?',
                        (state, now if ended else None, note, provider, machine_id))

    def machine_note(self, provider: str, machine_id: str, note: str) -> None:
        with self._tx() as cur:
            cur.execute('UPDATE machines SET note = ? WHERE provider = ? AND machine_id = ?',
                        (note, provider, machine_id))

    def machine_seen(self, provider: str, m, now: float, origin: str = 'unmanaged') -> bool:
        """Record that `provider` lists machine `m` (a providers.machine.Machine). A machine we did not know is
        added with `origin` (static or unmanaged); a known one gets its last_seen, provider state and addresses
        refreshed, and is never re-labelled. Returns True when the machine is new to the inventory."""
        with self._tx() as cur:
            row = cur.execute('SELECT state FROM machines WHERE provider = ? AND machine_id = ?',
                              (provider, m.id)).fetchone()
            if row is None or row[0] == 'gone':        # gone and back (e.g. a stopped VM resumed): count it again
                cur.execute('INSERT OR REPLACE INTO machines (provider, machine_id, name, origin, state, '
                            'provider_state, kind, price_per_hour, public_ipv4, public_ipv6, created_at, last_seen, '
                            "note) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?)",
                            (provider, m.id, m.name, origin, m.state, m.kind, m.price_per_hour, m.public_ipv4,
                             m.public_ipv6, m.created_at or now, now,
                             '' if origin == 'static' else 'not created by the planner; never deleted automatically'))
                return True
            # the name is never overwritten: boat may report its own name, and ours is also the k3s node name
            cur.execute('UPDATE machines SET last_seen = ?, provider_state = ?, '
                        'kind = CASE WHEN ? != \'\' THEN ? ELSE kind END, '
                        'price_per_hour = COALESCE(price_per_hour, ?), '
                        'public_ipv4 = COALESCE(?, public_ipv4), public_ipv6 = COALESCE(?, public_ipv6) '
                        'WHERE provider = ? AND machine_id = ?',
                        (now, m.state, m.kind, m.kind, m.price_per_hour, m.public_ipv4, m.public_ipv6,
                         provider, m.id))
            return False

    def machines_missing(self, provider: str, listed_ids, now: float) -> list[dict]:
        """Machines we do not own (static / unmanaged) that `provider` no longer lists are marked gone. The planner's
        own machines are never marked gone here: a list that misses one (paging bug, renamed machine) must not make
        the planner forget a machine that may still bill; the planner's own delete path ends those."""
        listed = set(listed_ids)
        gone = [r for r in self.machines(provider=provider)
                if r['origin'] != 'planner' and r['machine_id'] not in listed]
        for r in gone:
            self.machine_set_state(provider, r['machine_id'], 'gone', now)
        return gone

    # ------------------------------------------------------------ stats
    def capacity_snapshot(self, now: float | None = None) -> dict:
        """What the capacity planner needs: hotkeys waiting for capacity, slots in use, and per-node slots."""
        now = time.time() if now is None else now
        lim = self.limits
        with self._tx() as cur:
            waiting = cur.execute('SELECT COUNT(DISTINCT hotkey) FROM demand WHERE at >= ?',
                                  (now - lim.demand_window,)).fetchone()[0]
            used = cur.execute("SELECT COALESCE(SUM(slots), 0) FROM tasks WHERE state='running'").fetchone()[0]
            nused = dict(cur.execute("SELECT node, SUM(slots) FROM tasks WHERE state='running' AND node IS NOT NULL "
                                     'GROUP BY node').fetchall())
            nodes = {n: {'slots': s, 'used': nused.get(n, 0)} for n, s in cur.execute(
                'SELECT name, slots FROM nodes WHERE last_seen >= ?', (now - lim.node_ttl,))}
        group = max((e.group_size for e in self.envs.values()), default=8)
        return {'waiting': waiting, 'used': used, 'nodes': nodes, 'group_size': group}

    def stats(self, epoch: str | None = None) -> dict:
        with self._tx() as cur:
            q = "SELECT state, COUNT(*), COALESCE(SUM(slots), 0) FROM tasks" + (' WHERE epoch=?' if epoch else '') + \
                ' GROUP BY state'
            tasks = {s: {'n': n, 'slots': sl} for s, n, sl in cur.execute(q, (epoch,) if epoch else ())}
            workers = cur.execute('SELECT COUNT(*), COALESCE(SUM(capacity), 0) FROM workers WHERE last_seen >= ?',
                                  (time.time() - self.limits.worker_ttl,)).fetchone()
        return {'tasks': tasks, 'workers_live': workers[0], 'capacity_live': workers[1]}
