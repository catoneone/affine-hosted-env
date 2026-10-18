"""Reading the task's grading scripts: the scripts a test patch ships, a small shell-command parser, and
the repo files a test command or script executes (which the grader then protects)."""
from __future__ import annotations

import base64
import re
import shlex

from ..base import GradeError


_CMD_PATH_RE = re.compile(r'PATH=([^\s;"\']+)')
_CMD_INVOKE_RE = re.compile(r'(?:^|[\s;&|(])(?:exec\s+)?(?:bash|sh)\s+(\S+\.sh)'
                            r'|(?:^|[\s;&|(])(\./[^\s;&|)]+)')


def _repo_rel(p: str, cwd: str) -> str | None:
    """`p` as a path relative to the repo `cwd`, or None when it is outside the repo. Relative paths are taken as
    repo-relative; an absolute path is kept only if it is under cwd."""
    p = p.strip().strip('"\'')
    if not p or p in ('.', './'):
        return None
    if p.startswith('/'):
        pre = cwd.rstrip('/') + '/'
        if not p.startswith(pre):
            return None
        p = p[len(pre):]
    elif p.startswith('./'):
        p = p[2:]
    if not p or p.startswith('../') or '/../' in p:
        return None
    p = p.rstrip('/')
    if any(seg in ('.', '..', '...') for seg in p.split('/')):   # e.g. go's ./... wildcard is not a path
        return None
    return p


def command_repo_paths(test_command: str, cwd: str) -> list[str]:
    """Repo-relative directories put on PATH and scripts invoked directly by the test command: anything the model
    adds or changes under them decides what the grader runs, so they are protected like the test infrastructure.
    Absolute paths outside the repo are ignored. Heuristic over the command string (the inner script of a mimo
    task lives in the test patch and is reset separately). A script invoked INDIRECTLY -- `make test` running a
    recipe, `python run_tests.py` importing a helper -- is not caught here by design: the entry point itself
    (Makefile, setup.py, test_*.py) is covered by the protected-name and protected-dir rules instead."""
    out = set()
    for m in _CMD_PATH_RE.findall(test_command):
        for part in m.split(':'):
            rel = _repo_rel(part, cwd)
            if rel is not None and not rel.startswith('$'):
                out.add(rel)
    for sh, dot in _CMD_INVOKE_RE.findall(test_command):
        rel = _repo_rel(sh or dot, cwd)
        if rel is not None and '$' not in rel:
            out.add(rel)
    return sorted(out)


MIMO_BUNDLE = 'mimo_build_env.tar.gz.b64'      # some MiMo test patches ship their scripts in this (-> .build_env/)
_BUNDLE_SCRIPT_MAX = 1_000_000


def test_patch_scripts(test_patch: str) -> list[tuple[str, str]]:
    """The grading scripts a test patch ships, as (repo-relative path, text): the added lines of every `.sh` file it
    touches (mimo_test_command.sh) and every `.sh` inside the base64 tar.gz bundle some MiMo tasks carry (unpacked
    at the repo root by `tar -xzf - -C <repo>`, members under .build_env/). The outer test command of a MiMo task
    is only `bash mimo_test_command.sh`; the commands that actually run (and the repo files they run) are in here.
    Parsed in the executor from our own task data; nothing the model wrote is read. A bundle that does not decode
    is our data or parser being wrong, so it raises (GradeError: infra) rather than grading without it."""
    shs: dict[str, list[str]] = {}
    bundle: list[str] = []
    cur, body = None, False
    for line in test_patch.splitlines():
        if line.startswith('diff --git '):
            cur, body = line.rsplit(' b/', 1)[-1], False
        elif cur is not None and not body:
            body = line.startswith('@@')                  # header lines (---/+++) come before the first hunk
        elif cur is not None and line.startswith('+'):    # content; base64 lines may themselves start with '++'
            if cur.endswith('.sh'):
                shs.setdefault(cur, []).append(line[1:])
            elif cur.rsplit('/', 1)[-1] == MIMO_BUNDLE:
                bundle.append(line[1:])
    out = [(p, '\n'.join(lines)) for p, lines in shs.items()]
    if bundle:
        import io
        import tarfile
        try:
            data = base64.b64decode(''.join(''.join(bundle).split()), validate=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith('.sh') and m.size <= _BUNDLE_SCRIPT_MAX:
                        out.append((_norm_rel(m.name) or m.name, tar.extractfile(m).read().decode('utf-8', 'replace')))
        except (ValueError, OSError, tarfile.TarError) as e:
            raise GradeError(f'test patch bundle {MIMO_BUNDLE} does not decode: {e}') from None
    return out


# a script argument with one of these extensions names a file the script runs or loads (`node build/run-tests.js`)
_SCRIPT_EXTS = ('.js', '.mjs', '.cjs', '.ts', '.mts', '.cts', '.jsx', '.tsx', '.py', '.sh', '.bash', '.rb', '.pl',
                '.php', '.go')
_ASSIGN_RE = re.compile(r'[A-Za-z_]\w*=')
# programs whose first non-flag argument is the script they run (python3.x is matched by pattern)
_INTERPRETERS = {'node', 'nodejs', 'bash', 'sh', 'dash', 'zsh', 'ruby', 'perl', 'php', 'deno', 'bun', 'ts-node',
                 'tsx', 'python', 'python3'}
# commands that run the rest of the line as the program (their own flags and numeric arguments are skipped)
_WRAPPERS = {'exec', 'env', 'timeout', 'nohup', 'time', 'xvfb-run', 'nice', 'stdbuf', 'command', 'sudo'}
# compound-command words that can stand before a command on the same line (`if ! node run.js; then ...`)
_KEYWORDS = {'if', 'then', 'else', 'elif', 'fi', 'do', 'done', 'while', 'until', '!', '{', '}', 'time'}
_SHELLS = {'bash', 'sh', 'dash', 'zsh', 'ksh'}       # their -c string and stdin are shell code, parsed again
_VALUE_FLAGS = {'-W', '-X', '-C', '--conditions', '--title', '--stack-size'}   # interpreter flags taking a value
_GO_VALUE_FLAGS = {'-tags', '-ldflags', '-gcflags', '-asmflags', '-mod', '-modfile', '-o', '-exec', '-p',
                   '-overlay', '-pgo', '-toolexec'}
_HEREDOC_RE = re.compile(r'''(?<!<)<<-?(?!<)\s*(['"]?)([A-Za-z_][\w-]*)\1''')   # <<EOF, <<-'PY'; not <<< here-strings
_SELF_RE = r'''"?\$(?:0|\{BASH_SOURCE(?:\[0\])?\}|BASH_SOURCE)"?'''           # "$0", "${BASH_SOURCE[0]}"
_DIRNAME_RE = rf'''\$\(\s*dirname\s+{_SELF_RE}\s*\)'''
_SCRIPT_DIR_RE = re.compile(rf'''\$\(\s*cd\s+"?{_DIRNAME_RE}"?\s*(?:&&|;)\s*pwd(?:\s+-P)?\s*\)|{_DIRNAME_RE}''')


def _norm_rel(p: str) -> str | None:
    """Normalise a repo-relative path ('a/./b/../c' -> 'a/c'): '' for the repo root itself, None if it climbs out."""
    parts: list[str] = []
    for seg in p.split('/'):
        if seg in ('', '.'):
            continue
        if seg == '..':
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return '/'.join(parts)


def _cd(cur: str | None, target: str, cwd: str) -> str | None:
    """The repo-relative directory after `cd target` from `cur` ('' = repo root); None once it cannot be followed
    (a variable, `cd -`, a path outside the repo, or a relative cd after an unknown one)."""
    if not target or any(c in target for c in '$~`') or target == '-':
        return None
    if target.startswith('/'):
        return '' if target.rstrip('/') == cwd.rstrip('/') else _repo_rel(target, cwd)
    return None if cur is None else _norm_rel(f'{cur}/{target}')


def _shell_commands(line: str, depth: int = 0) -> tuple[list[tuple[int, list[str]]], int]:
    """The simple commands on one shell line, as (subshell depth, words), and the depth at the end of the line:
    split at && || ; | & ( ) outside quotes, with redirections (`> f`, `2>&1`, `<<EOF`) and their targets dropped.
    The depth counts the enclosing `( )` / `$( )` (starting from `depth`, for a subshell opened on an earlier
    line), so a `cd` inside one can be undone when it closes. Backtick substitutions are read as `$( )`. Raises
    ValueError on an unclosed quote (the caller joins the next line: a quoted argument can span lines)."""
    lex = shlex.shlex(re.sub(r'`([^`]*)`', r'$(\1)', line), posix=True, punctuation_chars=';&|()<>')
    lex.whitespace_split = True
    lex.commenters = '#'
    cmds: list[tuple[int, list[str]]] = [(depth, [])]
    skip = False
    for tok in lex:
        if skip:
            skip = False
            continue
        if tok and set(tok) <= set(';&|()'):
            depth = max(0, depth + tok.count('(') - tok.count(')'))   # a case pattern's `)` cannot go below 0
            cmds.append((depth, []))
        elif tok and set(tok) <= set('<>&'):                 # a redirection: drop it and its target
            skip = True
            if cmds[-1][1] and cmds[-1][1][-1].isdigit():    # the fd number of `2>`
                cmds[-1][1].pop()
        else:
            cmds[-1][1].append(tok)
    return [(d, c) for d, c in cmds if c], depth


def script_named_files(script: str, cwd: str, script_dir: str = '') -> tuple[set[str], set[str], set[str]]:
    """Repo files a grading script EXECUTES, and repo directories it puts on PATH: (exact, suffix, dirs), all
    repo-relative.

    Only the program position counts: the command word when it is a path (`./run_tests`, `bin/goyacc`,
    `node_modules/.bin/jest`), the script an interpreter runs (`node build/run-tests.js`, `python manage.py test`,
    `bash ci/test.sh`, `go run tools/gen.go`), a file node preloads (`node -r ./setup.js`), and a file the shell
    `source`s / `.`s. Commands are found inside `if`/`for`/`while`/`!`/`{ }`, `$( )` and backticks, the command
    string of `bash -c` / `sh -lc`, and a here-doc fed to a shell (`bash <<EOF`). Arguments are not named, nor is
    `python -m <module>`: a full scan of the MiMo tasks (EXP-018 inner_scan.py) found scripts that `cp` the fixed
    sources into site-packages, pass source files to doctest, or run the program under test as a module --
    protecting those would revert the fix itself.

    `cd` is followed (undone when a subshell closes), so a name is resolved against the directory the script is in
    at that point (`cd packages/vuetify && node build/run-tests.js`). A name is EXACT once the script has done an
    absolute cd (`cd /testbed`, or `$(dirname "$0")`, which is known); before that the start directory is assumed
    to be the repo root, so the resolved name is a SUFFIX (it also matches deeper paths ending in it). After a cd
    that cannot be followed (`cd "$DIR"`) a multi-segment name is kept as written, as a suffix. Variable or glob
    names (`$BIN`, `*.js`) are skipped. Heuristic over shell text: what it misses is left to the per-test-id
    second check."""
    exact: set[str] = set()
    suffix: set[str] = set()
    dirs: set[str] = set()

    def parse(text: str, cur: str | None, anchored: bool, level: int) -> None:
        def name(tok: str):
            if not tok or any(c in tok for c in '$*?[]{}`~<>\'",=: ') or '://' in tok:
                return
            if tok.startswith('/'):
                rel = _repo_rel(tok, cwd)
                if rel:
                    exact.add(rel)
                return
            t = tok[2:] if tok.startswith('./') else tok
            if cur is None:                                  # after a cd we could not follow: as written
                w = _norm_rel(t)
                if w and not t.startswith('../'):
                    (suffix if '/' in w else exact).add(w)
                return
            p = _norm_rel(f'{cur}/{t}' if cur else t)
            if p:
                (exact if anchored else suffix).add(p)

        heredoc: str | None = None
        heredoc_body: list[str] | None = None        # collected when the here-doc is a shell's stdin
        pending: list[str] = []                      # lines of a quoted argument that spans lines
        stack: list[tuple[str | None, bool]] = []    # the directory outside each open subshell (across lines)
        for raw in re.sub(r'\\\r?\n', ' ', text).splitlines():        # join `\` continuation lines first
            line = raw.strip()
            if heredoc is not None:
                if line == heredoc:
                    if heredoc_body is not None and level < 4:
                        parse('\n'.join(heredoc_body), cur, anchored, level + 1)
                    heredoc = heredoc_body = None
                elif heredoc_body is not None:
                    heredoc_body.append(raw)
                continue                             # otherwise: input to another program, not shell commands
            if not pending and (not line or line.startswith('#')):
                continue
            if pending:
                line = '\n'.join(pending + [line])
            try:
                commands, end_depth = _shell_commands(line, len(stack))
            except ValueError:                       # an unclosed quote: read on (bounded), else give the line up
                pending = line.split('\n') if len(pending) < 50 else []
                continue
            pending = []
            shell_reads_stdin = False
            for depth, toks in commands:
                while len(stack) > depth:            # a subshell closed: its cd is gone
                    cur, anchored = stack.pop()
                while len(stack) < depth:            # one opened: remember the outer directory
                    stack.append((cur, anchored))
                while toks and toks[0] in _KEYWORDS:  # if / then / do / ! / { ... : the command follows
                    toks = toks[1:]
                if toks[:1] == ['export']:
                    toks = toks[1:]
                while toks and _ASSIGN_RE.match(toks[0]):            # VAR=value prefixes / export VAR=value
                    var, _, val = toks.pop(0).partition('=')
                    if var == 'PATH':
                        for part in val.split(':'):
                            rel = (_repo_rel(part, cwd) if part.startswith('/') else
                                   _norm_rel(f'{cur}/{part}') if cur is not None and part and '$' not in part
                                   else None)
                            if rel and '$' not in rel:
                                dirs.add(rel)
                while toks and toks[0] in _WRAPPERS:                 # exec / env / timeout 600 / xvfb-run ...
                    toks.pop(0)
                    while toks and (toks[0].startswith('-') or toks[0].isdigit() or _ASSIGN_RE.match(toks[0])
                                    or re.fullmatch(r'\d+[smhd]?', toks[0])):
                        toks.pop(0)
                if toks[:1] == ['npx'] and len(toks) > 1 and toks[1] in _INTERPRETERS:
                    toks = toks[1:]
                if not toks:
                    continue
                if toks[0] == 'cd':
                    target = toks[1] if len(toks) > 1 else ''
                    cur = _cd(cur, target, cwd)
                    anchored = anchored or (cur is not None and target.startswith('/'))
                    continue
                if toks[0] in ('source', '.'):       # the shell runs the file in place
                    name(toks[1] if len(toks) > 1 else '')
                    continue
                prog = toks[0].rsplit('/', 1)[-1]
                if '/' in toks[0]:
                    name(toks[0])                                    # the program itself is a repo path
                if prog == 'go':
                    if toks[1:2] == ['run']:                        # go run [build flags] a.go b.go [args]
                        rest, seen = iter(toks[2:]), False
                        for tok in rest:
                            if tok.startswith('-'):
                                if seen:
                                    break                            # a flag after the files: the program's own
                                if tok in _GO_VALUE_FLAGS:
                                    next(rest, '')
                                continue
                            if not tok.endswith('.go'):
                                break                                # a package path: not a file
                            name(tok)
                            seen = True
                    continue
                if prog in _SHELLS:
                    args, found = iter(toks[1:]), False
                    for tok in args:
                        if tok.startswith('--'):                     # --norc, --login
                            continue
                        if tok[:1] in '-+' and len(tok) > 1 and tok[1:].isalpha():
                            if 'c' in tok[1:]:                       # -c / -lc / -ec: a command string follows
                                if level < 4:
                                    parse(next(args, ''), cur, anchored, level + 1)
                                found = True
                                break
                            if tok[-1] in 'oO':                      # -o pipefail / -O extglob: a value follows
                                next(args, '')
                            if 's' in tok[1:]:                       # -s: commands come from stdin
                                break
                            continue
                        name(tok)                                    # the script file
                        found = True
                        break
                    shell_reads_stdin = shell_reads_stdin or not found
                    continue
                if prog not in _INTERPRETERS and not re.fullmatch(r'python[0-9.]*', prog):
                    continue
                args = iter(toks[1:])
                for tok in args:
                    if tok in ('-m', '-c', '-e', '-p', '--eval', '--print', '-'):   # code / a module, not a file
                        break
                    if tok in ('-r', '--require', '--import', '--loader', '--experimental-loader'):
                        pre = next(args, '')                         # node preloads this before the script;
                        if pre.startswith(('./', '../', '/')):       # a bare name is a package, not a repo file
                            name(pre)
                        continue
                    if tok in _VALUE_FLAGS:                          # python -W ignore / -X dev: skip the value
                        next(args, '')
                        continue
                    if tok.startswith('-'):
                        continue
                    if tok.endswith(_SCRIPT_EXTS) or '/' in tok:
                        name(tok)
                    break                                            # the script; the rest are its arguments
            while len(stack) > end_depth:            # subshells closed at the end of the line
                cur, anchored = stack.pop()
            while len(stack) < end_depth:            # `(` opened at the end of the line, closed on a later one
                stack.append((cur, anchored))
            if m := _HEREDOC_RE.search(line):
                heredoc = m.group(2)
                heredoc_body = [] if shell_reads_stdin else None   # `bash <<EOF`: the body runs as a script

    # `$(dirname "$0")`, `$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`: the script's own directory, which we know
    abs_dir = cwd.rstrip('/') + ('/' + script_dir if script_dir else '')
    parse(_SCRIPT_DIR_RE.sub(abs_dir, script), '', False, 0)
    return exact, suffix, dirs


def script_protected(touched: set[str], exact: set[str], suffix: set[str], dirs: set[str]) -> set[str]:
    """Touched paths a grading script runs: a named file (exact or suffix set), a file whose path ends with a
    multi-segment suffix name (the script's start directory was assumed, or a cd could not be followed), or
    anything under a named PATH directory. A bare name is matched only exactly, so `node index.js` does not
    protect every index.js in the repo."""
    ends = tuple('/' + f for f in suffix if '/' in f)
    return {t for t in touched
            if t in exact or t in suffix or (ends and t.endswith(ends)) or any(t.startswith(d + '/') for d in dirs)}
