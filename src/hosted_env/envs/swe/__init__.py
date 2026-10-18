"""SWE environment (the `code` split of Xiaomi MiMo-V2.6-RL-oss): everything about running and grading code-fix
tasks lives in this package, none of it in the infrastructure.

- env.py      SweEnv: task loading (expected-tests gate), setup, patch capture, the isolated grade
- cleanup.py  rollout-container preparation (anti-hack cleanup, git history strip)
- protect.py  files the model's patch may not change (protected names, manifests, shadow modules, JS/TS configs)
- scripts.py  reading the task's grading scripts (shell parsing, files a script executes)
- reports.py  the second check (structured test reports, per-test-id verdict, go shim)
- baseline.py deriving each task's expected tests (`hosted-env baseline`)
"""
from .cleanup import anti_hack_cleanup, future_commits, git_strip
from .env import (DROP_PRIVS, GRADER_BASH, GRADER_UID, GRADER_VERSION, MAX_MODEL_PATCH_BYTES, MODEL_PATCH_REMOTE,
                  TEST_PATCH_REMOTE, SweEnv, as_grader)
from .protect import (GRADER_PARSERS, STRUCTURED_MANIFESTS, jsts_config_refs, manifest_reverts, protected_paths,
                      shadow_added)
from .reports import GO_REPORT_SHIM, report_verdict
from .scripts import MIMO_BUNDLE, command_repo_paths, script_named_files, script_protected, test_patch_scripts

__all__ = ['DROP_PRIVS', 'GO_REPORT_SHIM', 'GRADER_BASH', 'GRADER_PARSERS', 'GRADER_UID', 'GRADER_VERSION',
           'MAX_MODEL_PATCH_BYTES', 'MIMO_BUNDLE', 'MODEL_PATCH_REMOTE', 'STRUCTURED_MANIFESTS', 'TEST_PATCH_REMOTE',
           'SweEnv', 'anti_hack_cleanup', 'as_grader', 'command_repo_paths', 'future_commits', 'git_strip',
           'jsts_config_refs', 'manifest_reverts', 'protected_paths', 'report_verdict', 'script_named_files',
           'script_protected', 'shadow_added', 'test_patch_scripts']
