"""Split reasoning out of model-generated text and parse tool calls (the XML format shared by Qwen3.5 / MiMo).

Copied from the EXP-014 harness (aligned with Xiaomi mimoagent's bash-only config):
`<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>`; one leading and one
trailing newline are stripped from each parameter value.
"""
from __future__ import annotations

import re

_TC_RE = re.compile(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', re.S)
_PARAM_RE = re.compile(r'<parameter=([^>\n]+)>(.*?)</parameter>', re.S)


def _strip_one_newline(s: str) -> str:
    if s.startswith('\n'):
        s = s[1:]
    if s.endswith('\n'):
        s = s[:-1]
    return s


def split_reasoning(text: str, prompt_opened_think: bool) -> tuple[str | None, str, bool]:
    """Split out reasoning and content, returning (reasoning, content, think_closed).

    prompt_opened_think: the prompt already ends with `<think>` (Qwen3.5 with thinking on), so the generation
    only contains `</think>`. With MiMo thinking on, the prompt ends with `assistant\\n` and the model writes
    `<think>` itself.
    """
    if '</think>' in text:
        r, c = text.split('</think>', 1)
        r = r.split('<think>', 1)[-1]
        return r.strip('\n'), c.lstrip('\n'), True
    if prompt_opened_think or text.lstrip().startswith('<think>'):
        return text.split('<think>', 1)[-1].strip('\n'), '', False   # stopped while still thinking
    return None, text, True


def parse_tool_calls(content: str) -> tuple[str, list[dict], str | None]:
    """Return (text before the calls, [{'name', 'arguments'}], format error or None)."""
    calls = []
    for m in _TC_RE.finditer(content):
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            args[pm.group(1).strip()] = _strip_one_newline(pm.group(2))
        calls.append({'name': m.group(1).strip(), 'arguments': args})
    first = content.find('<tool_call>')
    text = (content if first < 0 else content[:first]).strip()
    err = None
    n_open = content.count('<tool_call>')
    if n_open != len(calls):
        err = (f'Found {n_open} <tool_call> block(s) but could parse only {len(calls)}. Each tool call must be '
               '<tool_call><function=NAME><parameter=NAME>VALUE</parameter></function></tool_call>.')
    return text, calls, err
