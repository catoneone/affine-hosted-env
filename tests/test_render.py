"""render.py: task prompt and tool-output deltas.

The real-tokenizer tests need the MiMo / Qwen3.5 tokenizer directories: point the HOSTED_ENV_TOKENIZERS
environment variable at a directory containing hf-tokenizer-mimo-9b/ and hf-tokenizer-qwen35-9b/; the tests are
skipped when the directory is absent.
"""
import os

import pytest

from hosted_env.envs.base import BASH_TOOL, INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
from hosted_env.render import CharRenderer, RenderError, TemplateRenderer

TOK_ROOT = os.environ.get('HOSTED_ENV_TOKENIZERS', '')
TOK = {n: os.path.join(TOK_ROOT, f'hf-tokenizer-{n}-9b') for n in ('mimo', 'qwen35')}

SYS = {'role': 'system', 'content': SYSTEM_TEMPLATE.format(cwd='/testbed')}
USER = {'role': 'user', 'content': INSTANCE_TEMPLATE.format(task='bug')}
ASST = {'role': 'assistant', 'content': 'Looking.', 'reasoning_content': 'need to grep',
        'tool_calls': [{'type': 'function', 'function': {'name': 'bash', 'arguments': {'command': 'ls -la'}}}]}
TOOL_A = {'role': 'tool', 'content': "a.py\n\nTool metadata: {'returncode': 0}"}
TOOL_B = {'role': 'tool', 'content': 'Error: second'}


def split_full(r: TemplateRenderer, tools_msgs):
    """Render the whole conversation with the template itself and cut it into P / M / T text: P = the prompt
    with only the task prompt, M = up to the assistant end marker, T = the rest (tool outputs + start of the next
    assistant turn)."""
    # With thinking off the prompt already contains an empty think block and the model generates after it, so
    # the historical assistant turn has no reasoning either
    asst = ASST if r.enable_thinking else {**ASST, 'reasoning_content': ''}
    p = r.render_text([SYS, USER], [BASH_TOOL], True)
    full = r.render_text([SYS, USER, asst] + tools_msgs, [BASH_TOOL], True)
    assert full.startswith(p), 'template is not prefix-stable from prompt to first assistant turn'
    rest = full[len(p):]
    cut = rest.index(r.turn_end) + len(r.turn_end)
    return p, rest[:cut], rest[cut:]


def check_renderer(r: TemplateRenderer):
    for tools_msgs in ([TOOL_A], [TOOL_A, TOOL_B]):
        p, m, t = split_full(r, tools_msgs)
        assert r.prompt([SYS, USER], [BASH_TOOL]) == r.encode(p)
        assert r.tool_delta(tools_msgs, [BASH_TOOL]) == r.encode(t)
        # M ends with a stop token (a special token), so appending delta T equals encoding the whole thing; no
        # BPE merge across the boundary
        assert r.encode(m) + r.tool_delta(tools_msgs, [BASH_TOOL]) == r.encode(m + t)
        assert r.encode(m)[-1] == r.turn_end_id and r.turn_end_id in r.stop_ids
        assert r.decode_turn(r.encode(m)) == m[:-len(r.turn_end)]


def test_char_renderer():
    for thinking in (False, True):
        r = CharRenderer(enable_thinking=thinking)
        check_renderer(r)
        assert r.prompt_opens_think is False


def test_bad_template_detected():
    class Rerender(CharRenderer):
        def render_text(self, messages, tools, add_generation_prompt):   # template that rewrites history once tool messages are added
            s = super().render_text(messages, tools, add_generation_prompt)
            return s.replace('<think>z</think>', '') if any(m['role'] == 'tool' for m in messages) else s
    with pytest.raises(RenderError, match='re-renders history'):
        Rerender().tool_delta([TOOL_A], None)


@pytest.mark.parametrize('name', ['mimo', 'qwen35'])
@pytest.mark.parametrize('thinking', [False, True])
def test_hf_renderer(name, thinking):
    if not os.path.exists(os.path.join(TOK[name], 'tokenizer.json')):
        pytest.skip('no local tokenizer')
    pytest.importorskip('transformers')
    from hosted_env.render import HFRenderer
    r = HFRenderer(TOK[name], enable_thinking=thinking)
    check_renderer(r)
    assert r.prompt_opens_think == (thinking and name == 'qwen35')      # Qwen3.5 with thinking on: the prompt already opens think
    delta = r.decode(r.tool_delta([TOOL_A], [BASH_TOOL]))
    assert TOOL_A['content'] in delta and delta.rstrip().endswith(('assistant', '<think>', '</think>'))
    if name == 'mimo':
        assert delta.startswith('<|im_start|>tool\n')
