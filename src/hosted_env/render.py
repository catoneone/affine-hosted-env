"""Token-level rendering (design §6, §8): the worker renders the task prompt into prompt tokens (P) and renders
tool outputs into tokens (T) that follow the model's current turn; the miner appends them to the context verbatim
and keeps sampling.

Append-only context, no re-rendering (token in, token out): assistant turns in the history are the raw tokens the
model generated at the time and are never re-normalized by the chat template (dropping newlines, reordering tool
call arguments, etc.). This differs from the EXP-014 harness, which re-renders the whole conversation every turn;
the MiMo template keeps historical reasoning, so the main differences are only in whitespace and how tool calls
are written.

The tool-output delta is computed by diff rendering: take a fake [user, assistant] conversation, render it once
"up to the end of the assistant turn" and once "plus these tool messages, opening the next assistant turn", and
encode the text after the assistant end marker. This avoids hand-writing the format for every template.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

_DUMMY = [{'role': 'user', 'content': 'x'},
          # With reasoning: Qwen-family templates render the think block differently for the "last assistant"
          # and a "middle assistant"; with reasoning present both positions render the same, so prefixes match
          {'role': 'assistant', 'content': 'y', 'reasoning_content': 'z'}]


class RenderError(RuntimeError):
    pass


class TemplateRenderer:
    """Subclasses implement render_text / encode / decode / token_id / vocab_size."""

    def __init__(self, turn_end: str = '<|im_end|>', extra_stops: Sequence[str] = ('<|endoftext|>',)):
        self.turn_end = turn_end               # end marker of assistant messages in the template; each model turn ends with it
        self.turn_end_id = self.token_id(turn_end)
        self.stop_ids = [self.turn_end_id] + [self.token_id(s) for s in extra_stops]
        self.prompt_opens_think = self.render_text([{'role': 'user', 'content': 'x'}], None, True) \
            .rstrip().endswith('<think>')

    # ---- implemented by subclasses
    def render_text(self, messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool) -> str:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def token_id(self, token: str) -> int:
        raise NotImplementedError

    vocab_size: int = 0

    # ---- used by the worker
    def prompt(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        return self.encode(self.render_text(messages, tools, True))

    def tool_delta(self, tool_messages: list[dict], tools: list[dict] | None) -> list[int]:
        base = self.render_text(_DUMMY, tools, False)
        full = self.render_text(_DUMMY + list(tool_messages), tools, True)
        cut = base.rfind(self.turn_end)
        if cut < 0:
            raise RenderError('template does not end assistant turns with turn_end')
        cut += len(self.turn_end)
        if full[:cut] != base[:cut]:
            raise RenderError('template re-renders history when tool messages are appended')
        ids = self.encode(full[cut:])
        if not ids:
            raise RenderError('empty tool delta')
        return ids

    def decode_turn(self, ids: Sequence[int]) -> str:
        """Decode one model turn's tokens, dropping the trailing stop token."""
        if ids and ids[-1] in self.stop_ids:
            ids = ids[:-1]
        return self.decode(ids)


class HFRenderer(TemplateRenderer):
    """Uses the model's own HF tokenizer and chat template."""

    def __init__(self, path: str, *, enable_thinking: bool | None = None, turn_end: str = '<|im_end|>',
                 extra_stops: Sequence[str] = ('<|endoftext|>',)):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enable_thinking = enable_thinking
        self._lock = threading.Lock()          # HF fast tokenizer occasionally raises "Already borrowed" on concurrent encodes
        self.vocab_size = len(self.tok)
        super().__init__(turn_end, extra_stops)

    def render_text(self, messages, tools, add_generation_prompt):
        kw = {} if self.enable_thinking is None else {'enable_thinking': self.enable_thinking}
        with self._lock:
            out = self.tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                               add_generation_prompt=add_generation_prompt, **kw)
        if not isinstance(out, str):
            raise RenderError('chat template did not return text')
        return out

    def encode(self, text):
        with self._lock:
            return list(self.tok(text, add_special_tokens=False)['input_ids'])

    def decode(self, ids):
        with self._lock:
            return self.tok.decode(list(ids), skip_special_tokens=False)

    def token_id(self, token):
        tid = self.tok.convert_tokens_to_ids(token)
        if tid is None or tid == self.tok.unk_token_id:
            raise RenderError(f'unknown special token {token!r}')
        return int(tid)


class CharRenderer(TemplateRenderer):
    """For tests / local demos: a ChatML-like template (assistant turns written like MiMo), one token per
    character. Does not depend on transformers."""

    SPECIAL = {'<|im_start|>': 1, '<|im_end|>': 2, '<|endoftext|>': 3}
    OFFSET = 16

    def __init__(self, enable_thinking: bool = False):
        import json
        self._json = json
        self.enable_thinking = enable_thinking
        self.vocab_size = 0x110000 + self.OFFSET
        self._rev = {v: k for k, v in self.SPECIAL.items()}
        super().__init__('<|im_end|>', ('<|endoftext|>',))

    def render_text(self, messages, tools, add_generation_prompt):
        out = []
        if tools:
            out.append('<|im_start|>system\n<tools>' + ''.join('\n' + self._json.dumps(t, sort_keys=True)
                                                               for t in tools) + '\n</tools><|im_end|>')
        for m in messages:
            if m['role'] == 'assistant':
                calls = ''.join('<tool_call><function=' + c['function']['name'] + '>' +
                                ''.join(f'<parameter={k}>{v}</parameter>' for k, v in c['function']['arguments'].items())
                                + '</function></tool_call>' for c in m.get('tool_calls') or [])
                out.append(f"<|im_start|>assistant\n<think>{m.get('reasoning_content') or ''}</think>"
                           f"{m.get('content') or ''}{calls}<|im_end|>")
            else:
                out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            out.append('<|im_start|>assistant\n' + ('' if self.enable_thinking else '<think></think>'))
        return ''.join(out)

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for s, tid in self.SPECIAL.items():
                if text.startswith(s, i):
                    ids.append(tid)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]) + self.OFFSET)
                i += 1
        return ids

    def decode(self, ids):
        return ''.join(self._rev[t] if t in self._rev else chr(t - self.OFFSET) for t in ids)

    def token_id(self, token):
        return self.SPECIAL[token]


def make_renderer(cfg: dict) -> TemplateRenderer:
    """The renderer section of the worker config: {"kind": "hf", "path": ..., "enable_thinking": true} or
    {"kind": "char"}."""
    kind = cfg.get('kind', 'hf')
    if kind == 'char':
        return CharRenderer(enable_thinking=bool(cfg.get('enable_thinking', False)))
    if kind == 'hf':
        return HFRenderer(cfg['path'], enable_thinking=cfg.get('enable_thinking'),
                          turn_end=cfg.get('turn_end', '<|im_end|>'),
                          extra_stops=tuple(cfg.get('extra_stops', ('<|endoftext|>',))))
    raise ValueError(f'unknown renderer kind {kind!r}')
