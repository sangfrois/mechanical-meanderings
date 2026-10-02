"""Speech-to-visual prompt agent; transport(url, payload, timeout) returns a dict."""
import json
import math
import re
import threading
import time
from urllib.request import Request, urlopen


class StaleGeneration(Exception):
    """A reset superseded this request; do not emit its output."""


def normalize(content):
    if not isinstance(content, str):
        raise ValueError('Ollama returned no text content')
    content = re.sub(r'<think\b[^>]*>.*?(?:</think\s*>|$)', '', content, flags=re.S | re.I)
    content = re.sub(r'^\s*```[^\n]*\n|\n?```\s*$', '', content)
    content = re.sub(r'^\s*(?:visual\s+)?prompt\s*:\s*', '', content, flags=re.I)
    content = ' '.join(content.strip().strip('\"\'`“”').split()[:20])
    if not content:
        raise ValueError('Ollama returned empty prompt')
    if len(content) > 8192:
        raise ValueError('Ollama prompt exceeds 8192 characters')
    return content


def http_chat(url, payload, timeout):
    request = Request(url, json.dumps(payload).encode('utf-8'), {'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Agent:
    def __init__(self, system_prompt, transport=http_chat, model='gemma4:12b',
                 ollama_url='http://localhost:11434', timeout=120, history_turns=3,
                 reset_after=None, clock=time.monotonic):
        if history_turns < 0:
            raise ValueError('history_turns must be nonnegative')
        if reset_after is not None and reset_after <= 0:
            raise ValueError('reset_after must be positive')
        self.history_turns = history_turns
        self.history = {}
        self.system_prompt = system_prompt
        self.transport = transport
        self.model = model
        self.url = ollama_url.rstrip('/') + '/api/chat'
        self.timeout = timeout
        self.reset_after = reset_after
        self.clock = clock
        self.last_activity = {}
        self.epochs = {}
        self.lock = threading.RLock()

    def reset(self, channel=None):
        with self.lock:
            channels = set(self.epochs) | set(self.history) if channel is None else {channel}
            for ch in channels:
                self.epochs[ch] = self.epochs.get(ch, 0) + 1
                self.history.pop(ch, None)
                self.last_activity.pop(ch, None)

    def generate(self, text, channel='default', expected_epoch=None, settings=None, telemetry=None):
        with self.lock:
            if expected_epoch is not None and self.epochs.get(channel, 0) != expected_epoch:
                raise StaleGeneration('reset before generation')
            now = self.clock()
            if (self.reset_after is not None and channel in self.last_activity
                    and now - self.last_activity[channel] >= self.reset_after):
                self.reset(channel)
            self.last_activity[channel] = now
            epoch = self.epochs.setdefault(channel, 0)
            turns = self.history_turns if settings is None else settings.history
            history = self.history.get(channel, [])[-2 * turns:] if turns else []
            payload = {'model': self.model, 'think': False, 'stream': False,
                       'messages': ([{'role': 'system', 'content': self.system_prompt}]
                                    + history
                                    + [{'role': 'user', 'content': text}])}
            if settings is not None and settings.options():
                payload['options'] = settings.options()
        response = self.transport(self.url, payload, self.timeout)
        if telemetry is not None:
            for key in ('eval_count', 'eval_duration', 'prompt_eval_count', 'prompt_eval_duration', 'total_duration', 'load_duration'):
                value = response.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                    telemetry[key] = value
            duration = telemetry.get('eval_duration', 0)
            telemetry['eval_tokens_per_second'] = (telemetry['eval_count'] * 1e9 / duration
                if duration > 0 and 'eval_count' in telemetry else None)
        result = normalize(response['message']['content'])
        with self.lock:
            if self.epochs[channel] != epoch:
                raise StaleGeneration('reset during generation')
            self.history[channel] = ((payload['messages'][1:] + [{'role': 'assistant', 'content': result}])[-2 * turns:]
                                     if turns else [])
        return result
