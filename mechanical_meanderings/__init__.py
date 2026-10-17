"""Speech-to-visual prompt agent; transport(url, payload, timeout) returns a dict."""
import json
import math
import re
import threading
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import http.client
import inspect
import socket


class GenerationTimeout(TimeoutError):
    """The total realtime generation budget expired."""


class GenerationControl:
    """Cancellation owns the socket, never a background inference task."""

    def __init__(self, budget):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('generation budget must be finite and positive')
        self.deadline = time.monotonic() + budget
        self.reason = None
        self.lock = threading.Lock()
        self.sock = None
        self.changed = threading.Event()

    def cancel(self, reason='superseded'):
        with self.lock:
            if self.reason is None:
                self.reason = reason
            sock = self.sock
        self.changed.set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def attach(self, sock):
        with self.lock:
            self.sock = sock
            reason = self.reason
        if reason is not None:
            self.cancel(reason)
        self.check()

    def check(self):
        if self.reason is None and time.monotonic() >= self.deadline:
            self.cancel('timeout')
        if self.reason == 'timeout':
            raise GenerationTimeout('realtime generation deadline exceeded')
        if self.reason is not None:
            raise StaleGeneration(self.reason)


def _controlled_chat(url, payload, timeout, control):
    """Stream into a private buffer; only a complete response leaves here.

    One joined watchdog interrupts header/body stalls by shutting down the
    actual socket, including continuously arriving token streams at deadline.
    """
    control.check()
    parsed = urlsplit(url)
    cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    remaining = max(.001, control.deadline - time.monotonic())
    connection = cls(parsed.hostname, parsed.port, timeout=min(timeout, remaining))
    finished = threading.Event()

    def watch():
        while not finished.is_set():
            if control.reason is not None:
                return
            remaining = control.deadline - time.monotonic()
            if remaining <= 0:
                control.cancel('timeout')
                return
            finished.wait(min(.025, remaining))

    watchdog = threading.Thread(target=watch, name='ollama-deadline')
    watchdog.start()
    response = None
    try:
        connection.connect()
        control.attach(connection.sock)
        request_payload = dict(payload, stream=True)
        connection.request('POST', parsed.path + ('?' + parsed.query if parsed.query else ''),
                           json.dumps(request_payload).encode('utf-8'),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status >= 400:
            raise ValueError(f'Ollama HTTP {response.status}')
        pending = b''
        content = []
        final = None

        def consume(line):
            nonlocal final
            if not line.strip():
                return
            record = json.loads(line)
            if record.get('error'):
                raise ValueError(str(record['error']))
            content.append(record.get('message', {}).get('content', ''))
            if record.get('done'):
                final = record
            # Compatibility with nonstreaming Ollama-compatible HTTP fixtures.
            elif 'done' not in record and 'message' in record:
                final = record

        while final is None:
            control.check()
            block = response.read1(65536)
            control.check()
            if not block:
                if pending:
                    consume(pending)
                    pending = b''
                break
            pending += block
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                consume(line)
        control.check()
        if final is None:
            raise ValueError('Ollama stream ended before completion')
        return dict(final, message={'content': ''.join(content)})
    except socket.timeout:
        # Socket timeouts also enforce the remaining total budget. Windows may
        # deliver this before the watchdog's next scheduling slice.
        if remaining <= timeout:
            control.cancel('timeout')
        control.check()
        raise
    except (OSError, http.client.HTTPException):
        control.check()
        raise
    finally:
        finished.set()
        # shutdown precedes buffered response.close, which can otherwise block.
        with control.lock:
            sock = control.sock
            control.sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if response is not None:
            response.close()
        connection.close()
        watchdog.join()


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
                 ollama_url='http://127.0.0.1:11434', timeout=120, history_turns=3,
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

    def generate(self, text, channel='default', expected_epoch=None, settings=None, telemetry=None,
                 history_override=None, commit=True, control=None):
        """Generate without committing when the caller must validate freshness first.

        history_override is an explicit chronological message snapshot; legacy
        callers retain the configured generated-turn history unchanged.
        """
        with self.lock:
            if expected_epoch is not None and self.epochs.get(channel, 0) != expected_epoch:
                raise StaleGeneration('reset before generation')
            now = self.clock()
            if (history_override is None and self.reset_after is not None and channel in self.last_activity
                    and now - self.last_activity[channel] >= self.reset_after):
                self.reset(channel)
            self.last_activity[channel] = now
            epoch = self.epochs.setdefault(channel, 0)
            turns = self.history_turns if settings is None else settings.history
            history = (list(history_override) if history_override is not None else
                       self.history.get(channel, [])[-2 * turns:] if turns else [])
            payload = {'model': self.model, 'think': False, 'stream': False,
                       'messages': ([{'role': 'system', 'content': self.system_prompt}]
                                    + history
                                    + [{'role': 'user', 'content': text}])}
            if settings is not None and settings.options():
                payload['options'] = settings.options()
        http_started = time.perf_counter()
        try:
            if control is not None:
                control.check()
            if control is not None and self.transport is http_chat:
                response = _controlled_chat(self.url, payload, self.timeout, control)
            else:
                # Existing three-positional-argument transports stay synchronous.
                # Cooperative injections may explicitly accept control; never
                # abandon an injected blocking call in a new inference thread.
                try:
                    parameters = inspect.signature(self.transport).parameters if control is not None else {}
                except (TypeError, ValueError):
                    parameters = {}
                if 'control' in parameters:
                    response = self.transport(self.url, payload, self.timeout, control=control)
                else:
                    response = self.transport(self.url, payload, self.timeout)
            if control is not None:
                control.check()
            if telemetry is not None:
                telemetry['generation_status'] = 'complete'
        except GenerationTimeout:
            if telemetry is not None:
                telemetry['generation_status'] = 'timeout'
            raise
        except StaleGeneration:
            if telemetry is not None:
                telemetry['generation_status'] = control.reason if control is not None else 'superseded'
            raise
        except Exception:
            if telemetry is not None:
                telemetry['generation_status'] = 'error'
            raise
        finally:
            if telemetry is not None:
                telemetry['http_elapsed'] = time.perf_counter() - http_started
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
            if commit:
                self.history[channel] = ((payload['messages'][1:] + [{'role': 'assistant', 'content': result}])[-2 * turns:]
                                         if turns else [])
        return result
