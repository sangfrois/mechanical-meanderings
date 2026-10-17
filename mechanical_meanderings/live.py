"""Immutable next-request controls. None leaves Ollama's model default intact."""
from dataclasses import asdict, dataclass, replace
import math
import json
import os
import queue
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path


class Session:
    """Append-before-display journal; the UI may drop old events, disk never does.

    Each record is flushed and fsynced. Disk failure is latched visibly rather
    than killing OSC/inference. Records after failure are not claimed durable.
    """
    def __init__(self, directory='sessions', capacity=256):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.jsonl')
        self.file = self.path.open('x', encoding='utf-8')
        self.events = queue.Queue(maxsize=capacity)
        self.lock = threading.RLock()
        self.counts = Counter()
        self.latencies = deque(maxlen=120)
        self.started = time.monotonic()
        self.dropped = 0
        self.disk_error = None
        self.last_result = None
        self.active = None
        self.record({'kind': 'session', 'schema': 1})

    def record(self, event):
        with self.lock:
            event = dict(event, timestamp=datetime.now(timezone.utc).isoformat())
            try:
                self.file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                self.file.flush()
                os.fsync(self.file.fileno())
            except (OSError, ValueError) as exc:
                self.disk_error = type(exc).__name__
            self.counts[event['kind']] += 1
            if event['kind'] == 'start':
                self.active = event
            if event['kind'] in ('result', 'error', 'discard') and event.get('request_id'):
                self.active = None
            if event['kind'] == 'result':
                self.latencies.append(event['elapsed'])
                self.last_result = event
            if self.events.full():
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass  # The UI drained it after full() was checked.
                else:
                    self.dropped += 1
            self.events.put_nowait(event)

    def metrics(self):
        with self.lock:
            return dict(counts=dict(self.counts), latencies=list(self.latencies),
                        last=self.last_result, active=self.active, dropped=self.dropped,
                        disk_error=self.disk_error, uptime=time.monotonic() - self.started)

    def close(self):
        with self.lock:
            if not self.file.closed:
                self.record({'kind': 'session_end'})
                self.file.close()



@dataclass(frozen=True)
class Settings:
    temperature: float | None = None
    num_ctx: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    num_predict: int | None = None
    history: int = 3
    debounce: float = .6
    interval: float = 2.0

    def __post_init__(self):
        bounds = {'temperature': (0, 2), 'num_ctx': (256, 131072),
                  'top_p': (0, 1), 'top_k': (0, 1000),
                  'repeat_penalty': (0, 3), 'seed': (-1, 2147483647),
                  'num_predict': (1, 4096), 'history': (0, 64),
                  'debounce': (0, 60), 'interval': (.05, 300)}
        integers = {'num_ctx', 'top_k', 'seed', 'num_predict', 'history'}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if value is None and key not in ('history', 'debounce', 'interval'):
                continue
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or not low <= value <= high
                    or (key in integers and not isinstance(value, int))):
                raise ValueError(f'{key}: expected {low} … {high}' + (' (integer)' if key in integers else ''))

    def updated(self, **changes):
        return replace(self, **changes)

    def snapshot(self):
        return asdict(self)

    def options(self):
        return {k: v for k, v in asdict(self).items()
                if k not in ('history', 'debounce', 'interval') and v is not None}


PRESETS = {
    'Model defaults': {},
    'Close reflection': dict(temperature=.35, top_p=.85, top_k=30, repeat_penalty=1.1),
    'Associative drift': dict(temperature=1.25, top_p=.95, top_k=60, repeat_penalty=1.15),
    'Wild material': dict(temperature=1.5, top_p=.98, top_k=100, repeat_penalty=1.2),
}
