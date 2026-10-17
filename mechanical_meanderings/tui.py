"""Bounded live instrument panel; the bridge, not the UI, owns request snapshots."""
import asyncio
import queue
import statistics
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Sparkline, Static
from .live import PRESETS


class MeanderingsApp(App):
    TITLE = 'MECHANICAL MEANDERINGS'
    SUB_TITLE = 'a live instrument for situated transformation'
    BINDINGS = [Binding(key, action, label, priority=True) for key, action, label in
                [('ctrl+a', 'apply', 'Apply controls'), ('ctrl+p', 'pause', 'Pause/resume'),
                 ('ctrl+r', 'reset', 'Reset memory'), ('ctrl+enter', 'send', 'Send text'),
                 ('ctrl+q', 'quit', 'Quit')]]
    CSS = """
    Screen { background: #101820; color: #dbe5e8; }
    Header { background: #182b35; color: #7de2cf; }
    Footer { background: #182b35; }
    #body { height: 1fr; }
    #controls { width: 37; min-width: 30; background: #15232c; padding: 0 1; border-right: solid #34535d; }
    .section { color: #7de2cf; text-style: bold; margin-top: 1; }
    .hint { color: #97aeb7; height: auto; }
    #controls Input { height: 3; margin: 0; }
    #controls Label { height: 1; margin-top: 1; color: #aec3cb; }
    #stage { width: 1fr; padding: 0 1; }
    #route { height: 3; color: #97aeb7; }
    #status { height: 2; color: #7de2cf; text-style: bold; }
    #metrics { height: 5; background: #182b35; padding: 0 1; }
    #latency { height: 2; color: #e6b877; margin-bottom: 1; }
    #trajectory { height: 1fr; border: round #34535d; background: #111f28; }
    #notice { height: 2; color: #e6b877; }
    #manual-row { height: 3; }
    #manual { width: 1fr; }
    #channel { width: 14; }
    #send { min-width: 8; width: 8; }
    #actions { height: 3; margin-top: 1; }
    #actions Button { min-width: 10; width: 1fr; }
    #apply { width: 100%; margin-top: 1; }
    #session { height: 2; color: #97aeb7; }
    """
    LABELS = {'temperature': 'Temperature · 0–2', 'num_ctx': 'Context tokens · 256–131072',
              'top_p': 'Top-p · 0–1', 'top_k': 'Top-k · 0–1000',
              'repeat_penalty': 'Repeat penalty · 0–3', 'seed': 'Seed · -1 = random',
              'num_predict': 'Output budget · 1–4096 tokens', 'history': 'History · 0–64 successful pairs',
              'debounce': 'Debounce · 0–60 seconds', 'interval': 'Start interval · .05–300 seconds'}

    def __init__(self, bridge, session):
        super().__init__()
        self.bridge = bridge
        self.session = session
        self.bridge.on_event = session.record

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id='body'):
            with VerticalScroll(id='controls'):
                yield Static('SAMPLING LAB', classes='section')
                yield Static('Draft → Apply → next request.\nBlank sampling fields use model defaults; effective defaults are unknown.', classes='hint')
                yield Select([(name, name) for name in PRESETS], prompt='Explore a preset', id='preset')
                for key, label in self.LABELS.items():
                    yield Label(label)
                    value = getattr(self.bridge.settings, key)
                    yield Input('' if value is None else str(value), id=key, placeholder='model default')
                yield Button('Apply all · Ctrl+A', id='apply', variant='primary')
                yield Static('Presets edit the draft, not the live request. High temperature invites variation, not transport failures. Larger context may compete with diffusion for VRAM.', classes='hint')
            with Vertical(id='stage'):
                yield Static(f'OSC {self.bridge.address[0]}:{self.bridge.address[1]} → {self.bridge.agent.model}\nUDP → {self.bridge.output_target}{self.bridge.output_address}', id='route', markup=False)
                yield Static('LISTENING', id='status', markup=False)
                yield Static('', id='metrics', markup=False)
                yield Sparkline([], id='latency')
                yield RichLog(id='trajectory', max_lines=300, wrap=True, markup=False, highlight=False)
                yield Static('Explore the same words with different controls. Interactive visuals target 40–60 words; legacy visuals cap at 20.', id='notice', markup=False)
                with Horizontal(id='manual-row'):
                    yield Input(placeholder='Manual words / image / correction…', id='manual', max_length=self.bridge.MAX_TEXT)
                    yield Input('ch1' if self.bridge.channel_mode else 'default', id='channel', max_length=32)
                    yield Button('Send', id='send', variant='primary')
                with Horizontal(id='actions'):
                    yield Button('Pause', id='pause')
                    yield Button('Reset memory', id='reset', variant='warning')
                yield Static(f'JSONL · {self.session.path.name}\nPrivate speech text · flushed + fsynced per event', id='session', markup=False)
        yield Footer()

    def on_mount(self):
        self.bridge.start()
        self.set_interval(.1, self.refresh_live)
        self.refresh_live()

    async def on_unmount(self):
        await asyncio.to_thread(self.bridge.close)
        self.session.close()

    @on(Select.Changed, '#preset')
    def preset_changed(self, event):
        if event.value not in PRESETS:
            return
        overrides = PRESETS[event.value]
        for key in ('temperature', 'top_p', 'top_k', 'repeat_penalty'):
            self.query_one('#' + key, Input).value = str(overrides[key]) if key in overrides else ''
        self.query_one('#notice', Static).update(f'{event.value} staged. Ctrl+A applies the complete draft.')

    def action_apply(self):
        try:
            values = {}
            for key in self.LABELS:
                raw = self.query_one('#' + key, Input).value.strip()
                cast = int if key in ('num_ctx', 'top_k', 'seed', 'num_predict', 'history') else float
                values[key] = cast(raw) if raw else None
            settings = self.bridge.configure(**values)
        except ValueError as exc:
            self.query_one('#notice', Static).update(f'NOT APPLIED · {exc}')
            return
        self.query_one('#notice', Static).update(f'Applied atomically for next request · temperature {settings.temperature if settings.temperature is not None else "model default"} · history {settings.history}')

    def action_pause(self):
        self.bridge.set_paused(not self.bridge.paused)
        self.query_one('#pause', Button).label = 'Resume' if self.bridge.paused else 'Pause'
        self.query_one('#notice', Static).update('Pause holds new starts; in-flight output and raw forwarding continue. Reset discards in-flight output.' if self.bridge.paused else 'Resumed. Latest pending speech is eligible again.')

    def action_reset(self):
        self.bridge.receive('/reset', 1)
        self.query_one('#notice', Static).update('Memory and pending work cleared; in-flight output invalidated. Controls unchanged.')

    def action_send(self):
        widget = self.query_one('#manual', Input)
        try:
            channel = self.query_one('#channel', Input).value.strip()
            if not channel:
                raise ValueError('channel must not be empty')
            self.bridge.submit(widget.value, channel)
        except ValueError as exc:
            self.query_one('#notice', Static).update(str(exc))
            return
        widget.value = ''

    @on(Input.Submitted, '#manual')
    def manual_enter(self):
        self.action_send()

    @on(Button.Pressed)
    def button_pressed(self, event):
        actions = {'apply': self.action_apply, 'pause': self.action_pause,
                   'reset': self.action_reset, 'send': self.action_send}
        action = actions.get(event.button.id)
        if action:
            action()

    def refresh_live(self):
        # A timer tick can arrive after Textual removes the screen children.
        logs = list(self.query('#trajectory'))
        if not logs:
            return
        log = logs[0]
        for _ in range(64):
            try:
                event = self.session.events.get_nowait()
            except queue.Empty:
                break
            kind = event['kind']
            stamp = event['timestamp'][11:19]
            ch = event.get('channel', '')
            prefix = f'{stamp}  {kind.upper():8} {ch} '
            if kind == 'input':
                body = event['input'] or '(subtitle clear)'
            elif kind == 'result':
                s = event['settings']
                body = f'#{event["request_id"]}  {event["elapsed"]:.2f}s · T={s["temperature"]} H={s["history"]} ctx={s["num_ctx"]}\n    IN  {event["input"]}\n    OUT {event["output"]}'
            elif kind == 'error':
                body = f'TECHNICAL FAILURE · {event["stage"]} / {event["error"]} (not creative divergence)'
            elif kind == 'start':
                body = f'#{event["request_id"]}  {event["input"]}'
            elif kind == 'settings':
                body = 'Next request controls applied'
            else:
                body = event.get('reason', '')
            style = '#ed8f89' if kind in ('error', 'drop') else '#7de2cf' if kind == 'result' else '#aec3cb'
            log.write(Text(prefix + body, style=style))
        m = self.session.metrics()
        counts = m['counts']
        last = m['last'] or {}
        lat = m['latencies']
        avg = f'{statistics.mean(lat):.2f}s' if lat else 'unavailable'
        elapsed = f'{last["elapsed"]:.2f}s' if last else 'unavailable'
        speed = last.get('telemetry', {}).get('eval_tokens_per_second')
        speed = f'{speed:.1f} tok/s' if speed is not None else 'unavailable'
        rate = counts.get('result', 0) * 60 / max(m['uptime'], .001)
        with self.bridge.lock:
            backlog = self.bridge.pending.qsize() + len(self.bridge.dirty)
            pairs = sum(len(v) // 2 for v in self.bridge.agent.history.values())
        active = m['active']
        state = 'PAUSED' if self.bridge.paused else 'GENERATING' if active else 'LISTENING'
        self.query_one('#status', Static).update(f'{state}  ·  pending {backlog}  ·  retained pairs {pairs}' + (f'  ·  request #{active["request_id"]}' if active else ''))
        self.query_one('#metrics', Static).update(
            f'BRIDGE  last {elapsed} · mean≤120 {avg} · {rate:.1f} sends/min since launch\n'
            f'OLLAMA  reported decode {speed} · UDP sends {counts.get("result", 0)} · technical errors {counts.get("error", 0)}\n'
            f'STT latency / audio levels: unavailable · TD frames / FPS / delivery: unavailable\n'
            f'Input drops {counts.get("drop", 0)} · superseded {counts.get("discard", 0)} · UI-only evictions {m["dropped"]} · UDP ≠ rendered frames')
        self.query_one('#latency', Sparkline).data = lat or [0]
        if m['disk_error']:
            self.query_one('#session', Static).update(f'RECORDING FAILED · {m["disk_error"]}\nNew events are NOT guaranteed durable. Check disk/permissions.')
