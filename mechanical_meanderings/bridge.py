"""Nonblocking OSC receiver with one bounded serial generation worker."""
import logging
import queue
import re
import threading
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from . import StaleGeneration
from .live import Settings
from .timing import generation_timing

log = logging.getLogger(__name__)


class Bridge:
    MAX_CHANNELS = 64
    MAX_TEXT = 8192

    def __init__(self, agent, listen_host='localhost', listen_port=9001,
                 output_host='localhost', output_port=9000, queue_size=8,
                 channel_mode=False, forward_raw=False, output_address='/prompt',
                 dynamic=False, debounce=0.6, update_interval=2.0, on_event=None):
        if debounce < 0 or update_interval <= 0:
            raise ValueError('debounce must be nonnegative and update_interval positive')
        if not output_address.startswith('/'):
            raise ValueError('output_address must start with /')
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        self.dynamic = dynamic
        self.debounce = debounce
        self.update_interval = update_interval
        self.settings = Settings(history=agent.history_turns, debounce=debounce, interval=update_interval)
        self.on_event = on_event
        self.paused = False
        self.request_id = 0
        self.dirty = {}
        self.last_started = {}
        self.channel_mode = channel_mode
        self.forward_raw = forward_raw
        self.output_address = output_address
        self.output_target = f'{output_host}:{output_port}'
        self.trigger_address = '/trigger' if output_address == '/transcription' else output_address.rstrip('/') + '/trigger'
        self.agent = agent
        self.pending = queue.Queue(maxsize=queue_size)
        self.latest = {}
        self.epochs = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.receive)
        try:
            self.server = BlockingOSCUDPServer((listen_host, listen_port), dispatcher)
        except OSError as exc:
            raise RuntimeError(f'OSC input {listen_host}:{listen_port} cannot bind: port already in use '
                'or access denied. Stop the previous bridge before launching another copy.') from exc
        self.client = SimpleUDPClient(output_host, output_port)
        self.address = self.server.server_address
        self.receiver = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker = threading.Thread(target=self._work, daemon=True)

    def emit(self, kind, **data):
        if self.on_event:
            try:
                self.on_event(dict(kind=kind, **data))
            except Exception as exc:
                log.error('Event observer failed (%s)', type(exc).__name__)

    def configure(self, **changes):
        with self.lock:
            candidate = self.settings.updated(**changes)
            self.settings = candidate
            self.debounce, self.update_interval = candidate.debounce, candidate.interval
            self.emit('settings', settings=candidate.snapshot())
            return candidate

    def set_paused(self, value):
        with self.lock:
            self.paused = bool(value)
            self.emit('pause', paused=self.paused)

    def start(self):
        self.emit('runtime', model=self.agent.model, ollama_url=self.agent.url,
                  system_prompt=self.agent.system_prompt, settings=self.settings.snapshot(),
                  listen=list(self.address), output=self.output_target, output_address=self.output_address,
                  dynamic=self.dynamic, channel_mode=self.channel_mode, forward_raw=self.forward_raw)
        self.worker.start()
        self.receiver.start()

    def _admit(self, channel):
        if channel not in self.epochs and len(self.epochs) >= self.MAX_CHANNELS:
            self.emit('drop', reason='channel capacity', channel=channel)
            return False
        self.epochs.setdefault(channel, 0)
        return True

    def _enqueue(self, channel, text):
        try:
            self.pending.put_nowait((channel, text, self.epochs[channel], time.perf_counter()))
        except queue.Full:
            log.error('OSC queue overflow: dropped newest phrase')
            self.emit('drop', reason='queue overflow', channel=channel, input=text)

    def submit(self, text, channel='default'):
        """Manual input uses the same bounded worker, without a synthetic OSC send."""
        text = ' '.join(text.split())
        if not text or len(text) > self.MAX_TEXT:
            raise ValueError(f'manual text must contain 1 … {self.MAX_TEXT} characters')
        with self.lock:
            if self._admit(channel):
                self.emit('input', channel=channel, input=text, source='manual')
                self._enqueue(channel, text)

    def receive(self, address, *args):
        log.debug('RX %s %r', address, args)
        channel = 'default'
        if address != '/reset':
            if self.channel_mode:
                match = re.fullmatch(r'/(ch\d+)/(transcription|trigger)', address)
                if not match:
                    return
                channel, kind = match.groups()
                address = '/' + kind
            elif address not in ('/transcription', '/trigger'):
                return
        with self.lock:
            if address == '/reset':
                self.latest.clear()
                self.dirty.clear()
                self.last_started.clear()
                for ch in self.epochs:
                    self.epochs[ch] += 1
                self.agent.reset()
                while True:
                    try:
                        self.pending.get_nowait()
                        self.pending.task_done()
                    except queue.Empty:
                        break
                self.emit('reset')
                return
            if not self._admit(channel):
                return
            if address == '/transcription' and args and isinstance(args[0], str) and len(args[0]) > self.MAX_TEXT:
                self.emit('drop', channel=channel, reason='text exceeds 8192 characters')
                return
            if self.forward_raw and args:
                try:
                    self.client.send_message(address, list(args))
                except Exception as exc:
                    self.emit('error', stage='raw_udp', error=type(exc).__name__)
                    log.error('Raw OSC send failed (%s)', type(exc).__name__)
            if address == '/transcription' and args and isinstance(args[0], str):
                text = ' '.join(args[0].split())
                self.emit('input', channel=channel, input=text, source='osc')
                if self.dynamic:
                    if text and text != self.latest.get(channel):
                        self.dirty[channel] = time.perf_counter()
                        self.latest[channel] = text
                else:
                    self.latest[channel] = text
            elif not self.dynamic and address == '/trigger' and args and args[0] == 1:
                text = self.latest.pop(channel, '')
                if text:
                    self._enqueue(channel, text)

    def _take(self):
        """Selection and settings snapshot share the configure lock."""
        with self.lock:
            if self.paused:
                return None
            queued = False
            try:
                item = self.pending.get_nowait()
                queued = True
            except queue.Empty:
                item = None
                if self.dynamic:
                    now = time.perf_counter()
                    for ch, changed in list(self.dirty.items()):
                        due = now - changed >= self.debounce
                        spaced = now - self.last_started.get(ch, -float('inf')) >= self.update_interval
                        if due and spaced:
                            item = (ch, self.latest[ch], self.epochs[ch], changed)
                            self.dirty.pop(ch)
                            self.last_started[ch] = now
                            break
            if item is None:
                return None
            return item, queued, self.settings

    def _work(self):
        while not self.stop.is_set():
            selected = self._take()
            if selected is None:
                self.stop.wait(.025)
                continue
            (channel, text, epoch, received), queued, settings = selected
            started = time.perf_counter()
            self.request_id += 1
            detail = dict(request_id=self.request_id, channel=channel, input=text,
                          model=self.agent.model, settings=settings.snapshot(),
                          queue_wait=started - received)
            telemetry = {}
            stage = 'ollama'
            prompt = None
            try:
                with self.lock:
                    if self.epochs.get(channel, 0) != epoch:
                        raise StaleGeneration('reset before generation')
                    with self.agent.lock:
                        agent_epoch = self.agent.epochs.setdefault(channel, 0)
                self.emit('start', **detail)
                log.info('GENERATE %s model=%s input=%r', channel, self.agent.model, text)
                prompt = self.agent.generate(text, channel, expected_epoch=agent_epoch,
                                             settings=settings, telemetry=telemetry)
                with self.lock:
                    if self.epochs.get(channel, 0) != epoch or self.stop.is_set():
                        raise StaleGeneration('reset or shutdown before send')
                    stage = 'udp'
                    self.client.send_message(self.output_address, prompt)
                    self.client.send_message(self.trigger_address, 1)
                    self.emit('result', **detail, output=prompt, telemetry=telemetry,
                              elapsed=time.perf_counter() - started, udp_sent=True)
                    log.info('TX %s %s %.2fs prompt=%r; %s', self.output_target, self.output_address,
                             time.perf_counter() - started, prompt,
                             generation_timing(telemetry, detail['queue_wait'],
                                               time.perf_counter() - started, settings.num_predict))
            except StaleGeneration:
                self.emit('discard', **detail, reason='reset or shutdown', elapsed=time.perf_counter() - started)
                log.info('Discarded generation superseded by reset or shutdown')
            except Exception as exc:
                self.emit('error', **detail, stage=stage, error=type(exc).__name__,
                          output=prompt, telemetry=telemetry, elapsed=time.perf_counter() - started,
                          udp_sent=False)
                log.error('Generation failed (%s): no completed prompt/trigger send; %s', type(exc).__name__,
                          generation_timing(telemetry, detail['queue_wait'],
                                            time.perf_counter() - started, settings.num_predict))
            finally:
                if queued:
                    self.pending.task_done()

    def close(self):
        self.stop.set()
        if self.receiver.is_alive():
            self.server.shutdown()
            self.receiver.join(timeout=2)
        self.server.server_close()
        if self.worker.is_alive():
            self.worker.join(timeout=self.agent.timeout + 1)
        self.client.close()
