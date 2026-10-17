"""Opt-in completed-speech scheduling; raw OSC is a separate immediate lane."""
from collections import OrderedDict, deque
import logging
import re
import threading
import time

from . import StaleGeneration, GenerationControl
from .bridge import Bridge
from .timing import generation_timing

log = logging.getLogger(__name__)


class FinalTurnBridge(Bridge):
    """One running request and at most one newest completed request per channel.

    /final TEXT is atomic. The legacy-compatible two-packet protocol requires
    /transcription TEXT before /trigger 1; an early trigger is never latched.
    New partial speech invalidates that channel's pending and running visuals.
    Completed speech survives coalescing in a separate bounded context ledger.
    """

    def __init__(self, agent, *args, speech_history=128, generation_budget=None, **kwargs):
        if speech_history < 1:
            raise ValueError('speech-history must be positive')
        if generation_budget is None:
            generation_budget = agent.timeout
        GenerationControl(generation_budget)  # Validate before allocating sockets.
        kwargs['dynamic'] = False
        super().__init__(agent, *args, **kwargs)
        self.speech_history = speech_history
        self.completed = {}
        self.completed_at = {}
        self.revisions = {}
        self.ready = OrderedDict()
        self.accepted_epochs = {}
        self.shutdown_thread = None
        # Applies only to realtime final/boundary requests, not interactive Agent.
        self.generation_budget = generation_budget
        self.active_generations = {}

    def start(self):
        self.emit('final_turn_protocol', speech_history=self.speech_history,
                  protocol='/final TEXT or /transcription TEXT then /trigger 1')
        super().start()

    def _forward(self, address, args):
        if self.forward_raw and args:
            try:
                self.client.send_message(address, list(args))
            except Exception as exc:
                self.emit('error', stage='raw_udp', error=type(exc).__name__)

    def _supersede(self, channel):
        control = self.active_generations.get(channel)
        if control is not None:
            control.cancel('superseded')
        self.revisions[channel] = self.revisions.get(channel, 0) + 1
        self.ready.pop(channel, None)

    def _complete(self, channel, text):
        # Called with only the bridge lock; never wait for the model/agent lock
        # on the receive lane, even when the previous HTTP request is blocked.
        now = self.agent.clock()
        last = self.completed_at.get(channel)
        if (self.agent.reset_after is not None and last is not None
                and now - last >= self.agent.reset_after):
            self.completed.pop(channel, None)
            self.epochs[channel] += 1
        self.completed_at[channel] = now
        ledger = self.completed.setdefault(channel, deque(maxlen=self.speech_history))
        ledger.append(text)
        self._supersede(channel)
        self.ready[channel] = (text, self.epochs[channel], self.revisions[channel],
                               time.perf_counter())
        self.latest.pop(channel, None)
        self.emit('complete', channel=channel, input=text, retained=len(ledger))

    def submit(self, text, channel='default'):
        text = ' '.join(text.split())
        if not text or len(text) > self.MAX_TEXT:
            raise ValueError(f'manual text must contain 1 … {self.MAX_TEXT} characters')
        with self.lock:
            if not self.stop.is_set() and self._admit(channel):
                self.emit('input', channel=channel, input=text, source='manual')
                self._complete(channel, text)

    def receive(self, address, *args):
        if self.stop.is_set():
            return
        if address == '/reset':
            with self.lock:
                self.latest.clear()
                self.ready.clear()
                self.completed.clear()
                self.completed_at.clear()
                for control in self.active_generations.values():
                    control.cancel('reset')
                for channel in self.epochs:
                    self.epochs[channel] += 1
                    self._supersede(channel)
                self.emit('reset')
            # No bridge lock while waiting for an agent lock.
            self.agent.reset()
            return
        channel = 'default'
        if self.channel_mode:
            match = re.fullmatch(r'/(ch\d+)/(transcription|trigger|final)', address)
            if not match:
                return
            channel, kind = match.groups()
            address = '/' + kind
        elif address not in ('/transcription', '/trigger', '/final'):
            return
        if address in ('/transcription', '/final'):
            if not args or not isinstance(args[0], str):
                return
            if len(args[0]) > self.MAX_TEXT:
                self.emit('drop', channel=channel, reason='text exceeds 8192 characters')
                return
        with self.lock:
            if not self._admit(channel):
                return
        # Raw transcript delivery precedes scheduling and never waits for
        # inference. Atomic finals are forwarded as transcripts, not /final.
        self._forward('/transcription' if address == '/final' else address, args)
        with self.lock:
            if self.stop.is_set():
                return
            if address in ('/transcription', '/final'):
                text = ' '.join(args[0].split())
                self.emit('input', channel=channel, input=text, source='osc')
                if address == '/final':
                    if text:
                        self._complete(channel, text)
                elif text != self.latest.get(channel):
                    self._supersede(channel)
                    self.latest[channel] = text
            elif args and args[0] == 1:
                text = self.latest.pop(channel, '')
                if text:
                    self._complete(channel, text)
                else:
                    self.emit('drop', channel=channel, reason='trigger without preceding text')

    def _take_final(self):
        with self.lock:
            if self.paused or not self.ready or self.stop.is_set():
                return None
            channel, (text, epoch, revision, received) = self.ready.popitem(last=False)
            context = [{'role': 'user', 'content': value}
                       for value in list(self.completed[channel])[:-1]]
            return channel, text, epoch, revision, received, context, self.settings

    def _fresh(self, channel, epoch, revision):
        return (not self.stop.is_set() and self.epochs.get(channel) == epoch
                and self.revisions.get(channel) == revision)

    def _work(self):
        while not self.stop.is_set():
            item = self._take_final()
            if item is None:
                self.stop.wait(.025)
                continue
            channel, text, epoch, revision, received, context, settings = item
            started = time.perf_counter()
            self.request_id += 1
            detail = dict(request_id=self.request_id, channel=channel, input=text,
                          model=self.agent.model, settings=settings.snapshot(),
                          queue_wait=started - received)
            telemetry = {}
            stage, prompt = 'ollama', None
            control = GenerationControl(self.generation_budget)
            try:
                # Lock order is agent -> bridge, never bridge -> agent. The
                # receiver takes no agent lock for transcript/final admission.
                with self.agent.lock:
                    with self.lock:
                        if not self._fresh(channel, epoch, revision):
                            raise StaleGeneration('superseded before generation')
                        if self.accepted_epochs.get(channel) != epoch:
                            self.agent.history.pop(channel, None)
                            self.accepted_epochs[channel] = epoch
                        agent_epoch = self.agent.epochs.setdefault(channel, 0)
                        self.active_generations[channel] = control
                self.emit('start', **detail)
                log.info('GENERATE final %s model=%s input=%r', channel, self.agent.model, text)
                prompt = self.agent.generate(text, channel, expected_epoch=agent_epoch,
                    settings=settings, telemetry=telemetry, history_override=context, commit=False,
                    control=control)
                with self.agent.lock:
                    with self.lock:
                        if (not self._fresh(channel, epoch, revision)
                                or self.agent.epochs.get(channel) != agent_epoch):
                            raise StaleGeneration('superseded before send')
                        stage = 'udp'
                        self.client.send_message(self.output_address, prompt)
                        self.client.send_message(self.trigger_address, 1)
                        # Only outputs accepted for delivery enter generated
                        # history. Final-mode payloads use the speech ledger,
                        # never visual paraphrases in place of missing speech.
                        history = self.agent.history.get(channel, [])
                        self.agent.history[channel] = ((history + [
                            {'role': 'user', 'content': text},
                            {'role': 'assistant', 'content': prompt}])[-2 * settings.history:]
                            if settings.history else [])
                        self.emit('result', **detail, output=prompt, telemetry=telemetry,
                                  elapsed=time.perf_counter() - started, udp_sent=True)
                        log.info('TX %s %s prompt=%r; %s', self.output_target, self.output_address, prompt,
                                 generation_timing(telemetry, detail['queue_wait'],
                                                   time.perf_counter() - started, settings.num_predict))
            except StaleGeneration:
                elapsed = time.perf_counter() - started
                self.emit('discard', **detail, reason='superseded, reset or shutdown',
                          telemetry=telemetry, elapsed=elapsed)
                log.info('DISCARD final %s; %s', channel,
                         generation_timing(telemetry, detail['queue_wait'], elapsed, settings.num_predict))
            except Exception as exc:
                self.emit('error', **detail, stage=stage, error=type(exc).__name__,
                          output=prompt, telemetry=telemetry, elapsed=time.perf_counter() - started,
                          udp_sent=False)
                log.error('Final generation failed (%s); %s', type(exc).__name__,
                          generation_timing(telemetry, detail['queue_wait'],
                                            time.perf_counter() - started, settings.num_predict))
            finally:
                with self.lock:
                    if self.active_generations.get(channel) is control:
                        self.active_generations.pop(channel, None)

    def close(self):
        # Real HTTP closes immediately. Legacy injected blocking transports
        # cannot be killed; retain bounded close and prohibit late commits.
        self.stop.set()
        with self.lock:
            for control in self.active_generations.values():
                control.cancel('shutdown')
        if self.receiver.is_alive():
            if self.shutdown_thread is None:
                self.shutdown_thread = threading.Thread(target=self.server.shutdown, daemon=True)
                self.shutdown_thread.start()
            self.shutdown_thread.join(timeout=.75)
            self.receiver.join(timeout=.1)
        self.server.server_close()
        if self.worker.is_alive():
            self.worker.join(timeout=.25)
        self.client.close()
