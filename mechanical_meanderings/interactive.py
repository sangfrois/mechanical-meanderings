"""Optional speech mediator and idle engagement on the existing serial OSC worker."""
from collections import deque
import json
import logging
import math
import re
import time
from . import StaleGeneration
from .bridge import Bridge
from .camera import FrameInbox
from .engagement import EngagementState
from .mediation import brief_invitation, invitation_objects

log = logging.getLogger(__name__)


class InteractiveBridge(Bridge):
    def __init__(self, agent, *args, engagement=None, frame_provider=None, frame_port=9012,
                 frame_timeout=3., reply_cooldown=12., message_ttl=12., invitation_language='auto', **kwargs):
        for name, value in dict(frame_timeout=frame_timeout, reply_cooldown=reply_cooldown, message_ttl=message_ttl).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if message_ttl > 60:
            raise ValueError('message_ttl must not exceed the TD 60-second limit')
        if not isinstance(invitation_language, str) or not 1 <= len(invitation_language) <= 64:
            raise ValueError('invalid invitation language')
        self.engagement = engagement or EngagementState()
        self.frames = FrameInbox(frame_port)
        self.frame_provider = frame_provider
        self.frame_timeout = frame_timeout
        self.reply_cooldown = reply_cooldown
        self.message_ttl = message_ttl
        self.invitation_language = invitation_language
        self.invitation_epoch = 0
        self.last_reply = -float('inf')
        self.recent_replies = deque(maxlen=8)
        self.recent_invitations = deque(maxlen=3)
        self.reply_inputs = {}
        try:
            super().__init__(agent, *args, **kwargs)
        except Exception:
            self.frames.close()
            raise

    def start(self):
        super().start()
        self.emit('interactive', frame_endpoint=self.frames.endpoint, idle_after=self.engagement.idle_after,
            presence_dwell=self.engagement.presence_dwell, invite_cooldown=self.engagement.invite_cooldown,
            scene_timeout=self.engagement.scene_timeout, max_invitations=self.engagement.max_invitations,
            reply_cooldown=self.reply_cooldown, message_ttl=self.message_ttl)

    def receive(self, address, *args):
        with self.lock:
            try:
                if address == '/scene':
                    if len(args) != 1:
                        raise ValueError('scene requires one JSON argument')
                    self.engagement.observe(args[0])
                    return
                if address in ('/speech/activity', '/engagement/enabled'):
                    if len(args) != 1 or args[0] not in (0, 1) or not isinstance(args[0], (int, float)):
                        raise ValueError('activity/enabled requires 0 or 1')
                    if address == '/speech/activity':
                        self.engagement.activity(bool(args[0]))
                    else:
                        self.engagement.set_enabled(bool(args[0]))
                    return
            except ValueError as exc:
                self.emit('drop', reason=str(exc), source=address)
                return
            if address == '/reset':
                self.engagement.reset()
                self.frames.cancel()
                self.invitation_epoch += 1
                self.recent_invitations.clear()
                self.recent_replies.clear()
                self.reply_inputs.clear()
                self.last_reply = -float('inf')
                if not self.stop.is_set():
                    try:
                        self._message('', 'reply')
                    except Exception as exc:
                        self.emit('error', stage='reset_caption', error=type(exc).__name__)
            else:
                channel = 'default'
                accepted = address == '/transcription' and not self.channel_mode
                if self.channel_mode:
                    match = re.fullmatch(r'/(ch\d+)/transcription', address)
                    accepted = match is not None
                    if accepted:
                        channel = match.group(1)
                if accepted and args and isinstance(args[0], str) and len(args[0]) <= self.MAX_TEXT:
                    if channel in self.epochs or len(self.epochs) < self.MAX_CHANNELS:
                        self.engagement.speech(args[0], channel)
                        text = ' '.join(args[0].split())
                        if text:
                            self.reply_inputs[channel] = text
            super().receive(address, *args)

    def submit(self, text, channel='default'):
        with self.lock:
            # Validate/admit through the original path before changing silence state.
            super().submit(text, channel)
            if channel in self.epochs:
                self.engagement.speech(text, channel)
                self.reply_inputs[channel] = ' '.join(text.split())

    def set_paused(self, value):
        with self.lock:
            if self.paused != bool(value):
                self.invitation_epoch += 1
            super().set_paused(value)
            if self.paused:
                self.frames.cancel()

    def _take(self):
        selected = super()._take()
        if selected is not None:
            return selected
        with self.lock:
            # Even debouncing speech outranks a camera/model invitation.
            if self.paused or self.stop.is_set() or self.dirty or not self.pending.empty():
                return None
            token = self.engagement.reserve()
            if token is None:
                return None
            return (('__invitation__', token, self.invitation_epoch, time.perf_counter()), False, self.settings)

    def _valid_invitation(self, token, epoch):
        return not self.stop.is_set() and not self.paused and epoch == self.invitation_epoch and self.engagement.valid(token)

    def _message(self, message, kind):
        # Text, kind and lifetime travel in ONE packet; no separate-field ordering race.
        self.client.send_message('/engagement/message', [message, kind, self.message_ttl])

    def _work(self):
        while not self.stop.is_set():
            selected = self._take()
            if selected is None:
                self.stop.wait(.025)
                continue
            (channel, text, epoch, received), queued, settings = selected
            invitation = channel == '__invitation__'
            token = text if invitation else None
            started = time.perf_counter()
            self.request_id += 1
            detail = dict(request_id=self.request_id, channel=channel,
                input='[quiet microphone, detected presence]' if invitation else text,
                model=self.agent.model, settings=settings.snapshot(), queue_wait=started - received)
            telemetry = {}
            stage = 'yolo' if invitation else 'ollama'
            output = None
            try:
                with self.lock:
                    if invitation:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('invitation superseded')
                    else:
                        if self.epochs.get(channel, 0) != epoch:
                            raise StaleGeneration('reset before mediation')
                        with self.agent.lock:
                            agent_epoch = self.agent.epochs.setdefault(channel, 0)
                self.emit('start', **detail)
                if invitation:
                    with self.lock:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('speech or departure before invitation')
                        scene = self.engagement.snapshot()
                    detail['scene'] = scene
                    stage = 'ollama_yolo'
                    output = self.agent.invite(None, scene, settings=settings, telemetry=telemetry,
                        language=self.invitation_language, previous=tuple(self.recent_invitations))
                    with self.lock:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('speech, departure or reset during invitation')
                        # An object can leave while the model runs: validate against the latest fresh scene.
                        current_scene = self.engagement.snapshot()
                        language = 'French' if output.startswith(('Dis ', 'Parle ')) else 'English'
                        output = brief_invitation(output, language, tuple(self.recent_invitations),
                            invitation_objects(current_scene))
                        detail['scene_at_send'] = current_scene
                        stage = 'udp'
                        self._message(output, 'invitation')
                        self.engagement.sent(token)
                        self.recent_invitations.append(output)
                        self.emit('result', **detail, output=output, telemetry=telemetry,
                            elapsed=time.perf_counter() - started, udp_sent=True, interaction='invitation')
                        log.info('INVITE TX %s %.2fs message=%r', self.output_target, time.perf_counter() - started, output)
                else:
                    log.info('MEDIATE %s input=%r', channel, text)
                    decision = self.agent.interpret(text, channel, expected_epoch=agent_epoch,
                        settings=settings, telemetry=telemetry)
                    output = decision.prompt
                    with self.lock:
                        if self.epochs.get(channel, 0) != epoch or self.stop.is_set():
                            raise StaleGeneration('reset or shutdown before send')
                        stage = 'udp'
                        metadata = dict(decision.metadata(), channel=channel, request_id=self.request_id,
                            judgement='model interpretation, not ground truth')
                        self.client.send_message('/mediation', json.dumps(metadata, ensure_ascii=False, allow_nan=False))
                        if decision.apply_visual:
                            self.client.send_message(self.output_address, decision.prompt)
                            self.client.send_message(self.trigger_address, 1)
                        reply = decision.reply
                        now = time.monotonic()
                        if (reply and self.engagement.enabled and not self.engagement.speaking
                                and self.reply_inputs.get(channel) == text
                                and now - self.last_reply >= self.reply_cooldown
                                and reply.casefold() not in self.recent_replies):
                            self._message(reply, 'reply')
                            self.last_reply = now
                            self.recent_replies.append(reply.casefold())
                        else:
                            reply = ''
                        self.emit('result', **detail, output=output, telemetry=telemetry, mediation=metadata,
                            reply=reply, elapsed=time.perf_counter() - started, udp_sent=True, interaction='speech')
                        log.info('TX %s category=%s preserve=%.2f specificity=%.2f visual=%s %.2fs prompt=%r reply=%r',
                            self.output_target, decision.category, decision.preserve_identity,
                            decision.visual_specificity, decision.apply_visual, time.perf_counter() - started, output, reply)
            except StaleGeneration:
                self.emit('discard', **detail, reason='superseded interaction', elapsed=time.perf_counter() - started)
                log.info('Discarded superseded %s', 'invitation' if invitation else 'mediation')
            except Exception as exc:
                self.emit('error', **detail, stage=stage, error=type(exc).__name__, output=output,
                    telemetry=telemetry, elapsed=time.perf_counter() - started, udp_sent=False)
                log.error('Interaction failed at %s (%s); no fallback appearance claims', stage, type(exc).__name__)
            finally:
                if invitation:
                    self.engagement.failed(token)
                if queued:
                    self.pending.task_done()

    def close(self):
        self.stop.set()
        self.frames.close()
        super().close()
