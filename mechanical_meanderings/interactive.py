"""Optional speech mediator and idle engagement on the existing serial OSC worker."""
from collections import deque
import json
import logging
import math
import re
import time
from . import StaleGeneration, GenerationControl
from .bridge import Bridge
from .boundary import BoundaryBridge
from .final_turn import FinalTurnBridge
from .camera import FrameInbox
from .engagement import EngagementState
from .mediation import brief_invitation, invitation_objects
from .invitation import InvitationPromptError, InvitationRejected
from .typography import parse_typography
from .timing import generation_timing

log = logging.getLogger(__name__)


class InteractiveBridge(BoundaryBridge):
    def __init__(self, agent, *args, engagement=None, frame_provider=None, frame_port=9012,
                 frame_timeout=3., reply_cooldown=6., message_ttl=12., invitation_language='auto',
                 camera_enabled=True, camera_cooldown=20., asides_enabled=True,
                 invitations_enabled=True, boundary_speech=None, speech_history=128,
                 generation_budget=None, **kwargs):
        requested_dynamic = kwargs.get('dynamic', False)
        if boundary_speech is not None and not isinstance(boundary_speech, bool):
            raise ValueError('boundary_speech must be boolean or None')
        # Explicit Python dynamic=True preserves the legacy embedding contract.
        # CLI selects boundary speech unless the operator explicitly opts out.
        self.boundary_speech = not requested_dynamic if boundary_speech is None else boundary_speech
        for name, value in dict(frame_timeout=frame_timeout, reply_cooldown=reply_cooldown,
                               message_ttl=message_ttl, camera_cooldown=camera_cooldown).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if message_ttl > 60:
            raise ValueError('message_ttl must not exceed the TD 60-second limit')
        if not isinstance(invitation_language, str) or not 1 <= len(invitation_language) <= 64:
            raise ValueError('invalid invitation language')
        if not isinstance(camera_enabled, bool):
            raise ValueError('camera_enabled must be boolean')
        if not isinstance(asides_enabled, bool):
            raise ValueError('asides_enabled must be boolean')
        if not isinstance(invitations_enabled, bool):
            raise ValueError('invitations_enabled must be boolean')
        self.invitations_enabled = invitations_enabled
        self.engagement = engagement or EngagementState()
        self.frames = FrameInbox(frame_port)
        self.frame_provider = frame_provider
        self.frame_timeout = frame_timeout
        self.camera_enabled = camera_enabled
        self.camera_cooldown = camera_cooldown
        self.last_frame_attempt = -float('inf')
        self.camera_epoch = 0
        self.asides_enabled = asides_enabled
        self.reply_cooldown = reply_cooldown
        self.message_ttl = message_ttl
        self.invitation_language = invitation_language
        self.invitation_epoch = 0
        self.last_reply = -float('inf')
        self.recent_replies = deque(maxlen=8)
        self.recent_invitations = deque(maxlen=12)
        self.recent_invitation_approaches = deque(maxlen=12)
        self.reply_inputs = {}
        try:
            super().__init__(agent, *args, speech_history=speech_history,
                             generation_budget=generation_budget, **kwargs)
            # Every speech channel feeds this output, not a channel-local renderer.
            self.render_key = (self.output_target, self.output_address, self.trigger_address)
            if self.boundary_speech:
                self.forward_raw = True
            else:
                self.dynamic = requested_dynamic
        except Exception:
            self.frames.close()
            raise

    def start(self):
        if self.boundary_speech:
            BoundaryBridge.start(self)
        else:
            Bridge.start(self)
        self.emit('interactive', frame_endpoint=self.frames.endpoint, idle_after=self.engagement.idle_after,
            presence_dwell=self.engagement.presence_dwell, invite_cooldown=self.engagement.invite_cooldown,
            scene_timeout=self.engagement.scene_timeout, max_invitations=self.engagement.max_invitations,
            reply_cooldown=self.reply_cooldown, message_ttl=self.message_ttl,
            camera_enabled=self.camera_enabled, camera_cooldown=self.camera_cooldown,
            frame_timeout=self.frame_timeout, boundary_speech=self.boundary_speech,
            generation_budget=self.generation_budget if self.boundary_speech else None)

    def _cancel_invitation(self):
        control = self.active_generations.get('__invitation__')
        if control is not None:
            control.cancel('superseded')

    def _complete(self, channel, text):
        super()._complete(channel, text)
        self.reply_inputs[channel] = text
        self._cancel_invitation()

    def _speech_address(self, address):
        if self.channel_mode:
            match = re.fullmatch(r'/(ch\d+)/(transcription|trigger|final|speech/activity)', address)
            return match.groups() if match else None
        if address in ('/transcription', '/trigger', '/final', '/speech/activity'):
            return 'default', address[1:]
        return None

    def receive(self, address, *args):
        if self.stop.is_set():
            return
        speech = self._speech_address(address)
        with self.lock:
            try:
                if address == '/creative/profile':
                    if len(args) != 1 or not isinstance(args[0], str) or len(args[0]) > 1024:
                        raise ValueError('invalid creative profile')
                    profile = json.loads(args[0])
                    if not isinstance(profile, dict) or set(profile) != {'disco', 'asides', 'camera'}:
                        raise ValueError('invalid creative profile fields')
                    if not all(isinstance(v, bool) for v in profile.values()):
                        raise ValueError('creative profile flags must be boolean')
                    changed = self.agent.set_disco(profile['disco'])
                    if self.asides_enabled != profile['asides']:
                        self.invitation_epoch += 1
                        self._cancel_invitation()
                    self.asides_enabled = profile['asides']
                    if self.camera_enabled != profile['camera']:
                        self._cancel_camera()
                    self.camera_enabled = profile['camera']
                    if changed:
                        self.invitation_epoch += 1
                        self._cancel_invitation()
                        self._cancel_camera()
                        if self.boundary_speech:
                            for control in self.active_generations.values():
                                control.cancel('superseded')
                        else:
                            for channel, latest in self.latest.items():
                                if latest:
                                    self.dirty[channel] = time.perf_counter()
                        self.emit('profile', **profile)
                    self.client.send_message('/creative/status', json.dumps(profile))
                    return
                if address == '/scene':
                    if len(args) != 1:
                        raise ValueError('scene requires one JSON argument')
                    present = self.engagement.people
                    changed = self.engagement.observe(args[0])
                    if changed and present and not self.engagement.people:
                        self._cancel_camera()
                        self._cancel_invitation()
                    return
                if address == '/engagement/enabled':
                    if len(args) != 1 or args[0] not in (0, 1) or not isinstance(args[0], (int, float)):
                        raise ValueError('activity/enabled requires 0 or 1')
                    if not bool(args[0]):
                        self._cancel_camera()
                        self._cancel_invitation()
                    self.engagement.set_enabled(bool(args[0]))
                    return
                if speech and speech[1] == 'speech/activity':
                    if len(args) != 1 or args[0] not in (0, 1) or not isinstance(args[0], (int, float)):
                        raise ValueError('activity/enabled requires 0 or 1')
                    if bool(args[0]):
                        self._cancel_camera()
                        self._cancel_invitation()
                    other_active = any(value for channel, value in self.speech_active.items()
                                       if channel != speech[0])
                    self.engagement.activity(bool(args[0]) or other_active)
                    if not self.boundary_speech:
                        return
            except (ValueError, TypeError) as exc:
                self.emit('drop', reason=str(exc), source=address)
                return
            if address == '/reset':
                self.engagement.reset()
                self._cancel_camera()
                self._cancel_invitation()
                self.invitation_epoch += 1
                self.recent_invitations.clear()
                self.recent_invitation_approaches.clear()
                self.recent_replies.clear()
                self.reply_inputs.clear()
                self.last_reply = -float('inf')
                if not self.stop.is_set():
                    try:
                        self._message('', 'reply')
                    except Exception as exc:
                        self.emit('error', stage='reset_caption', error=type(exc).__name__)
            elif speech and args:
                channel, kind = speech
                if (kind in ('transcription', 'final') and isinstance(args[0], str)
                        and len(args[0]) <= self.MAX_TEXT):
                    if channel in self.epochs or len(self.epochs) < self.MAX_CHANNELS:
                        self.engagement.speech(args[0], channel)
                        text = ' '.join(args[0].split())
                        if text:
                            self._cancel_invitation()
                            self._speech_preference(text)
                            if text != self.reply_inputs.get(channel):
                                self._cancel_camera()
                            self.reply_inputs[channel] = text
                elif kind == 'trigger' and args[0] == 0:
                    self._cancel_camera()
                    self._cancel_invitation()
        # No bridge lock while resetting or waiting for an agent lock.
        if self.boundary_speech:
            BoundaryBridge.receive(self, address, *args)
        else:
            Bridge.receive(self, address, *args)

    def submit(self, text, channel='default'):
        with self.lock:
            # Validate/admit through the original path before changing silence state.
            if self.boundary_speech:
                FinalTurnBridge.submit(self, text, channel)
            else:
                Bridge.submit(self, text, channel)
            if channel in self.epochs:
                self.engagement.speech(text, channel)
                text = ' '.join(text.split())
                self._speech_preference(text)
                if text != self.reply_inputs.get(channel):
                    self._cancel_camera()
                self.reply_inputs[channel] = text

    def _speech_preference(self, text):
        # The receive path applies a mute immediately, even during model inference.
        directive = self.agent.set_speech_preference(text)
        if directive is not None:
            self.invitation_epoch += 1
            self._cancel_camera()
            if directive and not self.stop.is_set():
                self._message('', 'reply')
            self.emit('speech_preference', muted=directive)

    def _can_invite(self):
        return self.invitations_enabled and not self.agent.speech_muted and self.agent.reply_mode != 'off'

    def set_paused(self, value):
        with self.lock:
            if self.paused != bool(value):
                self.invitation_epoch += 1
            super().set_paused(value)
            if self.paused:
                self._cancel_camera()

    def _cancel_camera(self):
        # Called with the bridge lock. Do not reset the cooldown with a reset/pause.
        self.camera_epoch += 1
        self.frames.cancel()

    def _camera_current(self, channel, text, epoch, camera_epoch):
        with self.lock:
            return (not self.stop.is_set() and not self.paused and self.engagement.enabled
                and self.camera_enabled and self.camera_epoch == camera_epoch
                and self.epochs.get(channel, 0) == epoch and self.reply_inputs.get(channel) == text)

    def _speech_frame(self, channel, text, epoch, camera_epoch, telemetry):
        def current():
            return self._camera_current(channel, text, epoch, camera_epoch)

        with self.lock:
            if not current():
                telemetry['frame_status'] = 'superseded'
                return None
            now = time.monotonic()
            if now - self.last_frame_attempt < self.camera_cooldown:
                telemetry['frame_status'] = 'cooldown'
                return None
            self.last_frame_attempt = now  # Failed/time-out attempts also consume the budget.
        def send(request_id, endpoint):
            with self.lock:
                if not current():
                    raise TimeoutError('camera request superseded')
                self.client.send_message('/camera/request', [request_id, endpoint])

        if self.frame_provider is not None:
            return self.frame_provider()  # Trusted injectable source; production uses the bounded inbox.
        return self.frames.request(send, timeout=self.frame_timeout)

    def _take(self):
        if self.boundary_speech:
            item = self._take_final()
            if item is not None:
                channel, text, epoch, revision, received, context, settings = item
                return ((channel, text, epoch, received, revision, context), False, settings)
        else:
            selected = Bridge._take(self)
            if selected is not None:
                return selected
        with self.lock:
            # Open/ambiguous speech outranks any synthetic engagement request.
            unfinished = (any(self.speech_active.values()) or bool(self.latest)
                          or any(self.trailing.values())) if self.boundary_speech else bool(self.dirty)
            if (self.paused or self.stop.is_set() or unfinished or self.ready
                    or not self.pending.empty() or not self._can_invite()):
                return None
            token = self.engagement.reserve()
            if token is None:
                return None
            return (('__invitation__', token, self.invitation_epoch, time.perf_counter()), False, self.settings)

    def _valid_invitation(self, token, epoch):
        return (not self.stop.is_set() and not self.paused and self._can_invite()
            and epoch == self.invitation_epoch and self.engagement.valid(token))

    def _message(self, message, kind, typography=None):
        # The optional fourth field preserves the original text/kind/lifetime wire prefix.
        style = json.dumps(parse_typography(typography), ensure_ascii=False, allow_nan=False)
        ttl = min(self.message_ttl, 8.) if kind == 'aside' else self.message_ttl
        self.client.send_message('/engagement/message', [message, kind, ttl, style])

    def _work(self):
        while not self.stop.is_set():
            selected = self._take()
            if selected is None:
                self.stop.wait(.025)
                continue
            item, queued, settings = selected
            channel, text, epoch, received = item[:4]
            revision, context = item[4:] if len(item) == 6 else (None, None)
            invitation = channel == '__invitation__'
            token = text if invitation else None
            started = time.perf_counter()
            self.request_id += 1
            detail = dict(request_id=self.request_id, channel=channel,
                input='[quiet microphone, detected presence]' if invitation else text,
                model=self.agent.model, settings=settings.snapshot(), queue_wait=started - received,
                source='engagement' if invitation else 'completed_speech' if self.boundary_speech else 'legacy_text')
            telemetry = {}
            udp_sent = False
            stage = 'yolo' if invitation else 'ollama'
            output = None
            control = GenerationControl(self.generation_budget) if self.boundary_speech else None
            try:
                # Never wait on agent.lock with bridge.lock held: the receiver
                # must keep forwarding speech even while this worker waits.
                with self.agent.lock, self.lock:
                    if invitation:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('invitation superseded')
                    else:
                        if (self.epochs.get(channel, 0) != epoch or
                                self.boundary_speech and not self._fresh(channel, epoch, revision)):
                            raise StaleGeneration('reset before mediation')
                        camera_epoch = self.camera_epoch
                        if self.boundary_speech and self.accepted_epochs.get(channel) != epoch:
                            self.agent.history.pop(channel, None)
                            self.accepted_epochs[channel] = epoch
                        agent_epoch = self.agent.epochs.setdefault(channel, 0)
                    if control is not None:
                        self.active_generations[channel] = control
                self.emit('start', **detail)
                if invitation:
                    with self.lock:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('speech or departure before invitation')
                        scene = self.engagement.snapshot()
                        scene['recent_speech'] = list(self.engagement.last_text.values())[-4:]
                        scene['recent_invitation_approaches'] = list(self.recent_invitation_approaches)
                        idle_kind = ('embodied' if self.engagement.invitation_count > 0 or
                            self.engagement.last_text else 'nudge') if self.asides_enabled else 'invitation'
                    detail['scene'] = scene
                    stage = 'ollama_yolo'
                    typography = {}
                    output = self.agent.invite(None, scene, settings=settings, telemetry=telemetry,
                        language=self.invitation_language, previous=tuple(self.recent_invitations), design=typography,
                        kind=idle_kind, **({'control': control} if control is not None else {}))
                    if control is not None:
                        control.check()
                    # File IO must not hold the receive lock or delay raw transcription.
                    prompt_info = telemetry.get('invitation_prompt')
                    if (prompt_info is not None and self.agent.invitation_prompt.read().sha256
                            != prompt_info['sha256']):
                        raise StaleGeneration('invitation prompt changed before send')
                    with self.lock:
                        if not self._valid_invitation(token, epoch):
                            raise StaleGeneration('speech, departure or reset during invitation')
                        # An object can leave while the model runs: validate against the latest fresh scene.
                        current_scene = self.engagement.snapshot()
                        language = 'French' if output.startswith(('Dis ', 'Parle ')) else 'English'
                        if idle_kind == 'invitation':
                            output = brief_invitation(output, language, tuple(self.recent_invitations),
                                invitation_objects(current_scene))
                        detail['scene_at_send'] = current_scene
                        stage = 'udp'
                        wire_kind = 'challenge' if idle_kind in ('nudge', 'embodied') else idle_kind
                        if control is not None:
                            control.check()
                        self._message(output, wire_kind, typography)
                        self.engagement.sent(token)
                        self.recent_invitations.append(output)
                        if telemetry.get('invitation_approach'):
                            self.recent_invitation_approaches.append(telemetry['invitation_approach'])
                        self.emit('result', **detail, output=output, telemetry=telemetry,
                            typography=typography, elapsed=time.perf_counter() - started, udp_sent=True, interaction=idle_kind)
                        log.info('INVITE TX %s %.2fs message=%r; %s', self.output_target,
                            time.perf_counter() - started, output,
                            generation_timing(telemetry, detail['queue_wait'], time.perf_counter() - started,
                                              telemetry.get('num_predict_max')))
                        if prompt_info is not None:
                            log.info('INVITE PROMPT %s sha256=%s', prompt_info['source'], prompt_info['sha256'])
                else:
                    log.info('MEDIATE %s input=%r', channel, text)
                    def current():
                        return self._camera_current(channel, text, epoch, camera_epoch)
                    def camera():
                        return self._speech_frame(channel, text, epoch, camera_epoch, telemetry)
                    decision = self.agent.interpret(text, channel, expected_epoch=agent_epoch,
                        settings=settings, telemetry=telemetry, render_key=self.render_key,
                        camera=camera if self.camera_enabled else None, camera_current=current,
                        **(dict(history_override=context, commit=False, control=control)
                           if self.boundary_speech else {}))
                    output = decision.prompt
                    with self.agent.lock, self.lock:
                        if control is not None:
                            control.check()
                        if self.epochs.get(channel, 0) != epoch or self.stop.is_set():
                            raise StaleGeneration('reset or shutdown before send')
                        if self.boundary_speech and (not self._fresh(channel, epoch, revision)
                                or self.agent.epochs.get(channel, 0) != agent_epoch):
                            raise StaleGeneration('new speech before send')
                        if telemetry.get('style_version') != self.agent.style_version:
                            raise StaleGeneration('creative profile changed before send')
                        if telemetry.get('frame_status') == 'attached' and not current():
                            raise StaleGeneration('camera turn superseded before send')
                        stage = 'udp'
                        metadata = dict(decision.metadata(), channel=channel, request_id=self.request_id,
                            judgement='model interpretation, not ground truth',
                            creative_profile=dict(disco=telemetry.get('disco', False)),
                            vision={key: telemetry.get(key) for key in
                                ('camera_requested', 'image_attached', 'frame_status', 'interpretation_passes')})
                        if telemetry.get('camera_requested'):
                            log.info('CAMERA %s image_attached=%s passes=%s', telemetry.get('frame_status'),
                                telemetry.get('image_attached'), telemetry.get('interpretation_passes'))
                        delivery = telemetry['delivery'] = dict(mediation=False, prompt=False, trigger=False, caption=False)
                        self.client.send_message('/mediation', json.dumps(metadata, ensure_ascii=False, allow_nan=False))
                        delivery['mediation'] = True
                        if decision.apply_visual:
                            self.client.send_message(self.output_address, decision.prompt)
                            delivery['prompt'] = True
                            self.client.send_message(self.trigger_address, 1)
                            delivery['trigger'] = True
                            udp_sent = True
                            # Both datagrams succeeded under the freshness locks. Caption
                            # delivery is independent and cannot undo this accepted visual.
                            self.agent.accept_visual(decision, channel, render_key=self.render_key)
                        pair = None
                        if self.boundary_speech:
                            # Generated captions/prompts are not speech context. Commit the
                            # delivered decision now; add a caption only after its send succeeds.
                            history = self.agent.history.get(channel, [])
                            pair = [dict(role='user', content=text), dict(role='assistant',
                                content=json.dumps(dict(prompt=decision.prompt, reply=''),
                                                   ensure_ascii=False, allow_nan=False))]
                            self.agent.history[channel] = ((history + pair)[-2 * settings.history:]
                                                           if settings.history else [])
                        reply = decision.reply
                        now = time.monotonic()
                        if (reply and not self.agent.speech_muted and self.engagement.enabled and not self.engagement.speaking
                                and self.reply_inputs.get(channel) == text
                                and now - self.last_reply >= self.reply_cooldown
                                and reply.casefold() not in self.recent_replies):
                            self._message(reply, 'reply', decision.typography)
                            delivery['caption'] = True
                            if pair is not None:
                                pair[1]['content'] = json.dumps(dict(prompt=decision.prompt, reply=reply),
                                                               ensure_ascii=False, allow_nan=False)
                            self.last_reply = now
                            self.recent_replies.append(reply.casefold())
                        else:
                            reply = ''
                        self.emit('result', **detail, output=output, telemetry=telemetry, mediation=metadata,
                            reply=reply, elapsed=time.perf_counter() - started, udp_sent=True, interaction='speech')
                        log.info('TX %s category=%s preserve=%.2f specificity=%.2f visual=%s %.2fs prompt=%r reply=%r; %s',
                            self.output_target, decision.category, decision.preserve_identity,
                            decision.visual_specificity, decision.apply_visual, time.perf_counter() - started, output, reply,
                            generation_timing(telemetry, detail['queue_wait'], time.perf_counter() - started,
                                              telemetry.get('num_predict_max')))
            except StaleGeneration:
                self.emit('discard', **detail, reason='superseded interaction', telemetry=telemetry,
                    elapsed=time.perf_counter() - started)
                log.info('Discarded superseded %s; %s', 'invitation' if invitation else 'mediation',
                    generation_timing(telemetry, detail['queue_wait'], time.perf_counter() - started,
                                      telemetry.get('num_predict_max')))
            except Exception as exc:
                rejection = exc.metadata() if isinstance(exc, InvitationRejected) else None
                self.emit('error', **detail, stage=stage, error=type(exc).__name__, output=output,
                    telemetry=telemetry, elapsed=time.perf_counter() - started, udp_sent=udp_sent,
                    **({'rejection': rejection} if rejection is not None else {}))
                if rejection is not None:
                    log.warning('INVITE REJECT kind=%s reason=%s words=%s chars=%s message=%r truncated=%s; %s; %s',
                        rejection['kind'], rejection['code'], rejection['words'], rejection['characters'],
                        rejection['candidate'], rejection['candidate_truncated'], rejection['reason'],
                        generation_timing(telemetry, detail['queue_wait'], time.perf_counter() - started,
                                          telemetry.get('num_predict_max')))
                else:
                    log.error('Interaction failed at %s (%s); no fallback appearance claims; %s', stage, type(exc).__name__,
                        generation_timing(telemetry, detail['queue_wait'], time.perf_counter() - started,
                                          telemetry.get('num_predict_max')))
                if isinstance(exc, InvitationPromptError):
                    log.error('%s; invitation skipped; correct the file for the next eligible attempt', exc)
            finally:
                if control is not None:
                    with self.lock:
                        if self.active_generations.get(channel) is control:
                            self.active_generations.pop(channel, None)
                if invitation:
                    self.engagement.failed(token)
                if queued:
                    self.pending.task_done()

    def close(self):
        self.stop.set()
        self.frames.close()
        if self.boundary_speech:
            FinalTurnBridge.close(self)
        else:
            Bridge.close(self)
