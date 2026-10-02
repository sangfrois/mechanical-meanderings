"""Bounded, presence-gated invitations; camera images never enter scene state."""
import json
import math
import threading
import time


def parse_scene(raw):
    if not isinstance(raw, str) or len(raw) > 32768:
        raise ValueError('invalid scene payload')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('scene must be an object')
    for key in ('people', 'sequence'):
        n = value.get(key)
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f'scene {key} must be a nonnegative integer')
    if value['people'] > 64:
        raise ValueError('scene person count exceeds 64')
    objects = value.get('objects', [])
    if not isinstance(objects, list) or len(objects) > 64:
        raise ValueError('scene objects must be a bounded list')
    clean = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError('invalid scene object')
        label, score = obj.get('label'), obj.get('confidence')
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            raise ValueError('invalid scene object label')
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid scene object confidence')
        # Low-confidence objects must not become assertions in the model context.
        if score >= .4:
            clean.append(dict(label=label, confidence=round(score, 3)))
    return dict(people=value['people'], sequence=value['sequence'], objects=clean)


class EngagementState:
    def __init__(self, clock=time.monotonic, idle_after=20., presence_dwell=8.,
                 invite_cooldown=90., scene_timeout=3., absence_reset=5.,
                 max_invitations=2, retry_after=20.):
        for name, value in dict(idle_after=idle_after, presence_dwell=presence_dwell,
                invite_cooldown=invite_cooldown, scene_timeout=scene_timeout,
                absence_reset=absence_reset, retry_after=retry_after).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if isinstance(max_invitations, bool) or not isinstance(max_invitations, int) or max_invitations < 1:
            raise ValueError('max_invitations must be a positive integer')
        self.clock = clock
        self.idle_after = idle_after
        self.presence_dwell = presence_dwell
        self.invite_cooldown = invite_cooldown
        self.scene_timeout = scene_timeout
        self.absence_reset = absence_reset
        self.max_invitations = max_invitations
        self.retry_after = retry_after
        self.lock = threading.RLock()
        self.enabled = True
        self.generation = 0
        self.serial = 0
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.last_speech = self.clock()
            self.speech_version = 0
            self.last_text = {}
            self.people = 0
            self.presence_since = None
            self.absent_since = None
            self.absence_cleared = False
            self.last_scene = None
            self.scene = None
            self.last_invitation = -float('inf')
            self.last_attempt = -float('inf')
            self.invitation_count = 0
            self.inflight = None
            self.speaking = False

    def set_enabled(self, enabled):
        with self.lock:
            if self.enabled != bool(enabled):
                self.generation += 1
                self.inflight = None
            self.enabled = bool(enabled)

    def activity(self, speaking):
        with self.lock:
            if speaking or self.speaking:
                self.last_speech = self.clock()
                self.speech_version += 1
            self.speaking = bool(speaking)

    def observe(self, raw):
        value = parse_scene(raw)
        with self.lock:
            now = self.clock()
            if self.scene and value['sequence'] == self.scene['sequence']:
                return False
            stale = self.last_scene is not None and now - self.last_scene > self.scene_timeout
            if stale:
                self.presence_since = None
            if self.absent_since is not None and now - self.absent_since >= self.absence_reset and not self.absence_cleared:
                self.generation += 1
                self.invitation_count = 0
                self.absence_cleared = True
            if value['people']:
                if not self.people or self.presence_since is None:
                    self.presence_since = now
                self.absent_since = None
                self.absence_cleared = False
            else:
                self.presence_since = None
                if self.absent_since is None:
                    self.absent_since = now
            self.people = value['people']
            self.scene = value
            self.last_scene = now
            return True

    def speech(self, text, channel='default'):
        text = ' '.join(text.split())
        with self.lock:
            if text and text != self.last_text.get(channel):
                self.last_text[channel] = text
                self.last_speech = self.clock()
                self.speech_version += 1

    def _present(self, now):
        return bool(self.people and self.last_scene is not None and now - self.last_scene <= self.scene_timeout)

    def reserve(self):
        with self.lock:
            now = self.clock()
            if not self.enabled or self.speaking or self.inflight is not None or not self._present(now):
                return None
            if self.presence_since is None or now - self.presence_since < self.presence_dwell:
                return None
            if now - self.last_speech < self.idle_after or now - self.last_invitation < self.invite_cooldown:
                return None
            if now - self.last_attempt < self.retry_after or self.invitation_count >= self.max_invitations:
                return None
            self.serial += 1
            self.last_attempt = now
            self.inflight = (self.generation, self.speech_version, self.serial)
            return self.inflight

    def valid(self, token):
        with self.lock:
            return bool(self.enabled and not self.speaking and self._present(self.clock())
                and token == self.inflight and token[:2] == (self.generation, self.speech_version))

    def failed(self, token):
        with self.lock:
            if token == self.inflight:
                self.inflight = None

    def sent(self, token):
        with self.lock:
            if not self.valid(token):
                return False
            self.invitation_count += 1
            self.last_invitation = self.clock()
            self.inflight = None
            return True

    def snapshot(self):
        with self.lock:
            return None if self.scene is None else json.loads(json.dumps(self.scene))
