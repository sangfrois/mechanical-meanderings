"""Fail-closed scheduling for the existing rt-stt phrase/subtitle protocol."""
from collections import deque
import logging
import re

from .bridge import Bridge
from .final_turn import FinalTurnBridge

log = logging.getLogger(__name__)


class BoundaryBridge(FinalTurnBridge):
    """Observed subtitle pages are best-effort context, not authoritative finals.

    Trigger 0 opens/extends a phrase; trigger 1 closes it. Optional activity 1
    vetoes generation until activity 0. No timer manufactures a boundary.
    Pages following closure are ambiguous drains/previews, not completed turns.
    Without a speech-start signal the wire cannot distinguish drains from new
    previews; prefer failing closed over generating from an ambiguous preview.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = {}
        self.trailing = {}
        self.observed = {}
        self.last_observed = {}
        self.closed_channels = set()
        self.speech_active = {}

    def start(self):
        log.info('Boundary mode: /trigger 1 phrase closure; optional /speech/activity 1 veto '
                 'until 0; no debounce generation. Subtitle context is best-effort; '
                 'rt-stt supplies no authoritative final text or live VAD start.')
        self.emit('boundary_protocol', speech_history=self.speech_history,
                  protocol='/trigger 1; optional /speech/activity 0/1',
                  context='best-effort observed subtitle pages')
        Bridge.start(self)

    def _accumulate(self, channel, text):
        if not text:
            return
        closed = channel in self.closed_channels
        # A redraw of the last observed page after closure is not a new page
        # or evidence of another utterance. Keep it in the verbatim observation
        # ledger/raw lane, but never carry it into the next completed phrase.
        # Once trigger 0 opens a new phrase, identical speech remains valid.
        if closed and text == self.last_observed.get(channel):
            return
        buffers = self.trailing if closed else self.pages
        pages = buffers.setdefault(channel, [])
        previous = ' '.join(pages).split()
        words = text.split()
        # Snapshot growth and sliding pages can overlap anywhere at the tail.
        # Exact repeated pages are indistinguishable from display repeats.
        overlap = 0
        for size in range(min(len(previous), len(words)), 0, -1):
            if previous[-size:] == words[:size]:
                overlap = size
                break
        if words[overlap:]:
            pages.append(' '.join(words[overlap:]))
        while len(' '.join(pages)) > self.MAX_TEXT and len(pages) > 1:
            pages.pop(0)
            self.emit('drop', channel=channel, reason='subtitle accumulation capacity')
        if not closed:
            self.latest[channel] = ' '.join(pages)

    def _begin(self, channel):
        if channel in self.closed_channels:
            self.closed_channels.remove(channel)
            # These may be new previews OR old queued pages. Carry them without
            # assigning them retroactively to the previous completed phrase.
            self.pages[channel] = self.trailing.pop(channel, [])
            if self.pages[channel]:
                self.latest[channel] = ' '.join(self.pages[channel])
        self._supersede(channel)

    def _boundary(self, channel):
        if channel in self.closed_channels:
            return
        text = self.latest.get(channel, '')
        if text:
            self._complete(channel, text)
            self.closed_channels.add(channel)
            self.pages.pop(channel, None)  # Already retained by _complete.
        else:
            self.emit('drop', channel=channel, reason='boundary without preceding text')

    def receive(self, address, *args):
        if address == '/reset':
            with self.lock:
                self.pages.clear()
                self.trailing.clear()
                self.observed.clear()
                self.last_observed.clear()
                self.closed_channels.clear()
                self.speech_active.clear()
            return super().receive(address, *args)
        original = address
        channel = 'default'
        if self.channel_mode:
            match = re.fullmatch(r'/(ch\d+)/(transcription|trigger|final|speech/activity)', address)
            if not match:
                return
            channel, kind = match.groups()
            address = '/' + kind
        if address == '/final':
            # Atomic explicit finals remain available, but are not required.
            return super().receive(original, *args)
        if self.stop.is_set() or address not in ('/transcription', '/trigger', '/speech/activity'):
            return
        if not args:
            return
        if address == '/transcription':
            if not isinstance(args[0], str) or len(args[0]) > self.MAX_TEXT:
                return
        elif args[0] not in (0, 1):
            return
        with self.lock:
            if not self._admit(channel):
                return
        # Preserve exact display packets (including duplicates and clears).
        self._forward(address, args)
        with self.lock:
            if self.stop.is_set():
                return
            if address == '/speech/activity':
                active = bool(args[0])
                if active and not self.speech_active.get(channel, False):
                    self._begin(channel)
                self.speech_active[channel] = active
                self.emit('speech_activity', channel=channel, active=active)

            elif address == '/trigger':
                if args[0] == 0:
                    self._begin(channel)
                else:
                    self._boundary(channel)
            else:
                text = ' '.join(args[0].split())
                self.emit('input', channel=channel, input=text, source='osc')
                closed = channel in self.closed_channels
                if text:
                    observations = self.observed.setdefault(
                        channel, deque(maxlen=self.speech_history))
                    if len(observations) == observations.maxlen:
                        self.emit('drop', channel=channel, reason='subtitle observation capacity')
                    observations.append((args[0], closed))
                self._accumulate(channel, text)
                # Clears/repeated redraws do not mean new speech. Any changed
                # nonempty snapshot might: invalidate even on a closed channel.
                if text and text != self.last_observed.get(channel):
                    self._supersede(channel)
                    self.last_observed[channel] = text

    def _take_final(self):
        with self.lock:
            if self.paused or self.stop.is_set():
                return None
            for channel in self.ready:
                if not self.speech_active.get(channel, False):
                    text, epoch, revision, received = self.ready.pop(channel)
                    context = [{'role': 'user', 'content': value}
                               for value in list(self.completed[channel])[:-1]]
                    # Preserve each distinct observed snapshot verbatim in the
                    # actual model payload, not just its words inside a merge.
                    represented = [text] + [m['content'] for m in context]
                    seen = set()
                    extras = []
                    for snapshot, ambiguous in self.observed.get(channel, ()):
                        if snapshot in seen or snapshot in represented:
                            continue
                        seen.add(snapshot)
                        extras.append({'role': 'user', 'content': snapshot})
                    if extras:
                        context.append({'role': 'system', 'content':
                            'The following messages are verbatim observed subtitle snapshots, '
                            'not additional completed turns. They may overlap or repeat. '
                            'Snapshots after closure may be old queued pages or new previews; '
                            'their utterance ownership is unknown. Retain their speech for '
                            'associations without treating them as authoritative finals.'})
                        context.extend(extras)
                    return channel, text, epoch, revision, received, context, self.settings
        return None

    def _fresh(self, channel, epoch, revision):
        return (not self.speech_active.get(channel, False)
                and super()._fresh(channel, epoch, revision))
