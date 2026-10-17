"""Existing rt-stt packet ordering, real OSC receiver, gated model transport."""
import importlib.util
from pathlib import Path
import queue
import time
from unittest.mock import patch

import pytest
import test_final_turn
from test_final_turn import rig, event, output
from mechanical_meanderings.boundary import BoundaryBridge
from mechanical_meanderings.__main__ import main


@pytest.fixture
def boundary_rig(rig, monkeypatch):
    monkeypatch.setattr(test_final_turn, 'FinalTurnBridge', BoundaryBridge)
    return rig


def test_active_pause_and_phrase_closure_cannot_override_vad(boundary_rig):
    r = boundary_rig(debounce=.01, update_interval=.05)
    r.sender.send_message('/speech/activity', 1)
    r.sender.send_message('/trigger', 0)
    r.sender.send_message('/transcription', 'a thought')
    event(r, 'input')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.15)
    r.sender.send_message('/transcription', 'a thought completed')
    r.sender.send_message('/trigger', 1)
    event(r, 'complete')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.15)
    r.sender.send_message('/speech/activity', 0)
    payload = r.requests.get(timeout=2)
    assert payload['messages'][-1]['content'] == 'a thought completed'
    assert {'role': 'user', 'content': 'a thought'} in payload['messages']
    assert payload['model'] == 'gemma4:12b'
    assert payload['options']['num_ctx'] == 16384
    r.responses.put('silent visual')
    output(r, 'silent visual')


def test_activity_silence_does_not_manufacture_trigger_closure(boundary_rig):
    r = boundary_rig()
    r.bridge.receive('/speech/activity', 1)
    r.bridge.receive('/transcription', 'live')
    r.bridge.receive('/transcription', 'live complete')
    r.bridge.receive('/speech/activity', 0)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.bridge.receive('/trigger', 1)
    assert r.requests.get(timeout=2)['messages'][-1]['content'] == 'live complete'
    r.responses.put('visual')
    output(r, 'visual')
    r.bridge.receive('/speech/activity', 0)
    assert r.requests.empty()


def test_pages_drains_duplicate_boundaries_keep_context_without_generating(boundary_rig):
    r = boundary_rig()
    r.bridge.set_paused(True)
    for address, value in [('/trigger', 0), ('/transcription', 'first'),
                           ('/transcription', 'first page'), ('/transcription', 'second page'),
                           ('/trigger', 1), ('/trigger', 1),
                           ('/transcription', 'last page'), ('/transcription', ''),
                           ('/trigger', 1)]:
        r.bridge.receive(address, value)
    assert list(r.bridge.completed['default']) == ['first page second page']
    assert r.bridge.trailing['default'] == ['last page']
    assert not r.bridge.ready  # Changed closed input invalidates pending work.
    r.bridge.set_paused(False)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.bridge.receive('/transcription', 'late display drain')
    r.bridge.receive('/trigger', 1)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/transcription', 'next phrase')
    r.bridge.receive('/trigger', 1)
    messages = r.requests.get(timeout=2)['messages']
    assert messages[-1]['content'] == 'last page late display drain next phrase'
    for text in ('first', 'first page', 'second page', 'last page', 'late display drain'):
        assert {'role': 'user', 'content': text} in messages
    r.responses.put('next visual')
    output(r, 'next visual')


def test_new_activity_discards_running_output_raw_stays_immediate(boundary_rig):
    r = boundary_rig(forward_raw=True)
    r.bridge.receive('/transcription', 'finished')
    assert r.packets.get(timeout=.3) == ('/transcription', ('finished',))
    r.bridge.receive('/trigger', 1)
    assert r.packets.get(timeout=.3) == ('/trigger', (1,))
    r.requests.get(timeout=2)
    r.sender.send_message('/speech/activity', 1)
    assert r.packets.get(timeout=.3) == ('/speech/activity', (1,))
    for text in ('next', 'next', '', 'next completed'):
        r.sender.send_message('/transcription', text)
        assert r.packets.get(timeout=.3) == ('/transcription', (text,))
    r.responses.put('stale visual')
    event(r, 'discard')
    assert not r.agent.history and r.requests.empty() and r.packets.empty()
    r.sender.send_message('/speech/activity', 0)
    assert r.packets.get(timeout=.3) == ('/speech/activity', (0,))
    assert r.requests.empty()
    r.sender.send_message('/trigger', 1)
    assert r.packets.get(timeout=.3) == ('/trigger', (1,))
    messages = r.requests.get(timeout=2)['messages']
    assert messages[-1]['content'] == 'next completed'
    for text in ('finished', 'next'):
        assert {'role': 'user', 'content': text} in messages
    r.responses.put('fresh visual')
    output(r, 'fresh visual')


def test_default_cli_constructs_boundary_raw_enabled_no_runtime():
    seen = []
    class Panel:
        def __init__(self, bridge, session):
            seen.append(bridge)
        def run(self):
            pass
    with patch('mechanical_meanderings.tui.MeanderingsApp', Panel), patch('mechanical_meanderings.__main__.Session'):
        assert main(['--tui', '--listen-port', '0', '--output-port', '9001',
                     '--history-turns', '40', '--num-ctx', '32768']) == 0
    assert isinstance(seen[0], BoundaryBridge)
    assert seen[0].forward_raw and not seen[0].dynamic
    assert seen[0].settings.history == 40 and seen[0].settings.num_ctx == 32768
    assert not seen[0].worker.is_alive()


def test_actual_sibling_phrase_tracker_replay(boundary_rig):
    # Read-only import of the exact existing protocol implementation; no audio,
    # TD, rt-stt writes, live sender or live inference. Portable without sibling.
    source = Path(__file__).resolve().parents[2] / 'rt-stt/src/scarlett_live_whisper/osc.py'
    if not source.exists():
        pytest.skip('optional sibling rt-stt checkout unavailable')
    spec = importlib.util.spec_from_file_location('readonly_rt_stt_osc', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = boundary_rig()
    now = [10.]
    class Sink:
        def transcription(self, channel, text):
            r.bridge.receive('/transcription', text)
        def trigger(self, channel, value):
            r.bridge.receive('/trigger', value)
        def phrase(self, *args):
            pass  # The existing wire omits this authoritative callback.
    tracker = module.PhraseTracker(Sink(), phrase_timeout=3, line_chars=12,
        lines=1, min_update=0, page_time=.1, hold=1, clock=lambda: now[0])
    tracker.partial(0, 'first words', start=0)
    tracker.partial(0, 'first words', start=0)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    tracker.add(0, 0, 1, 'first words second page last page')
    r.bridge.set_paused(True)
    tracker.idle(0, 5)  # closure before queued subtitle pages drain
    for _ in range(8):
        now[0] += .2
        tracker.tick()
    assert len(r.bridge.completed['default']) == 1
    assert list(r.bridge.completed['default']) == ['first words']
    # A post-closure redraw of the already completed page is not new speech.
    # The first page remains in completed context; only distinct drains carry.
    assert ' '.join(r.bridge.trailing['default']) == 'second page last page'
    assert not r.bridge.ready
    r.bridge.set_paused(False)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    tracker.partial(0, 'next beginning', start=10)
    tracker.partial(0, 'next beginning', start=10)
    now[0] += .2
    tracker.tick()
    tracker.add(0, 10, 1, 'next beginning ends')
    # A tiny screen queues the final word too: drain it before this closure.
    for _ in range(3):
        now[0] += .2
        tracker.tick()
    tracker.idle(0, 15)
    messages = r.requests.get(timeout=2)['messages']
    assert messages[-1]['content'] == 'second page last page next beginning ends'
    assert messages[1]['content'] == 'first words'
    r.responses.put('observed visual')
    output(r, 'observed visual')
    tracker.tick(now[0] + 5)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)


def test_trigger_open_supersedes_and_preserves_prior_completed_context(boundary_rig):
    r = boundary_rig()
    r.bridge.receive('/transcription', 'one')
    r.bridge.receive('/trigger', 1)
    r.requests.get(timeout=2)
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/transcription', 'two')
    r.responses.put('obsolete')
    event(r, 'discard')
    assert r.requests.empty() and r.agent.history == {}
    r.bridge.receive('/trigger', 1)
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'one', 'two']
    r.responses.put('two visual')
    output(r, 'two visual')


def test_scoped_activity_blocks_only_its_channel_and_reset_clears_pages(boundary_rig):
    r = boundary_rig(channel_mode=True)
    r.bridge.receive('/speech/activity', 1)  # ignored aggregate
    r.bridge.receive('/ch1/speech/activity', 1)
    r.bridge.receive('/ch1/transcription', 'active channel')
    r.bridge.receive('/ch1/trigger', 1)
    r.bridge.receive('/ch2/transcription', 'silent channel')
    r.bridge.receive('/ch2/trigger', 1)
    assert r.requests.get(timeout=2)['messages'][-1]['content'] == 'silent channel'
    r.responses.put('channel two visual')
    output(r, 'channel two visual')
    assert r.requests.empty()
    r.bridge.receive('/reset', 1)
    assert not r.bridge.pages and not r.bridge.closed_channels and not r.bridge.speech_active
    r.bridge.receive('/ch1/trigger', 1)
    r.bridge.receive('/ch1/transcription', 'fresh text')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.bridge.receive('/ch1/trigger', 1)
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'fresh text']
    r.responses.put('fresh visual')
    output(r, 'fresh visual')


def test_closed_previews_coalesced_turns_reach_actual_model_payload_verbatim(boundary_rig):
    r = boundary_rig()
    r.bridge.set_paused(True)
    r.bridge.receive('/transcription', 'previous completed phrase')
    r.bridge.receive('/trigger', 1)
    omitted = 'Yes, bon, vas-y. Je n’ai pas fini. Je fais un,deux,trois.Lets go.'
    r.bridge.receive('/transcription', omitted)
    r.bridge.receive('/transcription', '')
    assert not r.bridge.ready
    assert list(r.bridge.completed['default']) == ['previous completed phrase']
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/transcription', 'C’est fini. là. Supposé d’envoyer le texte.')
    r.bridge.receive('/transcription', 'le texte.')
    assert r.bridge.latest['default'].endswith('Supposé d’envoyer le texte.')
    r.bridge.receive('/trigger', 1)
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/transcription', 'subsequent turn')
    assert r.requests.empty()
    r.bridge.receive('/trigger', 1)
    r.bridge.set_paused(False)
    payload = r.requests.get(timeout=2)
    speech = [m['content'] for m in payload['messages'] if m['role'] == 'user']
    for exact in ('previous completed phrase', omitted,
                  'C’est fini. là. Supposé d’envoyer le texte.', 'le texte.', 'subsequent turn'):
        assert exact in speech
    assert omitted in speech[1]
    assert payload['model'] == 'gemma4:12b'
    assert payload['options']['num_ctx'] == 16384
    assert r.bridge.speech_history == 128
    r.responses.put('latest visual')
    output(r, 'latest visual')


def test_changed_closed_snapshot_discards_inflight_without_regenerating(boundary_rig):
    r = boundary_rig()
    r.bridge.receive('/transcription', 'completed')
    r.bridge.receive('/trigger', 1)
    r.requests.get(timeout=2)
    r.bridge.receive('/transcription', 'unknown ownership preview')
    r.bridge.receive('/trigger', 1)
    r.responses.put('obsolete')
    event(r, 'discard')
    assert not r.agent.history and r.requests.empty()
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/transcription', 'next known phrase')
    r.bridge.receive('/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert {'role': 'user', 'content': 'unknown ownership preview'} in payload['messages']
    r.responses.put('fresh')
    output(r, 'fresh')


def test_repeats_sliding_overlap_and_verbatim_snapshot_retention(boundary_rig):
    r = boundary_rig()
    r.bridge.receive('/transcription', 'one two three')
    r.bridge.receive('/transcription', 'two three four')
    r.bridge.receive('/transcription', 'two three four')
    assert r.bridge.latest['default'] == 'one two three four'
    r.bridge.receive('/trigger', 1)
    payload = r.requests.get(timeout=2)
    for exact in ('one two three', 'two three four'):
        assert {'role': 'user', 'content': exact} in payload['messages']
    r.responses.put('one')
    output(r, 'one')
    for _ in range(2):
        r.bridge.receive('/trigger', 0)
        r.bridge.receive('/transcription', 'same sentence.')
        r.bridge.receive('/trigger', 1)
        r.requests.get(timeout=2)
        r.responses.put('repeat')
        output(r, 'repeat')
    assert list(r.bridge.completed['default'])[-2:] == ['same sentence.', 'same sentence.']


def test_observation_lane_bounds_and_verbatim_whitespace(boundary_rig):
    r = boundary_rig(speech_history=3)
    r.bridge.set_paused(True)
    r.bridge.receive('/transcription', 'completed')
    r.bridge.receive('/trigger', 1)
    for text in ('old observation', '  Yes bon\nvas-y  ', 'second observation', 'third observation'):
        r.bridge.receive('/transcription', text)
    assert [text for text, _ in r.bridge.observed['default']] == [
        '  Yes bon\nvas-y  ', 'second observation', 'third observation']
    r.bridge.receive('/trigger', 0)
    r.bridge.receive('/trigger', 1)
    r.bridge.set_paused(False)
    payload = r.requests.get(timeout=2)
    assert {'role': 'user', 'content': '  Yes bon\nvas-y  '} in payload['messages']
    r.responses.put('retained')
    output(r, 'retained')
    r.bridge.receive('/reset', 1)
    assert not r.bridge.observed and not r.bridge.trailing and not r.bridge.last_observed


def test_actual_phrase_tracker_gap_without_open_retains_speech_in_payload(boundary_rig):
    source = Path(__file__).resolve().parents[2] / 'rt-stt/src/scarlett_live_whisper/osc.py'
    if not source.exists():
        pytest.skip('optional sibling rt-stt checkout unavailable')
    spec = importlib.util.spec_from_file_location('readonly_gap_osc', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = boundary_rig()
    now = [10.]
    wire, authoritative = [], []
    class Sink:
        def transcription(self, channel, text):
            wire.append(('/transcription', text))
            r.bridge.receive('/transcription', text)
        def trigger(self, channel, value):
            wire.append(('/trigger', value))
            r.bridge.receive('/trigger', value)
        def phrase(self, channel, text, *times):
            authoritative.append(text)  # Recorded only; NEVER supplied to bridge.
    tracker = module.PhraseTracker(Sink(), phrase_timeout=3, line_chars=200,
        lines=1, min_update=0, page_time=0, clock=lambda: now[0])
    tracker.add(0, 0, 1, 'first phrase')
    tracker.add(0, 10, 1, 'Yes bon vas-y Je n’ai pas fini')
    assert [value for address, value in wire if address == '/trigger'] == [0, 1]
    tracker.idle(0, 15)
    assert authoritative == ['first phrase', 'Yes bon vas-y Je n’ai pas fini']
    assert not r.bridge.ready and r.requests.empty()
    tracker.partial(0, 'subsequent phrase', start=20)
    tracker.partial(0, 'subsequent phrase', start=20)
    tracker.add(0, 20, 1, 'subsequent phrase')
    assert r.requests.empty()
    tracker.idle(0, 25)
    payload = r.requests.get(timeout=2)
    for exact in ('first phrase', 'Yes bon vas-y Je n’ai pas fini', 'subsequent phrase'):
        assert {'role': 'user', 'content': exact} in payload['messages']
    r.responses.put('gap associations')
    output(r, 'gap associations')
