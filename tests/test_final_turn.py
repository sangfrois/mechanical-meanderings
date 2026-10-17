"""Production scheduler + real loopback OSC; model transport is gated, not live."""
import queue
import threading
import time
import unittest
import test_bridge
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from mechanical_meanderings import Agent
from mechanical_meanderings.final_turn import FinalTurnBridge
from mechanical_meanderings.__main__ import main


@pytest.fixture
def rig():
    rigs = []

    def make(**options):
        requests, responses, packets, events = (queue.Queue() for _ in range(4))

        def transport(url, payload, timeout):
            requests.put(payload)
            response = responses.get(timeout=3)
            if isinstance(response, Exception):
                raise response
            return {'message': {'content': response}}

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(lambda address, *args: packets.put((address, args)))
        sink = BlockingOSCUDPServer(('127.0.0.1', 0), dispatcher)
        receiver = threading.Thread(target=sink.serve_forever, daemon=True)
        receiver.start()
        clock = options.pop('clock', time.monotonic)
        reset_after = options.pop('reset_after', None)
        agent = Agent('system', transport=transport, history_turns=12,
                      reset_after=reset_after, clock=clock)
        bridge = FinalTurnBridge(agent, listen_host='127.0.0.1', listen_port=0,
            output_host='127.0.0.1', output_port=sink.server_address[1],
            on_event=events.put, **options)
        bridge.configure(num_ctx=16384, num_predict=96, history=12)
        bridge.start()
        sender = SimpleUDPClient(*bridge.address)
        result = SimpleNamespace(agent=agent, bridge=bridge, sender=sender,
            requests=requests, responses=responses, packets=packets, events=events,
            sink=sink, receiver=receiver)
        rigs.append(result)
        return result

    yield make
    for r in rigs:
        r.bridge.close()
        r.responses.put('late teardown response')
        r.bridge.worker.join(timeout=1)
        r.sender.close()
        r.sink.shutdown()
        r.sink.server_close()
        r.receiver.join(timeout=1)


def event(r, kind):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        item = r.events.get(timeout=2)
        if item['kind'] == kind:
            return item
    raise AssertionError(f'missing {kind}')


def output(r, text):
    assert r.packets.get(timeout=2) == ('/prompt', (text,))
    assert r.packets.get(timeout=2) == ('/prompt/trigger', (1,))
    event(r, 'result')


def test_raw_duplicates_and_empty_clears_are_forwarded_without_generation(rig):
    r = rig(forward_raw=True)
    for text in ('same page', 'same page', ''):
        r.sender.send_message('/transcription', text)
        assert r.packets.get(timeout=.3) == ('/transcription', (text,))
    assert r.requests.empty()
    assert r.bridge.completed == {}


def test_partials_wait_even_after_debounce_then_final_snapshot(rig):
    r = rig(debounce=0, update_interval=.05)
    r.sender.send_message('/transcription', 'part')
    r.sender.send_message('/transcription', 'complete thought')
    event(r, 'input'); event(r, 'input')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.sender.send_message('/trigger', 0)
    r.sender.send_message('/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert payload['messages'] == [{'role': 'system', 'content': 'system'},
                                   {'role': 'user', 'content': 'complete thought'}]
    assert payload['model'] == 'gemma4:12b'
    assert payload['options']['num_ctx'] == 16384
    assert payload['options']['num_predict'] == 96
    assert payload['think'] is False and payload['stream'] is False
    r.responses.put('completed visual')
    output(r, 'completed visual')
    r.sender.send_message('/trigger', 1)
    event(r, 'drop')
    assert list(r.bridge.completed['default']) == ['complete thought']
    assert r.requests.empty()


def test_early_trigger_not_latched_or_assumed_to_pair_with_future_text(rig):
    r = rig()
    r.sender.send_message('/trigger', 1)
    event(r, 'drop')
    r.sender.send_message('/transcription', 'later partial')
    event(r, 'input')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    r.sender.send_message('/final', 'atomic completed thought')
    assert r.requests.get(timeout=2)['messages'][-1]['content'] == 'atomic completed thought'
    r.responses.put('atomic visual')
    output(r, 'atomic visual')


def test_raw_lane_continues_during_blocked_inference_and_partials_supersede(rig):
    r = rig(forward_raw=True)
    r.sender.send_message('/final', 'first final')
    assert r.packets.get(timeout=2) == ('/transcription', ('first final',))
    r.requests.get(timeout=2)  # HTTP seam remains blocked until responses.put.
    for text in ('new', 'new speech', 'new speech complete'):
        started = time.monotonic()
        r.sender.send_message('/transcription', text)
        assert r.packets.get(timeout=.3) == ('/transcription', (text,))
        assert time.monotonic() - started < .3
    event(r, 'input'); event(r, 'input'); event(r, 'input'); event(r, 'input')
    assert r.requests.empty()
    r.responses.put('obsolete visual')
    event(r, 'discard')
    assert r.agent.history == {}
    assert r.requests.empty()
    assert r.packets.empty()
    r.sender.send_message('/trigger', 1)
    assert r.packets.get(timeout=2) == ('/trigger', (1,))
    payload = r.requests.get(timeout=2)
    assert [m['content'] for m in payload['messages']] == ['system', 'first final', 'new speech complete']
    r.responses.put('fresh visual')
    output(r, 'fresh visual')
    assert 'obsolete visual' not in str(r.agent.history)


def test_coalescing_keeps_all_completed_speech_but_only_latest_pending_visual(rig):
    r = rig(queue_size=1)
    r.bridge.submit('one')
    r.requests.get(timeout=2)
    for text in ('two', 'three', 'four'):
        r.bridge.submit(text)
    assert len(r.bridge.ready) == 1
    r.responses.put('obsolete')
    event(r, 'discard')
    payload = r.requests.get(timeout=2)
    assert [m['content'] for m in payload['messages']] == ['system', 'one', 'two', 'three', 'four']
    assert r.agent.history == {}
    r.responses.put('four visual')
    output(r, 'four visual')
    assert r.agent.history['default'] == [{'role': 'user', 'content': 'four'},
                                         {'role': 'assistant', 'content': 'four visual'}]
    assert r.requests.empty()


def test_speech_ledger_bound_independent_of_generated_history_and_pause(rig):
    r = rig(speech_history=3)
    r.bridge.configure(history=0)
    r.bridge.set_paused(True)
    for text in ('one', 'two', 'three', 'four', 'five'):
        r.bridge.submit(text)
    assert r.requests.empty()
    assert list(r.bridge.completed['default']) == ['three', 'four', 'five']
    r.bridge.set_paused(False)
    payload = r.requests.get(timeout=2)
    assert [m['content'] for m in payload['messages']] == ['system', 'three', 'four', 'five']
    r.responses.put('five visual')
    output(r, 'five visual')
    assert r.agent.history['default'] == []


def test_new_partial_cancels_pending_final_without_erasing_completed_context(rig):
    r = rig()
    r.bridge.set_paused(True)
    r.bridge.receive('/final', 'finished one')
    r.bridge.receive('/transcription', 'starting two')
    r.bridge.set_paused(False)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.1)
    assert list(r.bridge.completed['default']) == ['finished one']
    r.bridge.receive('/final', 'finished two')
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'finished one', 'finished two']
    r.responses.put('second visual')
    output(r, 'second visual')


def test_channels_are_independent_and_aggregate_is_ignored(rig):
    r = rig(channel_mode=True)
    r.bridge.receive('/final', 'aggregate duplicate')
    r.bridge.receive('/ch1/final', 'channel one')
    r.requests.get(timeout=2)
    r.bridge.receive('/ch2/transcription', 'channel two')
    r.bridge.receive('/ch2/trigger', 1)
    r.responses.put('one visual')
    output(r, 'one visual')  # ch2 does not supersede ch1.
    payload = r.requests.get(timeout=2)
    assert [m['content'] for m in payload['messages']] == ['system', 'channel two']
    r.responses.put('two visual')
    output(r, 'two visual')
    assert set(r.bridge.completed) == {'ch1', 'ch2'}


def test_reset_discards_inflight_pending_and_both_contexts(rig):
    r = rig()
    r.bridge.submit('inflight')
    r.requests.get(timeout=2)
    r.bridge.submit('pending')
    r.bridge.receive('/reset', 1)
    r.responses.put('obsolete')
    event(r, 'discard')
    assert r.bridge.completed == {} and not r.bridge.ready
    assert r.agent.history == {} and r.packets.empty()
    r.bridge.submit('fresh')
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'fresh']
    r.responses.put('fresh visual')
    output(r, 'fresh visual')


def test_idle_reset_uses_completed_speech_clock_not_generation_clock(rig):
    now = [0.]
    r = rig(clock=lambda: now[0], reset_after=5)
    r.bridge.submit('old')
    r.requests.get(timeout=2)
    now[0] = 5.
    r.bridge.submit('after idle')
    r.responses.put('old visual')
    event(r, 'discard')
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'after idle']
    r.responses.put('new visual')
    output(r, 'new visual')


def test_shutdown_bounded_while_transport_blocked_no_late_send_or_commit(rig):
    r = rig()
    r.bridge.submit('slow')
    r.requests.get(timeout=2)
    start = time.monotonic()
    r.bridge.close()
    assert time.monotonic() - start < 1.5
    assert r.bridge.worker.is_alive()
    r.responses.put('late')
    r.bridge.worker.join(timeout=1)
    assert not r.bridge.worker.is_alive()
    assert r.packets.empty() and r.agent.history == {}
    assert r.bridge.client._sock.fileno() == -1


def test_failed_model_does_not_erase_completed_speech_or_commit_output(rig):
    r = rig()
    r.bridge.submit('first')
    r.requests.get(timeout=2)
    r.responses.put(ValueError('failed transport'))
    event(r, 'error')
    r.bridge.submit('second')
    assert [m['content'] for m in r.requests.get(timeout=2)['messages']] == ['system', 'first', 'second']
    r.responses.put('second visual')
    output(r, 'second visual')


def test_agent_lock_contention_cannot_hold_raw_receive_lane(rig):
    r = rig(forward_raw=True)
    # Hold the agent lock in this thread; worker will wait on it, not bridge.lock.
    with r.agent.lock:
        r.sender.send_message('/final', 'final')
        assert r.packets.get(timeout=.3) == ('/transcription', ('final',))
        event(r, 'complete')
        time.sleep(.05)
        r.sender.send_message('/transcription', 'next partial')
        assert r.packets.get(timeout=.3) == ('/transcription', ('next partial',))
        event(r, 'input')
    event(r, 'discard')
    assert r.requests.empty()


def test_cli_wires_final_mode_without_launching_runtime():
    seen = []

    class Panel:
        def __init__(self, bridge, session):
            seen.append(bridge)
        def run(self):
            pass

    with patch('mechanical_meanderings.tui.MeanderingsApp', Panel):
        # TUI construction exercises CLI wiring without start(), a model call,
        # or a real panel. No journal required: fake Session below.
        with patch('mechanical_meanderings.__main__.Session'):
            assert main(['--tui', '--final-turn', '--listen-port', '0',
                '--speech-history', '200', '--history-turns', '40', '--num-ctx', '32768']) == 0
    assert isinstance(seen[0], FinalTurnBridge)
    assert seen[0].speech_history == 200
    assert seen[0].settings.history == 40 and seen[0].settings.num_ctx == 32768
    assert not seen[0].worker.is_alive()


@pytest.mark.parametrize('args', [
    ['--final-turn', '--interactive'], ['--final-turn', '--text', 'test'],
    ['--final-turn', '--speech-history', '0'],
])
def test_invalid_final_cli_combinations_fail_before_runtime(args):
    assert main(args) == 1


def test_shutdown_bounded_even_with_reset_waiting_for_agent_lock(rig):
    r = rig()
    with r.agent.lock:
        r.sender.send_message('/reset', 1)
        event(r, 'reset')  # receiver is now waiting for the externally held lock
        start = time.monotonic()
        r.bridge.close()
        assert time.monotonic() - start < 1.5
    r.bridge.receiver.join(timeout=1)
    assert not r.bridge.receiver.is_alive()


class FinalHTTPTests(unittest.TestCase):
    setUp = test_bridge.BridgeTests.setUp
    tearDown = test_bridge.BridgeTests.tearDown

    def test_real_http_and_udp_final_pipeline_logs_safe_latency(self):
        self.bridge = FinalTurnBridge(self.agent, listen_host='127.0.0.1', listen_port=0,
            output_host='127.0.0.1', output_port=self.sink.server_address[1])
        self.bridge.start()
        self.sender = SimpleUDPClient(*self.bridge.address)
        with self.assertLogs('mechanical_meanderings.final_turn', level='INFO') as captured:
            self.sender.send_message('/transcription', 'partial input')
            with self.assertRaises(queue.Empty):
                self.requests.get(timeout=.1)
            self.sender.send_message('/final', 'authoritative full input')
            path, payload = self.requests.get(timeout=2)
            self.assertEqual(path, '/api/chat')
            self.assertEqual(payload['messages'][-1]['content'], 'authoritative full input')
            self.assertEqual(self.received.get(timeout=2), ('/prompt', ('A metal flower turns.',)))
            self.received.get(timeout=2)
            self.bridge.stop.set()
            self.bridge.worker.join(timeout=1)
        logs = '\n'.join(captured.output)
        self.assertIn('output_tokens=n/a', logs)
        self.assertIn('http=', logs)
        self.assertIn('request=', logs)
        self.assertIn('num_predict_max=model-default', logs)

