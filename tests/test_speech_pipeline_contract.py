"""Speech-only model input and concurrent raw OSC over real loopback sockets.

No live microphone, TD port, Ollama service, or synthetic installation input.
The HTTP fixture intentionally stalls; it is not a model-quality benchmark.
"""
import importlib.util
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from mechanical_meanderings.boundary import BoundaryBridge
from test_generation_budget import http_fixture


@pytest.fixture
def pipeline(http_fixture):
    agent, requests, disconnected = http_fixture
    packets, events = queue.Queue(), queue.Queue()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(lambda address, *args: packets.put((address, args)))
    sink = BlockingOSCUDPServer(('127.0.0.1', 0), dispatcher)
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    bridge = BoundaryBridge(agent, listen_host='127.0.0.1', listen_port=0,
        output_host='127.0.0.1', output_port=sink.server_address[1],
        output_address='/txt2img', forward_raw=True, on_event=events.put)
    # Preserve the operator's model/window/history; no smaller model or memory.
    bridge.configure(num_ctx=128000, history=10, debounce=.6, interval=3)
    sender = SimpleUDPClient(*bridge.address)
    bridge.start()
    rig = SimpleNamespace(agent=agent, bridge=bridge, sender=sender,
        requests=requests, disconnected=disconnected, packets=packets, events=events)
    yield rig
    sender.close()
    bridge.close()
    sink.shutdown()
    sink.server_close()
    thread.join(timeout=1)


def forwarded(rig, address, value):
    started = time.monotonic()
    rig.sender.send_message(address, value)
    assert rig.packets.get(timeout=.3) == (address, (value,))
    return time.monotonic() - started


def event(rig, kind):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = rig.events.get(timeout=2)
        if value['kind'] == kind:
            return value
    raise AssertionError(f'missing {kind}')


def output(rig):
    assert rig.packets.get(timeout=2) == ('/txt2img', ('private partial finished',))
    assert rig.packets.get(timeout=2) == ('/txt2img/trigger', (1,))
    return event(rig, 'result')


def speech(payload):
    return [message['content'] for message in payload['messages']
            if message['role'] == 'user']


@pytest.mark.parametrize('stall', ['headers', 'stall', 'tokens'])
def test_real_http_stall_keeps_raw_lane_live_and_next_speech_proceeds(pipeline, stall):
    r = pipeline
    assert r.bridge.generation_budget == r.bridge.agent.timeout
    # Accelerate the same whole-request deadline for this isolated regression.
    r.bridge.generation_budget = .4
    forwarded(r, '/transcription', stall)
    forwarded(r, '/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert payload['stream'] is True
    assert payload['model'] == 'gemma4:12b'
    assert payload['options']['num_ctx'] == 128000
    assert speech(payload) == [stall]
    started = time.monotonic()
    # Redraws and clears must neither block on the stalled HTTP read nor cancel
    # it as fake new speech. Inference is still active throughout these sends.
    latencies = [forwarded(r, '/transcription', value)
                 for value in (stall, '', stall, '')]
    with r.bridge.lock:
        assert r.bridge.active_generations['default'].reason is None
    result = event(r, 'error')
    assert result['error'] == 'GenerationTimeout'
    assert result['telemetry']['generation_status'] == 'timeout'
    assert time.monotonic() - started < 1
    assert max(latencies) < .3
    assert r.disconnected.get(timeout=1) == b''
    assert not r.agent.history and r.packets.empty() and r.requests.empty()
    print(f'{stall}: raw max={max(latencies):.4f}s; '
          f'HTTP={result["telemetry"]["http_elapsed"]:.4f}s; no visual/history')

    forwarded(r, '/trigger', 0)
    forwarded(r, '/transcription', 'success')
    forwarded(r, '/trigger', 1)
    next_payload = r.requests.get(timeout=2)
    assert speech(next_payload) == [stall, 'success']
    assert not any(message['role'] == 'assistant'
                   for message in next_payload['messages'])
    output(r)
    assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())


def test_untranscribed_addresses_generated_history_and_idle_never_become_speech(pipeline):
    r = pipeline
    foreign = 'UNTRANSCRIBED_FOREIGN_SENTINEL'
    r.agent.history['default'] = [
        {'role': 'user', 'content': foreign},
        {'role': 'assistant', 'content': 'UNTRANSCRIBED_GENERATED_SENTINEL'}]
    # These are real received OSC packets, but none is an accepted speech lane.
    for address in ('/txt2img', '/prompt', '/agent/message', '/scene/summary',
                    '/ch2/transcription'):
        r.sender.send_message(address, foreign)
    forwarded(r, '/trigger', 1)  # no previous transcript, never latched
    event(r, 'drop')
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.7)
    assert r.packets.empty()

    observed = ['une forêt', 'une forêt bleue, sans personne.']
    for value in observed:
        forwarded(r, '/transcription', value)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.7)  # longer than the operator's .6s debounce
    forwarded(r, '/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert set(speech(payload)) == set(observed)
    assert all('UNTRANSCRIBED' not in m['content'] for m in payload['messages'])
    assert not any(m['role'] == 'assistant' for m in payload['messages'])
    output(r)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.7)  # no timer-driven regeneration while idle

    forwarded(r, '/trigger', 0)
    forwarded(r, '/transcription', 'garde seulement la couleur.')
    forwarded(r, '/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert set(speech(payload)) == set(observed + ['garde seulement la couleur.'])
    assert not any(m['role'] == 'assistant' for m in payload['messages'])
    assert all('private partial finished' not in m['content'] for m in payload['messages'])
    output(r)


def test_agent_lock_and_stalled_http_cannot_hold_new_partial_osc(pipeline):
    r = pipeline
    forwarded(r, '/transcription', 'stall')
    forwarded(r, '/trigger', 1)
    r.requests.get(timeout=2)
    with r.agent.lock:
        # New speech cancels obsolete inference, but raw packets still arrive
        # while the model worker is unable to acquire the agent lock.
        forwarded(r, '/trigger', 0)
        for value in ('new', 'new sentence', '', 'new sentence completed.'):
            assert forwarded(r, '/transcription', value) < .3
        assert r.requests.empty()
        assert r.packets.empty()
    event(r, 'discard')
    assert r.disconnected.get(timeout=1) == b''
    assert not r.agent.history
    forwarded(r, '/trigger', 1)
    payload = r.requests.get(timeout=2)
    assert speech(payload)[-1] == 'new sentence completed.'
    assert {'role': 'user', 'content': 'stall'} in payload['messages']
    output(r)


def test_actual_rt_stt_partial_and_exact_timeout_boundary(pipeline):
    source = Path(__file__).resolve().parents[2] / 'rt-stt/src/scarlett_live_whisper/osc.py'
    if not source.exists():
        pytest.skip('optional sibling rt-stt unavailable')
    spec = importlib.util.spec_from_file_location('speech_contract_rt_stt', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    r = pipeline
    now = [10.]
    wire, authoritative = [], []

    class Sink:
        def transcription(self, channel, text):
            wire.append(('/transcription', text))
            forwarded(r, '/transcription', text)

        def trigger(self, channel, value):
            wire.append(('/trigger', value))
            forwarded(r, '/trigger', value)

        def phrase(self, channel, text, *times):
            authoritative.append(text)

    tracker = module.PhraseTracker(Sink(), phrase_timeout=3, line_chars=200,
        lines=1, min_update=0, page_time=0, clock=lambda: now[0])
    fragment = 'Hello, can you'
    tracker.partial(0, fragment, start=0)
    tracker.partial(0, fragment, start=0)
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.7)
    complete = 'Hello, can you make the forest blue, without people?'
    tracker.add(0, 0, 1, complete)
    tracker.idle(0, 3.999)
    tracker.idle(0, 4.0)  # producer closes only strictly beyond phrase_timeout
    assert not authoritative
    with pytest.raises(queue.Empty):
        r.requests.get(timeout=.7)
    tracker.idle(0, 4.001)
    payload = r.requests.get(timeout=2)
    assert authoritative == [complete]
    assert wire[-1] == ('/trigger', 1)
    assert speech(payload)[-1] == complete
    assert {'role': 'user', 'content': fragment} in payload['messages']
    output(r)
