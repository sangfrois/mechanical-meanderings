"""Completed-speech interactive mode; isolated HTTP/OSC, never live ports."""
import json
import queue
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from mechanical_meanderings.__main__ import main
from mechanical_meanderings.interactive import InteractiveBridge
from mechanical_meanderings.mediation import MediatingAgent
from test_mediation import judgement


@contextmanager
def rig(budget=None, stall_invitation=False, stall_server_timeout=2, progress_seconds=0, **options):
    requests, packets, events, disconnects = (queue.Queue() for _ in range(4))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.put(payload)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.end_headers()
            self.connection.settimeout(stall_server_timeout)
            invitation_call = 'message' in payload['format']['properties']
            if payload['messages'][-1]['content'] == 'stall' or stall_invitation and invitation_call:
                try:
                    self.wfile.write(json.dumps({'message': {'content': '{"prompt":"private'},
                        'done': False}).encode() + b'\n')
                    self.wfile.flush()
                    disconnects.put(self.connection.recv(1))
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    # Windows can report an abort rather than an EOF on cancellation.
                    # A timeout is NOT accepted as proof of disconnect.
                    disconnects.put(b'')
                return
            value = judgement(reply='', needs_camera=False)
            if progress_seconds:
                text = json.dumps(value)
                for index in range(6):
                    chunk = text[index * len(text) // 6:(index + 1) * len(text) // 6]
                    self.wfile.write(json.dumps({'message': {'content': chunk},
                        'done': False}).encode() + b'\n')
                    self.wfile.flush()
                    time.sleep(progress_seconds / 6)
            content = '' if progress_seconds else json.dumps(value)
            self.wfile.write(json.dumps({'message': {'content': content},
                'done': True, 'eval_count': 5, 'eval_duration': 100000000}).encode() + b'\n')
            self.wfile.flush()

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http.daemon_threads = True
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(lambda address, *args: packets.put((address, args)))
    sink = BlockingOSCUDPServer(('127.0.0.1', 0), dispatcher)
    sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
    sink_thread.start()
    agent = MediatingAgent('style', ollama_url=f'http://127.0.0.1:{http.server_port}',
        history_turns=10, reply_mode='sparse')
    bridge = None
    sender = None
    try:
        options.setdefault('camera_enabled', False)
        options.setdefault('invitations_enabled', False)
        bridge = InteractiveBridge(agent, listen_host='127.0.0.1', listen_port=0,
            output_host='127.0.0.1', output_port=sink.server_address[1],
            output_address='/txt2img', frame_port=0, on_event=events.put,
            **({} if budget is None else dict(generation_budget=budget)), **options)
        bridge.configure(num_ctx=128000, history=10, debounce=.6, interval=3)
        bridge.start()
        sender = SimpleUDPClient(*bridge.address)
        yield bridge, sender, requests, packets, events, disconnects
    finally:
        if sender is not None:
            sender.close()
        if bridge is not None:
            bridge.close()
        http.shutdown()
        http.server_close()
        sink.shutdown()
        sink.server_close()
        http_thread.join(1)
        sink_thread.join(1)


def raw(sender, packets, address, value):
    start = time.monotonic()
    sender.send_message(address, value)
    assert packets.get(timeout=.3) == (address, (value,))
    assert time.monotonic() - start < .3


def event(events, kind):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = events.get(timeout=2)
        if value['kind'] == kind:
            return value
    raise AssertionError(kind)


def accepted(packets, events):
    assert packets.get(timeout=2)[0] == '/mediation'
    assert packets.get(timeout=2)[0] == '/txt2img'
    assert packets.get(timeout=2) == ('/txt2img/trigger', (1,))
    return event(events, 'result')


def test_operator_cli_interactive_defaults_to_boundary_not_dynamic():
    seen = []

    class Panel:
        def __init__(self, bridge, session):
            seen.append(bridge)

        def run(self):
            pass

    with patch('mechanical_meanderings.tui.MeanderingsApp', Panel), \
            patch('mechanical_meanderings.__main__.Session'):
        assert main(['--tui', '--interactive', '--listen-host', '127.0.0.1',
            '--listen-port', '0', '--frame-port', '0', '--output-address', '/txt2img',
            '--history-turns', '10', '--forward-raw', '--update-interval', '3',
            '--debounce', '.6', '--num-ctx', '128000']) == 0
    bridge = seen[0]
    assert not bridge.dynamic
    assert bridge.boundary_speech and bridge.forward_raw
    assert bridge.settings.num_ctx == 128000 and bridge.settings.history == 10
    assert bridge.agent.model == 'gemma4:12b' and bridge.generation_budget == bridge.agent.timeout == 120


def test_partial_and_activity_veto_then_complete_speech_only_payload():
    with rig() as (bridge, sender, requests, packets, events, _):
        bridge.agent.history['default'] = [{'role': 'user', 'content': 'UNTRANSCRIBED'},
            {'role': 'assistant', 'content': 'invented visual'}]
        raw(sender, packets, '/speech/activity', 1)
        raw(sender, packets, '/transcription', 'Turn me into')
        with pytest.raises(queue.Empty):
            requests.get(timeout=.7)
        raw(sender, packets, '/transcription', 'Turn me into a blue forest.')
        raw(sender, packets, '/trigger', 1)
        with pytest.raises(queue.Empty):
            requests.get(timeout=.1)
        raw(sender, packets, '/speech/activity', 0)
        payload = requests.get(timeout=2)
        assert payload['stream'] is True
        assert payload['options']['num_ctx'] == 128000
        assert payload['messages'][-1]['content'] == 'Turn me into a blue forest.'
        assert {'role': 'user', 'content': 'Turn me into'} in payload['messages']
        assert not any(m['role'] == 'assistant' or 'UNTRANSCRIBED' in m['content']
                       for m in payload['messages'])
        accepted(packets, events)
        assert bridge.agent.history['default'][-2]['content'] == 'Turn me into a blue forest.'
        raw(sender, packets, '/trigger', 1)
        with pytest.raises(queue.Empty):
            requests.get(timeout=.7)


@pytest.mark.parametrize('end', ['timeout', 'superseded'])
def test_real_stall_does_not_block_transcription_or_commit_obsolete_visual(end):
    with rig(budget=.4 if end == 'timeout' else 5) as (
            bridge, sender, requests, packets, events, disconnects):
        raw(sender, packets, '/transcription', 'stall')
        raw(sender, packets, '/trigger', 1)
        requests.get(timeout=2)
        raw(sender, packets, '/transcription', 'stall')
        raw(sender, packets, '/transcription', '')
        if end == 'superseded':
            with bridge.agent.lock:
                raw(sender, packets, '/trigger', 0)
                for text in ('next', 'next speech complete.'):
                    raw(sender, packets, '/transcription', text)
                assert not bridge.agent.history and requests.empty()
        result = event(events, 'error' if end == 'timeout' else 'discard')
        assert result['telemetry']['generation_status'] == end
        assert result['telemetry']['http_elapsed'] < 1
        assert disconnects.get(timeout=1) == b''
        assert not bridge.agent.history and packets.empty()
        if end == 'timeout':
            raw(sender, packets, '/trigger', 0)
            raw(sender, packets, '/transcription', 'next speech complete.')
        raw(sender, packets, '/trigger', 1)
        payload = requests.get(timeout=2)
        assert payload['messages'][-1]['content'] == 'next speech complete.'
        assert {'role': 'user', 'content': 'stall'} in payload['messages']
        assert not any(m['role'] == 'assistant' for m in payload['messages'])
        accepted(packets, events)
        assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())


def test_explicit_five_second_stall_deadline_without_blocking_raw_transcription():
    with rig(budget=5, stall_server_timeout=7) as (
            bridge, sender, requests, packets, events, disconnects):
        assert bridge.generation_budget == 5
        raw(sender, packets, '/transcription', 'stall')
        raw(sender, packets, '/trigger', 1)
        requests.get(timeout=2)
        started = time.monotonic()
        # Prove an explicit whole-request budget, not only an accelerated test.
        # Real redraws continue over UDP while the same HTTP request is blocked.
        with pytest.raises(queue.Empty):
            requests.get(timeout=1)
        raw(sender, packets, '/transcription', 'stall')
        raw(sender, packets, '/transcription', '')
        deadline = time.monotonic() + 5
        result = None
        while time.monotonic() < deadline:
            value = events.get(timeout=5)
            if value['kind'] == 'error':
                result = value
                break
        assert result is not None
        assert result['error'] == 'GenerationTimeout'
        assert 4.7 < result['telemetry']['http_elapsed'] < 6
        assert time.monotonic() - started < 6
        assert disconnects.get(timeout=1) == b''
        assert not bridge.agent.history and packets.empty() and requests.empty()
        print(f'explicit five-second interactive stalled HTTP: {result["telemetry"]["http_elapsed"]:.3f}s; '
              'raw concurrent; no visual/history')
        raw(sender, packets, '/trigger', 0)
        raw(sender, packets, '/transcription', 'new complete speech.')
        raw(sender, packets, '/trigger', 1)
        assert requests.get(timeout=2)['messages'][-1]['content'] == 'new complete speech.'
        accepted(packets, events)


def test_idle_engagement_is_not_speech_or_visual_input_and_yields_to_transcription():
    from mechanical_meanderings.engagement import EngagementState
    from test_engagement import Clock, scene
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    clock.now = 1
    state.observe(scene(2))
    with rig(stall_invitation=True, invitations_enabled=True, engagement=state) as (
            bridge, sender, requests, packets, events, disconnects):
        invitation = requests.get(timeout=2)
        assert 'message' in invitation['format']['properties']
        assert 'scene' in json.loads(invitation['messages'][-1]['content'])
        assert event(events, 'start')['source'] == 'engagement'
        assert not bridge.agent.history and packets.empty()
        raw(sender, packets, '/speech/activity', 1)
        raw(sender, packets, '/transcription', 'Make a')
        discarded = event(events, 'discard')
        assert discarded['source'] == 'engagement'
        assert discarded['telemetry']['generation_status'] == 'superseded'
        assert disconnects.get(timeout=1) == b''
        assert packets.empty() and requests.empty() and not bridge.agent.history
        raw(sender, packets, '/transcription', 'Make a blue forest.')
        raw(sender, packets, '/trigger', 1)
        with pytest.raises(queue.Empty):
            requests.get(timeout=.1)
        raw(sender, packets, '/speech/activity', 0)
        payload = requests.get(timeout=2)
        speech = [m['content'] for m in payload['messages'] if m['role'] == 'user']
        assert set(speech) == {'Make a', 'Make a blue forest.'}
        assert not any(m['role'] == 'assistant' for m in payload['messages'])
        result = accepted(packets, events)
        assert result['source'] == 'completed_speech'
        assert '__invitation__' not in bridge.agent.history

