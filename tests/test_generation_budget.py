"""Isolated real HTTP sockets, never the user's Ollama service."""
import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mechanical_meanderings import Agent, GenerationControl, GenerationTimeout, StaleGeneration
from mechanical_meanderings.final_turn import FinalTurnBridge
from mechanical_meanderings.boundary import BoundaryBridge


@pytest.fixture
def http_fixture():
    requests = queue.Queue()
    disconnected = queue.Queue()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            text = payload['messages'][-1]['content']
            requests.put(payload)
            if text == 'headers':
                self.connection.settimeout(2)
                disconnected.put(self.connection.recv(1))
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.end_headers()

            def send(record):
                self.wfile.write(json.dumps(record).encode() + b'\n')
                self.wfile.flush()

            try:
                send({'message': {'content': 'private partial '}, 'done': False})
                if text == 'tokens':
                    while not release.wait(.025):
                        send({'message': {'content': 'token '}, 'done': False})
                elif text == 'stall':
                    self.connection.settimeout(2)
                    disconnected.put(self.connection.recv(1))
                    return
                else:
                    send({'message': {'content': 'finished'}, 'done': True,
                          'eval_count': 2, 'eval_duration': 100000000,
                          'total_duration': 200000000})
            except (OSError, socket.timeout):
                disconnected.put(b'')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    agent = Agent('system', ollama_url='http://127.0.0.1:%s' % server.server_port)
    yield agent, requests, disconnected
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(2)


@pytest.mark.parametrize('text', ['stall', 'headers', 'tokens'])
def test_total_budget_interrupts_real_socket_and_never_commits(http_fixture, text):
    agent, requests, disconnected = http_fixture
    telemetry = {}
    start = time.monotonic()
    with pytest.raises(GenerationTimeout):
        agent.generate(text, control=GenerationControl(.2), telemetry=telemetry)
    elapsed = time.monotonic() - start
    assert .15 < elapsed < .8
    assert telemetry['generation_status'] == 'timeout'
    assert telemetry['http_elapsed'] < .8
    assert agent.history == {}
    assert requests.get(timeout=1)['stream'] is True
    assert disconnected.get(timeout=1) == b''
    assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())


def test_complete_stream_buffers_and_keeps_final_telemetry(http_fixture):
    agent, requests, disconnected = http_fixture
    telemetry = {}
    assert agent.generate('success', control=GenerationControl(1), telemetry=telemetry) == 'private partial finished'
    assert telemetry['generation_status'] == 'complete'
    assert telemetry['eval_count'] == 2
    assert telemetry['eval_tokens_per_second'] == 20
    assert agent.timeout == 120


@pytest.mark.parametrize('bridge_type', [FinalTurnBridge, BoundaryBridge])
@pytest.mark.parametrize('action', ['superseded', 'reset', 'shutdown', 'timeout'])
def test_worker_interrupt_and_newest_proceeds(http_fixture, bridge_type, action):
    agent, requests, disconnected = http_fixture
    events = queue.Queue()
    bridge = bridge_type(agent, listen_host='127.0.0.1', listen_port=0,
                        generation_budget=.3 if action == 'timeout' else 5,
                        on_event=events.put)
    sent = []
    bridge.client.send_message = lambda address, value: sent.append((address, value))
    bridge.configure(num_ctx=16384, num_predict=96)

    def event(kind):
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            value = events.get(timeout=1.5)
            if value['kind'] == kind:
                return value
        raise AssertionError(kind)

    bridge.start()
    try:
        bridge.submit('stall')
        payload = requests.get(timeout=1)
        assert payload['model'] == 'gemma4:12b'
        assert payload['options']['num_ctx'] == 16384
        assert payload['options']['num_predict'] == 96
        started = time.monotonic()
        if action == 'superseded':
            bridge.set_paused(True)
            bridge.receive('/final', 'success')
        elif action == 'reset':
            bridge.receive('/reset', 1)
        elif action == 'shutdown':
            bridge.close()
        result = event('error' if action == 'timeout' else 'discard')
        assert time.monotonic() - started < .8
        assert result['telemetry']['generation_status'] == action
        assert result['output'] is None if action == 'timeout' else 'output' not in result
        assert disconnected.get(timeout=1) == b''
        assert agent.history == {}
        assert sent == []
        if action != 'shutdown':
            bridge.set_paused(False)
            if action != 'superseded':
                bridge.submit('success')
            requests.get(timeout=1)
            event('result')
            assert sent == [('/prompt', 'private partial finished'), ('/prompt/trigger', 1)]
            assert 'stall' not in str(agent.history)
        else:
            assert not bridge.worker.is_alive()
    finally:
        bridge.close()
    assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())


def test_legacy_and_cooperative_injected_transports_keep_contract():
    calls = []
    def legacy(url, payload, timeout):
        calls.append((payload, timeout))
        return {'message': {'content': 'legacy'}}
    assert Agent('s', transport=legacy, timeout=9).generate('x', control=GenerationControl(1)) == 'legacy'
    assert calls[0][0]['stream'] is False and calls[0][1] == 9
    def cooperative(url, payload, timeout, *, control):
        control.cancel('superseded')
        control.check()
    with pytest.raises(StaleGeneration):
        Agent('s', transport=cooperative).generate('x', control=GenerationControl(1))
