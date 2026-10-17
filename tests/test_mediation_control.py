"""Bounded mediation against isolated loopback HTTP, never live Ollama/audio."""
import base64
import copy
import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from mechanical_meanderings import GenerationControl, GenerationTimeout, StaleGeneration
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent
from mechanical_meanderings.typography import default_typography


FRAME = base64.b64encode(b'\xff\xd8\xff' + b'synthetic-frame-fixture' * 3 + b'\xff\xd9').decode()


def decision(**changes):
    value = dict(category='transformation', addressed_to_machine=True,
        address_confidence=.95, visual_specificity=.95, transformation_intent=.9,
        preserve_identity=.25, apply_visual=True, prompt='A glass ocean around the original speaker',
        reply='The ocean has arrived.', rationale='A deliberate image request.',
        needs_camera=False, typography=default_typography())
    value.update(changes)
    return value


def invitation():
    return dict(message='Feed the mic a strange idea.', visible_detail='', typography=default_typography())


@pytest.fixture
def http_fixture():
    state = SimpleNamespace(mode='complete', vision_mode='complete', show_mode='complete',
        needs_camera=False, first_delay=0., requests=queue.Queue(), disconnected=queue.Queue(),
        finish=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state.requests.put((self.path, payload))
            show = self.path == '/api/show'
            images = payload.get('messages', [{}])[-1].get('images')
            mode = state.show_mode if show else state.vision_mode if images else state.mode

            def disconnected():
                self.connection.settimeout(2)
                try:
                    data = self.connection.recv(1)
                except socket.timeout:
                    data = 'fixture_socket_timeout'
                except OSError:
                    data = b''  # A reset is also an observed client disconnect.
                state.disconnected.put((self.path, data))

            if mode == 'headers':
                disconnected()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if show else 'application/x-ndjson')
            self.end_headers()
            try:
                if show:
                    if mode == 'tokens':
                        self.wfile.write(b'{"capabilities":[')
                        self.wfile.flush()
                        while not state.finish.wait(.025):
                            self.wfile.write(b' ')
                            self.wfile.flush()
                    else:
                        self.wfile.write(json.dumps({'capabilities': ['vision']}).encode())
                        self.wfile.flush()
                    return
                if not images and state.first_delay:
                    state.finish.wait(state.first_delay)
                value = (invitation() if 'message' in payload['format']['properties'] else
                    decision(needs_camera=state.needs_camera and not images,
                             reply='A current glass cup.' if images else 'The ocean has arrived.'))
                content = json.dumps(value)
                if not payload['stream']:
                    self.wfile.write(json.dumps({'message': {'content': content}}).encode())
                    self.wfile.flush()
                    return

                def send(part, done=False):
                    record = dict(message=dict(content=part), done=done)
                    if done:
                        record.update(eval_count=4, eval_duration=200000000, total_duration=300000000)
                    self.wfile.write(json.dumps(record).encode() + b'\n')
                    self.wfile.flush()

                if mode == 'valid_partial':
                    send(content)
                    disconnected()
                    return
                halfway = len(content) // 2
                send(content[:halfway])
                if mode == 'body':
                    disconnected()
                    return
                if mode == 'tokens':
                    while not state.finish.wait(.025):
                        send(' ')
                    return
                if mode == 'eof':
                    return
                if mode == 'gated':
                    state.finish.wait(2)
                send(content[halfway:halfway + 10])
                send(content[halfway + 10:], done=True)
            except OSError:
                state.disconnected.put((self.path, b''))

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='mediation-http-fixture')
    thread.start()
    assert server.server_port != 11434
    state.agent = MediatingAgent('chosen visual style',
        ollama_url=f'http://127.0.0.1:{server.server_port}', vision_capable=True)
    try:
        yield state
    finally:
        state.finish.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def invoke(agent, operation, **kwargs):
    if operation == 'interpret':
        return agent.interpret('Make an ocean.', **kwargs)
    return agent.invite(None, {'objects': []}, language='English', **kwargs)


def no_watchdogs():
    assert not any(t.name == 'ollama-deadline' for t in threading.enumerate())


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
@pytest.mark.parametrize('mode', ['headers', 'body', 'tokens', 'valid_partial'])
def test_stalls_interrupt_socket_without_committing(http_fixture, operation, mode):
    state = http_fixture
    state.mode = mode
    telemetry = {}
    design = {'unchanged': True}
    kwargs = {'design': design} if operation == 'invite' else {}
    started = time.monotonic()
    with pytest.raises(GenerationTimeout):
        invoke(state.agent, operation, telemetry=telemetry, control=GenerationControl(.2), **kwargs)
    elapsed = time.monotonic() - started
    assert .12 < elapsed < .8
    assert telemetry['generation_status'] == 'timeout'
    assert .1 < telemetry['http_elapsed'] < .8
    assert state.agent.history == {} and state.agent.invitation_number == 0
    assert design == {'unchanged': True}
    assert state.requests.get(timeout=1)[1]['stream'] is True
    assert state.disconnected.get(timeout=1)[1] == b''
    assert state.requests.empty()  # No automatic retries or extra inference calls.
    no_watchdogs()


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_complete_json_stream_and_chosen_options(http_fixture, operation):
    telemetry, design = {}, {}
    settings = Settings(history=10, num_ctx=128000, num_predict=700, temperature=.65, top_p=.8)
    kwargs = {'design': design} if operation == 'invite' else {}
    result = invoke(http_fixture.agent, operation, telemetry=telemetry,
        control=GenerationControl(1), settings=settings, **kwargs)
    if operation == 'interpret':
        assert result.prompt == decision()['prompt']
        assert result.reply == decision()['reply']
        assert len(http_fixture.agent.history['default']) == 2
    else:
        assert result == invitation()['message']
        assert design == default_typography() and http_fixture.agent.invitation_number == 1
        assert not http_fixture.agent.history
    path, payload = http_fixture.requests.get(timeout=1)
    assert path == '/api/chat'
    assert payload['model'] == 'gemma4:12b'
    assert payload['options'] == settings.options()
    assert payload['stream'] is True and payload['think'] is False
    assert telemetry['generation_status'] == 'complete' and telemetry['http_elapsed'] > 0
    assert telemetry['eval_count'] == 4 and telemetry['eval_tokens_per_second'] == 20
    assert http_fixture.agent.timeout == 120
    no_watchdogs()


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_incomplete_stream_is_error_not_partial_json(http_fixture, operation):
    http_fixture.mode = 'eof'
    telemetry = {}
    with pytest.raises(ValueError, match='stream ended before completion'):
        invoke(http_fixture.agent, operation, control=GenerationControl(1), telemetry=telemetry)
    assert telemetry['generation_status'] == 'error' and telemetry['http_elapsed'] > 0
    assert not http_fixture.agent.history and http_fixture.agent.invitation_number == 0
    no_watchdogs()


def test_one_deadline_spans_text_and_vision_and_never_returns_draft(http_fixture):
    state = http_fixture
    state.needs_camera, state.first_delay, state.vision_mode = True, .2, 'body'
    telemetry = {}
    started = time.monotonic()
    with pytest.raises(GenerationTimeout):
        state.agent.interpret('Look at this ocean.', camera=lambda: FRAME,
            control=GenerationControl(.35), telemetry=telemetry, settings=Settings(num_ctx=128000))
    elapsed = time.monotonic() - started
    assert .28 < elapsed < .5  # Resetting the second budget would take at least .55s.
    assert telemetry['generation_status'] == 'timeout'
    assert .28 < telemetry['http_elapsed'] < .5 and telemetry['interpretation_passes'] == 2
    first, second = [state.requests.get(timeout=1)[1] for _ in range(2)]
    assert 'images' not in first['messages'][-1] and second['messages'][-1]['images'] == [FRAME]
    assert first['model'] == second['model'] == 'gemma4:12b'
    assert first['options'] == second['options'] == {'num_ctx': 128000, 'num_predict': 512}
    assert state.disconnected.get(timeout=1)[1] == b'' and state.agent.history == {}
    assert state.requests.empty()
    no_watchdogs()


@pytest.mark.parametrize('operation', ['interpret', 'invite', 'vision', 'discovery'])
def test_cancellation_interrupts_request_and_reports_reason(http_fixture, operation):
    state = http_fixture
    camera_pass = operation in ('vision', 'discovery')
    state.mode = 'complete' if camera_pass else 'body'
    state.needs_camera, state.vision_mode = camera_pass, 'body'
    if operation == 'discovery':
        state.agent.vision_capable, state.show_mode = None, 'headers'
    control, telemetry, outcomes = GenerationControl(3), {}, queue.Queue()

    def worker():
        try:
            kwargs = {'camera': lambda: FRAME} if camera_pass else {}
            outcomes.put(invoke(state.agent, 'interpret' if camera_pass else operation,
                control=control, telemetry=telemetry, **kwargs))
        except Exception as exc:
            outcomes.put(exc)

    thread = threading.Thread(target=worker, name='mediation-cancellation-caller')
    thread.start()
    try:
        state.requests.get(timeout=1)
        if operation == 'vision':
            assert state.requests.get(timeout=1)[1]['messages'][-1]['images'] == [FRAME]
        elif operation == 'discovery':
            assert state.requests.get(timeout=1)[0] == '/api/show'
        started = time.monotonic()
        control.cancel('superseded')
        outcome = outcomes.get(timeout=1)
        assert isinstance(outcome, StaleGeneration)
        assert time.monotonic() - started < .8
        assert telemetry['generation_status'] == 'superseded' and telemetry['http_elapsed'] > 0
        assert state.disconnected.get(timeout=1)[1] == b''
        assert state.agent.history == {} and state.agent.invitation_number == 0
    finally:
        control.cancel('shutdown')
        thread.join(2)
        assert not thread.is_alive()
    no_watchdogs()


@pytest.mark.parametrize('mode', ['headers', 'tokens'])
def test_capability_discovery_is_in_same_socket_budget(http_fixture, mode):
    state = http_fixture
    state.agent.vision_capable = None
    state.needs_camera, state.show_mode = True, mode
    telemetry = {}
    started = time.monotonic()
    with pytest.raises(GenerationTimeout):
        state.agent.interpret('Look.', camera=lambda: FRAME,
            control=GenerationControl(.2), telemetry=telemetry)
    assert time.monotonic() - started < .8
    assert [state.requests.get(timeout=1)[0] for _ in range(2)] == ['/api/chat', '/api/show']
    assert state.disconnected.get(timeout=1) == ('/api/show', b'')
    assert telemetry['generation_status'] == 'timeout' and telemetry['http_elapsed'] > .1
    assert not state.agent.history
    no_watchdogs()


def test_successful_discovery_and_vision_do_not_commit_images_or_speech_ledger(http_fixture):
    state = http_fixture
    state.agent.vision_capable, state.needs_camera = None, True
    visual = [dict(role='user', content='old accepted visual'), dict(role='assistant', content='old prompt')]
    state.agent.history['default'] = visual
    ledger = [dict(role='user', content='an earlier completed speech turn')]
    telemetry = {}
    result = state.agent.interpret('Look.', camera=lambda: FRAME, history_override=ledger,
        commit=False, control=GenerationControl(1), telemetry=telemetry)
    assert result.reply == 'A current glass cup.' and not result.needs_camera
    requests = [state.requests.get(timeout=1) for _ in range(3)]
    assert [path for path, _ in requests] == ['/api/chat', '/api/show', '/api/chat']
    assert requests[1][1] == {'model': 'gemma4:12b'}
    for _, payload in (requests[0], requests[2]):
        assert payload['messages'][1:-1] == ledger
        assert 'old accepted visual' not in json.dumps(payload)
    assert state.agent.history['default'] is visual
    assert FRAME not in json.dumps(state.agent.history) + json.dumps(telemetry)
    assert telemetry['generation_status'] == 'complete' and telemetry['interpretation_passes'] == 2
    no_watchdogs()


@pytest.mark.parametrize('text', ['An ocean.', 'Stop talking.', 'You can reply again.'])
@pytest.mark.parametrize('override', [None, [], [dict(role='user', content='earlier completed speech')]])
def test_noncommit_never_resets_or_writes_history_even_mute_paths(text, override):
    calls = []

    def transport(url, payload, timeout):
        calls.append(payload)
        return {'message': {'content': json.dumps(decision())}}

    agent = MediatingAgent('style', transport=transport, reset_after=1, clock=lambda: 100.)
    visual = [dict(role='user', content='old visual'), dict(role='assistant', content='old prompt')]
    agent.history['default'], agent.last_activity['default'], agent.epochs['default'] = visual, 0., 7
    before = copy.deepcopy(agent.history)
    result = agent.interpret(text, expected_epoch=7, history_override=override, commit=False)
    assert agent.history == before and agent.history['default'] is visual
    assert agent.epochs['default'] == 7
    if text == 'An ocean.':
        assert result.prompt
        assert calls[0]['messages'][1:-1] == (visual if override is None else override)
    else:
        assert not calls and not result.prompt and not result.reply


def test_override_is_not_truncated_by_visual_history_and_does_not_pollute_it():
    calls = []
    agent = MediatingAgent('style', transport=lambda url, payload, timeout:
        calls.append(copy.deepcopy(payload)) or {'message': {'content': json.dumps(decision())}},
        history_turns=1, reset_after=1, clock=lambda: 100.)
    agent.history['default'] = [dict(role='user', content='visual'), dict(role='assistant', content='prompt')]
    agent.last_activity['default'], agent.epochs['default'] = 0., 4
    ledger = [dict(role='user', content=f'completed speech {i}') for i in range(20)]
    agent.interpret('new speech', history_override=ledger, settings=Settings(history=0), commit=False)
    assert calls[-1]['messages'][1:-1] == ledger
    agent.interpret('accepted speech', expected_epoch=4, history_override=ledger, commit=True)
    assert calls[-1]['messages'][1:-1] == ledger and agent.epochs['default'] == 4
    assert [m['role'] for m in agent.history['default']] == ['user', 'assistant']
    assert 'completed speech' not in json.dumps(agent.history)
    assert agent.history['default'][0]['content'] == 'accepted speech'


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_injected_three_arg_transport_stays_synchronous_and_is_not_retried(operation):
    calls, caller = [], threading.get_ident()

    def legacy(url, payload, timeout):
        calls.append((threading.get_ident(), copy.deepcopy(payload), timeout))
        return {'message': {'content': json.dumps(decision() if operation == 'interpret' else invitation())}}

    agent = MediatingAgent('style', transport=legacy, timeout=9)
    invoke(agent, operation, control=GenerationControl(1))
    assert len(calls) == 1 and calls[0][0] == caller and calls[0][2] == 9
    assert calls[0][1]['stream'] is False
    no_watchdogs()


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_slow_legacy_return_is_checked_without_abandoning_inference_thread(operation):
    calls, caller, telemetry = [], threading.get_ident(), {}

    def legacy(url, payload, timeout):
        calls.append(threading.get_ident())
        time.sleep(.07)
        return {'message': {'content': json.dumps(decision() if operation == 'interpret' else invitation())}}

    agent = MediatingAgent('style', transport=legacy)
    with pytest.raises(GenerationTimeout):
        invoke(agent, operation, control=GenerationControl(.03), telemetry=telemetry)
    assert calls == [caller]
    assert telemetry['generation_status'] == 'timeout' and telemetry['http_elapsed'] >= .06
    assert not agent.history and agent.invitation_number == 0
    no_watchdogs()


def test_explicit_control_injection_gets_same_control_on_all_passes():
    controls = []
    control = GenerationControl(1)

    def cooperative(url, payload, timeout, *, control):
        controls.append(control)
        control.check()
        value = decision(needs_camera=len(controls) == 1) if 'format' in payload else {'capabilities': ['vision']}
        return {'message': {'content': json.dumps(value)}} if 'format' in payload else value

    agent = MediatingAgent('style', transport=cooperative, capability_transport=cooperative)
    agent.interpret('Look.', control=control, camera=lambda: FRAME, commit=False)
    assert controls == [control, control, control] and not agent.history


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_explicit_control_cancellation_is_not_swallowed(operation):
    control, telemetry = GenerationControl(1), {}

    def cooperative(url, payload, timeout, *, control):
        control.cancel('reset')
        # Even an injection returning a complete JSON after cancellation is rejected.
        return {'message': {'content': json.dumps(decision() if operation == 'interpret' else invitation())}}

    agent = MediatingAgent('style', transport=cooperative)
    with pytest.raises(StaleGeneration):
        invoke(agent, operation, control=control, telemetry=telemetry)
    assert telemetry['generation_status'] == 'reset'
    assert not agent.history and agent.invitation_number == 0


@pytest.mark.parametrize('preference', ['mute', 'disco'])
def test_receiver_preferences_do_not_wait_for_worker_holding_agent_lock(http_fixture, preference):
    state = http_fixture
    state.mode = 'gated'
    control, outcomes = GenerationControl(1), queue.Queue()

    def worker():
        try:
            with state.agent.lock:  # The bridge serializes worker-owned history acceptance.
                outcomes.put(state.agent.interpret('An ocean.', control=control, commit=False))
        except Exception as exc:
            outcomes.put(exc)

    thread = threading.Thread(target=worker, name='mediation-locked-worker')
    thread.start()
    try:
        state.requests.get(timeout=1)
        started = time.monotonic()
        if preference == 'mute':
            assert state.agent.set_speech_preference('Stop talking.') is True
            assert state.agent.speech_muted
        else:
            assert state.agent.set_disco(True)
            assert not state.agent.set_disco(True)
            assert state.agent.style_version == 1
        assert time.monotonic() - started < .15
        state.finish.set()
        result = outcomes.get(timeout=1)
        if preference == 'mute':
            assert not isinstance(result, Exception) and not result.reply and result.prompt
        else:
            assert isinstance(result, StaleGeneration)
        assert state.agent.history == {}
    finally:
        state.finish.set()
        control.cancel('shutdown')
        thread.join(2)
        assert not thread.is_alive()
    no_watchdogs()


def test_reset_preserves_channel_mute_and_style_but_full_reset_unmutes():
    agent = MediatingAgent('style')
    agent.set_speech_preference('Stop talking.')
    agent.set_disco(True)
    agent.history['default'] = [dict(role='user', content='old')]
    agent.reset('default')
    assert not agent.history and agent.speech_muted
    assert agent.disco and agent.style_version == 1
    agent.reset()
    assert not agent.speech_muted and agent.disco and agent.style_version == 1


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_uncontrolled_direct_http_keeps_nonstreaming_contract(http_fixture, operation):
    invoke(http_fixture.agent, operation)
    assert http_fixture.requests.get(timeout=1)[1]['stream'] is False
    no_watchdogs()


def test_transport_timeout_shorter_than_control_budget_cannot_return_vision_fallback(http_fixture):
    state = http_fixture
    state.needs_camera, state.vision_mode, state.agent.timeout = True, 'body', .08
    telemetry = {}
    with pytest.raises(GenerationTimeout):
        state.agent.interpret('Look.', camera=lambda: FRAME,
            control=GenerationControl(1), telemetry=telemetry)
    assert telemetry['generation_status'] == 'timeout' and telemetry['interpretation_passes'] == 2
    assert state.disconnected.get(timeout=1)[1] == b'' and not state.agent.history
    no_watchdogs()


@pytest.mark.parametrize('bounded', [False, True])
def test_camera_timeout_is_fallback_only_for_uncontrolled_calls(bounded):
    agent = MediatingAgent('style', vision_capable=True, transport=lambda *args:
        {'message': {'content': json.dumps(decision(needs_camera=True))}})
    telemetry = {}

    def camera():
        raise TimeoutError('synthetic camera timeout')

    if bounded:
        with pytest.raises(GenerationTimeout):
            agent.interpret('Look.', camera=camera, control=GenerationControl(1), telemetry=telemetry)
        assert not agent.history and telemetry['generation_status'] == 'timeout'
    else:
        result = agent.interpret('Look.', camera=camera, telemetry=telemetry)
        assert result.prompt == decision()['prompt'] and telemetry['frame_status'] == 'timeout'


@pytest.mark.parametrize('operation', ['interpret', 'invite'])
def test_control_is_checked_after_parsing_before_commit_or_return(monkeypatch, operation):
    from mechanical_meanderings import mediation
    control, telemetry = GenerationControl(1), {}
    name = 'parse_decision' if operation == 'interpret' else 'brief_invitation'
    original = getattr(mediation, name)

    def parse_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        control.cancel('shutdown')
        return result

    monkeypatch.setattr(mediation, name, parse_and_cancel)
    agent = MediatingAgent('style', transport=lambda *args:
        {'message': {'content': json.dumps(decision() if operation == 'interpret' else invitation())}})
    with pytest.raises(StaleGeneration):
        invoke(agent, operation, control=control, telemetry=telemetry)
    assert telemetry['generation_status'] == 'shutdown'
    assert not agent.history and agent.invitation_number == 0


def test_precancelled_mute_call_does_not_change_preference_or_history():
    agent = MediatingAgent('style', transport=lambda *args: pytest.fail('model must not be called'))
    control, telemetry = GenerationControl(1), {}
    control.cancel('reset')
    with pytest.raises(StaleGeneration):
        agent.interpret('Stop talking.', control=control, commit=False, telemetry=telemetry)
    assert not agent.speech_muted and not agent.history
    assert telemetry['generation_status'] == 'reset' and telemetry['http_elapsed'] == 0


def test_empty_override_suppresses_idle_reset_for_committing_calls():
    payloads = []
    agent = MediatingAgent('style', reset_after=1, clock=lambda: 100., transport=lambda url, payload, timeout:
        payloads.append(payload) or {'message': {'content': json.dumps(decision())}})
    agent.history['default'] = [dict(role='user', content='older visual'), dict(role='assistant', content='older prompt')]
    agent.epochs['default'], agent.last_activity['default'] = 3, 0.
    agent.interpret('new speech', history_override=[], expected_epoch=3)
    assert payloads[0]['messages'][1:-1] == [] and agent.epochs['default'] == 3
    assert len(agent.history['default']) == 4
