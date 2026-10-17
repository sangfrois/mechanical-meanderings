"""Editable invitation controls: real production payload/HTTP/OSC, synthetic model text.

These tests prove editability and transport, not a model's sense of humour.
"""
import hashlib
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from mechanical_meanderings import StaleGeneration
from mechanical_meanderings.__main__ import main
from mechanical_meanderings.engagement import EngagementState
from mechanical_meanderings.invitation import InvitationPrompt, InvitationPromptError, KINDS, MAX_PROMPT_BYTES
from mechanical_meanderings.invite_preview import preview_invitations
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent
from mechanical_meanderings.personality import INTERACTIVE_REFERENCE, PERSONALITY
from test_engagement import Clock, scene
from test_interactive import harness, event_of
from test_mediation import judgement


MESSAGE = 'Parle au micro pour changer cette image. Mon goût douteux a besoin de directives.'


def answer(message=MESSAGE):
    return {'message': {'content': json.dumps(dict(message=message, approach='machine with questionable taste',
        visible_detail='', typography={}))}, 'done': True, 'eval_count': 42, 'eval_duration': 100000000}


@pytest.fixture
def prompt_path(tmp_path):
    path = tmp_path / 'invitation.md'
    path.write_text('Operator voice A. No secret extra instructions.', encoding='utf-8')
    return path


@pytest.mark.parametrize('kind', KINDS)
def test_full_system_override_has_no_python_style_tail_and_preserves_context(prompt_path, kind):
    requests = []
    text = {'nudge': MESSAGE, 'embodied': MESSAGE, 'invitation': 'Tell the mic what to change.',
            'aside': 'The machine has questionable taste.'}[kind]
    agent = MediatingAgent('VISUAL_ONLY_SENTINEL', invitation_prompt=prompt_path,
        transport=lambda url, payload, timeout: requests.append(payload) or answer(text))
    telemetry = {}
    earlier = [f'earlier line {i}' for i in range(15)]
    agent.invite(None, dict(recent_speech=['actual speech'], recent_invitation_approaches=['prior angle']),
        kind=kind, language='French (Quebec)', previous=earlier, telemetry=telemetry,
        settings=Settings(num_ctx=8192, history=25, temperature=.9, num_predict=300))
    payload = requests[0]
    snapshot = InvitationPrompt(prompt_path).read()
    assert payload['messages'][0] == dict(role='system', content=snapshot.text)
    assert len(payload['messages']) == 2
    assert 'VISUAL_ONLY_SENTINEL' not in json.dumps(payload)
    assert PERSONALITY not in payload['messages'][0]['content']
    context = json.loads(payload['messages'][1]['content'])
    assert context['kind'] == kind and context['language'] == 'French (Quebec)'
    assert context['recent_speech'] == ['actual speech']
    assert context['recent_invitations'] == earlier[-12:]
    assert context['recent_invitation_approaches'] == ['prior angle']
    assert 'action' not in context and 'images' not in payload['messages'][1]
    assert payload['options'] == dict(num_ctx=8192, temperature=.9, num_predict=300)
    assert telemetry['invitation_prompt'] == snapshot.metadata()
    assert not agent.history and not agent.render_context


def test_same_agent_hot_reloads_atomic_editor_save_without_resetting_speech(prompt_path):
    requests = []
    agent = MediatingAgent(INTERACTIVE_REFERENCE, invitation_prompt=prompt_path,
        transport=lambda url, payload, timeout: requests.append(payload) or answer())
    agent.history['default'] = [dict(role='user', content='Earlier participant words.')]
    old_history = list(agent.history['default'])
    first = {}
    agent.invite(None, {}, kind='nudge', telemetry=first)
    replacement = prompt_path.with_suffix('.replacement')
    replacement.write_text('Operator voice B: blunt and entirely different.', encoding='utf-8')
    replacement.replace(prompt_path)
    second = {}
    agent.invite(None, {}, kind='embodied', telemetry=second)
    assert requests[0]['messages'][0]['content'].startswith('Operator voice A')
    assert requests[1]['messages'][0]['content'].startswith('Operator voice B')
    assert first['invitation_prompt']['sha256'] != second['invitation_prompt']['sha256']
    assert agent.history['default'] == old_history
    assert agent.invitation_number == 2 and agent.model == 'gemma4:12b'


@pytest.mark.parametrize('broken', ['empty', 'missing', 'invalid_utf8', 'oversized'])
def test_invalid_edit_skips_model_without_secret_fallback_and_recovers(prompt_path, broken):
    requests = []
    agent = MediatingAgent('style', invitation_prompt=prompt_path,
        transport=lambda *args: requests.append(args) or answer())
    if broken == 'missing':
        prompt_path.unlink()
    else:
        prompt_path.write_bytes({'empty': b' \n', 'invalid_utf8': b'\xff\xfe',
                                 'oversized': b'x' * (MAX_PROMPT_BYTES + 1)}[broken])
    with pytest.raises(InvitationPromptError):
        agent.invite(None, {}, kind='nudge')
    assert not requests and not agent.invitation_number
    prompt_path.write_text('Recovered operator voice.', encoding='utf-8')
    assert agent.invite(None, {}, kind='nudge') == MESSAGE
    assert requests[0][1]['messages'][0]['content'] == 'Recovered operator voice.'


def test_bom_whitespace_and_content_fingerprint(prompt_path):
    prompt_path.write_bytes(b'\xef\xbb\xbf  A UTF-8 prompt.\r\n')
    snapshot = InvitationPrompt(prompt_path).read()
    assert snapshot.text == 'A UTF-8 prompt.'
    assert snapshot.sha256 == hashlib.sha256(snapshot.text.encode()).hexdigest()


def test_edit_during_inference_discards_without_partial_design_commit(prompt_path):
    def transport(*args):
        prompt_path.write_text('Changed during generation.', encoding='utf-8')
        return answer()
    agent = MediatingAgent('style', invitation_prompt=prompt_path, transport=transport)
    design, telemetry = {'untouched': True}, {}
    with pytest.raises(StaleGeneration, match='prompt changed'):
        agent.invite(None, {}, kind='nudge', design=design, telemetry=telemetry)
    assert design == {'untouched': True} and not agent.invitation_number
    assert telemetry['generation_status'] == 'superseded'


def ready_state():
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1, invite_cooldown=10)
    state.observe(scene(1))
    clock.now = 1
    state.observe(scene(2))
    return state


@pytest.mark.parametrize('boundary_speech', [False, True])
def test_real_idle_worker_uses_custom_file_without_visual_or_speech_contamination(prompt_path, boundary_speech):
    requests = []
    with harness(lambda url, payload, timeout: requests.append(payload) or answer(),
            agent_options=dict(invitation_prompt=prompt_path), engagement=ready_state(),
            boundary_speech=boundary_speech) as (bridge, sender, packets, events):
        result = event_of(events, 'result')
        address, args = packets.get(timeout=1)
        assert address == '/engagement/message' and args[:2] == (MESSAGE, 'challenge')
        assert result['telemetry']['invitation_prompt'] == InvitationPrompt(prompt_path).read().metadata()
        assert requests[0]['messages'][0]['content'] == InvitationPrompt(prompt_path).read().text
        assert not bridge.agent.history and not bridge.completed
        with pytest.raises(queue.Empty):
            packets.get(timeout=.1)


def test_prompt_read_at_send_does_not_block_raw_speech(monkeypatch, prompt_path):
    began, release = threading.Event(), threading.Event()
    original, reads = InvitationPrompt.read, [0]
    def delayed_read(self):
        reads[0] += 1
        if reads[0] == 3:  # Start, model completion, then bridge pre-send check.
            began.set()
            assert release.wait(2)
        return original(self)
    monkeypatch.setattr(InvitationPrompt, 'read', delayed_read)
    with harness(lambda *args: answer(), agent_options=dict(invitation_prompt=prompt_path),
            engagement=ready_state(), boundary_speech=True) as (bridge, sender, packets, events):
        try:
            assert began.wait(1)
            sender.send_message('/transcription', 'new live speech')
            assert packets.get(timeout=.5) == ('/transcription', ('new live speech',))
        finally:
            release.set()
        discarded = event_of(events, 'discard')
        assert discarded['channel'] == '__invitation__'
        assert not bridge.recent_invitations


def test_bridge_rejects_edit_after_agent_returns_before_send(monkeypatch, prompt_path):
    original = MediatingAgent.invite
    def changed_after_return(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        prompt_path.write_text('Changed after model completion.', encoding='utf-8')
        return result
    monkeypatch.setattr(MediatingAgent, 'invite', changed_after_return)
    with harness(lambda *args: answer(), agent_options=dict(invitation_prompt=prompt_path),
            engagement=ready_state()) as (bridge, sender, packets, events):
        assert event_of(events, 'discard')['channel'] == '__invitation__'
        assert not bridge.recent_invitations
        with pytest.raises(queue.Empty):
            packets.get(timeout=.1)


def test_print_cli_is_exact_system_message_and_needs_no_runtime(monkeypatch, prompt_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Runtime/model must not be constructed by prompt inspection')
    monkeypatch.setattr(MediatingAgent, '__init__', forbidden)
    monkeypatch.setattr('mechanical_meanderings.bridge.Bridge.__init__', forbidden)
    assert main(['--print-invite-prompt', '--invite-prompt', str(prompt_path),
                 '--ollama-url', 'http://127.0.0.1:1']) == 0
    out = capsys.readouterr()
    assert out.out == InvitationPrompt(prompt_path).read().text + '\n'
    assert str(prompt_path) in out.err and 'sha256=' in out.err


def test_cli_custom_invite_does_not_replace_visual_reference(monkeypatch, prompt_path):
    seen = []
    def interpret(self, text, **kwargs):
        seen.append((self.system_prompt, self.invitation_prompt.read().text))
        return SimpleNamespace(metadata=lambda: dict(reply='synthetic fixture'))
    monkeypatch.setattr(MediatingAgent, 'interpret', interpret)
    assert main(['--interactive', '--text', 'how does this work?', '--invite-prompt', str(prompt_path)]) == 0
    assert seen == [(INTERACTIVE_REFERENCE, InvitationPrompt(prompt_path).read().text)]


def test_cli_missing_invite_fails_before_bridge_or_model(monkeypatch, prompt_path, capsys):
    prompt_path.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError('Must validate before sockets or inference')
    monkeypatch.setattr('mechanical_meanderings.interactive.InteractiveBridge.__init__', forbidden)
    monkeypatch.setattr(MediatingAgent, 'invite', forbidden)
    assert main(['--interactive', '--invite-prompt', str(prompt_path)]) == 1
    assert 'Cannot read invitation prompt' in capsys.readouterr().err


@pytest.fixture
def ollama_fixture():
    """Isolated actual HTTP server; all returned wording is explicitly synthetic."""
    state = dict(calls=[], payloads=[], models=[dict(name='gemma4:12b', context_length=8192)],
                 messages=[MESSAGE], ps_hook=None)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def send_json(self, value):
            data = (json.dumps(value) + '\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            assert self.path == '/api/ps'
            state['calls'].append('ps')
            if state['ps_hook'] is not None:
                state['ps_hook'](state)
            self.send_json(dict(models=state['models']))
        def do_POST(self):
            assert self.path == '/api/chat'
            state['calls'].append('chat')
            state['payloads'].append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_json(answer(state['messages'][min(len(state['payloads']) - 1, len(state['messages']) - 1)]))
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def test_preview_cli_real_http_reads_residency_captures_rejection_and_opens_no_osc(
        monkeypatch, prompt_path, capsys, ollama_fixture):
    url, state = ollama_fixture
    state['messages'] = [MESSAGE, MESSAGE]  # Exact repeat must remain visible as a rejected sample.
    def forbidden(*args, **kwargs):
        raise AssertionError('Preview must never create the installation bridge')
    monkeypatch.setattr('mechanical_meanderings.interactive.InteractiveBridge.__init__', forbidden)
    code = main(['--preview-invite', '--invite-prompt', str(prompt_path), '--invite-count', '2',
                 '--invitation-language', 'French (Quebec)', '--seed', '731', '--ollama-url', url])
    assert code == 1
    result = json.loads(capsys.readouterr().out)
    assert len(result['samples']) == 2
    first, second = result['samples']
    assert result['status'] == 'complete' and result['terminal_error'] is None
    assert first['accepted'] is True and first['message'] == MESSAGE
    assert second['accepted'] is False and second['error'] == 'repeated challenge'
    assert json.loads(second['raw_response'])['message'] == MESSAGE
    assert first['payload']['messages'][0]['content'] == InvitationPrompt(prompt_path).read().text
    assert first['telemetry']['eval_count'] == 42
    assert state['calls'] == ['ps', 'ps', 'chat', 'ps', 'chat']
    assert [p['options']['seed'] for p in state['payloads']] == [731, 732]
    assert all(p['options']['num_ctx'] == 8192 for p in state['payloads'])
    assert json.loads(state['payloads'][1]['messages'][-1]['content'])['recent_invitations'] == [MESSAGE]
    assert all(p['stream'] is True and p['think'] is False for p in state['payloads'])
    assert first['payload'] == state['payloads'][0]
    assert first['payload_kind'] == 'wire' and first['logical_payload']['stream'] is False


@pytest.mark.parametrize('failure', ['not_resident', 'unknown_context', 'changed_context', 'requested_context'])
def test_preview_aborts_without_loading_or_changing_context(prompt_path, ollama_fixture, failure):
    url, state = ollama_fixture
    settings = Settings()
    if failure == 'not_resident':
        state['models'] = []
    elif failure == 'unknown_context':
        state['models'][0].pop('context_length')
    elif failure == 'changed_context':
        state['ps_hook'] = lambda s: s['models'][0].update(
            context_length=8192 if len(s['calls']) == 1 else 4096)
    else:
        settings = Settings(num_ctx=4096)
    agent = MediatingAgent('style', invitation_prompt=prompt_path, ollama_url=url)
    original = agent.transport
    result = preview_invitations(agent, settings)
    assert result['status'] == 'aborted'
    assert result['terminal_error']['type'] == 'RuntimeError'
    assert not state['payloads'] and agent.transport is original
