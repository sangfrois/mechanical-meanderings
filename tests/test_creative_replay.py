"""Resident-model guards exercised through fake urllib, never live Ollama."""
import http.client
import io
import json
import urllib.error

import pytest

from scripts import creative_replay
from test_creative_direction import compact, response


URL = 'http://replay.invalid:11434'
MODEL = 'gemma4:12b'
CONTEXT = 128000
MESSAGES = (
    'Invent a landscape for the mic; this machine needs a holiday.',
    'Tell the microphone which impossible material should cover the scene.',
    'Describe a peculiar creature into the mic and challenge my imagination.',
    'Give the mic a visual paradox worth bending these pixels around.',
)


class Reply(io.BytesIO):
    def __init__(self, value, status=200):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())
        self.status = status


def resident(context=CONTEXT):
    return dict(models=[dict(name=MODEL, context_length=context)])


def mocked_replay(monkeypatch, *, fail_check=None, failure=None, status=200):
    calls, payloads = [], []
    checks, invitations = 0, 0

    def ps(url, timeout):
        nonlocal checks
        assert url == URL + '/api/ps' and timeout == 3
        calls.append('ps')
        checks += 1
        if checks == fail_check:
            if isinstance(failure, Exception):
                raise failure
            return Reply(failure, status=status)
        return Reply(resident())

    def chat(url, payload, timeout):
        nonlocal invitations
        assert url == URL + '/api/chat'
        calls.append('chat')
        payloads.append(payload)
        if 'approach' in payload['format']['properties']:
            message = MESSAGES[invitations]
            invitations += 1
            return response(dict(message=message, approach='distinct synthetic premise',
                                 visible_detail='', typography={}))
        return response(compact())

    monkeypatch.setattr(creative_replay.urllib.request, 'urlopen', ps)
    monkeypatch.setattr(creative_replay, 'http_chat', chat)
    return calls, payloads


def test_every_speech_and_invitation_checks_residency_before_chat(monkeypatch, tmp_path):
    calls, payloads = mocked_replay(monkeypatch)
    result = creative_replay.run(tmp_path / 'replay.json', url=URL, model=MODEL)
    assert len(result['speech']) == len(creative_replay.SPEECH)
    assert len(result['invitations']) == len(MESSAGES)
    assert all('error' not in row for row in result['invitations'])
    assert calls == ['ps'] + ['ps', 'chat'] * (len(creative_replay.SPEECH) + len(MESSAGES))
    assert all(p['model'] == MODEL and p['options']['num_ctx'] == CONTEXT for p in payloads)
    assert result['context_length'] == CONTEXT and result['history'] == 25
    assert [m['content'] for m in payloads[len(creative_replay.SPEECH) - 1]['messages']
            if m['role'] == 'user'] == list(creative_replay.SPEECH)


@pytest.mark.parametrize('request_number', [1, 3, 7, 9])
@pytest.mark.parametrize('failure', [dict(models=[]), resident(CONTEXT // 2)], ids=['evicted', 'context_changed'])
def test_eviction_or_context_change_aborts_entire_batch_before_chat(monkeypatch, tmp_path, request_number, failure):
    calls, payloads = mocked_replay(monkeypatch, fail_check=request_number + 1, failure=failure)
    output = tmp_path / 'replay.json'
    with pytest.raises(RuntimeError):
        creative_replay.run(output, url=URL, model=MODEL)
    assert calls == ['ps'] + ['ps', 'chat'] * (request_number - 1) + ['ps']
    assert len(payloads) == request_number - 1  # No retries, implicit reload or skipped invitation.
    if request_number > 1:
        saved = json.loads(output.read_text(encoding='utf-8'))
        assert len(saved['speech']) + len(saved['invitations']) == request_number - 1


@pytest.mark.parametrize('failure,status', [
    (resident(), 503),
    (b'{bad JSON', 200),
    (b'\xff', 200),
    ([], 200),
    ({}, 200),
    (dict(models=None), 200),
    (dict(models=[None]), 200),
    (dict(models=[{}, *resident()['models']]), 200),
    (dict(models=[dict(name=''), *resident()['models']]), 200),
    (dict(models=[dict(name=MODEL)]), 200),
    (resident(True), 200),
    (resident('128000'), 200),
    (resident(0), 200),
    (resident(-1), 200),
    (resident(1.5), 200),
    (dict(models=[dict(name=MODEL, context_length=CONTEXT)] * 2), 200),
    (urllib.error.URLError('synthetic connection failure'), 200),
    (urllib.error.HTTPError(URL + '/api/ps', 503, 'synthetic status', {}, None), 200),
    (TimeoutError('synthetic timeout'), 200),
    (http.client.IncompleteRead(b'{"models":'), 200),
])
@pytest.mark.parametrize('fail_check', [1, 8], ids=['initial', 'invitation'])
def test_raw_ps_errors_and_malformed_responses_fail_closed(monkeypatch, tmp_path, failure, status, fail_check):
    calls, payloads = mocked_replay(monkeypatch, fail_check=fail_check, failure=failure, status=status)
    with pytest.raises(RuntimeError):
        creative_replay.run(tmp_path / 'replay.json', url=URL, model=MODEL)
    successful = max(0, fail_check - 2)
    assert len(payloads) == successful
    assert calls == (['ps'] if fail_check == 1 else ['ps'] + ['ps', 'chat'] * successful + ['ps'])
