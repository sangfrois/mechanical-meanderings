"""Invitation rejections and mic-boilerplate repetition, all wording synthetic."""
import json
import logging
import queue

import pytest

from mechanical_meanderings.invitation import InvitationRejected
from mechanical_meanderings.mediation import MediatingAgent, repeated_invitation
from test_interactive import harness, event_of
from test_invitation_prompt import answer, MESSAGE, prompt_path, ready_state


@pytest.mark.parametrize('changes,previous,code', [
    ({'message': None}, [], 'invalid_message_type'),
    ({'message': 'Speak mic.'}, [], 'word_limit'),
    ({'message': 'Speak into the mic. ' + 'x ' * 21}, [], 'word_limit'),
    ({'message': 'Tell the mic ' + 'x' * 161}, [], 'character_limit'),
    ({'message': 'Describe an image. This software has questionable taste.'}, [], 'missing_microphone'),
    ({'message': MESSAGE}, [MESSAGE], 'repeated'),
    ({'visible_detail': None}, [], 'invalid_visible_detail'),
    ({'approach': None}, [], 'invalid_approach'),
])
def test_rejection_records_generated_candidate_and_exact_reason_without_partial_commit(
        prompt_path, changes, previous, code):
    body = dict(message=MESSAGE, visible_detail='', approach='software with questionable taste', typography={})
    body.update(changes)
    def transport(*args):
        response = answer()
        response['message']['content'] = json.dumps(body)
        return response
    agent = MediatingAgent('unused visual', invitation_prompt=prompt_path, transport=transport)
    telemetry, design = {}, {'untouched': True}
    with pytest.raises(InvitationRejected) as caught:
        agent.invite(None, {}, kind='nudge', previous=previous, telemetry=telemetry, design=design)
    rejection = caught.value.metadata()
    assert rejection['code'] == code and rejection['kind'] == 'nudge'
    assert telemetry['invitation_rejection'] == rejection
    assert telemetry['generation_status'] == 'error' and telemetry['eval_count'] == 42
    raw = body['message']
    if isinstance(raw, str):
        normalized = ' '.join(raw.split())
        assert rejection['candidate'] == normalized
        assert rejection['words'] == len(normalized.split()) and rejection['characters'] == len(normalized)
        assert not rejection['candidate_truncated']
    else:
        assert rejection['candidate'] is None and rejection['message_type'] == 'NoneType'
    assert design == {'untouched': True} and not agent.invitation_number and not agent.history


@pytest.mark.parametrize('content,code', [
    (None, 'invalid_content'),
    ('OPAQUE_PROVIDER_TEXT ' * 1500, 'invalid_content'),
    ('not JSON: OPAQUE_PROVIDER_TEXT', 'invalid_json'),
    ('["OPAQUE_PROVIDER_TEXT"]', 'invalid_json_shape'),
])
def test_malformed_content_never_leaks_whole_response_into_rejection_diagnostics(prompt_path, content, code):
    agent = MediatingAgent('unused visual', invitation_prompt=prompt_path, transport=lambda *args:
        {'message': {'content': content}})
    telemetry = {}
    with pytest.raises(InvitationRejected) as caught:
        agent.invite(None, {}, kind='nudge', telemetry=telemetry)
    assert caught.value.code == code
    assert 'OPAQUE_PROVIDER_TEXT' not in json.dumps(telemetry)
    assert caught.value.candidate is None


def test_large_candidate_is_explicitly_abbreviated_not_silently_changed(prompt_path):
    candidate = 'Speak into the mic ' + 'x' * 800
    agent = MediatingAgent('unused', invitation_prompt=prompt_path, transport=lambda *args: answer(candidate))
    with pytest.raises(InvitationRejected) as caught:
        agent.invite(None, {}, kind='nudge')
    data = caught.value.metadata()
    assert data['code'] == 'character_limit'
    assert len(data['candidate']) == 512 and data['candidate_truncated'] is True
    assert data['characters'] == len(candidate)
    assert not agent.invitation_number


@pytest.mark.parametrize('boundary_speech', [False, True])
def test_real_worker_logs_reason_and_candidate_without_forwarding_rejected_text(
        prompt_path, caplog, boundary_speech):
    candidate = 'Tell the mic ' + 'x' * 161
    caplog.set_level(logging.WARNING)
    with harness(lambda *args: answer(candidate), agent_options=dict(invitation_prompt=prompt_path),
            engagement=ready_state(), boundary_speech=boundary_speech) as (bridge, sender, packets, events):
        event = event_of(events, 'error')
        assert event['error'] == 'InvitationRejected' and not event['udp_sent']
        assert event['rejection']['code'] == 'character_limit'
        assert event['rejection']['candidate'] == candidate
        assert event['telemetry']['invitation_rejection'] == event['rejection']
        assert 'INVITE REJECT kind=nudge reason=character_limit' in caplog.text
        assert candidate in caplog.text and 'chars=' in caplog.text and 'words=' in caplog.text
        assert 'Interaction failed' not in caplog.text
        assert not bridge.recent_invitations and not bridge.agent.history and not bridge.completed
        with pytest.raises(queue.Empty):
            packets.get(timeout=.1)


def test_provider_value_error_is_not_mislabelled_as_wording_rejection(prompt_path, caplog):
    caplog.set_level(logging.ERROR)
    def transport(*args):
        raise ValueError('OPAQUE_PROVIDER_BODY')
    with harness(transport, agent_options=dict(invitation_prompt=prompt_path),
            engagement=ready_state()) as (bridge, sender, packets, events):
        event = event_of(events, 'error')
        assert event['error'] == 'ValueError' and 'rejection' not in event
        assert 'Interaction failed' in caplog.text and 'INVITE REJECT' not in caplog.text
        assert 'OPAQUE_PROVIDER_BODY' not in caplog.text and not bridge.agent.history


@pytest.mark.parametrize('first,second', [
    ('Décris une scène au micro pour changer le look. Le rendu a un goût de vieux filtre.',
     'Décris une scène au micro pour changer le look. Mon logiciel a raté son cours de goût.'),
    ('Describe an image into the microphone. My renderer needs adult supervision.',
     'Describe an image into the microphone. The software has dreadful taste.'),
    ('Parle au micro; mon logiciel a raté son cours de goût.',
     'Parle au micro; le rendu a un goût de vieux filtre.'),
    ('Dis au micro quoi changer. Sérieusement.', 'Dis au micro quoi changer. Logiciel typique.'),
    ('Tell the mic what to change. Typical software.',
     'Tell the mic what to change. Questionable taste.'),
])
def test_necessary_mic_instruction_can_repeat_with_a_different_punchline(first, second):
    assert not repeated_invitation(second, [first])
    assert repeated_invitation(first.upper(), [first])


@pytest.mark.parametrize('old,new', [
    ('Parle au micro pour changer cette image. Mon logiciel a raté son cours de goût.',
     'Décris une scène au microphone. Mon logiciel a raté son cours de goût.'),
    ('Describe an image into the microphone. The software has dreadful taste.',
     'Tell the mic what to change. The software has dreadful taste.'),
    ("Le micro mord pas. T'as un truc à lui dire?",
     "Le micro mord pas. T'as une idée à lui donner?"),
])
def test_same_joke_and_cosmetic_rewrites_are_still_rejected(old, new):
    assert repeated_invitation(new, [old])


def test_previously_false_repeated_challenge_is_now_accepted_without_changing_display_limits(prompt_path):
    previous = 'Décris une scène au micro pour changer le look. Le rendu a un goût de vieux filtre.'
    candidate = 'Décris une scène au micro pour changer le look. Mon logiciel a raté son cours de goût.'
    telemetry = {}
    agent = MediatingAgent('unused', invitation_prompt=prompt_path, transport=lambda *args: answer(candidate))
    assert agent.invite(None, {}, kind='nudge', previous=[previous], telemetry=telemetry) == candidate
    assert telemetry['generation_status'] == 'complete' and 'invitation_rejection' not in telemetry
    assert 3 <= len(candidate.split()) <= 24 and len(candidate) <= 160


def test_bad_optional_typography_is_not_a_fatal_invitation_rejection(prompt_path):
    body = json.loads(answer()['message']['content'])
    body['typography'] = dict(decoration='unsupported-decoration')
    telemetry = {}
    agent = MediatingAgent('unused', invitation_prompt=prompt_path,
        transport=lambda *args: {'message': {'content': json.dumps(body)}})
    assert agent.invite(None, {}, kind='nudge', telemetry=telemetry) == MESSAGE
    assert 'invitation_rejection' not in telemetry
