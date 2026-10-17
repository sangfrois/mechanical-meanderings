"""Personality, live palette and transport contracts (synthetic model fixtures)."""
import json
import pytest
from mechanical_meanderings import StaleGeneration
from mechanical_meanderings.mediation import MediatingAgent, brief_invitation
from test_mediation import judgement
from test_interactive import harness, event_of
from test_engagement import Clock, scene
from mechanical_meanderings.engagement import EngagementState


def test_toggle_changes_next_prompt_without_overriding_user_words():
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(judgement())}}
    agent = MediatingAgent('visual style', transport=transport)
    agent.set_disco(True)
    text = 'A quiet matte charcoal drawing, no glitter.'
    agent.interpret(text)
    assert 'DISCO = ON' in requests[-1]['messages'][0]['content']
    assert 'Clear requests against the palette win.' in requests[-1]['messages'][0]['content']
    assert requests[-1]['messages'][-1]['content'] == text
    agent.set_disco(False)
    agent.interpret('An ocean')
    assert 'DISCO = OFF' in requests[-1]['messages'][0]['content']
    assert 'DISCO = ON' not in requests[-1]['messages'][0]['content']
    assert set(json.loads(agent.history['default'][-1]['content'])) == {'prompt', 'reply'}
    with pytest.raises(ValueError):
        agent.set_disco(1)


def test_palette_change_invalidates_inflight_generation():
    agent = None
    def transport(*args):
        agent.set_disco(True)
        return {'message': {'content': json.dumps(judgement())}}
    agent = MediatingAgent('style', transport=transport)
    with pytest.raises(StaleGeneration):
        agent.interpret('An ocean')
    assert not agent.history


@pytest.mark.parametrize('text', ['Do not change the image.', "Don't alter my reflection.",
    "Ne modifie pas l'image."])
def test_explicit_no_image_change_overrides_model(text):
    agent = MediatingAgent('style', transport=lambda *a: {'message': {'content': json.dumps(judgement())}})
    decision = agent.interpret(text)
    assert not decision.apply_visual and decision.prompt == '' and decision.controls is None
    assert decision.reply


def test_personality_invitation_still_rejects_departed_objects():
    assert brief_invitation('Feed the mic a strange idea.', 'English') == 'Feed the mic a strange idea.'
    assert 'cup' not in brief_invitation('Say “floating cup” to the mic.', 'English')


def test_bad_optional_color_cannot_silence_the_reply():
    from mechanical_meanderings.mediation import parse_decision
    from mechanical_meanderings.typography import parse_typography
    d = parse_decision(json.dumps(judgement(typography=dict(font='Georgia', x=.7,
        color=[255, 255, 255], size=float('nan')))))
    assert d.reply == judgement()['reply'] and d.prompt == judgement()['prompt']
    assert d.typography['font'] == 'Georgia' and d.typography['x'] == .7
    assert all(0 <= c <= 1 for c in d.typography['color'])
    assert parse_typography(d.typography) == d.typography


def test_creative_profile_ack_and_invalid_profile_rejection():
    with harness(lambda *a: {'message': {'content': json.dumps(judgement())}}) as (bridge, sender, received, events):
        profile = dict(disco=True, asides=False, camera=False)
        sender.send_message('/creative/profile', json.dumps(profile))
        address, args = received.get(timeout=1)
        assert address == '/creative/status' and json.loads(args[0]) == profile
        assert bridge.agent.disco and not bridge.asides_enabled and not bridge.camera_enabled
        sender.send_message('/creative/profile', json.dumps(dict(profile, disco='true')))
        assert event_of(events, 'drop')['reason'] == 'creative profile flags must be boolean'
        assert bridge.agent.disco is True


@pytest.mark.parametrize('already_spoke', [False, True])
def test_next_idle_message_can_connect_imaginative_speech_and_movement(already_spoke):
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    if already_spoke:
        state.speech('Transforme-moi en arbre.')
    else:
        state.invitation_count = 1
    clock.now = 1
    state.observe(scene(2))
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(dict(message='Décris un monde impossible au micro, puis change de silhouette. Fais mentir cette machine.',
            visible_detail='', typography=dict(font='Georgia', x=.7, y=.25, decoration='slashes')))}}
    with harness(transport, engagement=state) as (_, sender, received, events):
        result = event_of(events, 'result')
        address, args = received.get(timeout=1)
        assert address == '/engagement/message' and args[1] == 'challenge' and args[2] == 12.
        assert result['interaction'] == 'embodied'
        assert 'CONTINUING PLAY' in requests[0]['messages'][0]['content']
        assert 'visually evocative' in requests[0]['messages'][0]['content']
        context = json.loads(requests[0]['messages'][-1]['content'])
        assert context['language'] == 'French (Quebec)'
        assert context['recent_speech'] == (['Transforme-moi en arbre.'] if already_spoke else [])
        assert 'images' not in requests[0]['messages'][-1]
        assert json.loads(args[3])['font'] == 'Georgia'
