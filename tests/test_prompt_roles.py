"""Prompt wiring and real OSC boundaries; model outputs are synthetic fixtures.

These tests do not establish a local model's semantic compliance or remove image-model bias.
"""
import json
import queue
from importlib.resources import files
import pytest
from mechanical_meanderings import Agent
from mechanical_meanderings.mediation import MediatingAgent
from mechanical_meanderings.personality import (
    INTERACTIVE_REFERENCE, MACHINE_KNOWLEDGE, PERSONALITY, VISUAL_DIRECTION,
)
from test_interactive import harness, event_of
from test_mediation import judgement


def explanation(**changes):
    value = judgement(category='conversation', apply_visual=False, prompt='',
        reply="Je donne les mots à StreamDiffusion. Les pixels, c'est lui, pas moi.",
        preserve_identity=1., transformation_intent=0., visual_specificity=0., needs_camera=False)
    value.update(changes)
    return value


def test_interpret_sends_role_and_body_direction_once_without_extra_inference():
    requests, frames = [], []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(explanation())}}
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=transport)
    result = agent.interpret('Tu fais les images ou les prompts?', camera=lambda: frames.append(True))
    assert len(requests) == 1 and not frames
    system = requests[0]['messages'][0]['content']
    for instructions in (PERSONALITY, MACHINE_KNOWLEDGE, VISUAL_DIRECTION):
        assert system.count(instructions) == 1
    assert requests[0]['messages'][-1]['content'] == 'Tu fais les images ou les prompts?'
    assert result.reply == explanation()['reply']
    assert not result.apply_visual and result.prompt == '' and result.controls is None
    assert len(agent.history['default']) == 2


@pytest.mark.parametrize('kind,message', [
    ('nudge', "T'as une idée? Donne-la au micro, crisse."),
    ('embodied', 'Décris un monde impossible au micro, puis change de silhouette. Fais mentir cette machine.'),
])
def test_challenges_share_casual_voice_and_honest_role(kind, message):
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(dict(message=message, visible_detail='', typography={}))}}
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=transport)
    assert agent.invite(None, {}, kind=kind) == message
    assert len(requests) == 1
    assert requests[0]['messages'][0]['content'] == agent.invitation_prompt.read().text
    assert 'Character and attitude' in requests[0]['messages'][0]['content']
    assert 'StreamDiffusion' in requests[0]['messages'][0]['content']
    assert MACHINE_KNOWLEDGE not in requests[0]['messages'][0]['content']
    assert not agent.history


def test_legacy_visual_prompt_receives_updated_body_direction():
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': 'Opened plum, rough linocut, dark pulp, irregular edges'}}
    system = files('mechanical_meanderings').joinpath('system.txt').read_text(encoding='utf-8')
    agent = Agent(system, transport=transport)
    assert agent.generate('La chair du fruit') == 'Opened plum, rough linocut, dark pulp, irregular edges'
    assert 'BODIES, BIAS AND ART DIRECTION' in requests[0]['messages'][0]['content']
    assert 'do not by themselves request nudity' in requests[0]['messages'][0]['content']
    assert 'do not default to young, thin, white women' in requests[0]['messages'][0]['content']
    assert requests[0]['messages'][-1]['content'] == 'La chair du fruit'


@pytest.mark.parametrize('text', [
    'Cette IA raconte de la bullshit. De quel blocage tu parles?',
    'Pourquoi tu fais toujours des femmes quand je dis chair?',
])
def test_critical_conversation_caption_is_not_an_image_or_renderer_command(text):
    with harness(lambda *args: {'message': {'content': json.dumps(explanation())}}) as (bridge, sender, received, events):
        sender.send_message('/transcription', text)
        result = event_of(events, 'result')
        packets = [received.get(timeout=1) for _ in range(2)]
        assert [address for address, args in packets] == ['/mediation', '/engagement/message']
        assert packets[1][1][0] == explanation()['reply']
        assert result['output'] == '' and not result['mediation']['apply_visual']
        assert result['mediation']['controls'] is None
        with pytest.raises(queue.Empty):
            received.get(timeout=.1)
