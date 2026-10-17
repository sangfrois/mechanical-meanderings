"""Concrete participant challenges: real scheduling/OSC, synthetic model responses."""
import json
from types import SimpleNamespace
import pytest
from mechanical_meanderings.engagement import EngagementState
from mechanical_meanderings.mediation import MediatingAgent
from test_engagement import Clock, scene
from test_interactive import harness, event_of
from test_td_interactivity import helper


def answer(text):
    return {'message': {'content': json.dumps(dict(message=text, visible_detail='', typography={}))}}


@pytest.mark.parametrize('new_arrival', [False, True])
def test_first_contact_and_new_arrival_get_nudge_not_previous_visitors_challenge(new_arrival):
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    if new_arrival:
        state.speech('Make me a tree.')
        state.invitation_count = 1
        state.observe(json.dumps(dict(people=0, sequence=2, objects=[])))
        clock.now = 6
        state.observe(scene(3))
        assert not state.last_text and state.invitation_count == 0
    clock.now += 1
    state.observe(scene(4))
    requests = []
    text = "Le micro mord pas. T'as une idée à lui donner?"
    def transport(url, payload, timeout):
        requests.append(payload)
        return answer(text)
    with harness(transport, engagement=state) as (_, sender, received, events):
        result = event_of(events, 'result')
        address, args = received.get(timeout=1)
        assert (address, args[0], args[1]) == ('/engagement/message', text, 'challenge')
        assert result['interaction'] == 'nudge'
        assert json.loads(requests[0]['messages'][-1]['content'])['recent_speech'] == []


def test_live_presence_context_does_not_recycle_other_visitors_history():
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return answer("Le micro mord pas. T'as une idée?")
    agent = MediatingAgent('style', transport=transport)
    agent.history['default'] = [dict(role='user', content='I was the last visitor.')]
    agent.invite(None, dict(recent_speech=[]), kind='nudge')
    assert json.loads(requests[0]['messages'][-1]['content'])['recent_speech'] == []


@pytest.mark.parametrize('kind,text', [
    ('embodied', 'Cover your face with your hands.'),
    ('nudge', 'Cover your face with your hands.'),
    ('embodied', ' '.join(['Move'] * 25)),
    ('embodied', 'Make ' + 'x' * 161 + ' now.'),
])
def test_wrong_action_or_overlong_challenge_is_rejected_not_truncated(kind, text):
    agent = MediatingAgent('style', transport=lambda *args: answer(text))
    with pytest.raises(ValueError):
        agent.invite(None, {}, kind=kind)
    assert not agent.history and agent.invitation_number == 0


def test_repeated_challenge_is_not_published_again():
    text = 'Cache ton visage avec tes mains. Voyons ce qui arrive.'
    agent = MediatingAgent('style', transport=lambda *args: answer(text))
    with pytest.raises(ValueError, match='repeated'):
        agent.invite(None, {}, kind='embodied', previous=[text.upper()])


@pytest.mark.parametrize('override', [False, True])
def test_interactive_cli_does_not_inherit_legacy_never_answer_contract(monkeypatch, tmp_path, override):
    from mechanical_meanderings.__main__ import main
    from mechanical_meanderings.personality import INTERACTIVE_REFERENCE
    captured = []
    def interpret(self, text, **kwargs):
        captured.append(self.system_prompt)
        return SimpleNamespace(metadata=lambda: {'reply': 'test fixture'})
    monkeypatch.setattr(MediatingAgent, 'interpret', interpret)
    args = ['--interactive', '--text', 'Comment ça marche?']
    if override:
        custom = tmp_path / 'reference.txt'
        custom.write_text('Custom visual reference', encoding='utf-8')
        args.extend(['--system-prompt', str(custom)])
    assert main(args) == 0
    assert captured == ['Custom visual reference' if override else INTERACTIVE_REFERENCE]
    assert 'not answer a chat' not in captured[0]


def test_body_challenge_survives_td_caption_limit_and_yields_to_speech():
    m = helper()
    nodes = dict(message=SimpleNamespace(text=''), last_message=SimpleNamespace(text=''),
                 display_text=SimpleNamespace(width=1280, height=720))
    b = SimpleNamespace(par=SimpleNamespace(Enabled=True, Fontsize=30), op=nodes.get)
    root = SimpleNamespace(op=lambda name: b if name == 'interactivity' else nodes['message'])
    m.apply_caption_layout = lambda *args: None
    text = 'Cache ton visage avec tes mains. Voyons si je décroche de l’humain.'
    try:
        m.show_message(root, text, 'challenge', 12., {'decoration': 'none'})
        assert json.loads(nodes['last_message'].text)['message'] == text
        assert m._message_kind == 'challenge' and m._caption_number == 1
        assert 'micro' not in nodes['message'].text
        m.speech(root, 'Ça marche comment?')
        assert nodes['message'].text == '' and m._expires == 0
        with pytest.raises(ValueError, match='caption limit'):
            m.show_message(root, 'Move ' * 25, 'challenge')
    finally:
        m.close()
