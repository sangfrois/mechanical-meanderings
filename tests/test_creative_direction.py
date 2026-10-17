"""Deterministic protocol tests; live language quality is checked by creative_replay.py."""
import json

import pytest

from mechanical_meanderings import GenerationControl, StaleGeneration
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import (
    DECISION_SCHEMA, MediatingAgent, parse_decision, repeated_invitation,
)
from mechanical_meanderings.personality import INTERACTIVE_REFERENCE
from mechanical_meanderings.typography import default_typography
from test_mediation import judgement
from test_interactive import harness, event_of
from test_interactive_boundary import rig, raw, accepted


LONG_PROMPT = ' '.join(f'detail{i}' for i in range(50))


def compact(**changes):
    value = judgement(prompt=LONG_PROMPT)
    value.pop('reply')
    value.update(caption=None, needs_camera=False)
    value.update(changes)
    return value


def response(value):
    return {'message': {'content': json.dumps(value)}}


def test_fifty_words_survive_compact_parsing_without_invisible_caption():
    result = parse_decision(json.dumps(compact()))
    assert result.prompt == LONG_PROMPT and len(result.prompt.split()) == 50
    assert not result.reply and result.typography == default_typography()
    assert result.controls
    assert 'reply' not in DECISION_SCHEMA['properties']
    assert 'typography' not in DECISION_SCHEMA['properties']
    assert DECISION_SCHEMA['properties']['caption']['anyOf'][0] == {'type': 'null'}


def test_visible_caption_preserves_the_existing_wire_contract():
    design = dict(default_typography(), font='Impact', size=45)
    result = parse_decision(json.dumps(compact(caption=dict(reply='Try a rougher surface.', typography=design))))
    assert result.reply == 'Try a rougher surface.' and result.typography == design
    assert 'caption' not in result.metadata()


def test_overlong_visual_keeps_complete_description_instead_of_dangling_suffix():
    complete = LONG_PROMPT + '.'
    result = parse_decision(json.dumps(compact(prompt=complete + ' ' + 'extra ' * 20)))
    assert result.prompt == complete


def test_malformed_approach_cannot_partially_commit_invitation():
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=lambda *args:
        response(dict(message='Describe an impossible scene into the mic and outdo this machine.',
                      visible_detail='', typography={}, approach=None)))
    design = {'untouched': True}
    with pytest.raises(ValueError):
        agent.invite(None, {}, kind='nudge', design=design, telemetry={})
    assert not agent.invitation_number and design == {'untouched': True}


@pytest.mark.parametrize('caption', ['', [], 2, {}, {'reply': '', 'typography': {}}, {'reply': 'hello'}])
def test_bad_compact_caption_is_rejected(caption):
    with pytest.raises(ValueError):
        parse_decision(json.dumps(compact(caption=caption)))


def test_render_memory_is_accepted_only_not_reconstructed_from_history_or_drafts():
    requests = []
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=lambda url, payload, timeout:
        requests.append(payload) or response(compact()), history_turns=25)
    ledger = [dict(role='user', content=f'actual observed speech {i}') for i in range(30)]
    agent.history['default'] = [dict(role='user', content='FOREIGN_HISTORY'),
                                dict(role='assistant', content='FOREIGN_GENERATED')]
    decision = agent.interpret('New visual.', history_override=ledger, commit=False, settings=Settings(history=25))
    assert not agent.render_context
    assert requests[-1]['messages'][1:-1] == ledger
    assert 'FOREIGN_' not in json.dumps(requests[-1])
    agent.accept_visual(decision)
    agent.interpret('Keep that subject; change only its surface.', history_override=ledger, commit=False)
    messages = requests[-1]['messages']
    assert messages[1:-2] == ledger  # Nothing deleted, paraphrased or turned into an assistant.
    assert messages[-2]['role'] == 'assistant'
    assert 'GENERATED RENDER CONTEXT' in messages[-2]['content']
    assert LONG_PROMPT in messages[-2]['content']
    assert messages[-1] == dict(role='user', content='Keep that subject; change only its surface.')
    assert not any(LONG_PROMPT in m['content'] for m in messages if m['role'] == 'user')
    assert len(requests) == 2  # No second model pass for a reset, length or quality repair.
    assert requests[0]['model'] == 'gemma4:12b' and requests[0]['options']['num_predict'] == 512
    assert agent.history['default'][0]['content'] == 'FOREIGN_HISTORY'


def test_bounded_render_context_retains_last_visual_across_nonvisual_turns_and_resets():
    agent = MediatingAgent(INTERACTIVE_REFERENCE)
    for i in range(8):
        agent.accept_visual(parse_decision(json.dumps(compact(prompt=f'accepted visual {i}'))))
    assert agent.render_context['default'] == [f'accepted visual {i}' for i in range(4, 8)]
    agent.accept_visual(parse_decision(json.dumps(compact(category='question', apply_visual=False, prompt=''))))
    assert agent.render_context['default'][-1] == 'accepted visual 7'
    agent.accept_visual(parse_decision(json.dumps(compact())), 'other')
    agent.reset('default')
    assert 'default' not in agent.render_context and 'other' in agent.render_context
    agent.reset()
    assert not agent.render_context


def test_cancelled_generation_never_accepts_a_render():
    control = GenerationControl(1)
    def transport(*args):
        control.cancel('superseded')
        return response(compact())
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=transport)
    with pytest.raises(StaleGeneration):
        agent.interpret('New visual.', control=control)
    assert not agent.render_context and not agent.history


def test_fifty_word_prompt_reaches_real_osc_without_changing_caption_route():
    with harness(lambda *args: response(compact()), boundary_speech=True) as (bridge, sender, packets, events):
        sender.send_message('/transcription', 'Invent something new.')
        sender.send_message('/trigger', 1)
        event_of(events, 'result')
        received = [packets.get(timeout=1) for _ in range(5)]
        by_address = dict(received)
        assert by_address['/txt2img'] == (LONG_PROMPT,)
        assert '/engagement/message' not in by_address
        assert json.loads(by_address['/mediation'][0])['prompt'] == LONG_PROMPT
        assert render_context(bridge.agent, bridge.render_key)['current_prompt'] == LONG_PROMPT


def test_actual_boundary_worker_supplies_only_delivered_render_as_separate_context(caplog):
    import logging
    caplog.set_level(logging.INFO)
    with rig() as (bridge, sender, requests, packets, events, _):
        raw(sender, packets, '/transcription', 'first scene')
        raw(sender, packets, '/trigger', 1)
        requests.get(timeout=2)
        accepted(packets, events)
        current = render_context(bridge.agent, bridge.render_key)['current_prompt']
        raw(sender, packets, '/trigger', 0)
        raw(sender, packets, '/transcription', 'Keep that subject, make it rough.')
        raw(sender, packets, '/trigger', 1)
        payload = requests.get(timeout=2)
        assert payload['messages'][-2]['role'] == 'assistant'
        assert current in payload['messages'][-2]['content']
        assert dict(role='user', content='first scene') in payload['messages']
        assert payload['messages'][-1]['content'] == 'Keep that subject, make it rough.'
        accepted(packets, events)
        assert 'input_eval=' in caplog.text and 'output_eval=' in caplog.text and 'queue=' in caplog.text


def test_cosmetic_rewrites_and_reused_hooks_are_rejected_but_actions_can_recur():
    old = "Le micro mord pas. T'as un truc à lui dire?"
    assert repeated_invitation("Le micro mord pas. T'as une idée à lui donner?", [old])
    assert repeated_invitation(old.upper(), [old])
    assert not repeated_invitation('Décris une scène au micro; la machine manque de vacances.', [old])
    assert not repeated_invitation('Describe a texture into the mic and reshape the camera silhouette.',
                                   ['Invent a scene for the mic; outdo the machine.'])


@pytest.mark.parametrize('kind', ['nudge', 'embodied'])
def test_challenges_use_exact_editable_prompt_and_twelve_line_anti_repeat_context(kind):
    requests = []
    message = 'Décris une scène impossible au micro, puis change doucement de silhouette. Fais mentir la machine.'
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=lambda url, payload, timeout:
        requests.append(payload) or response(dict(message=message, visible_detail='', typography={})))
    previous = [f'earlier distinct invitation {i}' for i in range(15)]
    assert agent.invite(None, dict(recent_speech=[]), kind=kind, previous=previous) == message
    payload = requests[0]
    system = payload['messages'][0]['content']
    assert system == agent.invitation_prompt.read().text
    assert json.loads(payload['messages'][-1]['content'])['kind'] == kind
    assert json.loads(payload['messages'][-1]['content'])['recent_invitations'] == previous[-12:]
    assert len(requests) == 1 and not agent.history and not agent.render_context


def render_context(agent, render_key):
    note = agent._render_note('default', render_key=render_key)
    return json.loads(note.rsplit('\n', 1)[1]) if note else None


def request_render(payload):
    notes = [m['content'] for m in payload['messages']
             if m['role'] == 'assistant' and 'GENERATED RENDER CONTEXT' in m['content']]
    return json.loads(notes[-1].rsplit('\n', 1)[1]) if notes else None


def spoken_turn(bridge, sender, packets, text, channel='default'):
    """Assert each exact raw packet independently of generated UDP output."""
    prefix = '' if channel == 'default' else '/' + channel
    def send(kind, value):
        sender.send_message(prefix + '/' + kind, value)
        assert packets.get(timeout=1) == ('/' + kind, (value,))
    if bridge.boundary_speech:
        send('trigger', 0)
    send('transcription', text)
    if bridge.boundary_speech:
        send('trigger', 1)


@pytest.mark.parametrize('boundary_speech', [True, False])
@pytest.mark.parametrize('expire_speech', [False, True])
def test_bridge_render_is_shared_across_channels_not_speech_epochs(boundary_speech, expire_speech):
    requests, now = [], [0.]
    prompts = iter(('DELIVERED_A', 'DELIVERED_B', 'EDITED_B'))
    def transport(url, payload, timeout):
        requests.append(payload)
        return response(compact(prompt=next(prompts)))
    with harness(transport, boundary_speech=boundary_speech, channel_mode=True,
            forward_raw=True, camera_enabled=False, invitations_enabled=False) as (bridge, sender, packets, events):
        bridge.agent.clock = lambda: now[0]
        bridge.agent.reset_after = 1 if expire_speech else None
        spoken_turn(bridge, sender, packets, 'First scene.', 'ch1')
        accepted(packets, events)
        now[0] = .5
        spoken_turn(bridge, sender, packets, 'A different scene.', 'ch2')
        accepted(packets, events)
        # First admission of ch2 must not erase the renderer's actual image.
        assert request_render(requests[1])['current_prompt'] == 'DELIVERED_A'
        now[0] = 2.
        spoken_turn(bridge, sender, packets, 'Keep this subject; make it rough.', 'ch1')
        accepted(packets, events)
        assert request_render(requests[2])['current_prompt'] == 'DELIVERED_B'
        assert requests[2]['messages'][-1] == dict(role='user', content='Keep this subject; make it rough.')
        if not expire_speech:
            assert [m['content'] for m in requests[2]['messages'] if m['role'] == 'user'] == [
                'First scene.', 'Keep this subject; make it rough.']
        assert [m['content'] for m in bridge.agent.history['ch2'] if m['role'] == 'user'] == ['A different scene.']
        if boundary_speech:
            assert list(bridge.completed['ch2']) == ['A different scene.']
        assert len(requests) == 3


@pytest.mark.parametrize('different_field', ['host', 'port', 'address'])
def test_output_targets_have_independent_render_context_without_forking_speech(different_field):
    from mechanical_meanderings.interactive import InteractiveBridge
    from contextlib import ExitStack
    requests = []
    agent = MediatingAgent(INTERACTIVE_REFERENCE, transport=lambda url, payload, timeout:
        requests.append(payload) or response(compact(prompt='UNDELIVERED_DRAFT')))
    options = dict(output_host='127.0.0.1', output_port=0, output_address='/txt2img')
    other = dict(options, **dict(host={'output_host': '127.0.0.2'},
        port={'output_port': 1}, address={'output_address': '/other'})[different_field])
    with ExitStack() as stack:
        bridges = []
        for output in (options, other):
            bridge = InteractiveBridge(agent, listen_host='127.0.0.1', listen_port=0,
                frame_port=0, camera_enabled=False, invitations_enabled=False, **output)
            stack.callback(bridge.close)
            bridges.append(bridge)
        # These bridges never start or send packets; only their destination keys are exercised.
        assert bridges[0].render_key != bridges[1].render_key
        for bridge, prompt in zip(bridges, ('TARGET_A', 'TARGET_B')):
            agent.accept_visual(parse_decision(json.dumps(compact(prompt=prompt))), 'ch1',
                                render_key=bridge.render_key)
        agent.reset('ch1')  # Speech reset is not an output reset.
        ledger = [dict(role='user', content=f'observed speech {i}') for i in range(30)]
        for bridge, prompt in zip(bridges, ('TARGET_A', 'TARGET_B')):
            agent.interpret('Keep this subject.', 'ch2', history_override=ledger,
                            render_key=bridge.render_key)
            assert request_render(requests[-1])['current_prompt'] == prompt
            assert requests[-1]['messages'][1:-2] == ledger
            # Legacy commit=True still commits speech pairs, but cannot accept a bridge draft.
            assert render_context(agent, bridge.render_key)['current_prompt'] == prompt
        assert [m['content'] for m in agent.history['ch2'] if m['role'] == 'user'] == ['Keep this subject.'] * 2
        agent.clear_render_context(render_key=bridges[0].render_key)
        assert render_context(agent, bridges[0].render_key) is None
        assert render_context(agent, bridges[1].render_key)['current_prompt'] == 'TARGET_B'
        agent.reset()
        assert not agent.render_context


@pytest.mark.parametrize('boundary_speech', [True, False])
def test_caption_send_failure_cannot_undo_delivered_visual_or_history(monkeypatch, boundary_speech):
    requests, at_caption = [], []
    values = iter((compact(prompt='DELIVERED_A'), compact(prompt='DELIVERED_B',
        caption=dict(reply='A new scene is ready.', typography=default_typography())), compact(prompt='EDITED_B')))
    def transport(url, payload, timeout):
        requests.append(payload)
        return response(next(values))
    with harness(transport, boundary_speech=boundary_speech, forward_raw=True,
            camera_enabled=False, invitations_enabled=False) as (bridge, sender, packets, events):
        send = bridge.client.send_message
        def fail_caption(address, value):
            if address == '/engagement/message':
                at_caption.append((render_context(bridge.agent, bridge.render_key),
                                   json.loads(bridge.agent.history['default'][-1]['content'])))
                raise OSError('synthetic caption send failure')
            return send(address, value)
        monkeypatch.setattr(bridge.client, 'send_message', fail_caption)
        spoken_turn(bridge, sender, packets, 'First scene.')
        accepted(packets, events)
        spoken_turn(bridge, sender, packets, 'A different scene.')
        error = event_of(events, 'error')
        assert error['error'] == 'OSError' and error['stage'] == 'udp' and error['udp_sent'] is True
        assert error['telemetry']['delivery'] == dict(mediation=True, prompt=True, trigger=True, caption=False)
        assert [packets.get(timeout=1)[0] for _ in range(3)] == ['/mediation', '/txt2img', '/txt2img/trigger']
        assert at_caption[0][0]['current_prompt'] == 'DELIVERED_B'
        assert at_caption[0][0]['earlier_directions'] == ['DELIVERED_A']
        assert at_caption[0][1]['prompt'] == 'DELIVERED_B'
        if boundary_speech:
            assert at_caption[0][1]['reply'] == ''  # The failed caption is not accepted history.
        spoken_turn(bridge, sender, packets, 'Keep this subject; make it rough.')
        accepted(packets, events)
        assert request_render(requests[2])['current_prompt'] == 'DELIVERED_B'
        assert [m['content'] for m in requests[2]['messages'] if m['role'] == 'user'] == [
            'First scene.', 'A different scene.', 'Keep this subject; make it rough.']
        assert len(requests) == 3


def test_successful_caption_updates_the_accepted_history_pair_without_duplication():
    with harness(lambda *args: response(compact(prompt='DELIVERED',
            caption=dict(reply='A new scene is ready.', typography=default_typography()))),
            boundary_speech=True, camera_enabled=False, invitations_enabled=False) as (bridge, sender, packets, events):
        spoken_turn(bridge, sender, packets, 'First scene.')
        result = accepted(packets, events)
        assert packets.get(timeout=1)[0] == '/engagement/message'
        assert result['telemetry']['delivery'] == dict(mediation=True, prompt=True, trigger=True, caption=True)
        assert bridge.agent.history['default'] == [dict(role='user', content='First scene.'),
            dict(role='assistant', content=json.dumps(dict(prompt='DELIVERED', reply='A new scene is ready.')))]
        assert render_context(bridge.agent, bridge.render_key) == dict(current_prompt='DELIVERED', earlier_directions=[])
        assert list(bridge.completed['default']) == ['First scene.']


@pytest.mark.parametrize('boundary_speech', [True, False])
@pytest.mark.parametrize('failed_address', ['/txt2img', '/txt2img/trigger'])
def test_failed_prompt_or_trigger_never_accepts_a_render(monkeypatch, boundary_speech, failed_address):
    requests = []
    prompts = iter(('DELIVERED_A', 'FAILED_B', 'EDITED_A'))
    def transport(url, payload, timeout):
        requests.append(payload)
        return response(compact(prompt=next(prompts)))
    with harness(transport, boundary_speech=boundary_speech, forward_raw=True,
            camera_enabled=False, invitations_enabled=False) as (bridge, sender, packets, events):
        spoken_turn(bridge, sender, packets, 'First scene.')
        accepted(packets, events)
        send = bridge.client.send_message
        def fail_visual(address, value):
            if address == failed_address:
                raise OSError('synthetic visual send failure')
            return send(address, value)
        monkeypatch.setattr(bridge.client, 'send_message', fail_visual)
        spoken_turn(bridge, sender, packets, 'A different scene.')
        error = event_of(events, 'error')
        assert error['error'] == 'OSError' and error['udp_sent'] is False
        prompt_sent = failed_address == '/txt2img/trigger'
        assert error['telemetry']['delivery'] == dict(mediation=True, prompt=prompt_sent, trigger=False, caption=False)
        assert packets.get(timeout=1)[0] == '/mediation'
        if prompt_sent:
            assert packets.get(timeout=1) == ('/txt2img', ('FAILED_B',))
        assert render_context(bridge.agent, bridge.render_key)['current_prompt'] == 'DELIVERED_A'
        if boundary_speech:
            assert [m['content'] for m in bridge.agent.history['default'] if m['role'] == 'user'] == ['First scene.']
        monkeypatch.setattr(bridge.client, 'send_message', send)
        spoken_turn(bridge, sender, packets, 'Keep this subject; make it rough.')
        accepted(packets, events)
        assert request_render(requests[2])['current_prompt'] == 'DELIVERED_A'
        assert [m['content'] for m in requests[2]['messages'] if m['role'] == 'user'] == [
            'First scene.', 'A different scene.', 'Keep this subject; make it rough.']
        assert len(requests) == 3
