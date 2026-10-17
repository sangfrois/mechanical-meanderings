"""Sparse captions and explicit silence: real agent logic and isolated OSC sockets."""
import json
import queue
import threading
import pytest
from mechanical_meanderings.mediation import MediatingAgent
from mechanical_meanderings.speech_policy import speech_directive
from mechanical_meanderings.engagement import EngagementState
from test_mediation import judgement
from test_interactive import harness, event_of
from test_engagement import Clock, scene


@pytest.mark.parametrize('text', ["Don't talk back.", 'Stop talking!', 'Please be quiet.',
    'Tais-toi.', 'Ta gueule!', 'Arrête de parler.', 'Ne réponds plus, svp.'])
def test_whole_utterance_mute_is_deterministic_and_model_free(text):
    def forbidden(*args):
        raise AssertionError('caption preference must not call the model')
    agent = MediatingAgent('style', transport=forbidden)
    telemetry = {}
    result = agent.interpret(text, telemetry=telemetry)
    assert agent.speech_muted and result.reply == '' and not result.apply_visual
    assert not result.prompt and result.controls is None
    assert telemetry['interpretation_passes'] == 0 and not agent.history
    assert not agent.interpret('You can reply again.').reply
    assert not agent.speech_muted


@pytest.mark.parametrize('text', ["He said don't talk back.", 'A sign reading "be quiet".',
    "Don't stop talking.", 'I imagine a silent robot.', 'Can you explain the phrase stop talking?',
    'Make a painting of someone saying shut up.'])
def test_quotes_negation_and_visual_content_do_not_mute(text):
    assert speech_directive(text) is None


def test_sparse_mode_keeps_prompt_controls_and_real_question_answers():
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        text = payload['messages'][-1]['content']
        value = (judgement(category='question', apply_visual=False, prompt='', reply='I write words; StreamDiffusion makes pixels.')
            if text == 'How does this work?' else judgement())
        return {'message': {'content': json.dumps(value)}}
    agent = MediatingAgent('style', transport=transport, reply_mode='sparse')
    visual = agent.interpret('Turn me into Batman')
    assert visual.apply_visual and visual.prompt and visual.controls and visual.reply == ''
    assert agent.interpret('Hello?').reply == ''
    question = agent.interpret('How does this work?')
    assert question.reply and not question.apply_visual
    assert 'CAPTION POLICY: sparse.' in requests[-1]['messages'][0]['content']


def test_muted_agent_still_generates_visuals_and_reset_restores_captions():
    agent = MediatingAgent('style', transport=lambda *args: {'message': {'content': json.dumps(judgement())}})
    agent.interpret("Don't talk back.")
    visual = agent.interpret('Turn me into Batman')
    assert visual.apply_visual and visual.prompt and visual.controls and not visual.reply
    assert json.loads(agent.history['default'][-1]['content'])['reply'] == ''
    agent.reset('default')  # Automatic per-channel history expiry is not permission to talk.
    assert agent.speech_muted
    agent.reset()
    assert not agent.speech_muted and agent.interpret('Turn me into Batman').reply


@pytest.mark.parametrize('option', ['disabled', 'muted', 'off'])
def test_invitation_disabled_without_camera_or_model_work(option):
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    clock.now = 1
    state.observe(scene(2))
    from mechanical_meanderings.interactive import InteractiveBridge
    def forbidden(*args):
        raise AssertionError('disabled invitation must not start work')
    agent = MediatingAgent('style', transport=forbidden, reply_mode='off' if option == 'off' else 'auto')
    agent.speech_muted = option == 'muted'
    bridge = InteractiveBridge(agent, listen_host='127.0.0.1', listen_port=0, frame_port=0,
        engagement=state, invitations_enabled=option != 'disabled')
    try:
        assert bridge._take() is None and state.inflight is None
    finally:
        bridge.close()


def test_osc_mute_clears_caption_but_preserves_prompt_and_raw_transcription():
    calls = []
    def transport(*args):
        calls.append(1)
        return {'message': {'content': json.dumps(judgement())}}
    with harness(transport, invitations_enabled=False, forward_raw=True) as (bridge, sender, received, events):
        sender.send_message('/transcription', "Don't talk back.")
        preference = event_of(events, 'speech_preference')
        assert preference['muted']
        assert event_of(events, 'result')['reply'] == '' and calls == []
        packets = [received.get(timeout=1) for _ in range(3)]
        assert dict(packets)['/engagement/message'][0] == ''
        assert dict(packets)['/transcription'] == ("Don't talk back.",)
        sender.send_message('/transcription', 'Turn me into Batman')
        result = event_of(events, 'result')
        assert result['output'] and not result['reply'] and len(calls) == 1
        packets = [received.get(timeout=1) for _ in range(4)]
        assert '/txt2img' in dict(packets) and '/txt2img/trigger' in dict(packets)
        assert all(address != '/engagement/message' for address, _ in packets)


def test_mute_during_inference_suppresses_old_reply_immediately():
    began, release = threading.Event(), threading.Event()
    def transport(*args):
        began.set()
        assert release.wait(1)
        return {'message': {'content': json.dumps(judgement())}}
    with harness(transport, invitations_enabled=False) as (bridge, sender, received, events):
        sender.send_message('/transcription', 'Turn me into Batman')
        assert began.wait(1)
        sender.send_message('/transcription', 'Stop talking.')
        event_of(events, 'speech_preference')
        assert bridge.agent.speech_muted
        assert received.get(timeout=1)[1][0] == ''
        release.set()
        first = event_of(events, 'result')
        assert first['output'] and not first['reply']
        assert event_of(events, 'result')['telemetry']['interpretation_passes'] == 0
        packets = [received.get(timeout=1) for _ in range(4)]
        assert all(address != '/engagement/message' for address, _ in packets)


def test_cli_sets_sparse_mode_without_changing_model_history_or_context(monkeypatch):
    from mechanical_meanderings.__main__ import main
    from types import SimpleNamespace
    captured = []
    def interpret(self, text, settings, **kwargs):
        captured.append((self.reply_mode, self.model, self.history_turns, settings.num_ctx))
        return SimpleNamespace(metadata=lambda: {})
    monkeypatch.setattr(MediatingAgent, 'interpret', interpret)
    assert main(['--interactive', '--text', 'How does this work?', '--history-turns', '10', '--num-ctx', '128000']) == 0
    assert captured == [('sparse', 'gemma4:12b', 10, 128000)]
