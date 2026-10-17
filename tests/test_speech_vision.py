"""Actual mediation/OSC behavior with explicit synthetic model/JPEG fixtures."""
import base64
import json
import threading
import pytest
from mechanical_meanderings import StaleGeneration
from mechanical_meanderings.mediation import MediatingAgent
from test_mediation import judgement
from test_interactive import harness, event_of

FRAME = base64.b64encode(b'\xff\xd8\xff' + b'fixture-not-a-real-camera' * 2 + b'\xff\xd9').decode()


def reply(**kw):
    return judgement(category='question', apply_visual=False, prompt='', reply='I need a glance.', needs_camera=True, **kw)


def test_one_current_image_refinement_commits_one_text_only_turn():
    requests = []
    def transport(url, payload, timeout):
        requests.append(json.loads(json.dumps(payload)))
        data = reply() if len(requests) == 1 else judgement(category='question', apply_visual=False,
            prompt='', reply='A glass cup / catching light.', needs_camera=False,
            typography=dict(font='Georgia', x=.26, y=.72, decoration='slashes'))
        return {'message':{'content':json.dumps(data)}}
    agent = MediatingAgent('style', transport=transport, vision_capable=True)
    telemetry = {}
    result = agent.interpret('What am I holding?', camera=lambda:FRAME, telemetry=telemetry)
    assert result.reply == 'A glass cup / catching light.'
    assert result.typography['font'] == 'Georgia'
    assert 'images' not in requests[0]['messages'][-1]
    assert requests[1]['messages'][-1]['images'] == [FRAME]
    assert len(requests) == 2 and telemetry['image_attached'] is True
    assert telemetry['frame_status'] == 'attached'
    assert len(agent.history['default']) == 2
    assert FRAME not in json.dumps(agent.history)
    assert FRAME not in json.dumps(telemetry)


@pytest.mark.parametrize('mode,status', [('timeout','timeout'),('none','unavailable'),
    ('bad','unavailable'),('unsupported','vision_unsupported'),('disabled','disabled')])
def test_camera_unavailable_preserves_text_decision(mode, status):
    calls=[]
    def transport(*args):
        calls.append(1)
        return {'message':{'content':json.dumps(reply())}}
    def camera():
        if mode=='timeout':
            raise TimeoutError()
        return 'invalid' if mode=='bad' else None
    agent=MediatingAgent('style', transport=transport, vision_capable=mode!='unsupported')
    telemetry={}
    result=agent.interpret('Look at this.', camera=None if mode=='disabled' else camera, telemetry=telemetry)
    assert result.reply == 'I need a glance.' and len(calls)==1
    assert telemetry['frame_status']==status and not telemetry['image_attached']


def test_nonvisual_speech_does_not_request_camera():
    agent=MediatingAgent('style',transport=lambda *args:{'message':{'content':json.dumps(judgement())}},vision_capable=True)
    def camera():
        raise AssertionError('unnecessary camera call')
    telemetry={}
    agent.interpret('Turn me into Batman',camera=camera,telemetry=telemetry)
    assert telemetry['frame_status']=='not_requested' and not telemetry['image_attached']


def test_bad_vision_response_cannot_erase_safe_text_decision():
    calls=[]
    def transport(*args):
        calls.append(1)
        return {'message':{'content':json.dumps(reply()) if len(calls)==1 else 'bad JSON'}}
    agent=MediatingAgent('style',transport=transport,vision_capable=True)
    telemetry={}
    result=agent.interpret('What is here?',camera=lambda:FRAME,telemetry=telemetry)
    assert result.reply=='I need a glance.' and telemetry['frame_status']=='vision_error'
    assert FRAME not in json.dumps(agent.history)


def test_reset_during_frame_wait_discards_generation():
    agent=MediatingAgent('style',transport=lambda *a:{'message':{'content':json.dumps(reply())}},vision_capable=True)
    def camera():
        agent.reset()
        return FRAME
    with pytest.raises(StaleGeneration):
        agent.interpret('Look.',camera=camera)
    assert not agent.history


def test_osc_speech_camera_cooldown_and_typography_packet():
    frames=[]
    payloads=[]
    def transport(url,payload,timeout):
        images=payload['messages'][-1].get('images')
        payloads.append(bool(images))
        data=reply() if not images else judgement(category='question',apply_visual=False,prompt='',
            reply='A luminous cup / yes.',needs_camera=False,typography=dict(font='Consolas',x=.7,y=.2,decoration='tilde'))
        return {'message':{'content':json.dumps(data)}}
    with harness(transport,frame_provider=lambda:frames.append(1) or FRAME,reply_cooldown=.01) as (bridge,sender,received,events):
        bridge.agent.vision_capable=True
        sender.send_message('/transcription','What am I holding?')
        first=event_of(events,'result')
        assert first['telemetry']['frame_status']=='attached'
        packets=dict(received.get(timeout=1) for _ in range(2))
        style=json.loads(packets['/engagement/message'][3])
        assert style['font']=='Consolas' and style['x']==.7
        assert json.loads(packets['/mediation'][0])['vision']['image_attached'] is True
        sender.send_message('/transcription','What about this other object?')
        second=event_of(events,'result')
        assert second['telemetry']['frame_status']=='cooldown'
        assert len(frames)==1 and payloads==[False,True,False]
