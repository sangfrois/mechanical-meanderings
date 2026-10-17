"""Real OSC transport tests; model and JPEG fixtures are explicitly synthetic."""
import json
import queue
import threading
import time
from contextlib import contextmanager
import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from mechanical_meanderings.mediation import MediatingAgent
from test_mediation import judgement
from test_engagement import Clock, scene


@contextmanager
def harness(transport, *, agent_options=None, **options):
    from mechanical_meanderings.interactive import InteractiveBridge
    received = queue.Queue()
    events = queue.Queue()
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(lambda address, *args: received.put((address, args)))
    sink = BlockingOSCUDPServer(('127.0.0.1', 0), dispatcher)
    receiver = threading.Thread(target=sink.serve_forever, daemon=True)
    receiver.start()
    agent = MediatingAgent('visual style', transport=transport, timeout=2, **(agent_options or {}))
    bridge = InteractiveBridge(agent, listen_host='127.0.0.1', listen_port=0,
        output_host='127.0.0.1', output_port=sink.server_address[1], output_address='/txt2img',
        dynamic=True, debounce=.01, update_interval=.05, on_event=events.put,
        frame_port=0, **options)
    bridge.start()
    sender = SimpleUDPClient(*bridge.address)
    try:
        yield bridge, sender, received, events
    finally:
        sender.close()
        bridge.close()
        sink.shutdown()
        sink.server_close()
        receiver.join(1)
        assert bridge.client._sock.fileno() == -1
        assert not bridge.worker.is_alive()
        assert not bridge.frames.thread.is_alive()


def event_of(events, kind):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        value = events.get(timeout=2)
        if value['kind'] == kind:
            return value
    raise AssertionError('missing event ' + kind)


def test_reply_and_metadata_never_become_image_prompt():
    def transport(*args):
        return {'message': {'content': json.dumps(judgement())}}
    with harness(transport) as (bridge, sender, received, events):
        sender.send_message('/transcription', 'Turn me into Batman')
        event_of(events, 'result')
        packets = [received.get(timeout=1) for _ in range(4)]
        by_address = dict(packets)
        assert by_address['/txt2img'] == (judgement()['prompt'],)
        assert by_address['/txt2img/trigger'] == (1,)
        assert by_address['/engagement/message'][:2] == (judgement()['reply'], 'reply')
        metadata = json.loads(by_address['/mediation'][0])
        assert metadata['category'] == 'transformation'
        assert metadata['controls']['amount'] > .5


def test_incidental_speech_publishes_judgement_but_no_replacement():
    def transport(*args):
        return {'message': {'content': json.dumps(judgement(category='incidental'))}}
    with harness(transport) as (bridge, sender, received, events):
        sender.send_message('/transcription', 'Did you park the car?')
        event_of(events, 'result')
        assert received.get(timeout=1)[0] == '/mediation'
        with pytest.raises(queue.Empty):
            received.get(timeout=.15)


def test_speech_during_yolo_invitation_inference_discards_invitation():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    clock.now = 1
    state.observe(scene(2))
    began = threading.Event()
    release = threading.Event()
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        if 'message' in payload['format'].get('properties', {}):
            began.set()
            assert release.wait(1)
            return {'message': {'content': json.dumps(dict(message='Speak into the microphone!', visible_detail=''))}}
        return {'message': {'content': json.dumps(judgement(category='incidental'))}}
    with harness(transport, engagement=state) as (bridge, sender, received, events):
        assert began.wait(1)
        sender.send_message('/transcription', 'new speech')
        event_of(events, 'input')
        release.set()
        discarded = event_of(events, 'discard')
        assert discarded['channel'] == '__invitation__'
        event_of(events, 'result')
        assert received.get(timeout=1)[0] == '/mediation'
        with pytest.raises(queue.Empty):
            received.get(timeout=.1)
        assert len(requests) == 2


def test_no_person_does_not_call_camera_or_model():
    called = []
    with harness(lambda *args: called.append('model'), frame_provider=lambda: called.append('frame')) as (bridge, sender, received, events):
        time.sleep(.08)
        assert called == []


def test_yolo_invitation_uses_detected_object_without_camera():
    from mechanical_meanderings.engagement import EngagementState
    clock = Clock()
    state = EngagementState(clock=clock, idle_after=.1, presence_dwell=.1)
    state.observe(scene(1))
    clock.now = 1
    state.observe(json.dumps(dict(people=1, sequence=2,
        objects=[dict(label='chair', confidence=.9)])))
    requests, frames = [], []
    def transport(url, payload, timeout):
        requests.append(payload)
        # A generic model answer must not erase the actual detector grounding.
        return {'message': {'content': json.dumps(dict(
            message='Speak into the microphone. Change your reflection.', visible_detail=''))}}
    with harness(transport, engagement=state, asides_enabled=False,
            frame_provider=lambda: frames.append(True) or 'synthetic-frame') as (_, sender, received, events):
        result = event_of(events, 'result')
        assert not frames
        assert 'images' not in requests[0]['messages'][-1]
        context = json.loads(requests[0]['messages'][-1]['content'])
        assert context['scene']['objects'] == [dict(label='chair', confidence=.9)]
        address, args = received.get(timeout=1)
        assert address == '/engagement/message' and args[1] == 'invitation'
        assert 'chair' in args[0].casefold()
        assert len(args[0].split()) <= 7 and len(args[0]) <= 48
        assert result['scene']['sequence'] == 2


def test_new_speech_suppresses_stale_reply_without_starving_visual_updates():
    began = threading.Event()
    release = threading.Event()
    calls = []
    def transport(url, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            began.set()
            assert release.wait(1)
            return {'message': {'content': json.dumps(judgement())}}
        return {'message': {'content': json.dumps(judgement(category='incidental'))}}
    with harness(transport) as (bridge, sender, received, events):
        sender.send_message('/transcription', 'Turn me into Batman')
        assert began.wait(1)
        sender.send_message('/transcription', 'different words now')
        deadline = time.monotonic() + 1
        while bridge.latest.get('default') != 'different words now' and time.monotonic() < deadline:
            time.sleep(.01)
        release.set()
        first = event_of(events, 'result')
        assert first['reply'] == ''
        assert first['output']  # Keep latest-text visual cadence, not blanket cancellation.
        event_of(events, 'result')
        packets = [received.get(timeout=1) for _ in range(4)]
        assert all(address != '/engagement/message' for address, _ in packets)


def test_reset_clears_machine_caption_and_discards_inflight_speech():
    began = threading.Event()
    release = threading.Event()
    def transport(*args):
        began.set()
        assert release.wait(1)
        return {'message': {'content': json.dumps(judgement())}}
    with harness(transport) as (bridge, sender, received, events):
        sender.send_message('/transcription', 'Turn me into Batman')
        assert began.wait(1)
        sender.send_message('/reset', 1)
        event_of(events, 'reset')
        release.set()
        event_of(events, 'discard')
        address, args = received.get(timeout=1)
        assert address == '/engagement/message' and args[0] == ''
        assert not bridge.agent.history
        with pytest.raises(queue.Empty):
            received.get(timeout=.1)


def test_message_lifetime_matches_td_limit():
    from mechanical_meanderings.interactive import InteractiveBridge
    with pytest.raises(ValueError):
        InteractiveBridge(MediatingAgent('style'), listen_port=0, frame_port=0, message_ttl=61)
