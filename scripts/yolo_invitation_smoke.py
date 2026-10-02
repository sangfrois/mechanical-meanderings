"""Read the live TD YOLO metadata, call real Ollama, optionally display captions.

No camera request or second production listener; can run alongside the bridge.
--show is a deliberate display test, not an automatic occupancy invitation.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from pythonosc.udp_client import SimpleUDPClient
from mechanical_meanderings.engagement import parse_scene
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent, brief_invitation, invitation_objects


def read_scene(instance):
    payload = dict(jsonrpc='2.0', id=1, method='tools/call', params=dict(
        name='td_execute_python', arguments=dict(target_instance=instance,
            code="import json\nb=op('/project1/interactivity');print(b.op('scene').text)")))
    request = Request('http://127.0.0.1:40404/mcp', json.dumps(payload).encode(),
        {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'})
    with urlopen(request, timeout=15) as response:
        body = response.read().decode()
    if body.lstrip().startswith('{'):
        result = json.loads(body)
    else:
        result = json.loads(next(line[6:] for line in body.splitlines() if line.startswith('data: ')))
    if 'error' in result or result.get('result', {}).get('isError'):
        raise RuntimeError('TD scene read failed')
    content = next(item['text'] for item in result['result']['content'] if item['type'] == 'text')
    scene, _ = json.JSONDecoder().raw_decode(content.lstrip())
    return parse_scene(json.dumps(scene))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--td-port', type=int, default=9001)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    agent = MediatingAgent('Interactive mirror', timeout=60, history_turns=0)
    client = SimpleUDPClient('127.0.0.1', args.td_port)
    result = dict(source='actual live TD YOLO metadata', camera_requested=False, invitations=[])
    previous = []
    try:
        for language in ('English', 'French'):
            scene = read_scene(args.instance)
            started = time.perf_counter()
            message = agent.invite(None, scene, settings=Settings(history=0, temperature=.5),
                language=language, previous=previous)
            current = read_scene(args.instance)
            assert current['sequence'] != scene['sequence'], 'detector sequence did not advance'
            message = brief_invitation(message, language, previous, invitation_objects(current))
            assert 3 <= len(message.split()) <= 7 and len(message) <= 48
            record = dict(language=language, message=message, scene=scene, scene_at_send=current,
                seconds=time.perf_counter() - started)
            result['invitations'].append(record)
            previous.append(message)
            print(json.dumps(record, ensure_ascii=True), flush=True)
            if args.show:
                client.send_message('/engagement/message', [message, 'invitation', 30.])
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
