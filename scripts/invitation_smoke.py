"""Real local English/French invitation check; one ephemeral cached camera frame.

Run only with the production bridge stopped and TD Frameport matching --frame-port.
No image is saved. --show sends the two real outputs to the live caption receiver.
"""
import argparse
import base64
import json
from pathlib import Path
import time
from pythonosc.udp_client import SimpleUDPClient
from mechanical_meanderings.camera import FrameInbox
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-port', type=int, default=9012)
    parser.add_argument('--td-port', type=int, default=9001)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    client = SimpleUDPClient('127.0.0.1', args.td_port)
    result = dict(camera_requested=True, invitations=[])
    try:
        with FrameInbox(args.frame_port) as inbox:
            image = inbox.request(lambda request_id, endpoint:
                client.send_message('/camera/request', [request_id, endpoint]), timeout=4.)
        result['camera'] = dict(actual_frame_received=True, jpeg_bytes=len(base64.b64decode(image)))
        agent = MediatingAgent('Interactive mirror', timeout=60, history_turns=0)
        previous = []
        for language in ('English', 'French'):
            started = time.perf_counter()
            message = agent.invite(image, {}, settings=Settings(history=0, temperature=.5),
                language=language, previous=previous)
            assert 3 <= len(message.split()) <= 10 and len(message) <= 72
            previous.append(message)
            record = dict(language=language, message=message, seconds=time.perf_counter()-started)
            result['invitations'].append(record)
            print(json.dumps(record), flush=True)
            if args.show:
                client.send_message('/engagement/message', [message, 'invitation', 45.])
        image = None
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
