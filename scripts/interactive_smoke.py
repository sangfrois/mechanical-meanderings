"""Explicit real local-model smoke; --vision requests one actual TD camera frame.

No microphone or original media is read. Test utterances are declared examples.
The optional JPEG stays in memory; the output records text judgements/timings only.
"""
import argparse
import base64
import json
from importlib.resources import files
from pathlib import Path
import time
from pythonosc.udp_client import SimpleUDPClient
from mechanical_meanderings.camera import FrameInbox
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent

EXAMPLES = ["I wanna look like Batman", "turn me into Batman",
    "I'm Batman and I walk around New York", "I walk around New York",
    "New York sunny day", "Did you park the car? We need to leave in ten minutes.",
    "Why are you changing my face?"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--td-port', type=int, default=9001)
    parser.add_argument('--frame-port', type=int, default=9012)
    args = parser.parse_args()
    system = files('mechanical_meanderings').joinpath('system.txt').read_text(encoding='utf-8')
    settings = Settings(history=0, temperature=.25, seed=42)
    result = dict(examples=[], failures=[], vision_requested=args.vision)
    for text in EXAMPLES:
        agent = MediatingAgent(system, history_turns=0, timeout=90)
        telemetry = {}
        started = time.perf_counter()
        try:
            decision = agent.interpret(text, settings=settings, telemetry=telemetry)
            record = dict(input=text, decision=decision.metadata(), elapsed=time.perf_counter()-started,
                telemetry=telemetry)
            result['examples'].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except Exception as exc:
            result['failures'].append(dict(input=text, error=type(exc).__name__))
            print(json.dumps(result['failures'][-1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if len(result['examples']) == len(EXAMPLES):
        decisions = [r['decision'] for r in result['examples']]
        if not decisions[0]['apply_visual'] or not decisions[1]['apply_visual']:
            result['failures'].append('Explicit Batman transformation not applied')
        if decisions[1]['controls'] and decisions[4]['controls'] and decisions[1]['controls']['amount'] <= decisions[4]['controls']['amount']:
            result['failures'].append('Transformation did not exceed scene-only strength')
        if decisions[5]['apply_visual']:
            result['failures'].append('Incidental logistical conversation changed the image')
        if not decisions[6]['reply'] or decisions[6]['apply_visual']:
            result['failures'].append('Addressed question did not yield separate reply-only output')
    if args.vision:
        client = SimpleUDPClient('127.0.0.1', args.td_port)
        try:
            with FrameInbox(args.frame_port) as inbox:
                image = inbox.request(lambda request_id, endpoint:
                    client.send_message('/camera/request', [request_id, endpoint]), timeout=3.)
                result['camera'] = dict(jpeg_bytes=len(base64.b64decode(image)), actual_frame_received=True)
                agent = MediatingAgent(system, history_turns=0, timeout=90)
                started = time.perf_counter()
                message = agent.invite(image, {}, settings=settings, language='English')
                result['invitation'] = dict(message=message, elapsed=time.perf_counter()-started)
                print(json.dumps(dict(camera=result['camera'], invitation=result['invitation'])), flush=True)
                image = None
        except Exception as exc:
            result['failures'].append(dict(vision_error=type(exc).__name__))
        finally:
            client.close()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(examples=len(result['examples']), failures=result['failures'], output=str(args.output))), flush=True)
    return int(bool(result['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
