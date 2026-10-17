"""Opt-in isolated wording probe; never sends OSC or loads a camera.

Run only with the operator's permission. Uses the already resident model/context.
Rechecks /api/ps before EVERY chat and aborts if residency/context cannot be verified.
The check and chat are not atomic: eviction or a context change between them can
still cause Ollama to reload. This is not an absolute no-reload guarantee.
This measures request time, not microphone or image-rendering latency.
"""
import argparse
import http.client
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanical_meanderings import http_chat
from mechanical_meanderings.live import Settings
from mechanical_meanderings.mediation import MediatingAgent
from mechanical_meanderings.personality import INTERACTIVE_REFERENCE


SPEECH = (
    'Cubism, naturalist, disco ball, discotheque.',
    "I don't want any portraits. I want a natural forest of discotheque balls.",
    'Give me something completely new as a visual.',
    'This is not new. I want something completely different, not the old style or objects.',
    "Why are you so stuck with cubism and kaleidoscopic textures? Drop them and diverge.",
    'Keep that new subject, but make its surface rough and its lighting overcast.',
)


def _resident_context(url, model):
    """Fail closed, including malformed successful responses and urllib failures."""
    try:
        with urllib.request.urlopen(url + '/api/ps', timeout=3) as response:
            if response.status != 200:
                raise RuntimeError('Cannot verify resident model: /api/ps was not successful')
            value = json.load(response)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # RuntimeError also bypasses the invitation-quality ValueError handler.
        raise RuntimeError('Cannot verify resident model: /api/ps failed') from exc
    if (not isinstance(value, dict) or not isinstance(value.get('models'), list)
            or not all(isinstance(m, dict) and isinstance(m.get('name'), str)
                       and m['name'] for m in value['models'])):
        raise RuntimeError('Cannot verify resident model: malformed /api/ps response')
    matches = [m for m in value['models'] if m.get('name') == model]
    context = matches[0].get('context_length') if len(matches) == 1 else None
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise RuntimeError('Selected model must already be resident with known context length')
    return context


def run(output, model='gemma4:12b', url='http://127.0.0.1:11434'):
    context = _resident_context(url, model)
    result = dict(model=model, context_length=context, history=25,
                  note='Isolated live Ollama requests; no OSC or camera. Not end-to-end latency.',
                  speech=[], invitations=[])
    requests = []
    responses = []

    def transport(endpoint, payload, timeout):
        if _resident_context(url, model) != context:
            raise RuntimeError('Resident model context changed; aborting replay')
        requests.append(payload)
        response = http_chat(endpoint, payload, timeout)
        responses.append(response['message']['content'])
        return response

    settings = Settings(history=25, num_ctx=context, seed=731)
    agent = MediatingAgent(INTERACTIVE_REFERENCE, model=model, ollama_url=url,
                           history_turns=25, reply_mode='sparse', transport=transport)
    ledger = []
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for text in SPEECH:
        telemetry = {}
        started = time.perf_counter()
        decision = agent.interpret(text, settings=settings, telemetry=telemetry,
                                   history_override=ledger)
        row = dict(input=text, decision=decision.metadata(), words=len(decision.prompt.split()),
                   elapsed=time.perf_counter() - started, telemetry=telemetry, payload=requests[-1],
                   raw_response=responses[-1])
        result['speech'].append(row)
        ledger.append(dict(role='user', content=text))
        save()
        print(json.dumps({k: row[k] for k in ('input', 'words', 'elapsed', 'decision')}, ensure_ascii=False), flush=True)
    previous = []
    previous_approaches = []
    for index, kind in enumerate(('nudge', 'embodied', 'nudge', 'embodied')):
        settings = Settings(history=25, num_ctx=context, seed=731 + index)
        telemetry = {}
        started = time.perf_counter()
        try:
            message = agent.invite(None, dict(people=1, objects=[], recent_speech=[],
                recent_invitation_approaches=previous_approaches),
                settings=settings, telemetry=telemetry, previous=previous, kind=kind)
            row = dict(kind=kind, message=message, elapsed=time.perf_counter() - started,
                       telemetry=telemetry, payload=requests[-1], raw_response=responses[-1])
            previous.append(message)
            previous_approaches.append(telemetry.get('invitation_approach', ''))
        except ValueError as exc:
            row = dict(kind=kind, error=str(exc), elapsed=time.perf_counter() - started,
                       telemetry=telemetry, payload=requests[-1], raw_response=responses[-1])
        result['invitations'].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ('payload', 'raw_response')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
