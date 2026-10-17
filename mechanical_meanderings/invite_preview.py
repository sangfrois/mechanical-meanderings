"""Opt-in invitation-only replay: resident Ollama, never OSC or camera."""
from copy import deepcopy
import http.client
import json
import time
import urllib.request

from . import GenerationControl, GenerationTimeout, StaleGeneration, http_chat
from .invitation import InvitationPromptError, KINDS


def resident_context(url, model):
    """Read-only discovery; unavailable or malformed residency fails closed."""
    try:
        with urllib.request.urlopen(url.rstrip('/') + '/api/ps', timeout=3) as response:
            if response.status != 200:
                raise RuntimeError('Cannot verify resident model: /api/ps was not successful')
            value = json.load(response)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise RuntimeError('Cannot verify resident model: /api/ps failed') from exc
    if (not isinstance(value, dict) or not isinstance(value.get('models'), list)
            or not all(isinstance(row, dict) for row in value['models'])):
        raise RuntimeError('Cannot verify resident model: malformed /api/ps response')
    matches = [row for row in value['models'] if row.get('name') == model]
    context = matches[0].get('context_length') if len(matches) == 1 else None
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise RuntimeError('Selected model must already be resident with known context length')
    return context


def _terminal_error(exc, request_failed=False):
    """Explain local failures without echoing HTTP bodies or server reasoning."""
    if (isinstance(exc, (GenerationTimeout, StaleGeneration, InvitationPromptError))
            or not request_failed and isinstance(exc, (ValueError, RuntimeError))):
        message = str(exc)
    elif isinstance(exc, OSError):
        message = 'Invitation preview request failed due to a network or I/O error'
    elif isinstance(exc, http.client.HTTPException):
        message = 'Invitation preview HTTP request failed'
    else:
        message = 'Invitation preview request failed' if request_failed else 'Invitation preview failed'
    return dict(type=type(exc).__name__, message=message or 'Invitation preview failed')


def preview_invitations(agent, settings, *, kind='nudge', language='auto', count=1, budget=120.):
    """Capture actual model requests/responses and rejected samples, not curated wins.

    Recheck residency before every request and abort on change. Discovery and
    chat are NOT atomic: Ollama cannot guarantee no reload in the intervening
    eviction race. Use only with operator authorization to consume inference.
    """
    if kind not in KINDS:
        raise ValueError('invalid engagement kind')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 12:
        raise ValueError('invite-count must be between 1 and 12')
    base_url = agent.url.rsplit('/api/', 1)[0]
    context = None
    requests, responses, previous, approaches = [], [], [], []
    original = agent.transport
    payload_kind = 'wire' if original is http_chat else 'logical'
    request_failed = False
    result = dict(model=agent.model, context_length=None, kind=kind, language=language,
                  note='Actual Ollama requests; no OSC or camera. Not installation latency.',
                  status='complete', terminal_error=None, samples=[])

    def capture(endpoint, payload, timeout, control=None):
        nonlocal request_failed
        request_payload = dict(payload, stream=True) if original is http_chat and control is not None else payload
        # Capture before residency validation too: a refused request still has an available payload.
        requests.append(dict(payload=deepcopy(request_payload), logical_payload=deepcopy(payload)))
        if resident_context(base_url, agent.model) != context:
            raise RuntimeError('Resident model context changed; aborting invitation preview')
        # Preserve the production transport's bounded HTTP path for a preview too.
        try:
            response = agent._request(request_payload, control, transport=original, url=endpoint, timeout=timeout)
        except Exception:
            request_failed = True
            raise
        responses.append(response['message']['content'])
        return response

    try:
        # Validate the ENTIRE seed sequence before even read-only model discovery.
        sample_settings = [(settings.updated(seed=settings.seed + index)
                            if settings.seed is not None else settings) for index in range(count)]
        GenerationControl(budget)  # Validate the budget before discovery; each sample gets its own deadline.
        agent.invitation_prompt.read()  # Prompt failures belong in this report, not the CLI's outer handler.
        context = resident_context(base_url, agent.model)
        result['context_length'] = context
        if settings.num_ctx is not None and settings.num_ctx != context:
            raise RuntimeError('Preview num-ctx must match the already resident context; no context reload requested')
        sample_settings = [sample.updated(num_ctx=context) for sample in sample_settings]
        agent.transport = capture
        for index, sample in enumerate(sample_settings):
            telemetry, design = {}, {}
            request_failed = False
            started = time.perf_counter()
            request_start, response_start = len(requests), len(responses)
            row = dict(index=index + 1)
            try:
                message = agent.invite(None, dict(people=1, objects=[], recent_speech=[],
                    recent_invitation_approaches=approaches), settings=sample,
                    telemetry=telemetry, design=design, language=language, previous=previous,
                    kind=kind, control=GenerationControl(budget))
                row.update(message=message, typography=design, accepted=True)
                previous.append(message)
                approaches.append(telemetry.get('invitation_approach', ''))
            except Exception as exc:
                if isinstance(exc, ValueError) and not isinstance(exc, InvitationPromptError) and not request_failed:
                    # Wording/parser rejections are samples, not terminal runtime failures.
                    row.update(error=str(exc), accepted=False)
                else:
                    error = _terminal_error(exc, request_failed)
                    result.update(status='aborted', terminal_error=error)
                    row.update(error=error['message'], accepted=False)
            finally:
                captured = requests[-1] if len(requests) > request_start else {}
                row.update(telemetry=telemetry, elapsed=time.perf_counter() - started,
                           payload=captured.get('payload'), logical_payload=captured.get('logical_payload'),
                           payload_kind=payload_kind,
                           raw_response=responses[-1] if len(responses) > response_start else None)
                result['samples'].append(row)
            if result['status'] == 'aborted':
                break
    except Exception as exc:
        result.update(status='aborted', terminal_error=_terminal_error(exc))
    finally:
        agent.transport = original
    return result
