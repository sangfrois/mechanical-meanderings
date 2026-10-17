"""Validate the shared OSC typography contract without a rendering dependency."""
import json
import pytest
from mechanical_meanderings.mediation import Decision, MediatingAgent, parse_decision
from mechanical_meanderings.typography import BOUNDS, FONTS, default_typography, parse_typography
from test_mediation import judgement

STYLE = dict(font='Georgia', size=43, x=.24, y=.29, align='left', italic=True, bold=False,
             tracking=2.5, color=[.25, .75, 1.], line_width=22, decoration='swung')


def test_model_design_and_typographic_marks_are_preserved():
    result = parse_decision(json.dumps(judgement(typography=STYLE, reply='~ a face / a thought // ⁓')))
    assert result.typography == STYLE
    assert result.reply == '~ a face / a thought // ⁓'
    assert result.metadata()['typography'] == STYLE


def test_legacy_decision_and_missing_style_have_independent_defaults():
    one = Decision(**judgement(), controls=None)
    two = parse_decision(json.dumps(judgement()))
    assert not one.needs_camera and one.typography == default_typography()
    one.typography['color'][0] = 0
    assert two.typography == default_typography()
    assert parse_typography(dict(font='Impact'))['font'] == 'Impact'


@pytest.mark.parametrize('font', FONTS)
def test_exact_font_allowlist(font):
    assert parse_typography(dict(font=font))['font'] == font


@pytest.mark.parametrize('key,bounds', BOUNDS.items())
def test_numeric_bounds_are_inclusive(key, bounds):
    for value in bounds:
        assert parse_typography({key: value})[key] == value
    for value in (bounds[0] - 1, bounds[1] + 1, True, '1', float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            parse_typography({key: value})


@pytest.mark.parametrize('style', [[], 'Verdana', {'unknown': 1}, {'font': 'Comic Sans'},
    {'font': []}, {'align': 'top'}, {'decoration': 'rainbow'}, {'italic': 1}, {'bold': 'true'},
    {'line_width': 22.5}, {'color': [1, 1]}, {'color': [1, 1, 1, 1]}, {'color': 'white'},
    {'color': [True, 0, 0]}, {'color': [0, float('nan'), 0]}, {'color': [0, 0, 1.01]}])
def test_malformed_wire_styles_rejected_but_model_reply_survives(style):
    with pytest.raises(ValueError):
        parse_typography(style)
    # Optional model design is repaired; strict OSC validation remains fail-closed.
    decision = parse_decision(json.dumps(judgement(typography=style)))
    assert decision.typography == default_typography()
    assert decision.reply == judgement()['reply']
    assert decision.prompt == judgement()['prompt']
    assert parse_typography(decision.typography) == decision.typography


def test_invite_keeps_string_api_and_collects_model_style_even_after_wording_fallback():
    requests = []
    def transport(url, payload, timeout):
        requests.append(payload)
        return {'message': {'content': json.dumps(dict(message='overly cryptic reflection',
                                                       visible_detail='', typography=STYLE))}}
    agent = MediatingAgent('style', transport=transport)
    design = {'old': 'discard'}
    telemetry = {}
    first = agent.invite(None, {}, design=design, telemetry=telemetry, language='English')
    second_design = {}
    second = agent.invite(None, {}, design=second_design, language='English', previous=[first])
    assert isinstance(first, str) and first != second
    assert design == second_design == STYLE  # No automatic rotating type treatments.
    assert not telemetry['image_attached'] and telemetry['frame_status'] == 'metadata_only'
    assert 'typography' in requests[0]['format']['required']
    assert all('images' not in message for request in requests for message in request['messages'])
    assert not agent.history
