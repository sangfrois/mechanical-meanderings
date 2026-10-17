import json
import math
from types import SimpleNamespace
import pytest
from test_td_interactivity import helper


def test_agent_design_overrides_rotation_and_preserves_play():
    m = helper()
    try:
        style = dict(font='Consolas', size=36, x=.73, y=.22, align='right', bold=True,
            italic=False, tracking=1, color=[1, .7, .1], line_width=40, decoration='swung')
        a = m.caption_layout('réel / irréel ~ oui', 0, 1280, 720, typography=style)
        b = m.caption_layout('réel / irréel ~ oui', 99, 1280, 720, typography=json.dumps(style))
        assert a == b
        assert a['text'] == '⁓ réel / irréel ~ oui ⁓'
        assert a['requested_center'] == [.73, .22]
        assert a['font'] == 'Consolas' and a['bold'] and a['align'] == 'right'
        assert a['x'] == pytest.approx(.23) and a['y'] == pytest.approx(.28)
    finally:
        m.close()


@pytest.mark.parametrize('bad', [dict(x=float('nan')), dict(size=True), dict(y=2),
    dict(color=[1, 0, float('inf')]), dict(font='C:/evil.ttf'), dict(align='eval'),
    dict(italic='true'), dict(fontfile='anything'), dict(decoration='script')])
def test_reject_untrusted_design_fields(bad):
    m = helper()
    try:
        with pytest.raises(ValueError):
            m.parse_typography(bad)
    finally:
        m.close()


def test_detector_ref_survives_renamed_component():
    m = helper()
    target = object()
    b = SimpleNamespace(par=SimpleNamespace(Yolo=SimpleNamespace(eval=lambda: target)))
    root = SimpleNamespace(op=lambda name: b if name == 'interactivity' else None)
    try:
        assert m.detector_component(root) is target
    finally:
        m.close()


def test_broken_explicit_detector_ref_does_not_silently_choose_other_camera():
    m = helper()
    b = SimpleNamespace(par=SimpleNamespace(Yolo=SimpleNamespace(eval=lambda: None)))
    root = SimpleNamespace(op=lambda name: b if name == 'interactivity' else object())
    try:
        with pytest.raises(ValueError, match='configured YOLO'):
            m.detector_component(root)
    finally:
        m.close()
