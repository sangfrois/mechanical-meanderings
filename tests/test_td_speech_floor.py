"""Floor/pacing regressions, using the real TD helper and shader expressions offline."""
from types import SimpleNamespace
import math
import pytest

from test_td_speech_controls import rig, visual, advance, values


def test_eight_second_transition_receipt_midpoint_endpoint_and_hold(rig):
    rig.morph.par.Smoothing = 8.
    before = values(rig)
    rig.module.apply_mediation(rig.root, visual(preserve=1.))
    assert values(rig) == before
    rig.module.interpolate_controls(rig.b, rig.clock.now)
    assert values(rig) == before
    advance(rig, 1.)
    assert .75 < float(rig.morph.par.Speechamount) < 1.
    assert values(rig)[2:] != (40, 49)
    advance(rig, 4.)
    assert float(rig.morph.par.Speechamount) == pytest.approx(.75)
    advance(rig, 8.)
    assert values(rig) == (.5, .05, 40, 49)
    advance(rig, 20.)
    assert values(rig) == (.5, .05, 40, 49)


def test_pause_or_slow_frame_does_not_fast_forward_to_target(rig):
    rig.morph.par.Smoothing = 8.
    rig.module.apply_mediation(rig.root, visual(preserve=1.))
    rig.clock.now += 60.
    rig.module.interpolate_controls(rig.b, rig.clock.now)
    assert .99 < float(rig.morph.par.Speechamount) < 1.
    assert values(rig)[2:] == (32, 36)
    assert rig.module._motion is not None


@pytest.mark.parametrize('speech,auto', [(True, False), (False, False), (False, True), (True, True)])
def test_morph_shader_has_point5_floor_even_with_illegal_inputs(rig, speech, auto):
    rig.morph.par.Speechdrive, rig.morph.par.Auto = speech, auto
    expression = rig.shader.par.vec0valuex.expr
    clock = SimpleNamespace(seconds=0.)
    env = {'op': rig.root.op, 'math': math, 'absTime': clock}
    for bad_input in [-100., .0, .12, .49, .5, 1., 100.]:
        rig.morph.par.Amount = bad_input
        rig.morph.par.Speechamount = bad_input
        for frame in range(121):
            clock.seconds = frame / 10.
            assert .5 <= float(eval(expression, env)) <= 1.


def test_every_speech_target_and_interpolation_sample_keeps_point5_floor(rig):
    rig.morph.par.Speechamount = .5
    for preserve in [0., .2, .5, .9, 1.]:
        target = rig.module.speech_targets(visual(preserve=preserve))
        assert .5 <= target['amount'] <= 1.
        rig.module.apply_mediation(rig.root, visual(preserve=preserve))
        for _ in range(241):
            rig.module.interpolate_controls(rig.b, rig.clock.now)
            assert .5 <= float(rig.morph.par.Speechamount) <= 1.
            rig.clock.now += 1. / 60.
    assert not any(name in ('Amount', 'Warp', 'Auto') for name, _ in rig.morph.writes)
