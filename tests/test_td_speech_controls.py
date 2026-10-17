"""Offline control motion tests. No TD process, camera readback, inference, or UDP sends."""
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGIC_FILE = ROOT / 'scripts' / 'td_interactivity.py'
CALLBACK_FILE = None  # Artifact verifier supplies the actual round-tripped callback.

# Existing TD sequence callback's operative body, also exercised with the real DAT.
BASE_CALLBACK = '''import re
def onValueChange(par, prev):
    m = re.match(r'^Tindexblock(\\d+)step$', par.name, re.IGNORECASE)
    if m:
        comp = parent()
        idx = int(m.group(1))
        seq = comp.par.Tindexblock.sequence
        val = par.eval()
        lo = seq[idx - 1].par.Step.eval() + 1 if idx > 0 else None
        hi = seq[idx + 1].par.Step.eval() - 1 if idx < seq.numBlocks - 1 else None
        if lo is not None and hi is not None:
            clamped = max(lo, min(val, hi)) if lo <= hi else lo
        elif lo is not None:
            clamped = max(val, lo)
        elif hi is not None:
            clamped = min(val, hi)
        else:
            clamped = val
        if clamped != val:
            par.val = clamped
            return
    parent().Updatesettings()
'''


def source(path):
    data = Path(path).read_bytes()
    if data.startswith(b'2\n'):
        assert int.from_bytes(data[23:27], 'big') == len(data) - 27
        data = data[27:]
    return data.decode('utf-8')


class Par:
    def __init__(self, owner, name, value):
        self.owner, self.name, self._value = owner, name, value
        self.expr = None
    @property
    def val(self):
        return self._value
    @val.setter
    def val(self, value):
        previous = self._value
        self._value = value
        self.owner.writes.append((self.name, value))
        if self.owner.on_change is not None and previous != value:
            self.owner.on_change(self, previous)
    def eval(self): return self._value
    def __float__(self): return float(self._value)
    def __int__(self): return int(self._value)
    def __bool__(self): return bool(self._value)
    def __gt__(self, other): return self._value > other
    def __lt__(self, other): return self._value < other
    def __rtruediv__(self, other): return other / self._value


class Pars:
    def __init__(self, owner, **values):
        object.__setattr__(self, '_values', {name: Par(owner, name, value) for name, value in values.items()})
    def __getitem__(self, name): return self._values[name]
    def __getattr__(self, name):
        if name not in self._values: raise AttributeError(name)
        return self._values[name]
    def __setattr__(self, name, value): self._values[name].val = value


class Node:
    def __init__(self, **values):
        self.writes, self.on_change = [], None
        self.par = Pars(self, **values)
    def appendCustomPage(self, name): return SimpleNamespace()


@pytest.fixture
def rig():
    module = ModuleType('td_speech_test')
    exec(compile(source(LOGIC_FILE), str(LOGIC_FILE), 'exec'), module.__dict__)
    module.close()  # Imported socket never sends anything in this fixture.
    clock = SimpleNamespace(now=100.)
    module.time = SimpleNamespace(monotonic=lambda: clock.now)
    sends = []
    module._send = lambda b, *args: sends.append(args)
    morph = Node(Amount=1., Warp=.05, Auto=False, Speechamount=1., Speechwarp=.05,
                 Speechdrive=True, Stepsdrive=True, Smoothing=4., Period=8.6)
    stream = Node(Tindexblock0step=32, Tindexblock1step=36, Tindexblock=0)
    blocks = [SimpleNamespace(par=SimpleNamespace(Step=stream.par['Tindexblock'+str(i)+'step'])) for i in range(2)]
    class Sequence(list):
        @property
        def numBlocks(self): return len(self)
    stream.par.Tindexblock.sequence = Sequence(blocks)
    updates = []
    stream.Updatesettings = lambda: updates.append((int(stream.par.Tindexblock0step), int(stream.par.Tindexblock1step)))
    unit = Node(Enabled=True, Controls=True, Agentport=9000)
    text_nodes = {name: SimpleNamespace(text='') for name in ('status', 'message', 'judgement', 'camera_status', 'scene')}
    unit.op = text_nodes.get
    shader = Node(vec0valuex=0., vec0valuey=0.)
    callback = SimpleNamespace(text=source(CALLBACK_FILE) if CALLBACK_FILE else BASE_CALLBACK.replace('    ', '\t'))
    nodes = {'interactivity': unit, 'morph_controls': morph, 'StreamDiffusionTD': stream,
             'source_morph': shader, 'StreamDiffusionTD/parexec_Updatelive': callback}
    root = SimpleNamespace(op=nodes.get)
    unit.parent = lambda: root
    spec = importlib.util.spec_from_file_location('speech_installer', ROOT / 'scripts/install_td_interactivity.py')
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    installer.install_speech_controls(root)
    cb_namespace = {'parent': lambda: stream}
    exec(compile(callback.text, 'step-callback', 'exec'), cb_namespace)
    stream.on_change = cb_namespace['onValueChange']
    return SimpleNamespace(module=module, root=root, b=unit, morph=morph, stream=stream,
        clock=clock, sends=sends, updates=updates, shader=shader, nodes=text_nodes, installer=installer)


def visual(preserve=.2, intent=1., specificity=.9):
    strength = (1. - preserve) * (.65 + .35 * max(intent, specificity))
    return dict(apply_visual=True, preserve_identity=preserve, transformation_intent=intent,
        visual_specificity=specificity, controls=dict(amount=.12 + .78 * strength, warp=.05 + .7 * strength))


def advance(rig, elapsed):
    end = 100. + elapsed
    while rig.clock.now + 1. / 60. < end:
        rig.clock.now += 1. / 60.
        rig.module.interpolate_controls(rig.b, rig.clock.now)
    rig.clock.now = end
    rig.module.interpolate_controls(rig.b, end)


def values(rig):
    return tuple(float(rig.morph.par[name]) for name in ('Speechamount', 'Speechwarp')) + tuple(
        int(rig.stream.par[name]) for name in ('Tindexblock0step', 'Tindexblock1step'))


def test_receipt_queues_without_snapping_or_touching_manual_controls(rig):
    before = values(rig)
    target = rig.module.apply_mediation(rig.root, visual())
    assert values(rig) == before and not rig.morph.writes
    assert target['amount'] == pytest.approx(.9)
    advance(rig, 2.)
    assert float(rig.morph.par.Speechamount) == pytest.approx(.95)
    assert not any(name in ('Amount', 'Warp', 'Auto') for name, _ in rig.morph.writes)
    assert (float(rig.morph.par.Amount), float(rig.morph.par.Warp), bool(rig.morph.par.Auto)) == (1., .05, False)


def test_all_control_targets_reached_exactly_then_held(rig):
    target = rig.module.apply_mediation(rig.root, visual(preserve=0.))
    for frame in range(241): advance(rig, frame / 60.)
    assert values(rig) == pytest.approx((target['amount'], target['warp'], 10, 31))
    assert rig.module._motion is None
    writes = len(rig.stream.writes) + len(rig.morph.writes)
    advance(rig, 20.)
    assert len(rig.stream.writes) + len(rig.morph.writes) == writes


def test_identity_preserving_scene_removes_old_point12_floor(rig):
    target = rig.module.apply_mediation(rig.root, visual(preserve=1.))
    assert target['amount'] == .5 and (target['step0'], target['step1']) == (40., 49.)
    advance(rig, 4.)
    assert values(rig) == (.5, .05, 40, 49)
    assert float(rig.morph.par.Amount) == 1.


def test_step_mapping_bounds_order_and_semantic_direction(rig):
    samples = [rig.module.speech_targets(visual(preserve=i / 1000.)) for i in range(1001)]
    for result in samples:
        first, second = rig.module.bounded_steps(result['step0'], result['step1'])
        assert 10 <= first < second <= 49 and second > 30
    assert samples[0]['step0'] < samples[-1]['step0']
    assert samples[0]['step1'] < samples[-1]['step1']


def test_every_interpolation_frame_and_backend_update_stays_in_bounds(rig):
    for start, end in [(100., visual(preserve=0.)), (104., visual(preserve=1.)), (108., visual(preserve=.2))]:
        rig.clock.now = start
        rig.module.apply_mediation(rig.root, end)
        for frame in range(241):
            rig.clock.now = start + frame / 60.
            rig.module.interpolate_controls(rig.b, rig.clock.now)
            a, b = values(rig)[2:]
            assert 10 <= a < b <= 49 and b > 30
    assert rig.updates and all(10 <= a < b <= 49 and b > 30 for a, b in rig.updates)
    assert len(rig.stream.par.Tindexblock.sequence) == 2


def test_retargeting_begins_at_current_values_without_jump(rig):
    rig.module.apply_mediation(rig.root, visual(preserve=1.))
    advance(rig, 2.)
    current = values(rig)
    rig.module.apply_mediation(rig.root, visual(preserve=0.))
    assert values(rig) == current
    rig.module.interpolate_controls(rig.b, rig.clock.now)
    assert values(rig) == current
    advance(rig, 6.)
    assert values(rig) == (1., .75, 10, 31)


def test_reply_only_question_and_invalid_json_do_not_replace_target(rig):
    rig.module.apply_mediation(rig.root, visual())
    target = rig.module._motion
    rig.module.handle(rig.root, '/mediation', [json.dumps(dict(apply_visual=False))])
    assert rig.module._motion is target
    assert json.loads(rig.nodes['judgement'].text)['apply_visual'] is False
    rig.module.handle(rig.root, '/mediation', ['{"apply_visual":'])
    assert rig.module._motion is target
    assert 'JSONDecodeError' in rig.nodes['status'].text


@pytest.mark.parametrize('field,value', [('preserve_identity', True), ('transformation_intent', float('nan')),
    ('visual_specificity', -1.), ('visual_specificity', 2.)])
def test_invalid_analysis_preserves_last_motion(rig, field, value):
    rig.module.apply_mediation(rig.root, visual())
    motion = rig.module._motion
    payload = visual(); payload[field] = value
    rig.module.handle(rig.root, '/mediation', [json.dumps(payload)])
    assert rig.module._motion is motion
    assert 'ValueError' in rig.nodes['status'].text


def test_step_automation_toggle_leaves_manual_steps_alone(rig):
    rig.morph.par.Stepsdrive = False
    rig.module.apply_mediation(rig.root, visual(preserve=0.))
    advance(rig, 4.)
    assert values(rig)[2:] == (32, 36) and not rig.stream.writes
    assert values(rig)[:2] == (1., .75)


@pytest.mark.parametrize('name', ['Enabled', 'Controls'])
def test_disabling_automation_stops_transition_without_reset(rig, name):
    rig.module.apply_mediation(rig.root, visual())
    advance(rig, 2.)
    previous = values(rig)
    rig.b.par[name].val = False
    advance(rig, 3.)
    assert values(rig) == previous and rig.module._motion is None


def test_actual_callback_clamps_manual_first_and_second_steps(rig):
    rig.stream.par.Tindexblock0step = -100
    assert int(rig.stream.par.Tindexblock0step) == 10
    rig.stream.par.Tindexblock1step = 0
    assert int(rig.stream.par.Tindexblock1step) == 31
    rig.stream.par.Tindexblock1step = 100
    assert int(rig.stream.par.Tindexblock1step) == 49


def test_write_order_handles_increasing_first_past_old_second(rig):
    rig.module._write_steps(rig.stream, 10, 31)
    rig.stream.writes.clear()
    rig.module._write_steps(rig.stream, 40, 49)
    assert rig.stream.writes == [('Tindexblock1step', 49), ('Tindexblock0step', 40)]
    assert values(rig)[2:] == (40, 49)


def test_shader_selects_speech_or_manual_without_changing_original_breathing(rig):
    env = {'op': rig.root.op, 'math': math, 'absTime': SimpleNamespace(seconds=2.)}
    rig.morph.par.Speechamount = .7
    rig.morph.par.Speechwarp = .2
    x, y = (rig.shader.par[name].expr for name in ('vec0valuex', 'vec0valuey'))
    assert float(eval(x, env)) == .7 and float(eval(y, env)) == .2
    rig.morph.par.Speechdrive = False
    assert float(eval(x, env)) == 1. and float(eval(y, env)) == .05
    rig.morph.par.Auto = True
    expected = min(1., max(.5, .5 - .62 * math.cos(2 * math.pi * 2. / 8.6)))
    assert eval(x, env) == pytest.approx(expected)


def test_status_retains_scene_reason_and_control_progress(rig):
    rig.module.apply_mediation(rig.root, visual())
    advance(rig, 1.)
    rig.module.set_status(rig.b, 'Waiting for a new YOLO detector frame; no scene sent')
    assert 'Waiting for' in rig.nodes['status'].text and 'Speech Morph' in rig.nodes['status'].text


def test_tick_interpolates_between_one_second_scene_polls(rig):
    rig.module.apply_mediation(rig.root, visual(preserve=1.))
    rig.module._last_poll = 100.
    rig.module._expires = 100.05
    rig.nodes['message'].text = 'old caption'
    rig.clock.now = 100.1
    rig.module.tick(rig.b)
    assert .5 < float(rig.morph.par.Speechamount) < 1.
    assert rig.nodes['message'].text == ''
    assert not rig.sends


def test_invalid_duration_is_rejected():
    module = ModuleType('td_transition_test')
    exec(source(LOGIC_FILE), module.__dict__)
    module.close()
    for seconds in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError): module.ControlTransition({'amount': 1.}, {'amount': 0.}, 100., seconds)
